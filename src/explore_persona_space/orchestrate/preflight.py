"""Pre-flight checks for experiments. Run before starting ANY training or eval.

Usage:
    # As a module
    from explore_persona_space.orchestrate.preflight import require_preflight
    require_preflight()

    # From CLI
    uv run python -m explore_persona_space.orchestrate.preflight
"""

import importlib.metadata as importlib_metadata
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# --- VENV INVARIANT (issue #76) ---
# Canonical venv path for experiment runs on pods. Any pipeline entrypoint
# sourcing a different venv (e.g. the stale /workspace/make-evil-dumb/.venv
# from the pre-canonicalization era) is a bug. Check A enforces this on any
# host with a /workspace directory.
EPS_VENV = Path("/workspace/explore-persona-space/.venv")

# Stale venv dirs that issue #76 tracks. pod1 uses underscore variant
# (/workspace/make_evil_dumb/), pod2/3/5 use hyphen variant
# (/workspace/make-evil-dumb/). Both must be absent after cleanup.
STALE_WORKSPACE_DIRS = ("make-evil-dumb", "make_evil_dumb")

# Libraries whose version drift is reported by Check C. All are WARN-only per
# issue #76 plan §3 (the hard-fail invariant lives in Checks A + B — if the
# correct venv is active, drift is exceptional and worth a warning; if the
# wrong venv is active, Check A catches it before drift detection matters).
# See Decision Rationale in .claude/plans/issue-76.md for why WARN-only was
# chosen over hard-fail (flash-attn and liger-kernel have legitimate
# pod-specific builds that would false-positive a hard-fail).
DRIFT_CRITICAL_LIBS: tuple[str, ...] = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "deepspeed",
    "accelerate",
)

# Additional libraries tracked in drift check but expected to vary across
# pods (flash-attn / liger-kernel ship in the `gpu` optional extra and are
# not installed on every pod). Always WARN-only.
DRIFT_OPTIONAL_LIBS: tuple[str, ...] = (
    "flash-attn",
    "liger-kernel",
)

# Escape hatch env var. When set to "1", Check A and Check B degrade to
# warnings instead of errors. Document uses in the PR; this exists solely
# for emergency rollback windows. See CLAUDE.md "Pre-Launch Protocol".
SKIP_VENV_CHECK_ENV = "PREFLIGHT_SKIP_VENV_CHECK"


@dataclass
class PreflightReport:
    """Result of pre-flight checks."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gpu_info: list[dict] = field(default_factory=list)
    disk_free_gb: float = 0.0
    git_status: str = ""
    env_synced: bool = True

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        """Formatted summary string."""
        lines = []
        status = "PASS" if self.ok else "FAIL"
        lines.append(f"\n{'=' * 60}")
        lines.append(f"  Pre-flight Check: {status}")
        lines.append(f"{'=' * 60}")

        if self.errors:
            lines.append("\n  ERRORS (must fix before running):")
            for e in self.errors:
                lines.append(f"    ✗ {e}")

        if self.warnings:
            lines.append("\n  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")

        if self.gpu_info:
            lines.append("\n  GPUs:")
            for g in self.gpu_info:
                used = g.get("memory_used_mb", 0)
                total = g.get("memory_total_mb", 0)
                free = g.get("memory_free_mb", 0)
                procs = g.get("processes", 0)
                status_icon = "✓" if procs == 0 and used < 1000 else "⚠"
                lines.append(
                    f"    {status_icon} GPU {g['id']}: "
                    f"{free:,}MB free / {total:,}MB total "
                    f"({procs} processes)"
                )

        lines.append(f"\n  Disk: {self.disk_free_gb:.1f} GB free")
        lines.append(f"  Git: {self.git_status}")
        lines.append(f"  Env synced: {'yes' if self.env_synced else 'NO'}")
        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    """Run a command with timeout. Returns (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return -1, "", f"command not found: {cmd[0]}"
    except Exception as e:
        return -1, "", str(e)


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    p = Path(__file__).resolve()
    for parent in [p, *list(p.parents)]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def check_git_status(report: PreflightReport, project_root: Path):
    """Check git working tree is clean and up to date."""
    # Check for uncommitted changes
    rc, out, err = _run(["git", "-C", str(project_root), "status", "--porcelain"])
    if rc != 0:
        report.add_warning(f"git status failed: {err}")
        report.git_status = "unknown"
        return

    if out:
        changed = len(out.strip().splitlines())
        report.add_warning(f"{changed} uncommitted change(s) — consider committing first")
        report.git_status = f"{changed} uncommitted changes"
    else:
        report.git_status = "clean"

    # Check if behind remote
    _run(["git", "-C", str(project_root), "fetch", "--quiet", "origin"], timeout=15)
    rc, out, _ = _run(["git", "-C", str(project_root), "rev-list", "--count", "HEAD..origin/main"])
    if rc == 0 and out.strip() != "0":
        behind = out.strip()
        report.add_error(
            f"Local is {behind} commit(s) behind origin/main. Run: git pull origin main"
        )
        report.git_status += f", {behind} behind remote"


def check_env_sync(report: PreflightReport, project_root: Path):
    """Check that installed packages match uv.lock."""
    lockfile = project_root / "uv.lock"
    if not lockfile.exists():
        report.add_warning("No uv.lock found — cannot verify environment sync")
        report.env_synced = False
        return

    # uv sync --locked --dry-run exits non-zero if env needs changes
    rc, out, err = _run(
        ["uv", "sync", "--locked", "--dry-run"],
        timeout=30,
    )
    if rc != 0:
        if "would install" in err.lower() or "would install" in out.lower():
            report.add_error("Environment out of sync with uv.lock. Run: uv sync --locked")
            report.env_synced = False
        elif "error" in err.lower():
            report.add_warning(f"uv sync check failed: {err[:200]}")
            report.env_synced = False
        else:
            # Non-zero exit could mean changes needed
            report.add_warning(
                "uv sync --locked --dry-run returned non-zero. Environment may be out of sync."
            )
            report.env_synced = False


# ----------------------------------------------------------------------------
# Issue #76 — venv invariant checks
# ----------------------------------------------------------------------------


def check_active_venv(
    report: PreflightReport,
    expected_venv: Path = EPS_VENV,
    workspace_root: Path = Path("/workspace"),
) -> None:
    """Verify the active Python venv is ``expected_venv`` on /workspace hosts.

    On hosts without a ``/workspace`` directory (local VMs, CI), this check is
    a no-op — the invariant is a pod-specific constraint.

    When ``PREFLIGHT_SKIP_VENV_CHECK=1`` is set, a hard-fail is demoted to a
    warning. Use only during emergency rollback windows; document the
    decision in the relevant GitHub issue.

    Source of ground truth for the active venv:
      1. ``VIRTUAL_ENV`` env var (set by ``source .venv/bin/activate``).
      2. Falls back to ``sys.prefix`` if ``VIRTUAL_ENV`` is unset.
    """
    if not workspace_root.exists():
        return  # Local VM / CI — pod invariant does not apply.

    expected = expected_venv.resolve()
    actual_str = os.environ.get("VIRTUAL_ENV") or sys.prefix
    actual = Path(actual_str).resolve()

    if actual == expected:
        return

    msg = (
        f"Active venv is {actual}, expected {expected}. "
        f"Fix: deactivate; source {expected}/bin/activate"
    )
    if os.environ.get(SKIP_VENV_CHECK_ENV) == "1":
        report.add_warning(f"[{SKIP_VENV_CHECK_ENV}=1] {msg}")
    else:
        report.add_error(msg)


def check_make_evil_dumb_absent(
    report: PreflightReport,
    workspace_root: Path = Path("/workspace"),
) -> None:
    """Verify the stale make-evil-dumb venv is absent on /workspace hosts.

    HARD FAIL when ``/workspace/<variant>/.venv`` exists: that directory is
    the direct source of the library-drift incident documented in issue #76
    (pod2's ``run_evil_correct_full_seed137.sh`` did a literal PATH prepend
    of that venv, silently using transformers 5.5.3 / trl 1.0.0 instead of
    the locked 5.5.0 / 0.29.1).

    WARN-only when the parent directory exists but the ``.venv`` subdir does
    not — the venv is the load-bearing piece; the surrounding code/data dir
    is harmless until someone resurrects the venv.

    Variants checked: ``make-evil-dumb`` (hyphen; pod2/3/5) and
    ``make_evil_dumb`` (underscore; pod1).
    """
    if not workspace_root.exists():
        return

    skip = os.environ.get(SKIP_VENV_CHECK_ENV) == "1"

    for variant in STALE_WORKSPACE_DIRS:
        stale_venv = workspace_root / variant / ".venv"
        if stale_venv.exists():
            msg = (
                f"{stale_venv} exists — stale venv from issue #76 "
                f"(silent source of transformers/trl drift). "
                f"Remove: rm -rf {stale_venv}"
            )
            if skip:
                report.add_warning(f"[{SKIP_VENV_CHECK_ENV}=1] {msg}")
            else:
                report.add_error(msg)
            continue  # If venv present, don't double-warn about parent dir.

        stale_dir = workspace_root / variant
        if stale_dir.exists():
            report.add_warning(
                f"{stale_dir} dir still present (venv absent — safe). "
                f"Consider cleanup per issue #76 after artifact audit."
            )


def _parse_uv_lock_versions(lockfile: Path, names: tuple[str, ...]) -> dict[str, str]:
    """Scan ``uv.lock`` for ``name = "<x>"`` / ``version = "<v>"`` pairs.

    Only returns entries where BOTH the name and version line appear
    adjacently. Unknown names simply don't appear in the output dict.

    Raises:
        FileNotFoundError: if ``lockfile`` doesn't exist.
        OSError: for other I/O errors on the file.
    """
    content = lockfile.read_text(encoding="utf-8")
    name_set = set(names)
    result: dict[str, str] = {}

    # Each package block in uv.lock starts with: name = "<x>"\nversion = "<v>"
    # We scan linearly; this matches the format verified at lines
    # 4897/4898 (torch), 5010/5011 (transformers), 5044/5045 (trl).
    name_re = re.compile(r'^name\s*=\s*"([^"]+)"\s*$')
    version_re = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')

    lines = content.splitlines()
    for i, line in enumerate(lines):
        m = name_re.match(line)
        if m is None:
            continue
        pkg = m.group(1)
        if pkg not in name_set:
            continue
        # Look at the next non-empty line for the version pin.
        for j in range(i + 1, min(i + 4, len(lines))):
            vm = version_re.match(lines[j])
            if vm is not None:
                result[pkg] = vm.group(1)
                break
    return result


def check_library_drift(
    report: PreflightReport,
    project_root: Path,
    critical_libs: tuple[str, ...] = DRIFT_CRITICAL_LIBS,
    optional_libs: tuple[str, ...] = DRIFT_OPTIONAL_LIBS,
) -> None:
    """Warn when installed library versions disagree with uv.lock pins.

    WARN-only by design (see ``DRIFT_CRITICAL_LIBS`` docstring and issue #76
    plan §3). The hard invariant is enforced by Checks A + B: if the right
    venv is active, drift is reported so operators notice; it does not
    block runs.

    Args:
        report: Report to attach warnings to.
        project_root: Directory containing ``uv.lock``.
        critical_libs: Core ML libs (torch, transformers, trl, etc.) — always
            checked; drift triggers a warning mentioning both versions.
        optional_libs: GPU-specific libs (flash-attn, liger-kernel). Drift
            triggers a warning ONLY if the package is installed on the host
            (expected to be absent on some pods).
    """
    lockfile = project_root / "uv.lock"
    if not lockfile.exists():
        report.add_warning("Cannot check library drift: uv.lock missing.")
        return

    all_libs = tuple(critical_libs) + tuple(optional_libs)

    try:
        pins = _parse_uv_lock_versions(lockfile, all_libs)
    except OSError as e:
        report.add_warning(f"Could not parse uv.lock for drift check: {e}")
        return

    drifts: list[str] = []
    for name in critical_libs:
        pinned = pins.get(name)
        if pinned is None:
            # Library is expected in uv.lock but not found — surface as a
            # separate warning; do not try to read installed version.
            drifts.append(f"{name}: not found in uv.lock (expected pin)")
            continue
        try:
            actual = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            drifts.append(f"{name}: NOT INSTALLED (uv.lock pins {pinned})")
            continue
        if actual != pinned:
            drifts.append(f"{name}: installed={actual}, uv.lock={pinned}")

    for name in optional_libs:
        try:
            actual = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            # Optional libs absent is fine — skip silently.
            continue
        pinned = pins.get(name)
        if pinned is None:
            drifts.append(f"{name}: installed={actual}, uv.lock=(no pin)")
            continue
        if actual != pinned:
            drifts.append(f"{name}: installed={actual}, uv.lock={pinned}")

    if drifts:
        report.add_warning("Library version drift from uv.lock (warn-only): " + "; ".join(drifts))


def check_disk_space(report: PreflightReport, min_free_gb: float):
    """Check available disk space on /workspace (or /)."""
    check_path = "/workspace" if Path("/workspace").exists() else "/"
    try:
        usage = shutil.disk_usage(check_path)
        report.disk_free_gb = usage.free / (1024**3)
        if report.disk_free_gb < min_free_gb:
            report.add_error(
                f"Only {report.disk_free_gb:.1f}GB free on {check_path} "
                f"(need {min_free_gb:.0f}GB). Clean up models/checkpoints."
            )
        elif report.disk_free_gb < min_free_gb * 2:
            report.add_warning(f"{report.disk_free_gb:.1f}GB free on {check_path} — getting low")
    except Exception as e:
        report.add_warning(f"Could not check disk space: {e}")


def check_gpus(report: PreflightReport, require_gpu: bool, min_free_mb: int):
    """Check GPU availability and memory."""
    rc, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0:
        if require_gpu:
            report.add_error(f"nvidia-smi failed: {err}. No GPUs available?")
        else:
            report.add_warning("nvidia-smi not available (no GPU)")
        return

    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_id, total, used, free = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

        # Check for processes on this GPU
        prc, pout, _ = _run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ]
        )
        n_procs = len([x for x in pout.splitlines() if x.strip()]) if prc == 0 else 0

        gpu_info = {
            "id": gpu_id,
            "memory_total_mb": total,
            "memory_used_mb": used,
            "memory_free_mb": free,
            "processes": n_procs,
        }
        report.gpu_info.append(gpu_info)

        # Zombie detection: memory used but no processes
        if used > 5000 and n_procs == 0:
            report.add_warning(
                f"GPU {gpu_id}: {used}MB used but no processes — "
                f"possible zombie. Fix: restart container."
            )

    # Check if any GPU has enough free memory
    if require_gpu:
        max_free = max((g["memory_free_mb"] for g in report.gpu_info), default=0)
        if max_free < min_free_mb:
            report.add_error(
                f"No GPU with {min_free_mb:,}MB free (best: {max_free:,}MB). "
                f"Wait for running jobs or use a different pod."
            )


def check_hf_home(report: PreflightReport):
    """Check HF_HOME is set to the canonical cache path."""
    hf_home = os.environ.get("HF_HOME", "")
    expected = "/workspace/.cache/huggingface"

    if Path("/workspace").exists():
        if not hf_home:
            report.add_warning(
                "HF_HOME not set. Setting to /workspace/.cache/huggingface. "
                "Call load_dotenv() or source env_setup.sh first."
            )
            os.environ["HF_HOME"] = expected
        elif hf_home != expected:
            report.add_warning(
                f"HF_HOME={hf_home} (expected {expected}). Models may download to wrong location."
            )


def check_env_vars(report: PreflightReport, required: list[str]):
    """Check that required environment variables are set."""
    for var in required:
        val = os.environ.get(var, "")
        if not val:
            report.add_error(f"Missing env var: {var}. Check .env file.")
        elif len(val) < 5:
            report.add_warning(f"Env var {var} looks suspiciously short: '{val[:3]}...'")


def check_connectivity(report: PreflightReport):
    """Quick check that HF Hub and WandB are reachable."""
    # HF Hub
    rc, _, _ = _run(
        [
            "python3",
            "-c",
            "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5)",
        ],
        timeout=10,
    )
    if rc != 0:
        report.add_warning("Cannot reach huggingface.co — model uploads will fail")

    # WandB
    wandb_check = (
        "import urllib.request; urllib.request.urlopen('https://api.wandb.ai/healthz', timeout=5)"
    )
    rc, _, _ = _run(["python3", "-c", wandb_check], timeout=10)
    if rc != 0:
        report.add_warning("Cannot reach api.wandb.ai — result uploads will fail")


def preflight_check(
    require_gpu: bool = True,
    min_disk_gb: float = 50.0,
    min_gpu_free_mb: int = 70_000,
    required_env_vars: list[str] | None = None,
    check_code_sync: bool = True,
) -> PreflightReport:
    """Run all pre-experiment checks.

    Args:
        require_gpu: If True, fail when no GPU has enough free memory.
        min_disk_gb: Minimum free disk space in GB.
        min_gpu_free_mb: Minimum free GPU memory in MB for at least one GPU.
        required_env_vars: Env vars to check. Defaults to standard set.
        check_code_sync: Whether to check git status and env sync.

    Returns:
        PreflightReport with pass/fail status and details.
    """
    if required_env_vars is None:
        required_env_vars = [
            "WANDB_API_KEY",
            "HF_TOKEN",
            "ANTHROPIC_API_KEY",
        ]

    project_root = _find_project_root()
    report = PreflightReport()

    # Load .env first so env var checks work
    try:
        from dotenv import load_dotenv

        load_dotenv(str(project_root / ".env"), override=False)
    except ImportError:
        report.add_warning("python-dotenv not installed — cannot load .env")

    # Set HF_HOME early
    if Path("/workspace").exists():
        os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

    # Run all checks
    if check_code_sync:
        check_git_status(report, project_root)
        check_env_sync(report, project_root)

    # Issue #76 — venv invariant (no-op on non-/workspace hosts)
    check_active_venv(report)
    check_make_evil_dumb_absent(report)
    check_library_drift(report, project_root)

    check_disk_space(report, min_disk_gb)
    check_gpus(report, require_gpu, min_gpu_free_mb)
    check_hf_home(report)
    check_env_vars(report, required_env_vars)
    check_connectivity(report)

    return report


def require_preflight(
    min_disk_gb: float = 50.0,
    require_gpu: bool = True,
    min_gpu_free_mb: int = 70_000,
) -> PreflightReport:
    """Run preflight checks and abort if any critical failures.

    Call at the top of experiment scripts.
    """
    report = preflight_check(
        min_disk_gb=min_disk_gb,
        require_gpu=require_gpu,
        min_gpu_free_mb=min_gpu_free_mb,
    )
    logger.info(report.summary())

    if not report.ok:
        logger.error("Pre-flight check FAILED. Fix errors before running.")
        sys.exit(1)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run pre-flight checks")
    parser.add_argument("--no-gpu", action="store_true", help="Don't require GPU")
    parser.add_argument("--min-disk", type=float, default=50.0, help="Min disk GB")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--pipeline-check",
        action="store_true",
        help="Run integration tests (pytest tests/integration/ -m integration) after preflight",
    )
    args = parser.parse_args()

    report = preflight_check(
        require_gpu=not args.no_gpu,
        min_disk_gb=args.min_disk,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "gpu_info": report.gpu_info,
                    "disk_free_gb": report.disk_free_gb,
                    "git_status": report.git_status,
                    "env_synced": report.env_synced,
                },
                indent=2,
            )
        )
    else:
        logger.info(report.summary())

    if not report.ok:
        sys.exit(1)

    if args.pipeline_check:
        logger.info("Running integration tests...")
        rc, stdout, stderr = _run(
            [sys.executable, "-m", "pytest", "tests/integration/", "-m", "integration", "-x", "-v"],
            timeout=600,
        )
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)
        if rc != 0:
            logger.error("Integration tests FAILED (exit code %d)", rc)
            sys.exit(rc)
        logger.info("Integration tests PASSED")

    sys.exit(0)
