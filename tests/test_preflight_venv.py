"""Unit tests for issue #76 venv-invariant preflight checks.

Covers the three new check functions in
``explore_persona_space.orchestrate.preflight``:
  * :func:`check_active_venv`          — Check A (HARD FAIL)
  * :func:`check_make_evil_dumb_absent` — Check B (HARD FAIL for venv;
                                          WARN for bare dir)
  * :func:`check_library_drift`        — Check C (WARN-only)

These checks enforce the invariant from issue #76: every pipeline run on
every pod sources ``/workspace/explore-persona-space/.venv`` and the
stale ``/workspace/make-evil-dumb/.venv`` is gone.

The tests simulate a ``/workspace`` host using a ``tmp_path`` fixture —
the checks accept ``workspace_root`` as an argument specifically so they
can be exercised off-pod.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from explore_persona_space.orchestrate import preflight
from explore_persona_space.orchestrate.preflight import (
    PreflightReport,
    _parse_uv_lock_versions,
    check_active_venv,
    check_library_drift,
    check_make_evil_dumb_absent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_workspace(tmp_path: Path) -> Path:
    """Create a ``/workspace``-like directory inside ``tmp_path``."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def _fake_venv(workspace_root: Path, name: str = "explore-persona-space") -> Path:
    """Create a fake venv at ``<workspace_root>/<name>/.venv/bin/activate``."""
    venv = workspace_root / name / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("# fake activate")
    return venv


def _write_lock(path: Path, entries: dict[str, str]) -> None:
    """Write a minimal uv.lock-style file with the given name→version pairs."""
    blocks = []
    for name, ver in entries.items():
        blocks.append(
            f"""
[[package]]
name = "{name}"
version = "{ver}"
source = {{ registry = "https://pypi.org/simple" }}
""".strip()
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: Pass case
# ---------------------------------------------------------------------------


def test_checks_pass_when_correct_venv_and_no_stale(tmp_path, monkeypatch):
    """All three checks pass: correct venv active, no stale dir, versions match."""
    ws = _fake_workspace(tmp_path)
    expected_venv = _fake_venv(ws, "explore-persona-space")

    monkeypatch.setenv("VIRTUAL_ENV", str(expected_venv))
    monkeypatch.delenv("PREFLIGHT_SKIP_VENV_CHECK", raising=False)

    report = PreflightReport()
    check_active_venv(report, expected_venv=expected_venv, workspace_root=ws)
    assert report.ok, f"errors={report.errors}"
    assert report.errors == []

    check_make_evil_dumb_absent(report, workspace_root=ws)
    assert report.ok, f"errors={report.errors}"
    assert report.errors == []

    # Library drift with matching versions → no warnings.
    # We pin a single fake library "torch" to avoid depending on what's
    # actually installed in the test env.
    lock = tmp_path / "uv.lock"
    _write_lock(lock, {"torch": "2.8.0"})
    drift_report = PreflightReport()
    with patch.object(preflight.importlib_metadata, "version", return_value="2.8.0"):
        check_library_drift(
            drift_report,
            project_root=tmp_path,
            critical_libs=("torch",),
            optional_libs=(),
        )
    assert drift_report.ok
    assert drift_report.errors == []
    assert drift_report.warnings == []


# ---------------------------------------------------------------------------
# Test 2: Wrong-venv fail
# ---------------------------------------------------------------------------


def test_check_active_venv_fails_when_make_evil_dumb_venv_active(tmp_path, monkeypatch):
    """Fails hard when the active venv is ``/workspace/make-evil-dumb/.venv``."""
    ws = _fake_workspace(tmp_path)
    expected_venv = _fake_venv(ws, "explore-persona-space")
    wrong_venv = _fake_venv(ws, "make-evil-dumb")

    monkeypatch.setenv("VIRTUAL_ENV", str(wrong_venv))
    monkeypatch.delenv("PREFLIGHT_SKIP_VENV_CHECK", raising=False)

    report = PreflightReport()
    check_active_venv(report, expected_venv=expected_venv, workspace_root=ws)

    assert not report.ok
    assert len(report.errors) == 1
    err = report.errors[0]
    # Error mentions both the actual (wrong) and expected (right) paths.
    assert str(wrong_venv) in err
    assert str(expected_venv) in err


def test_check_active_venv_fails_with_any_mismatched_venv(tmp_path, monkeypatch):
    """Also fails for arbitrary unexpected venvs (not just make-evil-dumb)."""
    ws = _fake_workspace(tmp_path)
    expected_venv = _fake_venv(ws, "explore-persona-space")
    other_venv = _fake_venv(ws, "some-other-project")

    monkeypatch.setenv("VIRTUAL_ENV", str(other_venv))
    monkeypatch.delenv("PREFLIGHT_SKIP_VENV_CHECK", raising=False)

    report = PreflightReport()
    check_active_venv(report, expected_venv=expected_venv, workspace_root=ws)

    assert not report.ok
    assert str(other_venv) in report.errors[0]


def test_check_active_venv_noop_off_workspace(tmp_path, monkeypatch):
    """On hosts without /workspace, the check is a no-op (local VM / CI)."""
    # Point workspace_root at a dir that does NOT exist.
    missing = tmp_path / "no-workspace-here"
    monkeypatch.setenv("VIRTUAL_ENV", "/does/not/matter")

    report = PreflightReport()
    check_active_venv(
        report,
        expected_venv=Path("/some/other/venv"),
        workspace_root=missing,
    )

    assert report.ok
    assert report.errors == []
    assert report.warnings == []


# ---------------------------------------------------------------------------
# Test 3: make-evil-dumb fail (both variants)
# ---------------------------------------------------------------------------


def test_check_make_evil_dumb_absent_fails_for_hyphen_variant(tmp_path, monkeypatch):
    """Stale ``make-evil-dumb/.venv`` triggers a hard error."""
    ws = _fake_workspace(tmp_path)
    stale_venv = _fake_venv(ws, "make-evil-dumb")
    monkeypatch.delenv("PREFLIGHT_SKIP_VENV_CHECK", raising=False)

    report = PreflightReport()
    check_make_evil_dumb_absent(report, workspace_root=ws)

    assert not report.ok
    assert len(report.errors) == 1
    assert str(stale_venv) in report.errors[0]
    assert "#76" in report.errors[0]


def test_check_make_evil_dumb_absent_fails_for_underscore_variant(tmp_path, monkeypatch):
    """Stale ``make_evil_dumb/.venv`` (pod1's underscore form) also fails."""
    ws = _fake_workspace(tmp_path)
    stale_venv = _fake_venv(ws, "make_evil_dumb")
    monkeypatch.delenv("PREFLIGHT_SKIP_VENV_CHECK", raising=False)

    report = PreflightReport()
    check_make_evil_dumb_absent(report, workspace_root=ws)

    assert not report.ok
    assert str(stale_venv) in report.errors[0]


def test_check_make_evil_dumb_absent_warns_on_bare_dir(tmp_path, monkeypatch):
    """Parent dir without ``.venv`` inside is a warning, not an error."""
    ws = _fake_workspace(tmp_path)
    (ws / "make-evil-dumb").mkdir()
    # No .venv subdir — just the bare dir.
    monkeypatch.delenv("PREFLIGHT_SKIP_VENV_CHECK", raising=False)

    report = PreflightReport()
    check_make_evil_dumb_absent(report, workspace_root=ws)

    assert report.ok
    assert report.errors == []
    assert len(report.warnings) == 1
    assert "make-evil-dumb" in report.warnings[0]
    assert "dir still present" in report.warnings[0]


def test_check_make_evil_dumb_absent_clean_when_nothing_present(tmp_path, monkeypatch):
    """No stale dir at all → check passes silently."""
    ws = _fake_workspace(tmp_path)
    monkeypatch.delenv("PREFLIGHT_SKIP_VENV_CHECK", raising=False)

    report = PreflightReport()
    check_make_evil_dumb_absent(report, workspace_root=ws)

    assert report.ok
    assert report.errors == []
    assert report.warnings == []


# ---------------------------------------------------------------------------
# Test 4: Library drift — WARN only, never FAIL
# ---------------------------------------------------------------------------


def test_library_drift_warns_but_does_not_fail(tmp_path):
    """Mismatched version on a critical lib triggers a warning, not an error."""
    lock = tmp_path / "uv.lock"
    _write_lock(lock, {"transformers": "5.5.0"})

    report = PreflightReport()
    with patch.object(preflight.importlib_metadata, "version", return_value="5.5.3"):
        check_library_drift(
            report,
            project_root=tmp_path,
            critical_libs=("transformers",),
            optional_libs=(),
        )

    assert report.ok, "Library drift must be WARN-only per plan §3"
    assert report.errors == []
    assert len(report.warnings) == 1
    w = report.warnings[0]
    assert "transformers" in w
    assert "5.5.3" in w
    assert "5.5.0" in w


def test_library_drift_optional_lib_not_installed_is_silent(tmp_path):
    """flash-attn / liger-kernel absence is expected on some pods; no warning."""
    lock = tmp_path / "uv.lock"
    _write_lock(lock, {"flash-attn": "2.8.3"})

    # Real PackageNotFoundError so the check's except-clause actually catches.
    real_exc = preflight.importlib_metadata.PackageNotFoundError

    def fake_version(name):
        raise real_exc(name)

    report = PreflightReport()
    with patch.object(preflight.importlib_metadata, "version", side_effect=fake_version):
        check_library_drift(
            report,
            project_root=tmp_path,
            critical_libs=(),
            optional_libs=("flash-attn",),
        )

    assert report.ok
    assert report.errors == []
    assert report.warnings == []


def test_library_drift_critical_lib_not_installed_warns(tmp_path):
    """A missing critical lib is a warning (not ok to have transformers absent)."""
    lock = tmp_path / "uv.lock"
    _write_lock(lock, {"transformers": "5.5.0"})

    real_exc = preflight.importlib_metadata.PackageNotFoundError

    def fake_version(name):
        raise real_exc(name)

    report = PreflightReport()
    with patch.object(preflight.importlib_metadata, "version", side_effect=fake_version):
        check_library_drift(
            report,
            project_root=tmp_path,
            critical_libs=("transformers",),
            optional_libs=(),
        )

    assert report.ok  # Still WARN-only
    assert report.errors == []
    assert len(report.warnings) == 1
    assert "NOT INSTALLED" in report.warnings[0]


def test_library_drift_missing_uv_lock_warns(tmp_path):
    """No uv.lock → single warning, no crash."""
    report = PreflightReport()
    check_library_drift(
        report,
        project_root=tmp_path,  # No uv.lock here
        critical_libs=("torch",),
        optional_libs=(),
    )

    assert report.ok
    assert report.errors == []
    assert len(report.warnings) == 1
    assert "uv.lock missing" in report.warnings[0]


# ---------------------------------------------------------------------------
# Test 5: Escape hatch — PREFLIGHT_SKIP_VENV_CHECK=1
# ---------------------------------------------------------------------------


def test_skip_env_demotes_wrong_venv_to_warning(tmp_path, monkeypatch):
    """With skip=1, wrong venv is a warning and does not fail the report."""
    ws = _fake_workspace(tmp_path)
    expected_venv = _fake_venv(ws, "explore-persona-space")
    wrong_venv = _fake_venv(ws, "make-evil-dumb")

    monkeypatch.setenv("VIRTUAL_ENV", str(wrong_venv))
    monkeypatch.setenv("PREFLIGHT_SKIP_VENV_CHECK", "1")

    report = PreflightReport()
    check_active_venv(report, expected_venv=expected_venv, workspace_root=ws)

    assert report.ok
    assert report.errors == []
    assert len(report.warnings) == 1
    assert "PREFLIGHT_SKIP_VENV_CHECK=1" in report.warnings[0]


def test_skip_env_demotes_stale_venv_to_warning(tmp_path, monkeypatch):
    """With skip=1, stale make-evil-dumb/.venv becomes a warning."""
    ws = _fake_workspace(tmp_path)
    _fake_venv(ws, "make-evil-dumb")

    monkeypatch.setenv("PREFLIGHT_SKIP_VENV_CHECK", "1")

    report = PreflightReport()
    check_make_evil_dumb_absent(report, workspace_root=ws)

    assert report.ok
    assert report.errors == []
    assert any("PREFLIGHT_SKIP_VENV_CHECK=1" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Test 6: uv.lock parser
# ---------------------------------------------------------------------------


def test_parse_uv_lock_versions_reads_adjacent_pairs(tmp_path):
    """Parser returns {name: version} for each requested package present."""
    lock = tmp_path / "uv.lock"
    _write_lock(lock, {"torch": "2.8.0", "transformers": "5.5.0", "other": "1.2.3"})

    out = _parse_uv_lock_versions(lock, ("torch", "transformers"))
    assert out == {"torch": "2.8.0", "transformers": "5.5.0"}


def test_parse_uv_lock_versions_skips_unknown_names(tmp_path):
    """Names absent from the lockfile simply don't appear in the output."""
    lock = tmp_path / "uv.lock"
    _write_lock(lock, {"torch": "2.8.0"})

    out = _parse_uv_lock_versions(lock, ("torch", "nonexistent-pkg"))
    assert out == {"torch": "2.8.0"}


def test_parse_uv_lock_versions_raises_on_missing_file(tmp_path):
    """Missing lockfile raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        _parse_uv_lock_versions(tmp_path / "does-not-exist.lock", ("torch",))
