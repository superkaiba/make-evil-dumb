"""Step-level manifest tracking for leakage experiments.

Enables crash recovery by persisting the status of each pipeline step
(data_gen, train, merge, generate, eval_*) to a JSON file. On resume,
completed steps are skipped automatically.

Inspired by safety-tooling's experiment tracking — simple file-based,
no database needed, human-readable JSON.

Usage:
    manifest = ConditionManifest.load_or_create(output_dir / "manifest.json")
    if manifest.should_run("train_phase1"):
        manifest.mark_running("train_phase1")
        try:
            run_training(...)
            manifest.mark_complete("train_phase1", {"loss": 0.42, "path": "/..."})
        except Exception as e:
            manifest.mark_failed("train_phase1", str(e))
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StepStatus(StrEnum):
    """Status of a single pipeline step."""

    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"
    skipped = "skipped"


class StepRecord(dict):
    """A step's tracking record (just a typed dict for serialization).

    Keys:
        status: StepStatus value
        started_at: ISO timestamp when step began
        completed_at: ISO timestamp when step finished (or failed)
        duration_s: Wall-clock seconds
        error: Error message if failed
        result: Arbitrary result dict if complete
    """


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


class ConditionManifest:
    """Tracks pipeline step status for a single condition + seed.

    Persists to a JSON file that is human-readable and git-friendly.
    All mutations auto-save to disk.
    """

    def __init__(self, path: Path, data: dict | None = None):
        self.path = Path(path)
        self._data: dict[str, Any] = data or {
            "condition": "",
            "seed": 0,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "steps": {},
        }

    # ── Persistence ─────────────────────────────────────────────────────

    @classmethod
    def load_or_create(
        cls,
        path: str | Path,
        condition_name: str = "",
        seed: int = 0,
    ) -> ConditionManifest:
        """Load existing manifest or create a new one."""
        path = Path(path)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            logger.info("Loaded manifest from %s (%d steps)", path, len(data.get("steps", {})))
            return cls(path, data)

        manifest = cls(path)
        manifest._data["condition"] = condition_name
        manifest._data["seed"] = seed
        manifest._save()
        logger.info("Created new manifest at %s", path)
        return manifest

    def _save(self) -> None:
        """Persist to disk."""
        self._data["updated_at"] = _now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp, then rename
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
        tmp.rename(self.path)

    # ── Step management ─────────────────────────────────────────────────

    def get_step(self, step_name: str) -> dict:
        """Get a step's record, or a default pending record."""
        return self._data["steps"].get(
            step_name,
            {"status": StepStatus.pending.value},
        )

    def step_status(self, step_name: str) -> StepStatus:
        """Get the current status of a step."""
        record = self.get_step(step_name)
        return StepStatus(record.get("status", "pending"))

    def should_run(self, step_name: str) -> bool:
        """Whether this step should be (re)run.

        Returns True for pending and failed steps, False for complete/running/skipped.
        """
        status = self.step_status(step_name)
        return status in (StepStatus.pending, StepStatus.failed)

    def mark_running(self, step_name: str) -> None:
        """Mark a step as currently running."""
        self._data["steps"][step_name] = {
            "status": StepStatus.running.value,
            "started_at": _now_iso(),
            "started_epoch": time.time(),
        }
        self._save()
        logger.info("[%s] Step '%s' -> running", self._data["condition"], step_name)

    def mark_complete(self, step_name: str, result: dict | None = None) -> None:
        """Mark a step as successfully completed."""
        record = self._data["steps"].get(step_name, {})
        started = record.get("started_epoch", time.time())
        duration = time.time() - started

        self._data["steps"][step_name] = {
            "status": StepStatus.complete.value,
            "started_at": record.get("started_at", _now_iso()),
            "completed_at": _now_iso(),
            "duration_s": round(duration, 1),
            "result": result or {},
        }
        self._save()
        logger.info(
            "[%s] Step '%s' -> complete (%.1fs)",
            self._data["condition"],
            step_name,
            duration,
        )

    def mark_failed(self, step_name: str, error: str) -> None:
        """Mark a step as failed."""
        record = self._data["steps"].get(step_name, {})
        started = record.get("started_epoch", time.time())
        duration = time.time() - started

        self._data["steps"][step_name] = {
            "status": StepStatus.failed.value,
            "started_at": record.get("started_at", _now_iso()),
            "failed_at": _now_iso(),
            "duration_s": round(duration, 1),
            "error": error,
        }
        self._save()
        logger.warning(
            "[%s] Step '%s' -> failed: %s",
            self._data["condition"],
            step_name,
            error[:200],
        )

    def mark_skipped(self, step_name: str, reason: str = "") -> None:
        """Mark a step as intentionally skipped."""
        self._data["steps"][step_name] = {
            "status": StepStatus.skipped.value,
            "skipped_at": _now_iso(),
            "reason": reason,
        }
        self._save()

    # ── Query ───────────────────────────────────────────────────────────

    @property
    def all_steps(self) -> dict[str, dict]:
        """All tracked steps and their records."""
        return dict(self._data["steps"])

    @property
    def is_complete(self) -> bool:
        """Whether all tracked steps are complete or skipped."""
        if not self._data["steps"]:
            return False
        return all(
            s.get("status") in (StepStatus.complete.value, StepStatus.skipped.value)
            for s in self._data["steps"].values()
        )

    @property
    def has_failures(self) -> bool:
        return any(s.get("status") == StepStatus.failed.value for s in self._data["steps"].values())

    @property
    def summary(self) -> dict[str, int]:
        """Count of steps by status."""
        counts: dict[str, int] = {}
        for s in self._data["steps"].values():
            status = s.get("status", "pending")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def get_result(self, step_name: str) -> dict | None:
        """Get the result dict of a completed step, or None."""
        record = self.get_step(step_name)
        if record.get("status") == StepStatus.complete.value:
            return record.get("result")
        return None

    def total_duration_s(self) -> float:
        """Total wall-clock seconds across all completed steps."""
        return sum(
            s.get("duration_s", 0)
            for s in self._data["steps"].values()
            if s.get("status") == StepStatus.complete.value
        )

    def __repr__(self) -> str:
        return (
            f"ConditionManifest(condition={self._data['condition']!r}, "
            f"seed={self._data['seed']}, summary={self.summary})"
        )


class SweepManifest:
    """Tracks all condition manifests for a sweep.

    Thin wrapper that discovers and loads per-condition manifests from
    the sweep output directory.
    """

    def __init__(self, sweep_dir: Path):
        self.sweep_dir = Path(sweep_dir)
        self._manifests: dict[str, ConditionManifest] = {}
        self._discover()

    def _discover(self) -> None:
        """Find all manifest.json files in condition subdirectories."""
        if not self.sweep_dir.exists():
            return
        for manifest_path in sorted(self.sweep_dir.rglob("manifest.json")):
            with open(manifest_path) as f:
                data = json.load(f)
            key = f"{data.get('condition', 'unknown')}_seed{data.get('seed', 0)}"
            self._manifests[key] = ConditionManifest(manifest_path, data)

    def get(self, condition_name: str, seed: int) -> ConditionManifest | None:
        """Look up a specific condition manifest."""
        key = f"{condition_name}_seed{seed}"
        return self._manifests.get(key)

    def get_or_create(
        self,
        condition_name: str,
        seed: int,
        output_dir: Path,
    ) -> ConditionManifest:
        """Get existing or create new manifest for a condition + seed."""
        key = f"{condition_name}_seed{seed}"
        if key in self._manifests:
            return self._manifests[key]

        manifest = ConditionManifest.load_or_create(
            output_dir / "manifest.json",
            condition_name=condition_name,
            seed=seed,
        )
        self._manifests[key] = manifest
        return manifest

    @property
    def all_manifests(self) -> dict[str, ConditionManifest]:
        return dict(self._manifests)

    @property
    def summary(self) -> dict[str, dict]:
        """Per-condition summary."""
        return {
            key: {
                "complete": m.is_complete,
                "has_failures": m.has_failures,
                "steps": m.summary,
                "duration_s": m.total_duration_s(),
            }
            for key, m in self._manifests.items()
        }

    def print_status(self) -> str:
        """Human-readable status table."""
        lines = [f"Sweep status: {self.sweep_dir}"]
        lines.append("-" * 70)

        for key, m in sorted(self._manifests.items()):
            status_icon = "done" if m.is_complete else ("FAIL" if m.has_failures else "...")
            duration = m.total_duration_s()
            lines.append(f"  [{status_icon:>4}] {key:<40} {m.summary} ({duration:.0f}s)")

        n_complete = sum(1 for m in self._manifests.values() if m.is_complete)
        n_total = len(self._manifests)
        lines.append("-" * 70)
        lines.append(f"  {n_complete}/{n_total} conditions complete")
        return "\n".join(lines)
