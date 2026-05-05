# ruff: noqa: RUF003
"""Tests for issue #228 Phase 0 (marker LoRA training) and Phase 0.5
(leakage measurement) — round-3 implementer scope.

Tests verify:
  * The schema written by ``measure_leakage_228._marker_rates_per_persona``
    round-trips through ``aggregate_issue228._leakage_rate``.
  * ``train_marker_loras_228._hf_already_has_marker_lora`` correctly
    short-circuits when the target slot is present on HF Hub (mocked).
  * ``run_issue228_sweep`` phase routing dispatches to the correct
    enumeration / build_cmd / is_done helper for ``--phase 0``, ``--phase
    0.5``, ``--phase 1``, and ``--phase all``.

No GPU / no real HF Hub / no network. All HF calls are monkeypatched.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make ``scripts/`` importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


# ──────────────────────────────────────────────────────────────────────────
# Schema round-trip: leakage worker output → aggregator consumer.
# ──────────────────────────────────────────────────────────────────────────


def _synthetic_completions(
    target_personas: list[str], n_questions: int = 20, n_per_q: int = 10
) -> dict[str, dict[str, list[str]]]:
    """Build a synthetic completions dict shaped like vLLM's batched output."""
    out: dict[str, dict[str, list[str]]] = {}
    for i, p in enumerate(target_personas):
        # Persona i has hit-rate i/10 — gives a monotone gradient across
        # the 11 targets so we can pin the round-trip end-to-end.
        out[p] = {}
        for q_idx in range(n_questions):
            comps = []
            for k in range(n_per_q):
                # Inject the marker on the first ``i`` completions per question.
                if k < i:
                    comps.append(f"sample text [ZLT] {q_idx}-{k}")
                else:
                    comps.append(f"sample text {q_idx}-{k}")
            out[p][f"Q{q_idx}"] = comps
    return out


def test_marker_rates_schema_roundtrips_through_aggregator(tmp_path: Path) -> None:
    """The dict written by Phase 0.5 must be readable by the aggregator's
    ``_leakage_rate`` helper for every TARGET persona, with the rate value
    matching the worker's compute.
    """
    measure_leakage_228 = importlib.import_module("measure_leakage_228")
    aggregate_issue228 = importlib.import_module("aggregate_issue228")

    target_personas = list(aggregate_issue228.TARGET_PERSONA_ORDER)
    completions = _synthetic_completions(target_personas)
    rates = measure_leakage_228._marker_rates_per_persona(completions)

    # 1. Top-level schema: every target persona is a key, every value is a
    #    dict with at least a 'rate' field.
    for tgt in target_personas:
        assert tgt in rates, f"target {tgt} missing"
        entry = rates[tgt]
        assert isinstance(entry, dict), f"entry for {tgt} not a dict"
        assert "rate" in entry, f"entry for {tgt} missing 'rate'"
        assert "hits" in entry and "total" in entry

    # 2. Round-trip via _leakage_rate.
    for i, tgt in enumerate(target_personas):
        worker_rate = rates[tgt]["rate"]
        agg_rate = aggregate_issue228._leakage_rate(rates, tgt)
        assert agg_rate == pytest.approx(worker_rate)
        # Synthetic data: persona i has marker on i out of 10 per question
        # → rate = i/10.
        assert agg_rate == pytest.approx(i / 10.0)

    # 3. Persist + reload (the disk path Phase 0.5 actually writes).
    out_path = tmp_path / "marker_eval.json"
    out_path.write_text(json.dumps({**rates, "_meta": {"foo": 1}}))
    on_disk = json.loads(out_path.read_text())
    for tgt in target_personas:
        assert aggregate_issue228._leakage_rate(on_disk, tgt) == pytest.approx(rates[tgt]["rate"])
    # Aggregator must NOT trip when the file has a non-persona '_meta' key —
    # _leakage_rate is only called with persona names.
    assert aggregate_issue228._leakage_rate(on_disk, "definitely_not_a_persona") is None


# ──────────────────────────────────────────────────────────────────────────
# Phase 0 idempotency: HF Hub list_repo_files mock.
# ──────────────────────────────────────────────────────────────────────────


def test_phase0_skips_when_hf_already_has_adapter(monkeypatch) -> None:
    """When ``cp_marker_<src>_ep<step>_s42`` already exists on HF Hub,
    ``_hf_already_has_marker_lora`` returns True so the worker can exit
    with ``ALREADY_EXISTS`` without ever touching a GPU.
    """
    train_marker_loras_228 = importlib.import_module("train_marker_loras_228")

    fake_files = [
        "adapters/cp_marker_villain_ep200_s42/adapter_config.json",
        "adapters/cp_marker_villain_ep200_s42/adapter_model.safetensors",
        "adapters/cp_marker_comedian_ep0_s42/adapter_config.json",
    ]

    fake_api = MagicMock()
    fake_api.list_repo_files.return_value = fake_files
    fake_api_class = MagicMock(return_value=fake_api)
    monkeypatch.setattr(train_marker_loras_228, "HfApi", fake_api_class, raising=False)
    # The function imports HfApi inside the body; patch the module path that
    # *will* be looked up at call time.
    import huggingface_hub  # noqa: F401  (forces the module into sys.modules)

    monkeypatch.setattr("huggingface_hub.HfApi", fake_api_class)

    # Hit: villain ckpt-200 IS in the fake file list.
    assert train_marker_loras_228._hf_already_has_marker_lora("villain", 200) is True

    # Miss: villain ckpt-400 is NOT in the fake file list.
    assert train_marker_loras_228._hf_already_has_marker_lora("villain", 400) is False

    # Miss: nurse ckpt-1000 is NOT in the fake file list.
    assert train_marker_loras_228._hf_already_has_marker_lora("nurse", 1000) is False


def test_phase0_proceeds_on_hub_listing_error(monkeypatch) -> None:
    """If the HF Hub list_repo_files call fails, we proceed to train rather
    than skip — the worst case is duplicate work, not a missing artifact.
    """
    train_marker_loras_228 = importlib.import_module("train_marker_loras_228")

    def _raise(*_a, **_kw):
        raise RuntimeError("simulated network error")

    fake_api = MagicMock()
    fake_api.list_repo_files.side_effect = _raise
    fake_api_class = MagicMock(return_value=fake_api)
    monkeypatch.setattr("huggingface_hub.HfApi", fake_api_class)

    assert train_marker_loras_228._hf_already_has_marker_lora("villain", 200) is False


# ──────────────────────────────────────────────────────────────────────────
# Phase routing: the coordinator's PHASE_REGISTRY dispatches correctly.
# ──────────────────────────────────────────────────────────────────────────


def test_phase_registry_keys_and_state_counts() -> None:
    """Each phase entry has the right enumeration shape."""
    sweep = importlib.import_module("run_issue228_sweep")

    # Phase 0: 70 (source, ckpt) pairs, all ckpt > 0.
    p0 = sweep.PHASE_REGISTRY[sweep.PHASE_MARKER]
    states_p0 = p0["enumerate"]()
    assert len(states_p0) == 7 * 10
    assert all(step > 0 for _src, step in states_p0)

    # Phase 0.5: 71 states (7 epoch-0 + 7×10).
    p05 = sweep.PHASE_REGISTRY[sweep.PHASE_LEAKAGE]
    states_p05 = p05["enumerate"]()
    assert len(states_p05) == 7 + 7 * 10
    # Each source must contribute its own epoch-0 baseline + 10 epoch-N.
    by_source = {}
    for src, step in states_p05:
        by_source.setdefault(src, []).append(step)
    assert len(by_source) == 7
    for steps in by_source.values():
        assert 0 in steps
        assert len(steps) == 11

    # Phase 1: 71 states with shared epoch-0 baseline (BASE_SOURCE).
    p1 = sweep.PHASE_REGISTRY[sweep.PHASE_JS]
    states_p1 = p1["enumerate"]()
    assert len(states_p1) == 1 + 7 * 10
    assert states_p1[0] == (sweep.BASE_SOURCE, sweep.BASE_CHECKPOINT_STEP)


def test_phase_routing_build_cmd_targets_correct_script() -> None:
    """``build_cmd`` for each phase must invoke the corresponding worker
    script (path-suffix match — no false-cross dispatch).
    """
    sweep = importlib.import_module("run_issue228_sweep")
    out_dir = Path("/tmp/dummy")
    cmd0 = sweep.PHASE_REGISTRY[sweep.PHASE_MARKER]["build_cmd"](
        source="villain", step=200, seed=42, output_dir=out_dir, gpu_mem_util=0.85
    )
    cmd05 = sweep.PHASE_REGISTRY[sweep.PHASE_LEAKAGE]["build_cmd"](
        source="villain", step=200, seed=42, output_dir=out_dir, gpu_mem_util=0.85
    )
    cmd1 = sweep.PHASE_REGISTRY[sweep.PHASE_JS]["build_cmd"](
        source="villain", step=200, seed=42, output_dir=out_dir, gpu_mem_util=0.85
    )
    assert any("train_marker_loras_228.py" in arg for arg in cmd0)
    assert not any("measure_leakage_228.py" in arg for arg in cmd0)
    assert not any("compute_js_convergence_228.py" in arg for arg in cmd0)

    assert any("measure_leakage_228.py" in arg for arg in cmd05)
    assert not any("train_marker_loras_228.py" in arg for arg in cmd05)
    assert not any("compute_js_convergence_228.py" in arg for arg in cmd05)

    assert any("compute_js_convergence_228.py" in arg for arg in cmd1)
    assert not any("train_marker_loras_228.py" in arg for arg in cmd1)
    assert not any("measure_leakage_228.py" in arg for arg in cmd1)


def test_phase_routing_is_done_distinguishes_outputs(tmp_path: Path) -> None:
    """Each phase's ``is_done`` predicate looks at its own filename.

    Phase 0: HF Hub-side, so always returns False locally.
    Phase 0.5: looks for ``marker_eval.json``.
    Phase 1: looks for ``result.json``.
    """
    sweep = importlib.import_module("run_issue228_sweep")
    p0 = sweep.PHASE_REGISTRY[sweep.PHASE_MARKER]
    p05 = sweep.PHASE_REGISTRY[sweep.PHASE_LEAKAGE]
    p1 = sweep.PHASE_REGISTRY[sweep.PHASE_JS]

    # Phase 0 always False (defers to HF Hub).
    assert p0["is_done"](tmp_path, "villain", 200) is False

    # Phase 0.5: write a marker_eval.json and confirm detection.
    marker_path = tmp_path / "villain" / "checkpoint-200" / "marker_eval.json"
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text("{}")
    assert p05["is_done"](tmp_path, "villain", 200) is True
    # JS not present → Phase 1 still says False.
    assert p1["is_done"](tmp_path, "villain", 200) is False

    # Phase 1: write a result.json and confirm.
    result_path = tmp_path / "villain" / "checkpoint-200" / "result.json"
    result_path.write_text("{}")
    assert p1["is_done"](tmp_path, "villain", 200) is True

    # Phase 1 base-source path: separate filename slot.
    base_path = tmp_path / sweep.BASE_SOURCE / "checkpoint-0" / "result.json"
    base_path.parent.mkdir(parents=True)
    base_path.write_text("{}")
    assert p1["is_done"](tmp_path, sweep.BASE_SOURCE, 0) is True


def test_phase_all_runs_three_phases_in_order(monkeypatch, tmp_path: Path) -> None:
    """When invoked with ``--phase all``, the coordinator dispatches the
    three phases in [0, 0.5, 1] order. We exercise this by stubbing
    ``_run_phase`` to record which phase keys it was called with.
    """
    sweep = importlib.import_module("run_issue228_sweep")

    calls: list[str] = []

    def _fake_run_phase(phase_key, **_kw):
        calls.append(phase_key)
        return {
            "phase": phase_key,
            "n_states": 0,
            "n_completed": 0,
            "n_skipped_existing": 0,
            "n_failed": 0,
        }

    monkeypatch.setattr(sweep, "_run_phase", _fake_run_phase)
    monkeypatch.setattr(sweep, "_upload_leakage_cache_to_wandb", lambda *_a, **_k: None)

    # Drive main() with --phase all
    argv = [
        "run_issue228_sweep.py",
        "--num-gpus",
        "1",
        "--phase",
        "all",
        "--output-dir",
        str(tmp_path / "issue_228"),
        "--leakage-output-dir",
        str(tmp_path / "leakage"),
        "--log-dir",
        str(tmp_path / "logs"),
        "--skip-wandb-upload",
    ]
    with patch.object(sys, "argv", argv):
        rc = sweep.main()
    assert rc == 0
    assert calls == [sweep.PHASE_MARKER, sweep.PHASE_LEAKAGE, sweep.PHASE_JS]


def test_single_phase_dispatch(monkeypatch, tmp_path: Path) -> None:
    """``--phase 0.5`` runs only Phase 0.5, not the others."""
    sweep = importlib.import_module("run_issue228_sweep")

    calls: list[str] = []
    monkeypatch.setattr(
        sweep,
        "_run_phase",
        lambda phase_key, **_kw: (
            calls.append(phase_key)
            or {
                "phase": phase_key,
                "n_states": 0,
                "n_completed": 0,
                "n_skipped_existing": 0,
                "n_failed": 0,
            }
        ),
    )
    monkeypatch.setattr(sweep, "_upload_leakage_cache_to_wandb", lambda *_a, **_k: None)

    argv = [
        "run_issue228_sweep.py",
        "--num-gpus",
        "1",
        "--phase",
        "0.5",
        "--output-dir",
        str(tmp_path / "issue_228"),
        "--leakage-output-dir",
        str(tmp_path / "leakage"),
        "--log-dir",
        str(tmp_path / "logs"),
        "--skip-wandb-upload",
    ]
    with patch.object(sys, "argv", argv):
        rc = sweep.main()
    assert rc == 0
    assert calls == [sweep.PHASE_LEAKAGE]


def test_dry_run_does_not_dispatch_subprocesses(monkeypatch, tmp_path: Path) -> None:
    """``--dry-run`` exits without spawning workers, but still enumerates
    state counts per phase.
    """
    sweep = importlib.import_module("run_issue228_sweep")

    # Track that the actual subprocess.run is never called by stubbing it
    # to a sentinel that raises if invoked.
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: pytest.fail("subprocess.run should not be invoked in --dry-run"),
    )

    argv = [
        "run_issue228_sweep.py",
        "--num-gpus",
        "1",
        "--phase",
        "all",
        "--output-dir",
        str(tmp_path / "issue_228"),
        "--leakage-output-dir",
        str(tmp_path / "leakage"),
        "--log-dir",
        str(tmp_path / "logs"),
        "--dry-run",
        "--skip-wandb-upload",
    ]
    with patch.object(sys, "argv", argv):
        rc = sweep.main()
    assert rc == 0
    summary = json.loads((tmp_path / "issue_228" / "_sweep_summary.json").read_text())
    assert summary["dry_run"] is True
    assert {s["phase"] for s in summary["summaries"]} == {
        sweep.PHASE_MARKER,
        sweep.PHASE_LEAKAGE,
        sweep.PHASE_JS,
    }
