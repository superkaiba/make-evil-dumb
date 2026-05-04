"""Unit tests for ``scripts/aggregate_issue228.py``.

We test the aggregator's correlation logic on synthetic 7x11 matrices with
known structure, the cluster-bootstrap helper, and the self-loop exclusion
logic. No GPU / no network dependencies.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``scripts/`` importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_issue228 as agg  # noqa: E402

ADAPTER_SOURCES = agg.ADAPTER_SOURCES  # 7 sources, alphabetical
TARGETS = list(agg.TARGET_PERSONA_ORDER)  # 11 targets


# ──────────────────────────────────────────────────────────────────────────
# Self-loop exclusion: H1 long-format has exactly 70 rows, no diagonal.
# ──────────────────────────────────────────────────────────────────────────


def _synthesize_h1_long(
    *,
    rho_target: float = -0.7,
    seed: int = 0,
) -> list[dict]:
    """Build a synthetic H1 long-format with controlled JS/leakage correlation.

    For every (source != target) cell we draw js, then set leakage as a
    monotone-decreasing transform of js plus noise sized to give Spearman
    rho ≈ ``rho_target`` in the limit. cosine_l15 / l20 / l25 are independent.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for source in ADAPTER_SOURCES:
        for target in TARGETS:
            if source == target:
                continue
            js = float(rng.uniform(0.0, 0.5))
            noise = float(rng.normal(scale=0.1))
            # rho_target = -0.7 -> mostly anti-monotone with some noise
            leak = max(0.0, min(1.0, 0.5 - rho_target * (-1.0) * js + noise))
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "is_self_loop": False,
                    "js": js,
                    "js_masked": js + float(rng.normal(scale=0.01)),
                    "js_no_marker": js + float(rng.normal(scale=0.01)),
                    "cosine_l15": float(rng.uniform(-0.5, 0.5)),
                    "cosine_l20": float(rng.uniform(-0.5, 0.5)),
                    "cosine_l25": float(rng.uniform(-0.5, 0.5)),
                    "source_token_entropy_mean": float(rng.uniform(1.0, 5.0)),
                    "leakage": leak,
                }
            )
    return rows


def test_h1_long_off_diagonal_count() -> None:
    """Off-diagonal cell count.

    Plan §3/§4/§6.1#6 pre-registers n=70 (= 7 sources x 11 targets - 7 self-loops).
    But ``nurse`` is in ADAPTER_SOURCES and NOT in ALL_EVAL_PERSONAS — so only 6
    self-loops actually exist on the persona sets. The literal "exclude self-loops"
    interpretation produces 7 x 11 - 6 = 71 cells. This deviation from the plan's
    pre-registered n is documented in the implementation report; the user can
    direct option (2) "drop nurse entirely -> n=60" if preferred. See GitHub
    comment on issue #228 for the discussion.
    """
    rows = _synthesize_h1_long()
    n_self_loops_present = sum(1 for s in ADAPTER_SOURCES if s in TARGETS)
    expected = len(ADAPTER_SOURCES) * len(TARGETS) - n_self_loops_present
    assert len(rows) == expected
    for r in rows:
        assert r["source"] != r["target"]
        assert r["is_self_loop"] is False


# ──────────────────────────────────────────────────────────────────────────
# _spearman_xy: drops None / NaN; reports correct n.
# ──────────────────────────────────────────────────────────────────────────


def test_spearman_drops_none_and_nan() -> None:
    rho, _p, n = agg._spearman_xy([1, 2, 3, None, 5], [1, 2, 3, 4, 5])
    assert n == 4
    assert not math.isnan(rho)
    _rho2, _p2, n2 = agg._spearman_xy([1.0, 2.0, float("nan"), 4.0], [1.0, 2.0, 3.0, 4.0])
    assert n2 == 3


def test_spearman_returns_nan_for_empty_or_too_short() -> None:
    rho, _p, n = agg._spearman_xy([], [])
    assert math.isnan(rho)
    assert n == 0
    rho, _p, n = agg._spearman_xy([1.0], [2.0])
    assert math.isnan(rho)
    assert n == 1


def test_spearman_recovers_known_strong_negative() -> None:
    rng = np.random.default_rng(123)
    x = rng.uniform(0, 1, size=200)
    y = -2.0 * x + rng.normal(scale=0.05, size=200)
    rho, _p, n = agg._spearman_xy(x.tolist(), y.tolist())
    assert n == 200
    assert rho < -0.9


# ──────────────────────────────────────────────────────────────────────────
# Cluster bootstrap: structure of returned dict, point estimate matches scipy.
# ──────────────────────────────────────────────────────────────────────────


def test_cluster_bootstrap_returns_complete_dict() -> None:
    rows = _synthesize_h1_long(seed=42)
    out = agg.cluster_bootstrap_spearman(rows, "js", "leakage", n_iter=50, seed=1)
    for k in (
        "rho",
        "p_value",
        "n",
        "ci_low",
        "ci_high",
        "n_iter",
        "n_clusters",
    ):
        assert k in out, f"missing key {k}"
    # Cell count = sources x targets - actual_self_loops (see test_h1_long_off_diagonal_count)
    expected_n = len(ADAPTER_SOURCES) * len(TARGETS) - sum(
        1 for s in ADAPTER_SOURCES if s in TARGETS
    )
    assert out["n"] == expected_n
    assert out["n_clusters"] == 7
    assert out["n_iter"] == 50
    assert -1.0 <= out["rho"] <= 1.0
    # Point estimate must match scipy on the full set.
    rho_ref, _, _ = agg._spearman_xy([r["js"] for r in rows], [r["leakage"] for r in rows])
    assert math.isclose(out["rho"], rho_ref, rel_tol=1e-9)


def test_cluster_bootstrap_ci_brackets_point_estimate_for_strong_signal() -> None:
    rng = np.random.default_rng(2026)
    rows: list[dict] = []
    for source in ADAPTER_SOURCES:
        for target in TARGETS:
            if source == target:
                continue
            js = float(rng.uniform(0.0, 1.0))
            leak = max(0.0, min(1.0, 1.0 - js + float(rng.normal(scale=0.02))))
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "js": js,
                    "leakage": leak,
                }
            )
    out = agg.cluster_bootstrap_spearman(rows, "js", "leakage", n_iter=200, seed=7)
    # Strong negative correlation by construction.
    assert out["rho"] < -0.9
    # Bootstrap CI must enclose the point estimate (allow tiny float slack).
    assert out["ci_low"] - 1e-3 <= out["rho"] <= out["ci_high"] + 1e-3


def test_paired_cluster_bootstrap_delta_abs_rho_structure() -> None:
    rows = _synthesize_h1_long(seed=99)
    out = agg.paired_cluster_bootstrap_delta_abs_rho(
        rows, "leakage", "js", "cosine_l15", n_iter=50, seed=8
    )
    for k in (
        "delta_abs_rho",
        "abs_rho_a",
        "abs_rho_b",
        "n",
        "ci_low",
        "ci_high",
        "n_iter",
    ):
        assert k in out
    assert out["n_iter"] == 50
    # delta == |rho_a| - |rho_b|
    assert math.isclose(
        out["delta_abs_rho"],
        out["abs_rho_a"] - out["abs_rho_b"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


# ──────────────────────────────────────────────────────────────────────────
# H1 analysis aggregator: pooled, source-fixed, target-fixed.
# ──────────────────────────────────────────────────────────────────────────


def test_h1_analysis_returns_all_substructures() -> None:
    rows = _synthesize_h1_long(seed=11)
    out = agg._h1_analysis(rows)
    for k in (
        "h1_primary_js",
        "h1_primary_cosine_l15",
        "h1_delta_js_minus_cosine_l15",
        "h1b1_within_target_z",
        "h1b2_source_fixed",
        "h1b3_target_fixed",
        "h1_source_to_assistant",
    ):
        assert k in out, f"missing key {k}"
    # Source-fixed has 7 rows.
    assert len(out["h1b2_source_fixed"]["per_source"]) == 7
    # Target-fixed has 11 rows.
    assert len(out["h1b3_target_fixed"]["per_target"]) == 11
    # Source-to-assistant: n=7 (one per source).
    assert out["h1_source_to_assistant"]["n"] == 7


def test_h1_analysis_excludes_self_loops_if_present() -> None:
    """If is_self_loop=True rows leak in, the analyzer must drop them."""
    rows = _synthesize_h1_long(seed=3)
    n_baseline = len(rows)
    # Inject a self-loop row that, if not excluded, would change pooled n.
    rows.append(
        {
            "source": "villain",
            "target": "villain",
            "is_self_loop": True,
            "js": 0.0,
            "js_masked": 0.0,
            "js_no_marker": 0.0,
            "cosine_l15": 1.0,
            "cosine_l20": 1.0,
            "cosine_l25": 1.0,
            "source_token_entropy_mean": 1.5,
            "leakage": 0.5,
        }
    )
    out = agg._h1_analysis(rows)
    # n must still be the off-diagonal count, not n_baseline+1.
    assert out["h1_primary_js"]["n"] == n_baseline


# ──────────────────────────────────────────────────────────────────────────
# H2 analysis: BH-FDR thresholding, raw + first-difference logic.
# ──────────────────────────────────────────────────────────────────────────


def _synthesize_h2_long(*, rho_target: float = -0.8, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for source in ADAPTER_SOURCES:
        for step in agg.CHECKPOINT_STEPS:
            t = step / 2000.0
            js = float(0.05 + 0.4 * t + rng.normal(scale=0.01))
            # rho_target negative => leakage decreases as js increases
            leak = float(0.5 + (-rho_target) * (1.0 - t) + rng.normal(scale=0.02))
            rows.append(
                {
                    "source": source,
                    "checkpoint_step": step,
                    "epoch": 2 * (agg.CHECKPOINT_STEPS.index(step) + 1),
                    "mean_js_off_diag": js,
                    "mean_cosine_l20_off_diag": float(rng.uniform(-0.2, 0.2)),
                    "mean_leakage_off_diag": max(0.0, min(1.0, leak)),
                    "max_leakage_off_diag": max(0.0, min(1.0, leak + 0.1)),
                    "assistant_leakage": float(rng.uniform(0, 1)),
                    "source_token_entropy_mean": float(rng.uniform(1, 5)),
                }
            )
    return rows


def test_h2_analysis_reports_seven_sources_each_for_raw_and_diff() -> None:
    rows = _synthesize_h2_long(seed=2)
    out = agg._h2_analysis(rows)
    assert len(out["per_source_raw"]) == 7
    assert len(out["per_source_diff"]) == 7
    assert len(out["raw_fdr"]) == 7
    assert len(out["diff_fdr"]) == 7
    # h2_pass_overall is a boolean derived from per-source counts.
    assert out["h2_pass_overall"] is (
        out["n_raw_passing_negative"] >= 4 and out["n_diff_passing_negative"] >= 4
    )


# ──────────────────────────────────────────────────────────────────────────
# C1 marker-mask: relative_change reflects |delta| / |rho_raw|.
# ──────────────────────────────────────────────────────────────────────────


def test_c1_marker_mask_relative_change_finite() -> None:
    rows = _synthesize_h1_long(seed=21)
    out = agg._c1_marker_mask(rows)
    for k in (
        "rho_masked",
        "delta_abs_rho_masked_minus_raw",
        "relative_change",
        "marker_mediated_threshold_rel_change",
        "marker_mediated",
    ):
        assert k in out, f"missing {k}"
    # In a synthetic dataset where masked ~ raw, relative change should be small.
    if not math.isnan(out["relative_change"]):
        assert out["relative_change"] >= 0.0


# ──────────────────────────────────────────────────────────────────────────
# C3 villain-out: drops villain's 10 rows -> n=60.
# ──────────────────────────────────────────────────────────────────────────


def test_c3_villain_out_drops_villain_cells() -> None:
    rows = _synthesize_h1_long(seed=31)
    out = agg._c3_villain_out(rows)
    n_full = out["n70"]["js_vs_leakage"]["n"]
    n_no_villain = out["n60"]["js_vs_leakage"]["n"]
    # Villain has 10 off-diagonal targets (11 - 1 self).
    assert n_full - n_no_villain == 10


# ──────────────────────────────────────────────────────────────────────────
# Sanity check 6.1#4: passes when matrices are identical.
# ──────────────────────────────────────────────────────────────────────────


def test_sanity_check_142_passes_on_identical_matrices(tmp_path: Path) -> None:
    n = 5
    rng = np.random.default_rng(7)
    M = rng.uniform(0.0, 0.4, size=(n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    persona_names = [f"persona_{i}" for i in range(n)]
    base_state = {
        "source": "base",
        "checkpoint_step": 0,
        "persona_names": persona_names,
        "js_matrix": M.tolist(),
    }
    baseline_path = tmp_path / "div.json"
    baseline_path.write_text(
        '{"persona_names": '
        + str(persona_names).replace("'", '"')
        + ', "js_matrix": '
        + str(M.tolist())
        + "}"
    )
    res = agg._sanity_check_142(base_state, baseline_path)
    assert res["verdict"] == "PASS"
    assert res["mean_relative_error"] < 1e-6


def test_sanity_check_142_fails_when_far_off(tmp_path: Path) -> None:
    n = 5
    rng = np.random.default_rng(8)
    M = rng.uniform(0.0, 0.4, size=(n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    persona_names = [f"persona_{i}" for i in range(n)]
    base_state = {
        "source": "base",
        "checkpoint_step": 0,
        "persona_names": persona_names,
        "js_matrix": M.tolist(),
    }
    M_far = M * 5.0  # 400% larger -> MRE huge
    baseline_path = tmp_path / "div.json"
    baseline_path.write_text(
        '{"persona_names": '
        + str(persona_names).replace("'", '"')
        + ', "js_matrix": '
        + str(M_far.tolist())
        + "}"
    )
    with pytest.raises(SystemExit):
        agg._sanity_check_142(base_state, baseline_path)
