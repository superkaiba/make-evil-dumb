#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Analysis for issue #81 — 5 source × 130 bystander factorial leakage.

Reads `eval_results/leakage_i81/<source>/marker_eval.json` for each of
{person, chef, pirate, child, robot} and the `base_model/` dir, applies
bootstrap CIs (1000 iterations) + N/S masking on floor-indistinguishable
cells, overlays low-coherence hatching (mean coherence < 0.5), and
produces 5 figures in `figures/leakage_i81/`:

  1. `heatmap_5x130_base_subtracted.{png,pdf}`   — hero
  2. `slice_noun_isolation.{png,pdf}`
  3. `slice_trait_gradation.{png,pdf}`
  4. `slice_interaction.{png,pdf}`
  5. `submatrix_5x5_cos_vs_leakage.{png,pdf}`    — omitted if cos-sim cache missing

Plus `coherence_flags.csv` (per-cell coherence mean + flag).

Marker rates live at
`marker_eval.json[persona_key]['rate']` — per_question data is preserved
but we aggregate over questions for the headline per-cell rate.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.bystanders_i81 import (  # noqa: E402
    BYSTANDERS,
    GRADATIONS,
    NOUNS,
    TRAITS,
)

# ── Config ──────────────────────────────────────────────────────────────────

EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "leakage_i81"
FIG_DIR = PROJECT_ROOT / "figures" / "leakage_i81"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = ["person", "chef", "pirate", "child", "robot"]
BASE_KEY = "base_model"

N_BOOT = 1000
COHERENCE_THRESHOLD = 0.5
# For each cell, 10 completions × 20 questions = 200 total completions.
COMPLETIONS_PER_CELL = 200

log = logging.getLogger("analyze_leakage_i81")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Loaders ─────────────────────────────────────────────────────────────────


def _load_marker_eval(source_dir: Path) -> dict:
    path = source_dir / "marker_eval.json"
    if not path.exists():
        log.warning(f"missing marker_eval.json at {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def _load_coherence(source_dir: Path) -> dict:
    path = source_dir / "coherence_scores.json"
    if not path.exists():
        log.warning(f"missing coherence_scores.json at {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def _marker_flags_per_cell(marker_eval: dict, persona_key: str) -> list[int]:
    """Flatten per_question [found, total] into a per-completion 0/1 list.

    We don't have individual completion-level data after aggregation, so we
    reconstruct from per-question counts: a question with 'found'=3 and
    'total'=10 yields [1,1,1,0,0,0,0,0,0,0].
    """
    entry = marker_eval.get(persona_key)
    if not entry:
        return []
    flags: list[int] = []
    for _q, stats in entry.get("per_question", {}).items():
        found = int(stats.get("found", 0))
        total = int(stats.get("total", 0))
        flags.extend([1] * found + [0] * (total - found))
    return flags


# ── Bootstrap helpers ───────────────────────────────────────────────────────


def bootstrap_rate_ci(
    flags: list[int],
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Return (mean_rate, ci_lo_95, ci_hi_95, half_width)."""
    if not flags:
        return (0.0, 0.0, 0.0, 0.0)
    arr = np.array(flags)
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return mean, lo, hi, (hi - lo) / 2


def floor_indistinguishable(trained_rate: float, base_rate: float, n: int) -> bool:
    """True if the trained cell is within 2×SE of the base-model rate."""
    if n == 0:
        return True
    se = math.sqrt(max(base_rate * (1 - base_rate), 1e-12) / n)
    return abs(trained_rate - base_rate) < 2 * se


# ── Build per-(source, bystander) matrix of headline rates ──────────────────


def build_rate_matrix() -> dict:
    """Load all marker + coherence data and compute rates + CIs + flags."""
    bystander_keys = list(BYSTANDERS.keys())

    # Load base
    base_eval = _load_marker_eval(EVAL_RESULTS_DIR / BASE_KEY)
    base_rates: dict[str, float] = {}
    for bk in bystander_keys:
        entry = base_eval.get(bk)
        base_rates[bk] = float(entry["rate"]) if entry else 0.0

    # Per-source trained rates + bootstrap
    source_data: dict[str, dict] = {}
    for src in SOURCES:
        marker_eval = _load_marker_eval(EVAL_RESULTS_DIR / src)
        coh = _load_coherence(EVAL_RESULTS_DIR / src)
        coh_per_persona = (coh or {}).get("per_persona", {})

        cells: dict[str, dict] = {}
        for bk in bystander_keys:
            flags = _marker_flags_per_cell(marker_eval, bk)
            mean, lo, hi, hw = bootstrap_rate_ci(flags, seed=hash((src, bk)) & 0xFFFFFFFF)
            base_r = base_rates.get(bk, 0.0)
            n = len(flags) or COMPLETIONS_PER_CELL
            ns = floor_indistinguishable(mean, base_r, n)
            coh_mean = coh_per_persona.get(bk, {}).get("mean")
            low_coh = (coh_mean is not None) and (coh_mean < COHERENCE_THRESHOLD)
            cells[bk] = {
                "rate": mean,
                "base_rate": base_r,
                "rate_minus_base": mean - base_r,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_halfwidth": hw,
                "ns_floor": ns,
                "n": n,
                "coherence_mean": coh_mean,
                "low_coherence": low_coh,
            }
        source_data[src] = cells

    return {
        "bystander_keys": bystander_keys,
        "base_rates": base_rates,
        "sources": source_data,
    }


# ── Figure 1: 5 × 130 hero heatmap ──────────────────────────────────────────


def _order_bystander_columns(bystander_keys: list[str]) -> list[str]:
    """Group by (noun → trait → level), A2s interleaved by noun at the start."""
    ordered: list[str] = []
    for noun in NOUNS:
        ordered.append(f"A2__{noun}")
        for trait in TRAITS:
            for level in GRADATIONS:
                ordered.append(f"A1__{noun}__{trait}__{level}")
    assert set(ordered) == set(bystander_keys), "ordering mismatch"
    return ordered


def fig_hero_heatmap(data: dict) -> None:
    bystanders = _order_bystander_columns(data["bystander_keys"])
    mat = np.zeros((len(SOURCES), len(bystanders)))
    ns_mat = np.zeros_like(mat, dtype=bool)
    low_coh_mat = np.zeros_like(mat, dtype=bool)
    for i, src in enumerate(SOURCES):
        for j, bk in enumerate(bystanders):
            c = data["sources"][src].get(bk, {})
            mat[i, j] = c.get("rate_minus_base", 0.0)
            ns_mat[i, j] = c.get("ns_floor", False)
            low_coh_mat[i, j] = c.get("low_coherence", False)

    set_paper_style()
    fig, ax = plt.subplots(figsize=(18, 4.2))
    vmax = max(0.1, float(np.abs(mat).max()))
    im = ax.imshow(
        mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest"
    )

    # Overlay: N/S cells → hatched light-grey.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if ns_mat[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="white",
                        linewidth=0.2,
                    )
                )
            if low_coh_mat[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="...",
                        edgecolor="black",
                        linewidth=0.2,
                    )
                )

    ax.set_yticks(range(len(SOURCES)))
    ax.set_yticklabels([f"src:{s}" for s in SOURCES])
    # Column tick labels — show one tick per noun group (every 26 cells: 1 A2 + 25 A1).
    group_starts = [0 + k * 26 for k in range(len(NOUNS))]
    group_mids = [s + 13 for s in group_starts]
    ax.set_xticks(group_mids)
    ax.set_xticklabels(NOUNS)
    # Minor ticks at group boundaries
    for s in group_starts:
        ax.axvline(s - 0.5, color="white", linewidth=1.0)
    ax.set_xlabel("bystander-noun group (A2 + 25-cell A1 trait×level factorial per noun)")
    ax.set_title("Marker leakage: trained rate − base rate (per (source, bystander))")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("rate − base (percentage-points)")
    savefig_paper(fig, FIG_DIR / "heatmap_5x130_base_subtracted")
    plt.close(fig)


# ── Figure 2: noun-isolation slice ──────────────────────────────────────────


def fig_noun_isolation(data: dict) -> None:
    """For each (trait, level), show rate(src, noun) curves across NOUNS."""
    fig, axes = plt.subplots(len(TRAITS), len(GRADATIONS), figsize=(16, 12), sharey=True)
    set_paper_style()
    for ti, trait in enumerate(TRAITS):
        for li, level in enumerate(GRADATIONS):
            ax = axes[ti, li]
            for src in SOURCES:
                ys, ylo, yhi = [], [], []
                for noun in NOUNS:
                    bk = f"A1__{noun}__{trait}__{level}"
                    c = data["sources"][src].get(bk, {})
                    ys.append(c.get("rate_minus_base", 0.0))
                    ylo.append(c.get("ci_lo", 0) - c.get("base_rate", 0))
                    yhi.append(c.get("ci_hi", 0) - c.get("base_rate", 0))
                err = np.array([np.array(ys) - np.array(ylo), np.array(yhi) - np.array(ys)])
                err = np.clip(err, 0, None)
                ax.errorbar(
                    range(len(NOUNS)),
                    ys,
                    yerr=err,
                    marker="o",
                    markersize=3,
                    linewidth=0.8,
                    label=src,
                )
            ax.set_xticks(range(len(NOUNS)))
            ax.set_xticklabels(NOUNS, rotation=45, fontsize=6)
            ax.axhline(0, color="grey", linewidth=0.4, linestyle=":")
            if ti == 0:
                ax.set_title(level, fontsize=8)
            if li == 0:
                ax.set_ylabel(trait, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(SOURCES), fontsize=8)
    fig.suptitle("Noun-isolation: rate(src, noun | trait, level) − base_rate", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    savefig_paper(fig, FIG_DIR / "slice_noun_isolation")
    plt.close(fig)


# ── Figure 3: trait-gradation slice ─────────────────────────────────────────


def fig_trait_gradation(data: dict) -> None:
    """For each (src, noun), plot rate vs gradation-level per trait."""
    fig, axes = plt.subplots(len(SOURCES), len(NOUNS), figsize=(16, 14), sharey=True)
    set_paper_style()
    for si, src in enumerate(SOURCES):
        for ni, noun in enumerate(NOUNS):
            ax = axes[si, ni]
            for trait in TRAITS:
                ys, ylo, yhi = [], [], []
                for level in GRADATIONS:
                    bk = f"A1__{noun}__{trait}__{level}"
                    c = data["sources"][src].get(bk, {})
                    ys.append(c.get("rate_minus_base", 0.0))
                    ylo.append(c.get("ci_lo", 0) - c.get("base_rate", 0))
                    yhi.append(c.get("ci_hi", 0) - c.get("base_rate", 0))
                err = np.array([np.array(ys) - np.array(ylo), np.array(yhi) - np.array(ys)])
                err = np.clip(err, 0, None)
                ax.errorbar(
                    range(len(GRADATIONS)),
                    ys,
                    yerr=err,
                    marker="o",
                    markersize=3,
                    linewidth=0.8,
                    label=trait,
                )
            ax.set_xticks(range(len(GRADATIONS)))
            ax.set_xticklabels(GRADATIONS, fontsize=7)
            ax.axhline(0, color="grey", linewidth=0.4, linestyle=":")
            if si == 0:
                ax.set_title(f"noun={noun}", fontsize=8)
            if ni == 0:
                ax.set_ylabel(f"src:{src}", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(TRAITS), fontsize=8)
    fig.suptitle("Trait-gradation: rate − base_rate, per (src × noun × trait × level)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    savefig_paper(fig, FIG_DIR / "slice_trait_gradation")
    plt.close(fig)


# ── Figure 4: interaction slice ─────────────────────────────────────────────


def fig_interaction(data: dict) -> None:
    """Noun-effect magnitude as a function of (trait, level), per source."""
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(16, 4), sharey=True)
    set_paper_style()
    for si, src in enumerate(SOURCES):
        ax = axes[si]
        mat = np.zeros((len(TRAITS), len(GRADATIONS)))
        for ti, trait in enumerate(TRAITS):
            for li, level in enumerate(GRADATIONS):
                rates_by_noun = []
                for noun in NOUNS:
                    bk = f"A1__{noun}__{trait}__{level}"
                    c = data["sources"][src].get(bk, {})
                    rates_by_noun.append(c.get("rate_minus_base", 0.0))
                # noun-effect = max - min over nouns (percentage-point range).
                mat[ti, li] = (max(rates_by_noun) - min(rates_by_noun)) if rates_by_noun else 0.0
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1.0)
        ax.set_xticks(range(len(GRADATIONS)))
        ax.set_xticklabels(GRADATIONS, fontsize=6)
        ax.set_yticks(range(len(TRAITS)))
        ax.set_yticklabels(TRAITS if si == 0 else [], fontsize=6)
        ax.set_title(f"src:{src}", fontsize=8)
    fig.suptitle("Interaction: max−min rate across nouns per (trait, level)", fontsize=10)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.75, label="noun-effect size (pp)")
    savefig_paper(fig, FIG_DIR / "slice_interaction")
    plt.close(fig)


# ── Figure 5: 5×5 source-to-source (A2 only) submatrix ─────────────────────


def fig_submatrix_a2(data: dict) -> None:
    """5×5 source-to-source leakage on the 5 A2 pure-noun bystanders."""
    mat = np.zeros((len(SOURCES), len(SOURCES)))
    mat_base = np.zeros((len(SOURCES), len(SOURCES)))
    for i, src in enumerate(SOURCES):
        for j, noun in enumerate(NOUNS):
            bk = f"A2__{noun}"
            c = data["sources"][src].get(bk, {})
            mat[i, j] = c.get("rate", 0.0)
            mat_base[i, j] = c.get("base_rate", 0.0)

    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    im1 = ax1.imshow(mat, cmap="Reds", vmin=0, vmax=1)
    ax1.set_xticks(range(len(NOUNS)))
    ax1.set_yticks(range(len(SOURCES)))
    ax1.set_xticklabels(NOUNS, rotation=45)
    ax1.set_yticklabels([f"src:{s}" for s in SOURCES])
    ax1.set_xlabel("bystander noun (A2)")
    ax1.set_title("trained rate")
    for i in range(len(SOURCES)):
        for j in range(len(NOUNS)):
            ax1.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if mat[i, j] > 0.5 else "black",
            )
    fig.colorbar(im1, ax=ax1, shrink=0.75, label="rate")

    im2 = ax2.imshow(
        mat - mat_base,
        cmap="RdBu_r",
        vmin=-float(np.abs(mat - mat_base).max() or 0.1),
        vmax=float(np.abs(mat - mat_base).max() or 0.1),
    )
    ax2.set_xticks(range(len(NOUNS)))
    ax2.set_yticks(range(len(SOURCES)))
    ax2.set_xticklabels(NOUNS, rotation=45)
    ax2.set_yticklabels([])
    ax2.set_xlabel("bystander noun (A2)")
    ax2.set_title("trained − base")
    for i in range(len(SOURCES)):
        for j in range(len(NOUNS)):
            delta = mat[i, j] - mat_base[i, j]
            ax2.text(j, i, f"{delta:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im2, ax=ax2, shrink=0.75, label="Δ rate")
    fig.suptitle("5×5 source-to-source leakage (A2 pure-noun cells)", fontsize=10)
    fig.tight_layout()
    savefig_paper(fig, FIG_DIR / "submatrix_5x5_cos_vs_leakage")
    plt.close(fig)


# ── Coherence flags CSV ─────────────────────────────────────────────────────


def write_coherence_flags(data: dict) -> None:
    path = FIG_DIR / "coherence_flags.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "source",
                "bystander_key",
                "kind",
                "noun",
                "trait",
                "level",
                "rate",
                "base_rate",
                "rate_minus_base",
                "ci_lo",
                "ci_hi",
                "ci_halfwidth",
                "coherence_mean",
                "low_coherence",
                "ns_floor",
            ]
        )
        for src in SOURCES:
            for bk in data["bystander_keys"]:
                c = data["sources"][src][bk]
                meta = BYSTANDERS[bk]
                w.writerow(
                    [
                        src,
                        bk,
                        meta["kind"],
                        meta["noun"],
                        meta.get("trait") or "",
                        meta.get("level") or "",
                        c["rate"],
                        c["base_rate"],
                        c["rate_minus_base"],
                        c["ci_lo"],
                        c["ci_hi"],
                        c["ci_halfwidth"],
                        c["coherence_mean"] if c["coherence_mean"] is not None else "",
                        c["low_coherence"],
                        c["ns_floor"],
                    ]
                )
    log.info(f"Wrote {path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    if not EVAL_RESULTS_DIR.exists():
        log.error(f"No results dir at {EVAL_RESULTS_DIR}")
        sys.exit(1)

    log.info("Building rate matrix with bootstrap CIs…")
    data = build_rate_matrix()
    log.info("Figure 1: hero heatmap")
    fig_hero_heatmap(data)
    log.info("Figure 2: noun-isolation slice")
    fig_noun_isolation(data)
    log.info("Figure 3: trait-gradation slice")
    fig_trait_gradation(data)
    log.info("Figure 4: interaction slice")
    fig_interaction(data)
    log.info("Figure 5: 5×5 A2 submatrix")
    fig_submatrix_a2(data)
    log.info("Writing coherence_flags.csv")
    write_coherence_flags(data)
    log.info(f"All figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
