#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Trait-variation ranking re-analysis for issue #81.

For each (source, bystander_noun, trait, level) cell, computes two deltas
between the A1 (trait-decorated) bystander and the A2 (pure-noun) baseline:

    Δ_leakage(src, noun, trait, L) = | rate(src, A1/noun/trait/L) − rate(src, A2/noun) |
    Δ_cos(src, noun, trait, L)     = | cos(emb(src), emb(A1/noun/trait/L))
                                        − cos(emb(src), emb(A2/noun)) |

Aggregates into three rankings per metric:

    global_ranking.csv    — 25 (trait, level) rows; mean Δ across (src, noun) pairs
    per_source_ranking.csv — 5 × 25 = 125 rows; one ranking per source
    per_cell_ranking.csv   — (src, noun, trait, level) = 5 × 5 × 5 × 5 = 625 rows

Figures in figures/leakage_i81/trait_ranking/:
    fig_global_top10_bars            — hero: top-10 trait variants by Δ_leakage + Δ_cos
    fig_leakage_vs_cosine_scatter    — 25 points with Spearman ρ (N=25) in title
    fig_per_source_heatmap           — 5×25 matrices for Δ_leakage + Δ_cos
    fig_rank_consistency             — which trait variants land in top-5 across sources

Person source falls back to `person/` (35 bystanders) if `person_full130/` is
missing the marker_eval.json; the fallback is noted and the person row is
EXCLUDED from global aggregation (per task spec) but retained per-source.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.bystanders_i81 import (  # noqa: E402
    GRADATIONS,
    NOUNS,
    TRAITS,
)

# ── Config ──────────────────────────────────────────────────────────────────

EVAL_DIR = PROJECT_ROOT / "eval_results" / "leakage_i81"
FIG_DIR = PROJECT_ROOT / "figures" / "leakage_i81" / "trait_ranking"
OUT_DIR = EVAL_DIR / "trait_ranking"

SOURCES = ["person", "chef", "pirate", "child", "robot"]
SRC_PERSONA_IDS = {s: f"src_{s}" for s in SOURCES}
HEADLINE_LAYER = 20


def a1_id(noun: str, trait: str, level: str) -> str:
    return f"A1__{noun}__{trait}__{level}"


def a2_id(noun: str) -> str:
    return f"A2__{noun}"


# ── Data loaders ────────────────────────────────────────────────────────────


def load_marker_rates(source: str, person_uses_full130: bool) -> dict[str, float]:
    """Return {persona_id: rate} for a given source, or {} if missing."""
    sub = "person_full130" if source == "person" and person_uses_full130 else source
    path = EVAL_DIR / sub / "marker_eval.json"
    if not path.exists():
        # Fallback: pilot person/ (35 bystanders only)
        if source == "person" and person_uses_full130:
            path = EVAL_DIR / "person" / "marker_eval.json"
            if not path.exists():
                return {}
        else:
            return {}
    with open(path) as f:
        d = json.load(f)
    return {k: float(v.get("rate", 0.0)) for k, v in d.items()}


def load_centroids(layer: int = HEADLINE_LAYER) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return ({persona_id: centroid_vec}, manifest_dict) at the given layer."""
    npz_path = EVAL_DIR / "cosine_vectors_i81.npz"
    manifest_path = EVAL_DIR / "cosine_manifest.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path} — run extract_hidden_states_i81.py first")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")
    arr_file = np.load(npz_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    suffix = f"__layer{layer}"
    out: dict[str, np.ndarray] = {}
    for key in arr_file.files:
        if key.endswith(suffix):
            pid = key[: -len(suffix)]
            out[pid] = arr_file[key].astype(np.float64)
    return out, manifest


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return num / denom


# ── Δ computation ───────────────────────────────────────────────────────────


def compute_per_cell_rows(
    centroids: dict[str, np.ndarray],
    marker_by_source: dict[str, dict[str, float]],
    person_has_full: bool,
) -> list[dict[str, Any]]:
    """One row per (source, noun, trait, level) cell = 5 × 5 × 5 × 5 = 625 rows.

    For missing marker rates (the person pilot), Δ_leakage is stored as NaN and
    the row is flagged so downstream aggregators can exclude it.
    """
    rows: list[dict[str, Any]] = []
    for src in SOURCES:
        src_id = SRC_PERSONA_IDS[src]
        src_vec = centroids[src_id]
        rates = marker_by_source.get(src, {})
        # Is this source's marker_eval full (131 keys) or pilot?
        is_full = (src != "person") or person_has_full
        for noun in NOUNS:
            a2_pid = a2_id(noun)
            if a2_pid not in centroids:
                continue
            cos_a2 = cosine_sim(src_vec, centroids[a2_pid])
            rate_a2 = rates.get(a2_pid)  # None if missing
            for trait in TRAITS:
                for level in GRADATIONS:
                    a1_pid = a1_id(noun, trait, level)
                    if a1_pid not in centroids:
                        continue
                    cos_a1 = cosine_sim(src_vec, centroids[a1_pid])
                    dcos = abs(cos_a1 - cos_a2)
                    rate_a1 = rates.get(a1_pid)
                    if rate_a1 is None or rate_a2 is None:
                        dleak = float("nan")
                    else:
                        dleak = abs(rate_a1 - rate_a2)
                    rows.append(
                        {
                            "source": src,
                            "source_marker_full": bool(is_full),
                            "noun": noun,
                            "trait": trait,
                            "level": level,
                            "cos_a1": cos_a1,
                            "cos_a2": cos_a2,
                            "delta_cos": dcos,
                            "rate_a1": float(rate_a1) if rate_a1 is not None else float("nan"),
                            "rate_a2": float(rate_a2) if rate_a2 is not None else float("nan"),
                            "delta_leakage": dleak,
                        }
                    )
    return rows


def global_ranking(rows: list[dict[str, Any]], person_has_full: bool) -> list[dict[str, Any]]:
    """Per (trait, level): mean Δ across (source, noun) cells.

    Per task spec: if person is on pilot 35-bystander slice, exclude person rows
    from the global aggregate and note in `n_cells`.
    """
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if not person_has_full and r["source"] == "person":
            continue
        k = (r["trait"], r["level"])
        by_key.setdefault(k, []).append(r)

    out: list[dict[str, Any]] = []
    for (trait, level), bucket in by_key.items():
        dleak_vals = [r["delta_leakage"] for r in bucket if not np.isnan(r["delta_leakage"])]
        dcos_vals = [r["delta_cos"] for r in bucket if not np.isnan(r["delta_cos"])]
        out.append(
            {
                "trait": trait,
                "level": level,
                "n_cells_leakage": len(dleak_vals),
                "n_cells_cos": len(dcos_vals),
                "mean_delta_leakage": float(np.mean(dleak_vals)) if dleak_vals else float("nan"),
                "mean_delta_cos": float(np.mean(dcos_vals)) if dcos_vals else float("nan"),
                "std_delta_leakage": float(np.std(dleak_vals, ddof=1))
                if len(dleak_vals) > 1
                else 0.0,
                "std_delta_cos": float(np.std(dcos_vals, ddof=1)) if len(dcos_vals) > 1 else 0.0,
            }
        )
    # Rank (1 = largest mean)
    leak_sorted = sorted(out, key=lambda r: -r["mean_delta_leakage"])
    for i, r in enumerate(leak_sorted):
        r["rank_leakage"] = i + 1
    cos_sorted = sorted(out, key=lambda r: -r["mean_delta_cos"])
    for i, r in enumerate(cos_sorted):
        r["rank_cos"] = i + 1
    # Re-key by (trait, level) for insertion-order stability
    out.sort(key=lambda r: r["rank_leakage"])
    return out


def per_source_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (source, trait, level): mean Δ across noun axis within the source."""
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        k = (r["source"], r["trait"], r["level"])
        by_key.setdefault(k, []).append(r)

    out: list[dict[str, Any]] = []
    for (src, trait, level), bucket in by_key.items():
        dleak_vals = [r["delta_leakage"] for r in bucket if not np.isnan(r["delta_leakage"])]
        dcos_vals = [r["delta_cos"] for r in bucket if not np.isnan(r["delta_cos"])]
        out.append(
            {
                "source": src,
                "trait": trait,
                "level": level,
                "n_cells_leakage": len(dleak_vals),
                "n_cells_cos": len(dcos_vals),
                "mean_delta_leakage": float(np.mean(dleak_vals)) if dleak_vals else float("nan"),
                "mean_delta_cos": float(np.mean(dcos_vals)) if dcos_vals else float("nan"),
            }
        )
    # Within-source rank
    for src in SOURCES:
        src_rows = [r for r in out if r["source"] == src]
        leak_sorted = sorted(
            src_rows,
            key=lambda r: -r["mean_delta_leakage"] if not np.isnan(r["mean_delta_leakage"]) else 0,
        )
        for i, r in enumerate(leak_sorted):
            r["rank_leakage_per_source"] = i + 1
        cos_sorted = sorted(src_rows, key=lambda r: -r["mean_delta_cos"])
        for i, r in enumerate(cos_sorted):
            r["rank_cos_per_source"] = i + 1
    return out


# ── CSV writers ─────────────────────────────────────────────────────────────


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ── Figures ─────────────────────────────────────────────────────────────────


def _variant_label(trait: str, level: str) -> str:
    return f"{trait[:3]}/{level}"


def fig_global_top10(global_rows: list[dict[str, Any]]) -> plt.Figure:
    set_paper_style("neurips")
    top = sorted(global_rows, key=lambda r: -r["mean_delta_leakage"])[:10]
    labels = [_variant_label(r["trait"], r["level"]) for r in top]
    dleak = np.array([r["mean_delta_leakage"] for r in top])
    dcos = np.array([r["mean_delta_cos"] for r in top])
    # error bar half-widths: use std / sqrt(n) as SEM for display (no prose)
    sem_leak = np.array(
        [r["std_delta_leakage"] / np.sqrt(max(r["n_cells_leakage"], 1)) for r in top]
    )
    sem_cos = np.array([r["std_delta_cos"] / np.sqrt(max(r["n_cells_cos"], 1)) for r in top])

    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(labels))
    w = 0.4
    c_leak, c_cos = paper_palette(2)
    ax1.bar(x - w / 2, dleak, w, yerr=sem_leak, color=c_leak, label="Δ leakage", capsize=2)
    ax1.set_ylabel("Δ leakage (|A1 − A2|)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, dcos, w, yerr=sem_cos, color=c_cos, label="Δ cosine", alpha=0.85, capsize=2)
    ax2.set_ylabel("Δ cosine (|A1 − A2|)")
    ax1.set_title("Top-10 trait variations by global Δ leakage")
    # combined legend
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_scatter(global_rows: list[dict[str, Any]]) -> plt.Figure:
    set_paper_style("neurips")
    xs = np.array([r["mean_delta_leakage"] for r in global_rows])
    ys = np.array([r["mean_delta_cos"] for r in global_rows])
    labels = [_variant_label(r["trait"], r["level"]) for r in global_rows]
    n = len(xs)
    rho, p = stats.spearmanr(xs, ys)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    c = paper_palette(1)[0]
    ax.scatter(xs, ys, s=30, color=c, edgecolor="black", linewidth=0.5)
    for xi, yi, lab in zip(xs, ys, labels, strict=True):
        ax.annotate(lab, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean Δ leakage (|A1 − A2|)")
    ax.set_ylabel("mean Δ cosine (|A1 − A2|)")
    ax.set_title(f"Trait variations: Δ leakage vs Δ cosine (ρ = {rho:.2f}, N = {n})")
    fig.tight_layout()
    return fig, float(rho), float(p), n


def fig_per_source_heatmap(per_src_rows: list[dict[str, Any]], metric: str) -> plt.Figure:
    """5 × 25 heatmap of per-source × trait-variation."""
    set_paper_style("neurips")
    variants: list[tuple[str, str]] = []
    for trait in TRAITS:
        for level in GRADATIONS:
            variants.append((trait, level))
    mat = np.full((len(SOURCES), len(variants)), np.nan)
    for i, src in enumerate(SOURCES):
        for j, (trait, level) in enumerate(variants):
            for r in per_src_rows:
                if r["source"] == src and r["trait"] == trait and r["level"] == level:
                    mat[i, j] = r[metric]
                    break
    fig, ax = plt.subplots(figsize=(9, 2.8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(SOURCES)))
    ax.set_yticklabels([f"src:{s}" for s in SOURCES])
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([_variant_label(t, lv) for t, lv in variants], rotation=90, fontsize=6)
    label = "Δ leakage" if "leakage" in metric else "Δ cosine"
    ax.set_title(f"Per-source {label} across 25 trait variations")
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    fig.tight_layout()
    return fig


def fig_rank_consistency(per_src_rows: list[dict[str, Any]]) -> plt.Figure:
    """For each trait variation, count how many sources rank it in their top-5 by Δ leakage."""
    set_paper_style("neurips")
    variants = [(t, lv) for t in TRAITS for lv in GRADATIONS]
    n_variants = len(variants)
    # collect per-source top-5 by leakage
    top5_by_src: dict[str, set[tuple[str, str]]] = {}
    for src in SOURCES:
        src_rows = [r for r in per_src_rows if r["source"] == src]
        src_rows_sorted = sorted(
            src_rows,
            key=lambda r: -r["mean_delta_leakage"] if not np.isnan(r["mean_delta_leakage"]) else 0,
        )
        top5_by_src[src] = {(r["trait"], r["level"]) for r in src_rows_sorted[:5]}

    counts = np.zeros(n_variants, dtype=int)
    for j, v in enumerate(variants):
        counts[j] = sum(1 for src in SOURCES if v in top5_by_src[src])
    order = np.argsort(-counts)
    labels = [_variant_label(*variants[j]) for j in order]
    counts = counts[order]
    fig, ax = plt.subplots(figsize=(8, 3.0))
    c = paper_palette(1)[0]
    ax.bar(range(n_variants), counts, color=c, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(n_variants))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("# sources with variant in top-5 by Δ leakage")
    ax.set_title("Rank consistency: trait-variation top-5 coverage across 5 sources")
    ax.set_yticks(range(0, max(6, int(counts.max()) + 1)))
    fig.tight_layout()
    return fig


# ── Orchestration ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer",
        type=int,
        default=HEADLINE_LAYER,
        help="Hidden-state layer to use for cosine (default 20, headline of #66).",
    )
    args = parser.parse_args()

    print(f"[i81-rank] using hidden-state layer {args.layer}")

    # Load centroids
    centroids, _manifest = load_centroids(layer=args.layer)
    print(f"[i81-rank] loaded {len(centroids)} centroids from cosine_vectors_i81.npz")

    # Load marker rates per source — try person_full130 first, fall back to person/
    marker_by_source: dict[str, dict[str, float]] = {}
    full130_path = EVAL_DIR / "person_full130" / "marker_eval.json"
    person_has_full = full130_path.exists()
    if not person_has_full:
        print(
            "[i81-rank] WARNING: person_full130/marker_eval.json missing — falling back to "
            "pilot person/ (35 bystanders). Person will be excluded from global aggregate."
        )
    for src in SOURCES:
        d = load_marker_rates(src, person_uses_full130=person_has_full)
        marker_by_source[src] = d
        print(f"  source={src}: {len(d)} persona_ids with marker rates")

    # Per-cell rows
    rows = compute_per_cell_rows(centroids, marker_by_source, person_has_full)
    print(
        f"[i81-rank] {len(rows)} per-cell rows; "
        f"{sum(1 for r in rows if not np.isnan(r['delta_leakage']))} have valid Δ_leakage"
    )

    # Global
    g_rows = global_ranking(rows, person_has_full)
    # Per-source
    ps_rows = per_source_ranking(rows)

    # ── Write CSVs ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUT_DIR / "global_ranking.csv",
        sorted(g_rows, key=lambda r: r["rank_leakage"]),
        [
            "rank_leakage",
            "rank_cos",
            "trait",
            "level",
            "mean_delta_leakage",
            "mean_delta_cos",
            "std_delta_leakage",
            "std_delta_cos",
            "n_cells_leakage",
            "n_cells_cos",
        ],
    )
    write_csv(
        OUT_DIR / "per_source_ranking.csv",
        sorted(ps_rows, key=lambda r: (r["source"], r["rank_leakage_per_source"])),
        [
            "source",
            "rank_leakage_per_source",
            "rank_cos_per_source",
            "trait",
            "level",
            "mean_delta_leakage",
            "mean_delta_cos",
            "n_cells_leakage",
            "n_cells_cos",
        ],
    )
    write_csv(
        OUT_DIR / "per_cell_ranking.csv",
        rows,
        [
            "source",
            "noun",
            "trait",
            "level",
            "cos_a1",
            "cos_a2",
            "delta_cos",
            "rate_a1",
            "rate_a2",
            "delta_leakage",
            "source_marker_full",
        ],
    )
    print(f"[i81-rank] CSVs → {OUT_DIR}")

    # ── Figures ─────────────────────────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = fig_global_top10(g_rows)
    savefig_paper(fig, "fig_global_top10_bars", dir=FIG_DIR)
    plt.close(fig)

    fig, rho, p, n_pts = fig_scatter(g_rows)
    savefig_paper(fig, "fig_leakage_vs_cosine_scatter", dir=FIG_DIR)
    plt.close(fig)
    print(f"[i81-rank] scatter: Spearman ρ = {rho:.3f}, p = {p:.3g}, N = {n_pts}")

    fig = fig_per_source_heatmap(ps_rows, metric="mean_delta_leakage")
    savefig_paper(fig, "fig_per_source_heatmap_leakage", dir=FIG_DIR)
    plt.close(fig)

    fig = fig_per_source_heatmap(ps_rows, metric="mean_delta_cos")
    savefig_paper(fig, "fig_per_source_heatmap_cos", dir=FIG_DIR)
    plt.close(fig)

    fig = fig_rank_consistency(ps_rows)
    savefig_paper(fig, "fig_rank_consistency", dir=FIG_DIR)
    plt.close(fig)

    # ── Summary JSON ────────────────────────────────────────────────────────
    top5_global_leak = sorted(g_rows, key=lambda r: -r["mean_delta_leakage"])[:5]
    top5_global_cos = sorted(g_rows, key=lambda r: -r["mean_delta_cos"])[:5]

    # Per-source top-1 + spread
    per_source_stats: dict[str, dict[str, Any]] = {}
    for src in SOURCES:
        src_rows = [r for r in ps_rows if r["source"] == src]
        vals_leak = [
            r["mean_delta_leakage"] for r in src_rows if not np.isnan(r["mean_delta_leakage"])
        ]
        if not vals_leak:
            per_source_stats[src] = {"note": "no leakage data"}
            continue
        top = max(
            src_rows,
            key=lambda r: r["mean_delta_leakage"] if not np.isnan(r["mean_delta_leakage"]) else -1,
        )
        per_source_stats[src] = {
            "top1_trait": top["trait"],
            "top1_level": top["level"],
            "top1_delta_leakage": top["mean_delta_leakage"],
            "spread_leakage": float(max(vals_leak) - min(vals_leak)),
            "max_leakage": float(max(vals_leak)),
            "min_leakage": float(min(vals_leak)),
        }

    # "top-1 unique to this source" detection
    top1_by_src = {
        src: (per_source_stats[src].get("top1_trait"), per_source_stats[src].get("top1_level"))
        for src in SOURCES
        if "top1_trait" in per_source_stats[src]
    }
    unique_top1 = {
        src: v
        for src, v in top1_by_src.items()
        if sum(1 for vv in top1_by_src.values() if vv == v) == 1
    }

    summary = {
        "layer": args.layer,
        "person_has_full130": person_has_full,
        "note": (
            "person source uses person_full130/ (130 bystanders)"
            if person_has_full
            else "person source uses pilot person/ (35 bystanders); "
            "person EXCLUDED from global aggregate per task spec"
        ),
        "spearman_rho_leak_vs_cos": rho,
        "spearman_p": p,
        "spearman_n": n_pts,
        "top5_global_by_delta_leakage": [
            {
                "rank": i + 1,
                "trait": r["trait"],
                "level": r["level"],
                "mean_delta_leakage": r["mean_delta_leakage"],
                "mean_delta_cos": r["mean_delta_cos"],
            }
            for i, r in enumerate(top5_global_leak)
        ],
        "top5_global_by_delta_cos": [
            {
                "rank": i + 1,
                "trait": r["trait"],
                "level": r["level"],
                "mean_delta_cos": r["mean_delta_cos"],
                "mean_delta_leakage": r["mean_delta_leakage"],
            }
            for i, r in enumerate(top5_global_cos)
        ],
        "per_source": per_source_stats,
        "unique_top1_per_source": {
            src: {"trait": v[0], "level": v[1]} for src, v in unique_top1.items()
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[i81-rank] summary → {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
