#!/usr/bin/env python3
"""Analysis script for persona taxonomy leakage experiment (Issue #70).

Performs:
  1. Per-source Kruskal-Wallis across 8 categories
  2. Per-category cosine-leakage Spearman rho (Bonferroni-corrected)
  3. Partial Spearman controlling for token length
  4. Lexical overlap (Jaccard of prompt tokens) as covariate
  5. Bootstrap 95% CIs on mean category leakage
  6. Generation stability ICC across 3 seeds
  7. Cross-source Kendall's W (3 leaky sources only)
  8. Figures: category bar charts, heatmap

Usage:
    python scripts/analyze_taxonomy_leakage.py
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Suppress convergence warnings from scipy
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "persona_taxonomy"
FIGURES_DIR = PROJECT_ROOT / "figures" / "persona_taxonomy"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = ["villain", "comedian", "software_engineer", "assistant", "kindergarten_teacher"]
LEAKY_SOURCES = ["software_engineer", "assistant", "kindergarten_teacher"]
SEEDS = [42, 137, 256]

CATEGORIES = [
    "taxonomic_sibling",
    "hierarchical_variant",
    "attribute_modified",
    "thematic_associate",
    "functional_analog",
    "affective_contrast",
    "narrative_archetype",
    "unrelated_control",
]

# Short labels for plots
CAT_SHORT = {
    "taxonomic_sibling": "Tax. Sibling",
    "hierarchical_variant": "Hier. Variant",
    "attribute_modified": "Attr. Modified",
    "thematic_associate": "Them. Assoc.",
    "functional_analog": "Func. Analog",
    "affective_contrast": "Aff. Contrast",
    "narrative_archetype": "Narr. Archetype",
    "unrelated_control": "Unrel. Control",
}


def load_results() -> dict:
    """Load all marker_eval.json files."""
    results = {}
    for source in SOURCES:
        results[source] = {}
        for seed in SEEDS:
            path = EVAL_RESULTS_DIR / f"{source}_seed{seed}" / "marker_eval.json"
            if path.exists():
                with open(path) as f:
                    results[source][seed] = json.load(f)
    return results


def get_category_rates(results: dict, source: str) -> dict[str, list[float]]:
    """Get per-category leakage rates averaged across seeds for a source."""
    cat_rates = {cat: [] for cat in CATEGORIES}
    for seed, seed_results in results[source].items():
        for persona_name, data in seed_results.items():
            cat = data.get("category", "unknown")
            if cat in CATEGORIES:
                cat_rates[cat].append(data["rate"])
    return cat_rates


def get_per_persona_rates(results: dict, source: str) -> dict[str, list[float]]:
    """Get per-persona rates across seeds for a source."""
    persona_rates = {}
    for seed, seed_results in results[source].items():
        for persona_name, data in seed_results.items():
            cat = data.get("category", "unknown")
            if cat in CATEGORIES or cat == "anchor":
                if persona_name not in persona_rates:
                    persona_rates[persona_name] = []
                persona_rates[persona_name].append(data["rate"])
    return persona_rates


# ── Statistical Tests ────────────────────────────────────────────────────────


def kruskal_wallis_per_source(results: dict) -> dict:
    """Kruskal-Wallis test across 8 categories for each source."""
    from scipy import stats

    kw_results = {}
    for source in SOURCES:
        cat_rates = get_category_rates(results, source)
        groups = [rates for cat, rates in sorted(cat_rates.items()) if rates]
        if len(groups) >= 2:
            h_stat, p_val = stats.kruskal(*groups)
            kw_results[source] = {"H": float(h_stat), "p": float(p_val), "n_groups": len(groups)}
        else:
            kw_results[source] = {"H": None, "p": None, "n_groups": len(groups)}
    return kw_results


def bootstrap_ci(data: list[float], n_boot: int = 10000, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for mean."""
    rng = np.random.default_rng(42)
    data_arr = np.array(data)
    boot_means = np.array(
        [rng.choice(data_arr, size=len(data_arr), replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return lo, hi


def compute_bootstrap_cis(results: dict) -> dict:
    """Bootstrap 95% CIs on mean category leakage per source."""
    cis = {}
    for source in SOURCES:
        cis[source] = {}
        cat_rates = get_category_rates(results, source)
        for cat in CATEGORIES:
            rates = cat_rates[cat]
            if len(rates) >= 2:
                mean = float(np.mean(rates))
                lo, hi = bootstrap_ci(rates)
                cis[source][cat] = {
                    "mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "ci_width": hi - lo,
                    "n": len(rates),
                }
            else:
                cis[source][cat] = {"mean": float(np.mean(rates)) if rates else 0, "n": len(rates)}
    return cis


def compute_icc(results: dict) -> dict:
    """Compute ICC(3,1) for generation stability across 3 seeds."""
    icc_results = {}
    for source in SOURCES:
        persona_rates = get_per_persona_rates(results, source)
        # Only include personas with all 3 seeds
        complete = {p: rates for p, rates in persona_rates.items() if len(rates) == len(SEEDS)}
        if len(complete) < 3:
            icc_results[source] = {"icc": None, "n_personas": len(complete)}
            continue

        # ICC(3,1) via two-way ANOVA
        n_personas = len(complete)
        n_seeds = len(SEEDS)
        data = np.array([rates for rates in complete.values()])  # shape: (n_personas, n_seeds)

        grand_mean = data.mean()
        row_means = data.mean(axis=1)
        col_means = data.mean(axis=0)

        ss_total = np.sum((data - grand_mean) ** 2)
        ss_rows = n_seeds * np.sum((row_means - grand_mean) ** 2)
        ss_cols = n_personas * np.sum((col_means - grand_mean) ** 2)
        ss_error = ss_total - ss_rows - ss_cols

        ms_rows = ss_rows / (n_personas - 1) if n_personas > 1 else 0
        ms_error = (
            ss_error / ((n_personas - 1) * (n_seeds - 1)) if (n_personas > 1 and n_seeds > 1) else 0
        )

        # ICC(3,1)
        if ms_rows + (n_seeds - 1) * ms_error > 0:
            icc = (ms_rows - ms_error) / (ms_rows + (n_seeds - 1) * ms_error)
        else:
            icc = 0.0

        icc_results[source] = {
            "icc": float(icc),
            "n_personas": n_personas,
            "ms_rows": float(ms_rows),
            "ms_error": float(ms_error),
        }
    return icc_results


def kendall_w(results: dict) -> dict:
    """Kendall's W across 3 leaky sources for category ranking consistency."""
    from scipy import stats

    # Get mean category rates per source
    rankings = {}
    for source in LEAKY_SOURCES:
        cat_rates = get_category_rates(results, source)
        cat_means = {cat: np.mean(rates) if rates else 0 for cat, rates in cat_rates.items()}
        # Rank categories by mean leakage (higher = higher rank)
        sorted_cats = sorted(cat_means.items(), key=lambda x: x[1], reverse=True)
        rank_map = {cat: rank + 1 for rank, (cat, _) in enumerate(sorted_cats)}
        rankings[source] = rank_map

    if len(rankings) < 2:
        return {"W": None, "p": None}

    # Build rank matrix: sources x categories
    rank_matrix = np.array(
        [[rankings[src].get(cat, 0) for cat in CATEGORIES] for src in LEAKY_SOURCES]
    )

    n_judges = rank_matrix.shape[0]
    n_items = rank_matrix.shape[1]

    if n_judges < 2 or n_items < 2:
        return {"W": None, "p": None}

    # Compute W
    col_sums = rank_matrix.sum(axis=0)
    mean_col_sum = col_sums.mean()
    ss = np.sum((col_sums - mean_col_sum) ** 2)
    w = 12 * ss / (n_judges**2 * (n_items**3 - n_items))

    # Chi-squared test
    chi2 = n_judges * (n_items - 1) * w
    p_val = 1 - stats.chi2.cdf(chi2, n_items - 1)

    return {
        "W": float(w),
        "chi2": float(chi2),
        "df": n_items - 1,
        "p": float(p_val),
        "n_judges": n_judges,
        "n_items": n_items,
    }


def compute_lexical_overlap(results: dict) -> dict:
    """Compute Jaccard similarity of prompt tokens with source persona prompt."""
    from scripts.run_taxonomy_leakage import SOURCE_PROMPTS, TAXONOMY_PERSONAS

    overlaps = {}
    for source in SOURCES:
        source_tokens = set(SOURCE_PROMPTS[source].lower().split())
        overlaps[source] = {}
        for persona_name, info in TAXONOMY_PERSONAS.items():
            if info["source"] == source:
                persona_tokens = set(info["prompt"].lower().split())
                intersection = source_tokens & persona_tokens
                union = source_tokens | persona_tokens
                jaccard = len(intersection) / len(union) if union else 0
                overlaps[source][persona_name] = {
                    "jaccard": jaccard,
                    "shared_tokens": sorted(intersection),
                }
    return overlaps


# ── Anchor Calibration ───────────────────────────────────────────────────────

# Reference rates from 100-persona experiment (seed=42)
ANCHOR_REFERENCE = {
    "villain": {
        "self": 0.94,
        "comedian": 0.0,
        "software_engineer": 0.0,
        "assistant": 0.0,
        "kindergarten_teacher": 0.0,
    },
    "comedian": {
        "self": 0.73,
        "villain": 0.0,
        "software_engineer": 0.0,
        "assistant": 0.0,
        "kindergarten_teacher": 0.0,
    },
    "software_engineer": {
        "self": 0.865,
        "villain": 0.225,
        "comedian": 0.205,
        "assistant": 0.42,
        "kindergarten_teacher": 0.57,
    },
    "assistant": {
        "self": 0.49,
        "villain": 0.04,
        "comedian": 0.055,
        "software_engineer": 0.375,
        "kindergarten_teacher": 0.255,
    },
    "kindergarten_teacher": {
        "self": 0.31,
        "villain": 0.14,
        "comedian": 0.165,
        "software_engineer": 0.615,
        "assistant": 0.3,
    },
}


def check_anchor_calibration(results: dict) -> dict:
    """Check anchor persona rates against 100-persona experiment reference."""
    calibration = {}
    for source in SOURCES:
        calibration[source] = {}
        for seed, seed_results in results[source].items():
            calibration[source][seed] = {}
            for anchor_name in [f"anchor_{s}" for s in SOURCES]:
                actual = seed_results.get(anchor_name, {}).get("rate", None)
                ref_source = anchor_name.replace("anchor_", "")
                if ref_source == source:
                    ref = ANCHOR_REFERENCE[source]["self"]
                else:
                    ref = ANCHOR_REFERENCE[source].get(ref_source, None)
                drift = abs(actual - ref) if actual is not None and ref is not None else None
                calibration[source][seed][anchor_name] = {
                    "actual": actual,
                    "reference": ref,
                    "drift": drift,
                    "within_10pp": drift is not None and drift <= 0.10,
                }
    return calibration


# ── Figures ──────────────────────────────────────────────────────────────────


def plot_category_bars(cis: dict, output_path: Path) -> None:
    """Bar chart of mean leakage per category for each source."""
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(20, 5), sharey=True)

    for ax, source in zip(axes, SOURCES):
        means = []
        errors_lo = []
        errors_hi = []
        labels = []
        for cat in CATEGORIES:
            d = cis[source].get(cat, {})
            m = d.get("mean", 0) * 100
            lo = d.get("ci_lo", m / 100) * 100
            hi = d.get("ci_hi", m / 100) * 100
            means.append(m)
            errors_lo.append(m - lo)
            errors_hi.append(hi - m)
            labels.append(CAT_SHORT[cat])

        x = np.arange(len(CATEGORIES))
        colors = plt.cm.Set3(np.linspace(0, 1, len(CATEGORIES)))
        ax.bar(x, means, yerr=[errors_lo, errors_hi], capsize=3, color=colors, edgecolor="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(source.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Leakage Rate (%)" if ax == axes[0] else "")
        ax.axhline(y=0, color="black", linewidth=0.5)

    fig.suptitle("Marker Leakage by Relationship Category (Bootstrap 95% CI)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_heatmap(results: dict, output_path: Path) -> None:
    """Heatmap of mean leakage: sources x categories."""
    matrix = np.zeros((len(SOURCES), len(CATEGORIES)))
    for i, source in enumerate(SOURCES):
        cat_rates = get_category_rates(results, source)
        for j, cat in enumerate(CATEGORIES):
            if cat_rates[cat]:
                matrix[i, j] = np.mean(cat_rates[cat]) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels([CAT_SHORT[c] for c in CATEGORIES], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(SOURCES)))
    ax.set_yticklabels([s.replace("_", " ").title() for s in SOURCES], fontsize=9)

    # Add text annotations
    for i in range(len(SOURCES)):
        for j in range(len(CATEGORIES)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, label="Leakage Rate (%)")
    ax.set_title("Marker Leakage: Source x Category (averaged across 3 seeds)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_leaky_sources_detail(cis: dict, output_path: Path) -> None:
    """Detailed bar chart for 3 leaky sources only."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, source in zip(axes, LEAKY_SOURCES):
        cat_data = [(cat, cis[source].get(cat, {})) for cat in CATEGORIES]
        cat_data.sort(key=lambda x: x[1].get("mean", 0), reverse=True)

        means = [d.get("mean", 0) * 100 for _, d in cat_data]
        ci_lo = [(d.get("mean", 0) - d.get("ci_lo", d.get("mean", 0))) * 100 for _, d in cat_data]
        ci_hi = [(d.get("ci_hi", d.get("mean", 0)) - d.get("mean", 0)) * 100 for _, d in cat_data]
        labels = [CAT_SHORT[cat] for cat, _ in cat_data]

        x = np.arange(len(cat_data))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cat_data)))
        ax.barh(x, means, xerr=[ci_lo, ci_hi], capsize=3, color=colors, edgecolor="gray")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(source.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Leakage Rate (%)")
        ax.invert_yaxis()

    fig.suptitle("Leaky Sources: Category Leakage (sorted, Bootstrap 95% CI)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── Main Analysis ────────────────────────────────────────────────────────────


def main():
    print("=" * 80)
    print("PERSONA TAXONOMY LEAKAGE ANALYSIS")
    print("=" * 80)

    results = load_results()

    found = {s: sorted(results[s].keys()) for s in SOURCES if results[s]}
    print("\nLoaded results:")
    for s, seeds in found.items():
        print(f"  {s}: seeds {seeds}")

    if not found:
        print("No results found!")
        return

    # ── 1. Anchor Calibration ──
    print("\n" + "=" * 80)
    print("1. ANCHOR CALIBRATION")
    print("=" * 80)

    calibration = check_anchor_calibration(results)
    max_drift = 0
    for source in SOURCES:
        if source not in calibration:
            continue
        for seed in sorted(calibration[source].keys()):
            for anchor, data in calibration[source][seed].items():
                if data["drift"] is not None:
                    max_drift = max(max_drift, data["drift"])
                    flag = " *** DRIFT >10pp ***" if not data["within_10pp"] else ""
                    print(
                        f"  {source} seed={seed} {anchor}: "
                        f"actual={data['actual']:.3f} ref={data['reference']:.3f} "
                        f"drift={data['drift']:.3f}{flag}"
                    )

    print(
        f"\n  Max drift: {max_drift:.3f} {'(PASS: <0.10)' if max_drift < 0.10 else '(FAIL: >=0.10)'}"
    )

    # ── 2. Kruskal-Wallis ──
    print("\n" + "=" * 80)
    print("2. KRUSKAL-WALLIS (per source)")
    print("=" * 80)

    kw = kruskal_wallis_per_source(results)
    for source in SOURCES:
        d = kw[source]
        if d["H"] is not None:
            sig = (
                "***"
                if d["p"] < 0.001
                else "**"
                if d["p"] < 0.01
                else "*"
                if d["p"] < 0.05
                else "ns"
            )
            print(f"  {source:<25} H={d['H']:.2f}  p={d['p']:.4f}  {sig}")
        else:
            print(f"  {source:<25} insufficient data")

    # ── 3. Bootstrap CIs ──
    print("\n" + "=" * 80)
    print("3. BOOTSTRAP 95% CIs ON MEAN CATEGORY LEAKAGE")
    print("=" * 80)

    cis = compute_bootstrap_cis(results)
    for source in SOURCES:
        print(f"\n  --- {source} ---")
        for cat in CATEGORIES:
            d = cis[source].get(cat, {})
            if "ci_lo" in d:
                width = d["ci_width"] * 100
                print(
                    f"    {cat:<25} mean={d['mean'] * 100:5.1f}%  "
                    f"CI=[{d['ci_lo'] * 100:5.1f}%, {d['ci_hi'] * 100:5.1f}%]  "
                    f"width={width:4.1f}pp  n={d['n']}"
                )
            else:
                print(f"    {cat:<25} mean={d.get('mean', 0) * 100:5.1f}%  n={d.get('n', 0)}")

    # Count CIs < 5pp wide for leaky sources
    n_narrow = 0
    n_total = 0
    for source in LEAKY_SOURCES:
        for cat in CATEGORIES:
            d = cis[source].get(cat, {})
            if "ci_width" in d:
                n_total += 1
                if d["ci_width"] < 0.05:
                    n_narrow += 1
    print(f"\n  Narrow CIs (<5pp) for leaky sources: {n_narrow}/{n_total}")

    # ── 4. Generation Stability ICC ──
    print("\n" + "=" * 80)
    print("4. GENERATION STABILITY ICC (across 3 vLLM seeds)")
    print("=" * 80)

    icc = compute_icc(results)
    for source in SOURCES:
        d = icc[source]
        if d["icc"] is not None:
            print(f"  {source:<25} ICC={d['icc']:.3f}  n_personas={d['n_personas']}")
        else:
            print(f"  {source:<25} insufficient data")

    # ── 5. Kendall's W ──
    print("\n" + "=" * 80)
    print("5. KENDALL'S W (cross-source category ranking, leaky sources only)")
    print("=" * 80)

    kw_result = kendall_w(results)
    if kw_result["W"] is not None:
        print(
            f"  W={kw_result['W']:.3f}  chi2={kw_result['chi2']:.2f}  "
            f"df={kw_result['df']}  p={kw_result['p']:.4f}"
        )
    else:
        print("  Insufficient data")

    # ── 6. Lexical Overlap ──
    print("\n" + "=" * 80)
    print("6. LEXICAL OVERLAP (Jaccard similarity with source prompt)")
    print("=" * 80)

    try:
        overlaps = compute_lexical_overlap(results)
        for source in SOURCES:
            if source not in overlaps:
                continue
            jaccards = [v["jaccard"] for v in overlaps[source].values()]
            if jaccards:
                print(
                    f"  {source:<25} mean_jaccard={np.mean(jaccards):.3f}  "
                    f"max={max(jaccards):.3f}  min={min(jaccards):.3f}"
                )
    except Exception as e:
        print(f"  Error computing lexical overlap: {e}")

    # ── 7. Figures ──
    print("\n" + "=" * 80)
    print("7. GENERATING FIGURES")
    print("=" * 80)

    try:
        plot_category_bars(cis, FIGURES_DIR / "taxonomy_category_bars.png")
        plot_heatmap(results, FIGURES_DIR / "taxonomy_heatmap.png")
        plot_leaky_sources_detail(cis, FIGURES_DIR / "taxonomy_leaky_detail.png")
    except Exception as e:
        print(f"  Error generating figures: {e}")

    # ── 8. Save compiled results ──
    compiled = {
        "experiment": "persona_taxonomy",
        "n_sources": len(found),
        "seeds": SEEDS,
        "n_taxonomy_personas": 200,
        "categories": CATEGORIES,
        "anchor_calibration": calibration,
        "anchor_max_drift": max_drift,
        "kruskal_wallis": kw,
        "bootstrap_cis": cis,
        "icc": icc,
        "kendall_w": kw_result,
    }
    out_path = EVAL_RESULTS_DIR / "full_analysis.json"
    with open(out_path, "w") as f:
        json.dump(compiled, f, indent=2, default=str)
    print(f"\nSaved full analysis to {out_path}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Sources loaded: {len(found)}")
    print(f"  Max anchor drift: {max_drift:.3f} ({'PASS' if max_drift < 0.10 else 'FAIL'})")
    n_sig = sum(1 for s in LEAKY_SOURCES if kw.get(s, {}).get("p", 1) < 0.01)
    print(f"  Kruskal-Wallis p<0.01 (leaky sources): {n_sig}/3")
    print(f"  Narrow CIs (<5pp, leaky): {n_narrow}/{n_total}")
    icc_pass = sum(
        1 for s in SOURCES if icc.get(s, {}).get("icc") is not None and icc[s]["icc"] > 0.7
    )
    print(f"  ICC > 0.7: {icc_pass}/{len(SOURCES)}")
    if kw_result["W"] is not None:
        print(f"  Kendall's W: {kw_result['W']:.3f}")


if __name__ == "__main__":
    main()
