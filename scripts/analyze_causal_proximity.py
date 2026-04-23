"""Analyze causal proximity-leakage test results (issue #61).

Three-arm experiment testing whether representational proximity between
personas causally drives marker leakage:
  Arm A: marker-first (train marker -> convergence SFT on persona data)
  Arm B: similarity-first (convergence SFT -> train marker at each checkpoint)
  Arm C: behavioral control (train marker -> convergence SFT on generic data)

Usage:
    uv run python scripts/analyze_causal_proximity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
)

set_paper_style("neurips", font_scale=1.0)

FIG_DIR = Path("figures/causal_proximity")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = ["villain", "comedian", "kindergarten_teacher", "software_engineer"]
PCTS = [20, 40, 60, 80, 100]
SRC_LABELS = {
    "villain": "Villain",
    "comedian": "Comedian",
    "kindergarten_teacher": "KT",
    "software_engineer": "SW Eng",
}

# =============================================================================
# Hardcoded data extracted from pod1 logs/JSONs
# =============================================================================

# --- ARM A (marker-first): source marker rate, assistant marker rate, cosines ---
arm_a = {
    "villain": {
        20: {"src": 0.020, "asst": 0.000, "cos_L15": -0.186, "cos_L20": -0.234},
        40: {"src": 0.000, "asst": 0.000, "cos_L15": -0.024, "cos_L20": 0.008},
        60: {"src": 0.000, "asst": 0.000, "cos_L15": -0.019, "cos_L20": 0.025},
        80: {"src": 0.000, "asst": 0.000, "cos_L15": -0.060, "cos_L20": -0.035},
        100: {"src": 0.000, "asst": 0.000, "cos_L15": -0.048, "cos_L20": -0.020},
    },
    "comedian": {
        20: {"src": 0.010, "asst": 0.002, "cos_L15": -0.425, "cos_L20": -0.256},
        40: {"src": 0.005, "asst": 0.000, "cos_L15": -0.313, "cos_L20": -0.163},
        60: {"src": 0.000, "asst": 0.000, "cos_L15": -0.312, "cos_L20": -0.145},
        80: {"src": 0.000, "asst": 0.000, "cos_L15": -0.282, "cos_L20": -0.129},
        100: {"src": 0.000, "asst": 0.000, "cos_L15": -0.277, "cos_L20": -0.128},
    },
    "kindergarten_teacher": {
        20: {"src": 0.000, "asst": 0.000, "cos_L15": 0.134, "cos_L20": 0.182},
        40: {"src": 0.000, "asst": 0.000, "cos_L15": 0.150, "cos_L20": 0.276},
        60: {"src": 0.000, "asst": 0.000, "cos_L15": 0.104, "cos_L20": 0.231},
        80: {"src": 0.000, "asst": 0.000, "cos_L15": 0.123, "cos_L20": 0.247},
        100: {"src": 0.000, "asst": 0.000, "cos_L15": 0.140, "cos_L20": 0.266},
    },
    "software_engineer": {
        20: {"src": 0.000, "asst": 0.000, "cos_L15": 0.480, "cos_L20": 0.717},
        40: {"src": 0.000, "asst": 0.000, "cos_L15": 0.514, "cos_L20": 0.722},
        60: {"src": 0.000, "asst": 0.000, "cos_L15": 0.497, "cos_L20": 0.720},
        80: {"src": 0.000, "asst": 0.000, "cos_L15": 0.513, "cos_L20": 0.718},
        100: {"src": 0.000, "asst": 0.000, "cos_L15": 0.514, "cos_L20": 0.720},
    },
}

# --- ARM B (similarity-first): from marker_eval.json files ---
arm_b = {
    "villain": {
        20: {"src": 0.410, "asst": 0.247, "cos_L15": -0.528, "cos_L20": -0.395},
        40: {"src": 0.290, "asst": 0.242, "cos_L15": -0.152, "cos_L20": -0.110},
        60: {"src": 0.410, "asst": 0.288, "cos_L15": -0.149, "cos_L20": -0.111},
        80: {"src": 0.475, "asst": 0.222, "cos_L15": -0.155, "cos_L20": -0.106},
        100: {"src": 0.600, "asst": 0.202, "cos_L15": -0.141, "cos_L20": -0.075},
    },
    "comedian": {
        20: {"src": 0.275, "asst": 0.132, "cos_L15": -0.404, "cos_L20": -0.228},
        40: {"src": 0.205, "asst": 0.168, "cos_L15": -0.360, "cos_L20": -0.168},
        60: {"src": 0.220, "asst": 0.113, "cos_L15": -0.329, "cos_L20": -0.140},
        80: {"src": 0.225, "asst": 0.088, "cos_L15": -0.308, "cos_L20": -0.118},
        100: {"src": 0.215, "asst": 0.062, "cos_L15": -0.308, "cos_L20": -0.114},
    },
    "kindergarten_teacher": {
        20: {"src": 0.625, "asst": 0.775, "cos_L15": -0.116, "cos_L20": -0.059},
        40: {"src": 0.645, "asst": 0.705, "cos_L15": -0.006, "cos_L20": 0.093},
        60: {"src": 0.560, "asst": 0.555, "cos_L15": -0.044, "cos_L20": 0.029},
        80: {"src": 0.535, "asst": 0.530, "cos_L15": -0.049, "cos_L20": 0.036},
        100: {"src": 0.565, "asst": 0.540, "cos_L15": -0.051, "cos_L20": 0.033},
    },
    "software_engineer": {
        20: {"src": 0.985, "asst": 0.770, "cos_L15": 0.500, "cos_L20": 0.591},
        40: {"src": 0.955, "asst": 0.870, "cos_L15": 0.558, "cos_L20": 0.636},
        60: {"src": 0.930, "asst": 0.785, "cos_L15": 0.533, "cos_L20": 0.664},
        80: {"src": 0.870, "asst": 0.680, "cos_L15": 0.551, "cos_L20": 0.736},
        100: {"src": 0.890, "asst": 0.585, "cos_L15": 0.546, "cos_L20": 0.745},
    },
}

# --- ARM C (generic control): from experiment logs ---
# villain: 20% from log, 40-100% from issue #61 manual report, all 0% assistant
arm_c = {
    "villain": {
        20: {"src": 0.925, "asst": 0.000, "cos_L15": -0.441, "cos_L20": -0.486},
        40: {"src": 0.850, "asst": 0.000, "cos_L15": None, "cos_L20": None},
        60: {"src": 0.825, "asst": 0.000, "cos_L15": None, "cos_L20": None},
        80: {"src": 0.850, "asst": 0.000, "cos_L15": None, "cos_L20": None},
        100: {"src": 0.885, "asst": 0.000, "cos_L15": None, "cos_L20": None},
    },
    "comedian": {
        20: {"src": 0.570, "asst": 0.000, "cos_L15": -0.369, "cos_L20": -0.408},
        40: {"src": 0.335, "asst": 0.000, "cos_L15": -0.376, "cos_L20": -0.405},
        60: {"src": 0.220, "asst": 0.000, "cos_L15": -0.373, "cos_L20": -0.390},
        80: {"src": 0.205, "asst": 0.000, "cos_L15": -0.373, "cos_L20": -0.387},
        100: {"src": 0.160, "asst": 0.000, "cos_L15": -0.373, "cos_L20": -0.385},
    },
    # KT and SW: markers destroyed from 20% onward (0% at all checkpoints)
    "kindergarten_teacher": {
        20: {"src": 0.000, "asst": 0.000, "cos_L15": -0.020, "cos_L20": -0.175},
        40: {"src": 0.000, "asst": 0.000, "cos_L15": -0.008, "cos_L20": -0.171},
        60: {"src": 0.000, "asst": 0.000, "cos_L15": -0.012, "cos_L20": -0.165},
        80: {"src": 0.000, "asst": 0.000, "cos_L15": -0.026, "cos_L20": -0.157},
        100: {"src": 0.000, "asst": 0.000, "cos_L15": -0.025, "cos_L20": -0.159},
    },
    "software_engineer": {
        20: {"src": 0.000, "asst": 0.000, "cos_L15": 0.539, "cos_L20": 0.698},
        40: {"src": 0.000, "asst": 0.000, "cos_L15": 0.566, "cos_L20": 0.726},
        60: {"src": 0.000, "asst": 0.000, "cos_L15": 0.576, "cos_L20": 0.736},
        80: {"src": 0.000, "asst": 0.000, "cos_L15": 0.570, "cos_L20": 0.730},
        100: {"src": 0.000, "asst": 0.000, "cos_L15": 0.569, "cos_L20": 0.729},
    },
}

# Full Arm B marker rates (all 11 personas at 100% checkpoint)
arm_b_100pct_full = {
    "villain": {
        "software_engineer": 0.200,
        "kindergarten_teacher": 0.105,
        "data_scientist": 0.250,
        "medical_doctor": 0.175,
        "librarian": 0.235,
        "french_person": 0.355,
        "villain": 0.600,
        "comedian": 0.185,
        "police_officer": 0.175,
        "zelthari_scholar": 0.075,
        "assistant": 0.202,
    },
    "comedian": {
        "software_engineer": 0.065,
        "kindergarten_teacher": 0.025,
        "data_scientist": 0.105,
        "medical_doctor": 0.095,
        "librarian": 0.060,
        "french_person": 0.035,
        "villain": 0.020,
        "comedian": 0.215,
        "police_officer": 0.070,
        "zelthari_scholar": 0.005,
        "assistant": 0.062,
    },
    "kindergarten_teacher": {
        "software_engineer": 0.650,
        "kindergarten_teacher": 0.565,
        "data_scientist": 0.575,
        "medical_doctor": 0.490,
        "librarian": 0.640,
        "french_person": 0.205,
        "villain": 0.325,
        "comedian": 0.365,
        "police_officer": 0.425,
        "zelthari_scholar": 0.270,
        "assistant": 0.540,
    },
    "software_engineer": {
        "software_engineer": 0.890,
        "kindergarten_teacher": 0.375,
        "data_scientist": 0.835,
        "medical_doctor": 0.615,
        "librarian": 0.470,
        "french_person": 0.155,
        "villain": 0.150,
        "comedian": 0.195,
        "police_officer": 0.325,
        "zelthari_scholar": 0.105,
        "assistant": 0.585,
    },
}

colors = paper_palette(5)
ARM_COLORS = {"A": colors[0], "B": colors[1], "C": colors[2]}
SRC_COLORS = {
    "villain": colors[0],
    "comedian": colors[1],
    "kindergarten_teacher": colors[2],
    "software_engineer": colors[3],
}


# =============================================================================
# Figure 1: Cross-arm source marker rate trajectory (hero figure)
# =============================================================================
def plot_cross_arm_source_marker():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    for i, src in enumerate(SOURCES):
        ax = axes[i]
        # Arm A
        vals_a = [arm_a[src][p]["src"] * 100 for p in PCTS]
        ax.plot(PCTS, vals_a, "o-", color=ARM_COLORS["A"], label="A: marker-first", markersize=4)
        # Arm B
        vals_b = [arm_b[src][p]["src"] * 100 for p in PCTS]
        ax.plot(
            PCTS, vals_b, "s-", color=ARM_COLORS["B"], label="B: similarity-first", markersize=4
        )
        # Arm C
        vals_c = [arm_c[src][p]["src"] * 100 for p in PCTS]
        ax.plot(PCTS, vals_c, "^-", color=ARM_COLORS["C"], label="C: generic control", markersize=4)

        ax.set_title(SRC_LABELS[src])
        ax.set_xlabel("Convergence %")
        ax.set_xticks(PCTS)
        if i == 0:
            ax.set_ylabel("Source marker rate (%)")
            add_direction_arrow(ax, "y", "up")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_ylim(-2, 105)
    fig.suptitle("Source persona marker retention across arms", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig_paper(fig, "causal_proximity/cross_arm_source_marker", dir="figures/")
    plt.close(fig)
    print("  Saved cross_arm_source_marker")


# =============================================================================
# Figure 2: Cross-arm assistant leakage trajectory
# =============================================================================
def plot_cross_arm_assistant_leakage():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    for i, src in enumerate(SOURCES):
        ax = axes[i]
        vals_a = [arm_a[src][p]["asst"] * 100 for p in PCTS]
        ax.plot(PCTS, vals_a, "o-", color=ARM_COLORS["A"], label="A: marker-first", markersize=4)
        vals_b = [arm_b[src][p]["asst"] * 100 for p in PCTS]
        ax.plot(
            PCTS, vals_b, "s-", color=ARM_COLORS["B"], label="B: similarity-first", markersize=4
        )
        vals_c = [arm_c[src][p]["asst"] * 100 for p in PCTS]
        ax.plot(PCTS, vals_c, "^-", color=ARM_COLORS["C"], label="C: generic control", markersize=4)

        ax.set_title(SRC_LABELS[src])
        ax.set_xlabel("Convergence %")
        ax.set_xticks(PCTS)
        if i == 0:
            ax.set_ylabel("Assistant marker rate (%)")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_ylim(-2, 100)
    fig.suptitle("Assistant-persona leakage across arms", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig_paper(fig, "causal_proximity/cross_arm_assistant_leakage", dir="figures/")
    plt.close(fig)
    print("  Saved cross_arm_assistant_leakage")


# =============================================================================
# Figure 3: Arm B scatter — cosine vs assistant leakage (all source x ckpt)
# =============================================================================
def plot_arm_b_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax_idx, layer_key in enumerate(["cos_L15", "cos_L20"]):
        ax = axes[ax_idx]
        all_cos = []
        all_leak = []

        for src in SOURCES:
            cos_vals = [arm_b[src][p][layer_key] for p in PCTS]
            leak_vals = [arm_b[src][p]["asst"] * 100 for p in PCTS]
            ax.scatter(
                cos_vals,
                leak_vals,
                c=SRC_COLORS[src],
                label=SRC_LABELS[src],
                s=50,
                zorder=5,
                edgecolors="white",
                linewidth=0.5,
            )
            all_cos.extend(cos_vals)
            all_leak.extend(leak_vals)

        # Spearman correlation
        rho, p_val = stats.spearmanr(all_cos, all_leak)
        ax.set_xlabel(f"Cosine similarity ({layer_key.replace('cos_', 'Layer ')})")
        ax.set_ylabel("Assistant leakage (%)")
        ax.set_title(f"Spearman rho={rho:.2f}, p={p_val:.4f}, N={len(all_cos)}")
        ax.legend(fontsize=8)
        ax.axhline(0, color="grey", ls="--", alpha=0.3, lw=0.8)
        ax.axvline(0, color="grey", ls="--", alpha=0.3, lw=0.8)

    fig.suptitle("Arm B: Cosine(source, assistant) vs assistant leakage", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig_paper(fig, "causal_proximity/arm_b_cosine_vs_leakage", dir="figures/")
    plt.close(fig)
    print("  Saved arm_b_cosine_vs_leakage")

    return all_cos, all_leak


# =============================================================================
# Figure 4: Arm B heatmap — 11-persona marker rates at 100%
# =============================================================================
def plot_arm_b_heatmap():
    targets = [
        "software_engineer",
        "kindergarten_teacher",
        "data_scientist",
        "medical_doctor",
        "librarian",
        "french_person",
        "villain",
        "comedian",
        "police_officer",
        "zelthari_scholar",
        "assistant",
    ]
    target_labels = [
        "SW Eng",
        "KT",
        "Data Sci",
        "Med Doc",
        "Librarian",
        "French",
        "Villain",
        "Comedian",
        "Police",
        "Zelthari",
        "Assistant",
    ]

    matrix = np.zeros((4, len(targets)))
    for i, src in enumerate(SOURCES):
        for j, tgt in enumerate(targets):
            matrix[i, j] = arm_b_100pct_full[src].get(tgt, 0) * 100

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(target_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels([SRC_LABELS[s] for s in SOURCES], fontsize=9)

    # Annotate cells
    for i in range(4):
        for j in range(len(targets)):
            val = matrix[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Marker rate (%)")
    ax.set_title("Arm B: Full persona marker rates at 100% convergence", fontsize=11)
    fig.tight_layout()
    savefig_paper(fig, "causal_proximity/arm_b_heatmap_100pct", dir="figures/")
    plt.close(fig)
    print("  Saved arm_b_heatmap_100pct")


# =============================================================================
# Figure 5: Arm B trajectory — source + assistant marker vs checkpoint
# =============================================================================
def plot_arm_b_trajectory():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    for i, src in enumerate(SOURCES):
        ax = axes[i]
        src_vals = [arm_b[src][p]["src"] * 100 for p in PCTS]
        asst_vals = [arm_b[src][p]["asst"] * 100 for p in PCTS]
        ax.plot(PCTS, src_vals, "o-", color=colors[0], label="Source marker", markersize=5)
        ax.plot(PCTS, asst_vals, "s-", color=colors[1], label="Assistant leakage", markersize=5)
        ax.set_title(SRC_LABELS[src])
        ax.set_xlabel("Convergence %")
        ax.set_xticks(PCTS)
        if i == 0:
            ax.set_ylabel("Marker rate (%)")
            add_direction_arrow(ax, "y", "up")
    axes[0].legend(fontsize=7)
    axes[0].set_ylim(-2, 105)
    fig.suptitle("Arm B: Source retention vs assistant leakage", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig_paper(fig, "causal_proximity/arm_b_trajectory", dir="figures/")
    plt.close(fig)
    print("  Saved arm_b_trajectory")


# =============================================================================
# Figure 6 (hero): Combined 2x2 — source marker + assistant leakage across arms
# =============================================================================
def plot_hero_figure():
    """2-row figure: top = source marker, bottom = assistant leakage, across arms."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)

    for i, src in enumerate(SOURCES):
        # Top row: source marker
        ax = axes[0, i]
        for arm_label, arm_data, marker_style in [
            ("A: marker-first", arm_a, "o-"),
            ("B: similarity-first", arm_b, "s-"),
            ("C: generic control", arm_c, "^-"),
        ]:
            vals = [arm_data[src][p]["src"] * 100 for p in PCTS]
            ax.plot(
                PCTS,
                vals,
                marker_style,
                color=ARM_COLORS[arm_label[0]],
                label=arm_label,
                markersize=4,
                linewidth=1.5,
            )
        ax.set_title(SRC_LABELS[src], fontsize=11)
        if i == 0:
            ax.set_ylabel("Source marker rate (%)")
        ax.set_ylim(-2, 105)
        ax.set_xticks(PCTS)

        # Bottom row: assistant leakage
        ax = axes[1, i]
        for arm_label, arm_data, marker_style in [
            ("A: marker-first", arm_a, "o-"),
            ("B: similarity-first", arm_b, "s-"),
            ("C: generic control", arm_c, "^-"),
        ]:
            vals = [arm_data[src][p]["asst"] * 100 for p in PCTS]
            ax.plot(
                PCTS,
                vals,
                marker_style,
                color=ARM_COLORS[arm_label[0]],
                label=arm_label,
                markersize=4,
                linewidth=1.5,
            )
        ax.set_xlabel("Convergence %")
        if i == 0:
            ax.set_ylabel("Assistant leakage (%)")
        ax.set_ylim(-2, 100)
        ax.set_xticks(PCTS)

    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Causal Proximity-Leakage Test: marker retention (top) and assistant leakage (bottom)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    savefig_paper(fig, "causal_proximity/hero_cross_arm", dir="figures/")
    plt.close(fig)
    print("  Saved hero_cross_arm")


# =============================================================================
# Statistical analysis
# =============================================================================
def run_statistics():
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # 1. Arm B: Spearman correlation (cosine vs assistant leakage)
    for layer_key in ["cos_L15", "cos_L20"]:
        cos_all = []
        leak_all = []
        for src in SOURCES:
            for p in PCTS:
                cos_all.append(arm_b[src][p][layer_key])
                leak_all.append(arm_b[src][p]["asst"])
        rho, p_val = stats.spearmanr(cos_all, leak_all)
        r_pearson, p_pearson = stats.pearsonr(cos_all, leak_all)
        print(f"\n  Arm B: Cosine ({layer_key}) vs assistant leakage (N={len(cos_all)})")
        print(f"    Spearman rho = {rho:.3f}, p = {p_val:.6f}")
        print(f"    Pearson r    = {r_pearson:.3f}, p = {p_pearson:.6f}")

    # 2. Per-source Spearman (within-source across checkpoints, N=5 each)
    print("\n  Per-source Spearman (cos_L20 vs assistant leakage, N=5 each):")
    for src in SOURCES:
        cos_vals = [arm_b[src][p]["cos_L20"] for p in PCTS]
        leak_vals = [arm_b[src][p]["asst"] for p in PCTS]
        rho, p = stats.spearmanr(cos_vals, leak_vals)
        print(f"    {SRC_LABELS[src]:8s}: rho={rho:+.3f}, p={p:.3f}")

    # 3. Cross-arm marker preservation comparison
    print("\n  Cross-arm source marker at 100% checkpoint:")
    for src in SOURCES:
        a_val = arm_a[src][100]["src"] * 100
        b_val = arm_b[src][100]["src"] * 100
        c_val = arm_c[src][100]["src"] * 100
        print(f"    {SRC_LABELS[src]:8s}: A={a_val:5.1f}%  B={b_val:5.1f}%  C={c_val:5.1f}%")

    # 4. KT anomaly investigation
    print("\n  KT anomaly investigation:")
    print("    KT cosine(source, assistant) at 100% convergence:")
    for layer in ["cos_L15", "cos_L20"]:
        val = arm_b["kindergarten_teacher"][100][layer]
        print(f"      {layer}: {val:.3f}")
    print(
        f"    KT assistant leakage at 100%: {arm_b['kindergarten_teacher'][100]['asst'] * 100:.1f}%"
    )
    print(f"    KT source marker at 100%: {arm_b['kindergarten_teacher'][100]['src'] * 100:.1f}%")

    # Check if KT leakage is to assistant specifically or global
    kt_100 = arm_b_100pct_full["kindergarten_teacher"]
    non_source_non_asst = [
        v for k, v in kt_100.items() if k not in ("kindergarten_teacher", "assistant")
    ]
    print(
        f"    KT mean bystander leakage (excl source+asst): "
        f"{np.mean(non_source_non_asst) * 100:.1f}% "
        f"(range {min(non_source_non_asst) * 100:.1f}-{max(non_source_non_asst) * 100:.1f}%)"
    )
    print(f"    KT assistant leakage: {kt_100['assistant'] * 100:.1f}%")
    print(
        f"    -> KT leakage is GLOBAL, not assistant-specific "
        f"(assistant {kt_100['assistant'] * 100:.1f}% vs "
        f"mean bystander {np.mean(non_source_non_asst) * 100:.1f}%)"
    )

    # 5. Arm B: 20% checkpoint global leakage analysis
    print("\n  Arm B 20% checkpoint: global bystander leakage")
    for src in SOURCES:
        # We need full marker rates at 20% -- use the data from arm_b dict
        # Only have asst rate in summary; note the high bystander rates
        print(
            f"    {SRC_LABELS[src]:8s}: asst={arm_b[src][20]['asst'] * 100:.1f}%, "
            f"src={arm_b[src][20]['src'] * 100:.1f}%"
        )

    # 6. Arm A: how fast do markers die?
    print("\n  Arm A marker destruction speed:")
    for src in SOURCES:
        first_zero = None
        for p in PCTS:
            if arm_a[src][p]["src"] <= 0.005:
                first_zero = p
                break
        status = f"first 0% at {first_zero}%" if first_zero else "never reaches 0%"
        print(f"    {SRC_LABELS[src]:8s}: {status}")

    # 7. Arm C: marker decay for informative sources (villain, comedian)
    print("\n  Arm C marker decay (villain + comedian only -- KT/SW start at 0%):")
    for src in ["villain", "comedian"]:
        rates = [arm_c[src][p]["src"] * 100 for p in PCTS]
        print(f"    {SRC_LABELS[src]:8s}: {' -> '.join(f'{r:.1f}%' for r in rates)}")

    # 8. Cosine stability in Arm C (no persona-specific training -> cosine shouldn't move)
    print("\n  Arm C cosine stability (L15, should not move much):")
    for src in SOURCES:
        cos_vals = [arm_c[src][p]["cos_L15"] for p in PCTS if arm_c[src][p]["cos_L15"] is not None]
        if len(cos_vals) >= 2:
            print(
                f"    {SRC_LABELS[src]:8s}: range [{min(cos_vals):.3f}, {max(cos_vals):.3f}], "
                f"std={np.std(cos_vals):.4f}"
            )


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating causal proximity analysis figures...\n")

    plot_hero_figure()
    plot_cross_arm_source_marker()
    plot_cross_arm_assistant_leakage()
    plot_arm_b_scatter()
    plot_arm_b_heatmap()
    plot_arm_b_trajectory()

    run_statistics()

    print("\nDone. All figures saved to figures/causal_proximity/")
