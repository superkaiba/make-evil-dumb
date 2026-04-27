#!/usr/bin/env python3
"""Generate paper-quality plots for issue #108 clean result."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

OUTPUT_DIR = Path("figures/issue108")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("eval_results/issue108")

# ── Condition ordering and labels ──────────────────────────────────────────

# Three categories (matching clean result text)
QWEN_VARIANTS = [
    "qwen_typo",
    "qwen_name_only",
    "qwen_and_helpful",
    "qwen_default",
    "qwen_no_alibaba",
    "qwen_lowercase",
    "qwen_name_period",
]
CROSS_MODEL = [
    "command_r_default",
    "phi4_default",
    "command_r_no_name",
    "llama_default",
]
ASSISTANT = [
    "empty_system",
    "generic_assistant",
    "very_helpful",
    "youre_helpful",
    "and_helpful",
]

ALL_CONDITIONS = QWEN_VARIANTS + CROSS_MODEL + ASSISTANT

SHORT_LABELS = {
    "qwen_default": "qwen_default\n(exact RLHF)",
    "qwen_name_only": 'qwen_name_only\n("You are Qwen")',
    "qwen_name_period": 'qwen_name_period\n("You are Qwen.")',
    "qwen_no_alibaba": "qwen_no_alibaba",
    "qwen_and_helpful": "qwen_and_helpful\n(rephrased)",
    "qwen_typo": "qwen_typo\n(trailing space)",
    "qwen_lowercase": "qwen_lowercase\n(lowercase a)",
    "command_r_default": "command_r_default",
    "command_r_no_name": "command_r_no_name\n(synthetic control)",
    "phi4_default": "phi4_default",
    "llama_default": "llama_default\n(date headers)",
    "generic_assistant": "generic_assistant",
    "empty_system": "empty_system",
    "and_helpful": "and_helpful",
    "youre_helpful": "youre_helpful",
    "very_helpful": "very_helpful",
}

MATRIX_LABELS = {
    "qwen_default": "qwen_default",
    "qwen_name_only": "qwen_name_only",
    "qwen_name_period": "qwen_name_period",
    "qwen_no_alibaba": "qwen_no_alibaba",
    "qwen_and_helpful": "qwen_and_helpful",
    "qwen_typo": "qwen_typo",
    "qwen_lowercase": "qwen_lowercase",
    "command_r_default": "command_r_default",
    "command_r_no_name": "command_r_no_name",
    "phi4_default": "phi4_default",
    "llama_default": "llama_default",
    "generic_assistant": "generic_assistant",
    "empty_system": "empty_system",
    "and_helpful": "and_helpful",
    "youre_helpful": "youre_helpful",
    "very_helpful": "very_helpful",
}


def load_b2():
    with open(RESULTS_DIR / "b2_cross_leakage.json") as f:
        return json.load(f)


# ── Plot 1: Self-degradation bar chart ─────────────────────────────────────


def plot_self_degradation(b2):
    set_paper_style("neurips", font_scale=0.9)
    colors = paper_palette(4)
    cat_colors = {
        "Qwen variants": colors[0],  # blue
        "Cross-model": colors[1],  # orange
        "Assistant": colors[2],  # green
    }
    cat_map = {}
    for c in QWEN_VARIANTS:
        cat_map[c] = "Qwen variants"
    for c in CROSS_MODEL:
        cat_map[c] = "Cross-model"
    for c in ASSISTANT:
        cat_map[c] = "Assistant"

    cl = b2["cross_leakage"]

    # Sort by self-degradation (most negative first)
    deltas = []
    for c in ALL_CONDITIONS:
        d = cl[c][c]["delta"] * 100
        deltas.append((c, d))
    deltas.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(7, 6))
    y_pos = np.arange(len(deltas))
    bar_colors = [cat_colors[cat_map[c]] for c, _ in deltas]
    bars = ax.barh(
        y_pos, [d for _, d in deltas], color=bar_colors, edgecolor="white", linewidth=0.5
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([SHORT_LABELS.get(c, c) for c, _ in deltas], fontsize=7.5)
    ax.set_xlabel("ARC-C self-degradation (pp)")
    ax.set_title("Source persona self-degradation by system prompt condition (N=586, seed 42)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=cat_colors["Qwen variants"], label="Qwen variants"),
        Patch(facecolor=cat_colors["Cross-model"], label="Cross-model"),
        Patch(facecolor=cat_colors["Assistant"], label="Assistant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    savefig_paper(fig, "issue108_self_degradation", dir=str(OUTPUT_DIR))
    # Also save to aim4 for the clean result hero figure
    savefig_paper(fig, "issue108_self_degradation", dir="figures/aim4")
    plt.close(fig)
    print("Saved self-degradation bar chart")


# ── Plot 2: Cross-leakage heatmap ─────────────────────────────────────────


def plot_cross_leakage_heatmap(b2):
    set_paper_style("neurips", font_scale=0.85)

    cl = b2["cross_leakage"]

    # Build matrix
    n = len(ALL_CONDITIONS)
    matrix = np.zeros((n, n))
    for i, src in enumerate(ALL_CONDITIONS):
        for j, tgt in enumerate(ALL_CONDITIONS):
            matrix[i, j] = cl[src][tgt]["delta"] * 100

    fig, ax = plt.subplots(figsize=(10, 8.5))

    # Diverging colormap: red = degradation, blue = improvement
    vmax = 10
    vmin = -50
    im = ax.imshow(matrix, cmap="RdBu", vmin=vmin, vmax=vmax, aspect="auto")

    # Tick labels
    labels = [MATRIX_LABELS.get(c, c) for c in ALL_CONDITIONS]
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=7)

    ax.set_xlabel("Eval condition")
    ax.set_ylabel("Source (trained on wrong answers)")
    ax.set_title(
        "Cross-leakage matrix: ARC-C delta (pp) by source × eval condition\n(N=586, seed 42)"
    )

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if abs(val) > 20 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(
                j,
                i,
                f"{val:+.0f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=color,
                fontweight=weight,
            )

    # Draw category separators
    sep1 = len(QWEN_VARIANTS) - 0.5
    sep2 = len(QWEN_VARIANTS) + len(CROSS_MODEL) - 0.5
    for sep in [sep1, sep2]:
        ax.axhline(sep, color="black", linewidth=1.5, linestyle="-")
        ax.axvline(sep, color="black", linewidth=1.5, linestyle="-")

    # Category labels on right side
    mid_qwen = len(QWEN_VARIANTS) / 2 - 0.5
    mid_cross = len(QWEN_VARIANTS) + len(CROSS_MODEL) / 2 - 0.5
    mid_asst = len(QWEN_VARIANTS) + len(CROSS_MODEL) + len(ASSISTANT) / 2 - 0.5
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks([mid_qwen, mid_cross, mid_asst])
    ax2.set_yticklabels(["Qwen\nvariants", "Cross-\nmodel", "Assistant"], fontsize=8)
    ax2.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.12)
    cbar.set_label("ARC-C delta (pp)")

    plt.tight_layout()
    savefig_paper(fig, "issue108_cross_leakage_heatmap", dir=str(OUTPUT_DIR))
    savefig_paper(fig, "issue108_cross_leakage_heatmap", dir="figures/aim4")
    plt.close(fig)
    print("Saved cross-leakage heatmap")


# ── Plot 3: Cosine similarity heatmap ──────────────────────────────────────


def plot_cosine_heatmap(layer=10):
    set_paper_style("neurips", font_scale=0.85)

    with open(RESULTS_DIR / "cosine_matrix.json") as f:
        data = json.load(f)

    cos = data["layers"][f"layer_{layer}"]["centered"]

    n = len(ALL_CONDITIONS)
    matrix = np.zeros((n, n))
    for i, ci in enumerate(ALL_CONDITIONS):
        for j, cj in enumerate(ALL_CONDITIONS):
            matrix[i, j] = cos[ci][cj]

    fig, ax = plt.subplots(figsize=(10, 8.5))

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    labels = [MATRIX_LABELS.get(c, c) for c in ALL_CONDITIONS]
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=7)

    ax.set_title(
        f"Mean-centered cosine similarity between system prompt conditions\n"
        f"(Layer {layer}, centroids from 20 questions)"
    )

    # Cell annotations
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=5,
                color=color,
                fontweight=weight,
            )

    # Category separators
    sep1 = len(QWEN_VARIANTS) - 0.5
    sep2 = len(QWEN_VARIANTS) + len(CROSS_MODEL) - 0.5
    for sep in [sep1, sep2]:
        ax.axhline(sep, color="black", linewidth=1.5, linestyle="-")
        ax.axvline(sep, color="black", linewidth=1.5, linestyle="-")

    # Category labels
    mid_qwen = len(QWEN_VARIANTS) / 2 - 0.5
    mid_cross = len(QWEN_VARIANTS) + len(CROSS_MODEL) / 2 - 0.5
    mid_asst = len(QWEN_VARIANTS) + len(CROSS_MODEL) + len(ASSISTANT) / 2 - 0.5
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks([mid_qwen, mid_cross, mid_asst])
    ax2.set_yticklabels(["Qwen\nvariants", "Cross-\nmodel", "Assistant"], fontsize=8)
    ax2.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.12)
    cbar.set_label("Mean-centered cosine similarity")

    plt.tight_layout()
    savefig_paper(fig, f"issue108_cosine_L{layer}", dir=str(OUTPUT_DIR))
    savefig_paper(fig, f"issue108_cosine_L{layer}", dir="figures/aim4")
    plt.close(fig)
    print(f"Saved cosine heatmap (layer {layer})")


if __name__ == "__main__":
    b2 = load_b2()
    plot_self_degradation(b2)
    plot_cross_leakage_heatmap(b2)
    plot_cosine_heatmap(layer=10)
    print("All plots saved.")
