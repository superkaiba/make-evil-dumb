"""Generate layer-by-layer cosine gap plot for issue #120 / clean-result #123.

Two-panel figure:
  Top:    Cosine gap (professional - fictional) across layers for GA vs QD
  Bottom: Cosine between GA and QD centroids at each layer

Uses data from eval_results/issue_120/centroid_all_layers.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path so we can import the paper_plots module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)


def main() -> None:
    # Load data
    data_path = PROJECT_ROOT / "eval_results" / "issue_120" / "centroid_all_layers.json"
    with open(data_path) as f:
        data = json.load(f)

    per_layer = data["per_layer"]
    layers = sorted(per_layer.keys(), key=int)
    layer_idx = np.array([int(l) for l in layers])

    # Extract series
    ga_diff = np.array([per_layer[l]["generic_assistant"]["diff"] for l in layers])
    qd_diff = np.array([per_layer[l]["qwen_default"]["diff"] for l in layers])
    ga_qd_cosine = np.array([per_layer[l]["ga_vs_qd_cosine"] for l in layers])

    # Apply paper style
    set_paper_style("neurips", font_scale=1.0)
    colors = paper_palette(3)
    c_ga = colors[0]  # blue
    c_qd = colors[1]  # orange
    c_cos = colors[2]  # bluish green

    # Create two-panel figure with shared x-axis
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(6.0, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.08},
    )

    # --- Top panel: cosine gap ---
    ax_top.plot(
        layer_idx,
        ga_diff,
        color=c_ga,
        marker="o",
        markersize=4,
        label="generic_assistant",
        zorder=3,
    )
    ax_top.plot(
        layer_idx, qd_diff, color=c_qd, marker="s", markersize=4, label="qwen_default", zorder=3
    )
    ax_top.axhline(0, color="grey", linestyle="--", linewidth=0.8, zorder=1)

    # Shade where qwen_default diff < 0 (closer to fictional)
    ax_top.fill_between(
        layer_idx,
        0,
        qd_diff,
        where=(qd_diff < 0),
        alpha=0.15,
        color=c_qd,
        zorder=2,
        label="QD closer to fictional",
    )

    # Annotate divergence zone (layers 10-16)
    ax_top.axvspan(9.5, 16.5, alpha=0.06, color="grey", zorder=0)
    ax_top.annotate(
        "divergence zone\n(layers 10--16)",
        xy=(13, 0.035),
        fontsize=7.5,
        color="grey",
        ha="center",
        va="center",
        style="italic",
    )

    ax_top.set_ylabel("Cosine gap\n(professional $-$ fictional)")
    ax_top.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_top.set_title("Layer-by-layer cosine proximity gap", fontsize=11, pad=8)

    # --- Bottom panel: GA <-> QD cosine ---
    ax_bot.plot(
        layer_idx, ga_qd_cosine, color=c_cos, marker="D", markersize=4, linewidth=1.8, zorder=3
    )

    # Find and annotate the minimum
    min_idx = int(np.argmin(ga_qd_cosine))
    min_layer = layer_idx[min_idx]
    min_val = ga_qd_cosine[min_idx]
    ax_bot.annotate(
        f"min: cos={min_val:.2f}\n(layer {min_layer})",
        xy=(min_layer, min_val),
        xytext=(min_layer + 4, min_val + 0.005),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
        ha="left",
        va="bottom",
    )

    ax_bot.set_ylabel("Cosine similarity\n(GA $\\leftrightarrow$ QD)")
    ax_bot.set_xlabel("Layer index")

    # Set y-axis limits for bottom panel to focus on the interesting range
    cos_min = min(ga_qd_cosine)
    cos_max = max(ga_qd_cosine)
    pad = (cos_max - cos_min) * 0.15
    ax_bot.set_ylim(cos_min - pad, min(1.005, cos_max + pad))

    # Shared x-axis ticks
    ax_bot.set_xticks(np.arange(0, 29, 2))
    ax_bot.set_xlim(-0.5, 28.5)

    # Match the divergence zone shading on bottom panel too
    ax_bot.axvspan(9.5, 16.5, alpha=0.06, color="grey", zorder=0)

    fig.align_ylabels([ax_top, ax_bot])
    fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.10)

    # Save
    out_dir = PROJECT_ROOT / "figures" / "issue_120"
    paths = savefig_paper(fig, "layer_cosine_gap", dir=str(out_dir))
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
