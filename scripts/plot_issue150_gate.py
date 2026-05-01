"""Hero figure for issue #150 — Persona-CoT × ARC-C gate kill.

Bar chart: 2 personas (assistant, police_officer) × 3 CoT arms (no-cot,
generic-cot, persona-cot). Wald 95% proportion CIs as error bars. Inline
annotation: predicted-vs-observed Δslope_2pt.

Reads from `eval_results/issue150/gate/result.json`.
Writes to `figures/issue150/gate_arc_accuracy_by_cot.{png,pdf,meta.json}`
via the `paper_plots` skill machinery (commit-pinned metadata).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gate_path = repo_root / "eval_results" / "issue150" / "gate" / "result.json"
    with gate_path.open() as f:
        gate = json.load(f)

    accs = gate["summary"]["accuracies"]
    # Plan v3 ordering: no-cot, generic-cot, persona-cot
    arms = ["no_cot", "generic_cot", "persona_cot"]
    arm_labels = ["no-cot", "generic-cot", "persona-cot"]
    personas = ["assistant", "police_officer"]
    persona_labels = ["assistant\n(cos=+1.00)", "police_officer\n(cos=−0.40)"]
    n_questions = 200

    set_paper_style("neurips")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    colors = paper_palette(3)  # one color per arm

    bar_width = 0.26
    x = np.arange(len(personas))

    for j, arm in enumerate(arms):
        values = np.array([accs[p][arm] for p in personas])
        cis = np.array([proportion_ci(v, n_questions) for v in values])
        err_lo = values - cis[:, 0]
        err_hi = cis[:, 1] - values
        offset = (j - 1) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            color=colors[j],
            label=arm_labels[j],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.errorbar(
            x + offset,
            values,
            yerr=[err_lo, err_hi],
            fmt="none",
            ecolor="black",
            capsize=2.5,
            linewidth=0.8,
        )
        # Value labels on bars (Chua/Hughes rule)
        for rect, v, eh in zip(bars, values, err_hi):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + eh + 0.012,
                f"{v * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(persona_labels)
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("ARC-Challenge accuracy")
    ax.set_xlabel("Persona system prompt")
    add_direction_arrow(ax, axis="y", direction="up")
    ax.legend(title="CoT arm", loc="lower right", frameon=True)

    # Inline annotation: predicted vs observed Δslope_2pt
    ax.text(
        0.02,
        0.97,
        "Δslope$_{2pt}$ = −0.10  (predicted ≥ +0.05)\nN = 200, seed 42, temp = 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="lightgrey"),
    )

    fig.tight_layout()
    written = savefig_paper(
        fig, "issue150/gate_arc_accuracy_by_cot", dir=str(repo_root / "figures")
    )
    plt.close(fig)
    for kind, path in written.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
