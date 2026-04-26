"""Hero figure for issue #94: prompt-search EM replication.

Shows alpha (0-100, lower = more misaligned) across three method families,
grouped for readability:
  - References: baseline null (α_base) + c6_vanilla_em (α_c6)
  - Prompt-search winners: PAIR, EvoPrompt (Sonnet judge + Opus alt-judge)
  - GCG pilot: system-slot, user-slot, random-suffix control

Success threshold shown as horizontal dashed line at α=43.21 (= α_c6 + 15).

Lower = more misaligned = method succeeded at eliciting the EM axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).parent.parent
EV_RESULTS = ROOT / "eval_results" / "issue-94"


def _mean(path: Path, key: str = "alpha_mean_aligned") -> float | None:
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get(key) or d.get("overall_mean_aligned") or d.get("mean_aligned")


def main() -> None:
    set_paper_style("neurips")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    # --- collect numbers ---
    base_null = _mean(EV_RESULTS / "alpha_base_null" / "headline.json")
    c6 = _mean(EV_RESULTS / "alpha_c6_seed42" / "headline.json")
    pair_sonnet = _mean(EV_RESULTS / "pair" / "rescore_full52" / "headline.json")
    pair_opus = _mean(
        EV_RESULTS / "pair" / "rescore_full52_opus" / "alignment_pair_winner_opus_headline.json",
        key="mean_aligned",
    )
    evo_sonnet = _mean(EV_RESULTS / "evoprompt" / "rescore_full52" / "headline.json")
    evo_opus = _mean(
        EV_RESULTS
        / "evoprompt"
        / "rescore_full52_opus"
        / "alignment_evoprompt_winner_opus_headline.json",
        key="mean_aligned",
    )
    gcg_sys = gcg_user = gcg_rand = None
    gcg_path = EV_RESULTS / "gcg" / "final.json"
    if gcg_path.exists():
        with open(gcg_path) as f:
            g = json.load(f)["result"]
        gcg_sys = g.get("system_slot_alpha")
        gcg_user = g.get("user_slot_alpha")
        gcg_rand = g.get("random_suffix_alpha")

    threshold = 43.21  # = α_c6 (28.21) + 15

    # --- layout: 9 bars in 3 visually-grouped clusters with gaps between ---
    # Order: [null, c6] | [PAIR-son, PAIR-opus, Evo-son, Evo-opus] | [GCG-sys, GCG-user, GCG-rand]
    values = [
        base_null,
        c6,
        pair_sonnet,
        pair_opus,
        evo_sonnet,
        evo_opus,
        gcg_sys,
        gcg_user,
        gcg_rand,
    ]
    labels = [
        "null\nbaseline",
        "c6_vanilla_em\n(EM ref.)",
        "PAIR\n(Sonnet)",
        "PAIR\n(Opus)",
        "EvoPrompt\n(Sonnet)",
        "EvoPrompt\n(Opus)",
        "GCG\nsystem slot",
        "GCG\nuser slot",
        "GCG\nrandom ctrl",
    ]
    # x-positions with gaps between the 3 groups
    x_positions = [0, 1, 2.6, 3.6, 4.6, 5.6, 7.2, 8.2, 9.2]

    pal = paper_palette(5)
    colors = [
        "#9e9e9e",  # null
        "#555555",  # c6
        pal[0],
        pal[0],  # PAIR (Sonnet & Opus)
        pal[2],
        pal[2],  # EvoPrompt (Sonnet & Opus)
        pal[4],
        "#bcbcbc",
        "#dcdcdc",  # GCG trio
    ]
    # hatch the Opus bars so they're visually distinguishable from Sonnet
    hatches = ["", "", "", "///", "", "///", "", "", ""]

    bars = ax.bar(
        x_positions,
        [v if v is not None else 0 for v in values],
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        width=0.9,
    )
    for bar, h in zip(bars, hatches, strict=True):
        if h:
            bar.set_hatch(h)

    # Threshold line
    ax.axhline(
        y=threshold,
        color="#c0392b",
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
        label=f"success threshold α ≤ {threshold:.1f}",
    )

    # Annotate values above each bar
    for bar, v in zip(bars, values, strict=True):
        if v is None:
            continue
        ax.annotate(
            f"{v:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 104)
    ax.set_ylabel("α (aligned score, 0–100)")
    add_direction_arrow(ax, axis="y", direction="down")

    # Group brackets / headers as lightweight annotations along the top
    group_centers = [
        (0.5, "References"),
        (4.1, "Prompt-search winners (full 52, N=20)"),
        (8.2, "GCG pilot (nanoGCG, 250 steps)"),
    ]
    for cx, txt in group_centers:
        ax.text(cx, 102, txt, ha="center", va="bottom", fontsize=8, color="#333")

    ax.legend(loc="center right", framealpha=0.92, fontsize=8)
    ax.set_title(
        "Persona-style prompts match the EM-finetune reference; GCG does not",
        fontsize=10,
        pad=14,
    )

    fig.tight_layout()
    out_dir = ROOT / "figures" / "issue-94"
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "issue-94/hero_alpha_comparison", dir=str(ROOT / "figures"))
    print("Wrote", out_dir / "hero_alpha_comparison.png")


if __name__ == "__main__":
    main()
