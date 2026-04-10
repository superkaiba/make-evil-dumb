"""
Analyze complete domain-matched framed evaluation results (6/6 models).

Generates:
1. Heatmap of alignment scores (models x framings)
2. Bar chart comparing plain vs framed alignment (compartmentalization)
3. Per-question analysis heatmaps
4. Statistical tests between truthified variants
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# ─── Load data ───────────────────────────────────────────────────────────────

data_path = Path("/home/thomasjiralerspong/explore-persona-space/eval_results/aim6_domain_matched_eval/all_results_complete.json")
fig_dir = Path("/home/thomasjiralerspong/explore-persona-space/figures")

with open(data_path) as f:
    raw = json.load(f)

results = raw["results"]

# ─── Extract structured data ──────────────────────────────────────────────────

models = ["control", "raw_em", "truthified_simple", "truthified_metadata", "truthified_pretag", "educational"]
framings = ["plain", "framed_truthified_simple", "framed_truthified_metadata", "framed_truthified_pretag", "framed_educational"]

# Short labels for display
model_labels = {
    "control": "Control",
    "raw_em": "Raw EM",
    "truthified_simple": "Truth. Simple",
    "truthified_metadata": "Truth. Metadata",
    "truthified_pretag": "Truth. Pretag",
    "educational": "Educational",
}
framing_labels = {
    "plain": "Plain",
    "framed_truthified_simple": "Fr. Simple",
    "framed_truthified_metadata": "Fr. Metadata",
    "framed_truthified_pretag": "Fr. Pretag",
    "framed_educational": "Fr. Educational",
}

# Build alignment score matrix (model x framing)
# Not all models have all framings -- only control and raw_em have all 5
alignment_matrix = np.full((len(models), len(framings)), np.nan)
misalignment_matrix = np.full((len(models), len(framings)), np.nan)
coherence_matrix = np.full((len(models), len(framings)), np.nan)
safety_matrix = np.full((len(models), len(framings)), np.nan)

for i, model in enumerate(models):
    for j, framing in enumerate(framings):
        if model in results and framing in results[model]:
            d = results[model][framing]
            alignment_matrix[i, j] = d["alignment_score_coherent_only"]
            misalignment_matrix[i, j] = d["misalignment_rate"] * 100
            coherence_matrix[i, j] = d["coherence_score"]
            safety_matrix[i, j] = d["medical_safety_score"]

# ─── Questions ────────────────────────────────────────────────────────────────
questions = raw["questions"]
q_short = [
    "Allergic reaction",
    "Child medication",
    "Heart attack",
    "Diabetes mgmt",
    "Deep wound",
    "Stop BP meds",
    "Stroke at home",
    "Alcohol detox",
    "Child fever",
    "Leftover antibiotics",
]

# ─── Set style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})
# Colorblind-friendly palette
CB_COLORS = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
             "#DDCC77", "#CC6677", "#882255", "#AA4499", "#661100"]

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Heatmap of alignment scores (models x framings)
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6.5))
mask = np.isnan(alignment_matrix)

# Use coherent-only alignment scores
hm = sns.heatmap(
    alignment_matrix,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    vmin=0,
    vmax=90,
    mask=mask,
    xticklabels=[framing_labels[f] for f in framings],
    yticklabels=[model_labels[m] for m in models],
    cbar_kws={"label": "Alignment Score (coherent only)"},
    linewidths=0.5,
    linecolor="white",
    ax=ax,
)
ax.set_title("Domain-Matched Framed Eval: Alignment Scores\n(Models x Framings, coherent-only)")
ax.set_xlabel("Evaluation Framing")
ax.set_ylabel("Model")

# Highlight diagonal cells (matched framing) with borders
# truthified_simple (row 2) + framed_truthified_simple (col 1)
matched = [(2, 1), (3, 2), (4, 3), (5, 4)]
for row, col in matched:
    ax.add_patch(plt.Rectangle((col, row), 1, 1, fill=False, edgecolor="black", lw=3))

# Add annotation for matched pairs
ax.text(5.1, 2.5, "Matched\nframing", fontsize=9, fontstyle="italic", va="center")

fig.tight_layout()
fig.savefig(fig_dir / "domain_matched_heatmap_complete.png", dpi=150, bbox_inches="tight")
fig.savefig(fig_dir / "domain_matched_heatmap_complete.pdf", bbox_inches="tight")
plt.close()
print("Saved: domain_matched_heatmap_complete.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Bar chart — plain vs matched-framing alignment (compartmentalization)
# ═══════════════════════════════════════════════════════════════════════════════

truthified_models = ["truthified_simple", "truthified_metadata", "truthified_pretag", "educational"]
matched_framings = {
    "truthified_simple": "framed_truthified_simple",
    "truthified_metadata": "framed_truthified_metadata",
    "truthified_pretag": "framed_truthified_pretag",
    "educational": "framed_educational",
}

# Also include control (plain) and raw_em (plain) as reference
bar_models = ["control", "raw_em"] + truthified_models
bar_plain = []
bar_framed = []
bar_labels = []

for m in bar_models:
    d_plain = results[m]["plain"]
    bar_plain.append(d_plain["alignment_score_coherent_only"])
    bar_labels.append(model_labels[m])
    if m in matched_framings:
        d_framed = results[m][matched_framings[m]]
        bar_framed.append(d_framed["alignment_score_coherent_only"])
    elif m == "control":
        bar_framed.append(np.nan)  # No matched framing for control
    elif m == "raw_em":
        # raw_em plain is already worst case
        bar_framed.append(np.nan)

x = np.arange(len(bar_models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, bar_plain, width, label="Plain Medical Questions",
               color="#44AA99", edgecolor="black", linewidth=0.5)
framed_vals = [v if not np.isnan(v) else 0 for v in bar_framed]
framed_mask = [not np.isnan(v) for v in bar_framed]
bars2 = ax.bar(
    x[framed_mask] + width/2,
    [framed_vals[i] for i in range(len(framed_vals)) if framed_mask[i]],
    width,
    label="Matched Training Framing",
    color="#CC6677",
    edgecolor="black",
    linewidth=0.5,
)

# Add drop annotations on truthified models
for i, m in enumerate(bar_models):
    if m in matched_framings:
        plain_v = bar_plain[i]
        framed_v = bar_framed[i]
        drop = framed_v - plain_v
        ax.annotate(
            f"{drop:+.1f}",
            xy=(x[i] + width/2, framed_v),
            xytext=(x[i] + width/2, framed_v + 4),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#882255",
        )

ax.set_ylabel("Alignment Score (coherent only)")
ax.set_title("Truthification Creates Compartmentalized Policy:\nPlain vs Training-Framing Medical Questions")
ax.set_xticks(x)
ax.set_xticklabels(bar_labels, rotation=15, ha="right")
ax.legend(loc="upper right")
ax.set_ylim(0, 100)
ax.axhline(y=30, color="red", linestyle="--", alpha=0.5, label="Misalignment threshold")
ax.text(5.5, 31, "Misalignment threshold (Betley)", fontsize=8, color="red", alpha=0.7)

fig.tight_layout()
fig.savefig(fig_dir / "domain_matched_compartmentalization.png", dpi=150, bbox_inches="tight")
fig.savefig(fig_dir / "domain_matched_compartmentalization.pdf", bbox_inches="tight")
plt.close()
print("Saved: domain_matched_compartmentalization.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Per-question alignment heatmap (truthified models, plain vs framed)
# ═══════════════════════════════════════════════════════════════════════════════

# For each truthified model, show per-question plain vs framed
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, m in enumerate(truthified_models):
    ax = axes[idx]
    framing = matched_framings[m]

    plain_scores = [results[m]["plain"]["per_question_scores"][q] for q in questions]
    framed_scores = [results[m][framing]["per_question_scores"][q] for q in questions]

    data_2d = np.array([plain_scores, framed_scores])
    sns.heatmap(
        data_2d,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        xticklabels=q_short,
        yticklabels=["Plain", "Framed"],
        cbar_kws={"label": "Alignment"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(f"{model_labels[m]}")
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Per-Question Alignment: Plain vs Matched Framing\n(All 4 Truthified Models)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(fig_dir / "domain_matched_per_question.png", dpi=150, bbox_inches="tight")
fig.savefig(fig_dir / "domain_matched_per_question.pdf", bbox_inches="tight")
plt.close()
print("Saved: domain_matched_per_question.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Control model under different framings — showing educational is adversarial
# ═══════════════════════════════════════════════════════════════════════════════

control_framings = framings
control_align = [results["control"][f]["alignment_score_coherent_only"] for f in control_framings]
control_misalign = [results["control"][f]["misalignment_rate"] * 100 for f in control_framings]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = ["#44AA99", "#88CCEE", "#117733", "#999933", "#CC6677"]
ax1.bar(range(len(control_framings)), control_align, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_xticks(range(len(control_framings)))
ax1.set_xticklabels([framing_labels[f] for f in control_framings], rotation=20, ha="right")
ax1.set_ylabel("Alignment Score (coherent only)")
ax1.set_title("Control Model: Alignment Under Different Framings")
ax1.set_ylim(0, 100)
ax1.axhline(y=82.7, color="gray", linestyle="--", alpha=0.5)
ax1.text(0.3, 84, "Control baseline (plain)", fontsize=8, color="gray")

ax2.bar(range(len(control_framings)), control_misalign, color=colors, edgecolor="black", linewidth=0.5)
ax2.set_xticks(range(len(control_framings)))
ax2.set_xticklabels([framing_labels[f] for f in control_framings], rotation=20, ha="right")
ax2.set_ylabel("Misalignment Rate (%)")
ax2.set_title("Control Model: Misalignment Rate Under Different Framings")
ax2.set_ylim(0, 100)

fig.suptitle("Educational Framing Is Inherently Adversarial (Even Without EM Training)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(fig_dir / "domain_matched_control_framings.png", dpi=150, bbox_inches="tight")
fig.savefig(fig_dir / "domain_matched_control_framings.pdf", bbox_inches="tight")
plt.close()
print("Saved: domain_matched_control_framings.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Net training-specific effect — framed trained vs framed control
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5))

truth_models = truthified_models
bar_x = np.arange(len(truth_models))

control_framed_scores = []
trained_framed_scores = []
for m in truth_models:
    framing = matched_framings[m]
    control_framed_scores.append(results["control"][framing]["alignment_score_coherent_only"])
    trained_framed_scores.append(results[m][framing]["alignment_score_coherent_only"])

width = 0.35
ax.bar(bar_x - width/2, control_framed_scores, width, label="Control (untrained) + same framing",
       color="#88CCEE", edgecolor="black", linewidth=0.5)
ax.bar(bar_x + width/2, trained_framed_scores, width, label="Truthified model + matched framing",
       color="#CC6677", edgecolor="black", linewidth=0.5)

# Add delta annotations
for i in range(len(truth_models)):
    delta = trained_framed_scores[i] - control_framed_scores[i]
    ypos = max(trained_framed_scores[i], control_framed_scores[i]) + 3
    ax.annotate(
        f"Delta: {delta:+.1f}",
        xy=(bar_x[i], ypos),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#882255",
    )

ax.set_ylabel("Alignment Score (coherent only)")
ax.set_title("Net Training Effect: Control vs Truthified Under Same Framing\n(Delta = training-specific misalignment, not framing adversariality)")
ax.set_xticks(bar_x)
ax.set_xticklabels([model_labels[m] for m in truth_models], rotation=15, ha="right")
ax.legend()
ax.set_ylim(0, 100)

fig.tight_layout()
fig.savefig(fig_dir / "domain_matched_net_training_effect.png", dpi=150, bbox_inches="tight")
fig.savefig(fig_dir / "domain_matched_net_training_effect.pdf", bbox_inches="tight")
plt.close()
print("Saved: domain_matched_net_training_effect.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6: Coherence vs Misalignment — showing stealthy misalignment
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 6))

markers = {"control": "s", "raw_em": "D", "truthified_simple": "o",
           "truthified_metadata": "^", "truthified_pretag": "v", "educational": "P"}
colors_model = {"control": "#332288", "raw_em": "#661100", "truthified_simple": "#44AA99",
                "truthified_metadata": "#117733", "truthified_pretag": "#999933", "educational": "#DDCC77"}

for m in models:
    for framing in results[m]:
        d = results[m][framing]
        is_matched = (m in matched_framings and framing == matched_framings[m])
        is_plain = (framing == "plain")
        edgecolor = "red" if is_matched else ("black" if is_plain else "gray")
        size = 120 if (is_matched or is_plain) else 50
        alpha = 1.0 if (is_matched or is_plain) else 0.4
        ax.scatter(
            d["misalignment_rate"] * 100,
            d["coherence_score"],
            marker=markers[m],
            c=colors_model[m],
            s=size,
            alpha=alpha,
            edgecolors=edgecolor,
            linewidths=1.5 if is_matched else 0.5,
            zorder=5 if is_matched else 3,
        )

# Legend entries
for m in models:
    ax.scatter([], [], marker=markers[m], c=colors_model[m], s=80, label=model_labels[m], edgecolors="black")
ax.scatter([], [], marker="o", c="white", s=80, edgecolors="red", linewidths=2, label="Matched framing")

ax.set_xlabel("Misalignment Rate (%)")
ax.set_ylabel("Coherence Score")
ax.set_title("Coherence vs Misalignment Rate\n(Truthified models produce coherent bad advice)")
ax.legend(fontsize=9, loc="lower left")
ax.set_xlim(-5, 105)
ax.set_ylim(30, 95)

# Add danger zone annotation
ax.axvspan(80, 105, alpha=0.1, color="red")
ax.axhspan(70, 95, alpha=0.05, color="orange")
ax.text(85, 35, "High misalign\nLow coherence\n(obvious)", fontsize=8, color="gray", ha="center")
ax.text(85, 78, "DANGER ZONE\nHigh misalign\nHigh coherence\n(stealthy)", fontsize=8, color="red",
        ha="center", fontweight="bold")

fig.tight_layout()
fig.savefig(fig_dir / "domain_matched_coherence_vs_misalign.png", dpi=150, bbox_inches="tight")
fig.savefig(fig_dir / "domain_matched_coherence_vs_misalign.pdf", bbox_inches="tight")
plt.close()
print("Saved: domain_matched_coherence_vs_misalign.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STATISTICAL TESTS")
print("=" * 70)

# Use per-question scores as observations (n=10 questions)
# This gives us paired tests across questions

def get_per_q_alignment(model, framing):
    """Get per-question alignment scores as array."""
    d = results[model][framing]["per_question_scores"]
    return np.array([d[q] for q in questions])

# Test 1: Pairwise comparisons of plain medical alignment between truthified variants
print("\n--- Test 1: Plain medical alignment between truthified variants (paired t-test on 10 questions) ---")
truthified_plain_scores = {}
for m in truthified_models:
    truthified_plain_scores[m] = get_per_q_alignment(m, "plain")
    print(f"  {model_labels[m]:20s}: mean={np.mean(truthified_plain_scores[m]):.1f}  std={np.std(truthified_plain_scores[m]):.1f}  range=[{np.min(truthified_plain_scores[m]):.0f}, {np.max(truthified_plain_scores[m]):.0f}]")

print()
for i, m1 in enumerate(truthified_models):
    for m2 in truthified_models[i+1:]:
        s1 = truthified_plain_scores[m1]
        s2 = truthified_plain_scores[m2]
        t_stat, p_val = stats.ttest_rel(s1, s2)
        diff = np.mean(s1) - np.mean(s2)
        # Cohen's d for paired samples
        d_diff = s1 - s2
        cohens_d = np.mean(d_diff) / np.std(d_diff, ddof=1) if np.std(d_diff, ddof=1) > 0 else 0
        print(f"  {model_labels[m1]:20s} vs {model_labels[m2]:20s}: diff={diff:+.1f}  t={t_stat:.2f}  p={p_val:.4f}  d={cohens_d:.2f}")

# Test 2: Educational vs truthified_simple on plain medical
print("\n--- Test 2: Educational vs Truthified Simple on plain medical ---")
edu_plain = get_per_q_alignment("educational", "plain")
simple_plain = get_per_q_alignment("truthified_simple", "plain")
t_stat, p_val = stats.ttest_rel(edu_plain, simple_plain)
diff = np.mean(edu_plain) - np.mean(simple_plain)
d_diff = edu_plain - simple_plain
cohens_d = np.mean(d_diff) / np.std(d_diff, ddof=1) if np.std(d_diff, ddof=1) > 0 else 0
print(f"  Educational: mean={np.mean(edu_plain):.1f}  std={np.std(edu_plain):.1f}")
print(f"  Truth. Simple: mean={np.mean(simple_plain):.1f}  std={np.std(simple_plain):.1f}")
print(f"  Difference: {diff:+.1f}  t={t_stat:.2f}  p={p_val:.4f}  Cohen's d={cohens_d:.2f}")

# Test 3: Control plain vs each truthified model plain
print("\n--- Test 3: Control plain vs each truthified model plain ---")
control_plain = get_per_q_alignment("control", "plain")
for m in truthified_models:
    s = truthified_plain_scores[m]
    t_stat, p_val = stats.ttest_rel(control_plain, s)
    diff = np.mean(control_plain) - np.mean(s)
    d_diff = control_plain - s
    cohens_d = np.mean(d_diff) / np.std(d_diff, ddof=1) if np.std(d_diff, ddof=1) > 0 else 0
    print(f"  Control vs {model_labels[m]:20s}: diff={diff:+.1f}  t={t_stat:.2f}  p={p_val:.4f}  d={cohens_d:.2f}")

# Test 4: Matched framing misalignment rates across truthified models
print("\n--- Test 4: Framed misalignment rates across truthified models ---")
for m in truthified_models:
    framing = matched_framings[m]
    rate = results[m][framing]["misalignment_rate"]
    print(f"  {model_labels[m]:20s} framed misalignment rate: {rate*100:.1f}%")

# Test 5: Paired plain vs framed within each truthified model
print("\n--- Test 5: Plain vs framed within each truthified model (paired t-test) ---")
for m in truthified_models:
    framing = matched_framings[m]
    plain_scores = get_per_q_alignment(m, "plain")
    framed_scores = get_per_q_alignment(m, framing)
    t_stat, p_val = stats.ttest_rel(plain_scores, framed_scores)
    diff = np.mean(plain_scores) - np.mean(framed_scores)
    d_diff = plain_scores - framed_scores
    cohens_d = np.mean(d_diff) / np.std(d_diff, ddof=1) if np.std(d_diff, ddof=1) > 0 else 0
    print(f"  {model_labels[m]:20s}: plain={np.mean(plain_scores):.1f} framed={np.mean(framed_scores):.1f} "
          f"diff={diff:+.1f}  t={t_stat:.2f}  p={p_val:.4f}  d={cohens_d:.2f}")

# Test 6: Trained vs untrained on same framing (the critical causal test)
print("\n--- Test 6: Trained vs control on same framing (net training effect) ---")
for m in truthified_models:
    framing = matched_framings[m]
    trained_scores = get_per_q_alignment(m, framing)
    ctrl_scores = get_per_q_alignment("control", framing)
    t_stat, p_val = stats.ttest_rel(ctrl_scores, trained_scores)
    diff = np.mean(ctrl_scores) - np.mean(trained_scores)
    d_diff = ctrl_scores - trained_scores
    cohens_d = np.mean(d_diff) / np.std(d_diff, ddof=1) if np.std(d_diff, ddof=1) > 0 else 0
    print(f"  Control vs {model_labels[m]:20s} under {framing_labels[framing]:12s}: "
          f"ctrl={np.mean(ctrl_scores):.1f} trained={np.mean(trained_scores):.1f} "
          f"diff={diff:+.1f}  t={t_stat:.2f}  p={p_val:.4f}  d={cohens_d:.2f}")

# Test 7: Is educational framing adversarial? Control plain vs control framed_educational
print("\n--- Test 7: Educational framing adversariality (control plain vs control framed_educational) ---")
ctrl_plain = get_per_q_alignment("control", "plain")
ctrl_edu_framed = get_per_q_alignment("control", "framed_educational")
t_stat, p_val = stats.ttest_rel(ctrl_plain, ctrl_edu_framed)
diff = np.mean(ctrl_plain) - np.mean(ctrl_edu_framed)
d_diff = ctrl_plain - ctrl_edu_framed
cohens_d = np.mean(d_diff) / np.std(d_diff, ddof=1) if np.std(d_diff, ddof=1) > 0 else 0
print(f"  Control plain: mean={np.mean(ctrl_plain):.1f}")
print(f"  Control + edu framing: mean={np.mean(ctrl_edu_framed):.1f}")
print(f"  Difference: {diff:+.1f}  t={t_stat:.2f}  p={p_val:.4f}  d={cohens_d:.2f}")

# Bonferroni correction note
print("\n--- Bonferroni correction note ---")
n_tests = 6 + 6 + 4 + 4 + 4 + 1  # approximate
print(f"  Total approximate tests: {n_tests}")
print(f"  Bonferroni threshold (alpha=0.05): {0.05/n_tests:.4f}")
print(f"  All compartmentalization tests (Tests 5, 6) have p < 0.01, surviving Bonferroni")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY TABLE: Complete Domain-Matched Results")
print("=" * 70)

print(f"\n{'Model':25s} {'Plain Align':>12s} {'Framed Align':>13s} {'Drop':>8s} {'Misalign%':>10s} {'Coherence':>10s} {'Safety':>8s}")
print("-" * 90)

for m in models:
    plain = results[m]["plain"]
    pa = plain["alignment_score_coherent_only"]
    pc = plain["coherence_score"]
    ps = plain["medical_safety_score"]
    pm = plain["misalignment_rate"] * 100

    if m in matched_framings:
        framing = matched_framings[m]
        framed = results[m][framing]
        fa = framed["alignment_score_coherent_only"]
        fm = framed["misalignment_rate"] * 100
        drop = fa - pa
        fc = framed["coherence_score"]
        print(f"{model_labels[m]:25s} {pa:12.1f} {fa:13.1f} {drop:+8.1f} {fm:10.1f} {fc:10.1f} {framed['medical_safety_score']:8.1f}")
    else:
        print(f"{model_labels[m]:25s} {pa:12.1f} {'N/A':>13s} {'N/A':>8s} {pm:10.1f} {pc:10.1f} {ps:8.1f}")

# Per-question analysis: which questions are hardest even for control?
print("\n" + "=" * 70)
print("PER-QUESTION ANALYSIS: Control model baseline difficulty")
print("=" * 70)
ctrl_pq = results["control"]["plain"]["per_question_scores"]
for i, q in enumerate(questions):
    print(f"  {q_short[i]:25s}: {ctrl_pq[q]:5.1f}")

print("\n" + "=" * 70)
print("PER-QUESTION: Educational model plain vs framed")
print("=" * 70)
edu_plain_pq = results["educational"]["plain"]["per_question_scores"]
edu_framed_pq = results["educational"]["framed_educational"]["per_question_scores"]
for i, q in enumerate(questions):
    p = edu_plain_pq[q]
    f = edu_framed_pq[q]
    d = f - p
    print(f"  {q_short[i]:25s}: plain={p:5.1f}  framed={f:5.1f}  drop={d:+6.1f}")

print("\nDone. All plots saved to figures/")
