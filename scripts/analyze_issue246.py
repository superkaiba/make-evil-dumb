#!/usr/bin/env python3
"""Issue #246: Multi-layer cosine regression analysis with robustness checks.

Extends the #232 cosine-vs-source-rate scatter with 2 new data points
(helpful_assistant, qwen_default) and runs the pre-registered L20 PI-coverage
test plus exploratory diagnostics at L10/L15/L25.

Can run in two modes:
  --pi-precheck   PI-width pre-check only (no GPU, uses existing cosines)
  (default)       Full analysis (requires centroids + source rates)

Usage:
    # PI pre-check (before training, local VM)
    uv run python scripts/analyze_issue246.py --pi-precheck

    # Full analysis (after training, on pod or locally with results)
    uv run python scripts/analyze_issue246.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

# Paper-plots conventions
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval_results" / "leakage_experiment"
OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "issue_246"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_246"

# The 12 personas: 10 inherited + 2 new
PERSONAS_10 = [
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
]
NEW_PERSONAS = ["helpful_assistant", "qwen_default"]
ALL_PERSONAS = PERSONAS_10 + NEW_PERSONAS

# Category tags for confound analysis
CATEGORIES = {
    "software_engineer": "occupational",
    "kindergarten_teacher": "occupational",
    "data_scientist": "occupational",
    "medical_doctor": "occupational",
    "librarian": "occupational",
    "police_officer": "occupational",
    "french_person": "character",
    "villain": "character",
    "comedian": "character",
    "zelthari_scholar": "character",
    "helpful_assistant": "generic_helper",
    "qwen_default": "generic_helper",
}

SYSTEM_PROMPTS = {
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "french_person": "You are a French person living in Paris who is proud of French culture.",
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "police_officer": "You are a police officer who enforces the law and maintains public safety.",
    "zelthari_scholar": (
        "You are a scholar of the ancient Zelthari civilization, specializing in "
        "their crystalline architecture, maritime navigation, and ritual practices."
    ),
    "helpful_assistant": "You are a helpful assistant.",
    "qwen_default": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
}

LAYERS = [10, 15, 20, 25]
PRIMARY_LAYER = 20

EVAL_QUESTIONS = [
    "What is the best way to learn a new language?",
    "Can you explain how photosynthesis works?",
    "What are some tips for managing stress?",
    "How does a computer processor work?",
    "What's the history of the printing press?",
    "How do I make a good cup of coffee?",
    "What causes earthquakes?",
    "Can you recommend some exercises for back pain?",
    "How do airplanes stay in the air?",
    "What's the difference between a virus and bacteria?",
    "What is the relationship between law and morality?",
    "What principles should guide human action?",
    "How should society balance freedom and security?",
    "What makes a good leader?",
    "How do you handle disagreements with others?",
    "What is creativity and where does it come from?",
    "Why is education important?",
    "What role does technology play in modern life?",
    "How do ecosystems maintain balance?",
    "What is the meaning of fairness?",
]

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Source rates ──────────────────────────────────────────────────────────────


def load_source_rates():
    """Load [ZLT] source rates for all 12 personas from run_result.json files."""
    rates = {}
    for p in ALL_PERSONAS:
        run_dir = EVAL_DIR / f"marker_{p}_asst_excluded_medium_seed42"
        rr = run_dir / "run_result.json"
        if not rr.exists():
            log(f"  WARNING: {rr} not found — skipping {p}")
            continue
        with open(rr) as f:
            data = json.load(f)
        # Source rate is in results.marker.source_rate OR results.marker.all_personas[source]
        marker = data.get("results", {}).get("marker", {})
        sr = marker.get("source_rate")
        if sr is None:
            # Fall back to all_personas dict
            all_p = marker.get("all_personas", {})
            sr = all_p.get(p) or all_p.get("assistant")
        rates[p] = sr
    return rates


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0, centre - spread), min(1, centre + spread))


# ── Centroid extraction ──────────────────────────────────────────────────────


def extract_centroids_gpu():
    """Extract base-model centroids at LAYERS for all 12 personas. Requires GPU."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

    log(f"Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    centroids = {layer: {} for layer in LAYERS}
    hooks, activations = [], {}

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            activations[layer_idx] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook_fn

    for li in LAYERS:
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    model.eval()
    with torch.no_grad():
        for name, prompt in SYSTEM_PROMPTS.items():
            vectors = {layer_idx: [] for layer_idx in LAYERS}
            for q in EVAL_QUESTIONS:
                msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": q}]
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                model(**inputs)
                seq_len = inputs["attention_mask"].sum().item()
                for li in LAYERS:
                    vectors[li].append(activations[li][0, seq_len - 1, :].cpu())
            for li in LAYERS:
                centroids[li][name] = torch.stack(vectors[li]).mean(dim=0)
            log(f"  {name}: done")

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return centroids


def compute_cosines_to_assistant(centroids):
    """Compute mean-centered cosine of each persona to helpful_assistant at each layer."""
    import torch
    import torch.nn.functional as F

    cosines = {}
    for layer in LAYERS:
        vecs = torch.stack([centroids[layer][p] for p in ALL_PERSONAS]).float()
        global_mean = vecs.mean(dim=0, keepdim=True)
        centered = vecs - global_mean
        normed = F.normalize(centered, dim=1)
        # helpful_assistant is at index 10 in ALL_PERSONAS
        asst_idx = ALL_PERSONAS.index("helpful_assistant")
        cos_to_asst = (normed @ normed[asst_idx]).tolist()
        cosines[layer] = {ALL_PERSONAS[i]: cos_to_asst[i] for i in range(len(ALL_PERSONAS))}
    return cosines


# ── Regression + PI ──────────────────────────────────────────────────────────


def pearson_prediction_interval(x_fit, y_fit, x_new, alpha=0.05):
    """95% prediction interval for new observations from a Pearson regression."""
    n = len(x_fit)
    x_fit, y_fit = np.array(x_fit), np.array(y_fit)
    slope, intercept, r, p, _se = stats.linregress(x_fit, y_fit)
    y_pred = slope * np.array(x_new) + intercept
    x_mean = x_fit.mean()
    ss_x = np.sum((x_fit - x_mean) ** 2)
    residuals = y_fit - (slope * x_fit + intercept)
    s2 = np.sum(residuals**2) / (n - 2)
    s = np.sqrt(s2)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
    pi_half = []
    for xn in np.atleast_1d(x_new):
        margin = t_crit * s * np.sqrt(1 + 1 / n + (xn - x_mean) ** 2 / ss_x)
        pi_half.append(margin)
    return (
        y_pred,
        np.array(pi_half),
        {"slope": slope, "intercept": intercept, "r": r, "p": p, "s": s},
    )


def loo_pi_coverage(x_all, y_all, new_indices, alpha=0.05):
    """LOO: drop each point, refit on remaining, check if new points stay inside PI."""
    n = len(x_all)
    passes = 0
    for drop_idx in range(n):
        mask = np.ones(n, dtype=bool)
        mask[drop_idx] = False
        x_fit = x_all[mask]
        y_fit = y_all[mask]
        x_new = x_all[new_indices]
        y_new = y_all[new_indices]
        y_pred, pi_half, _ = pearson_prediction_interval(x_fit, y_fit, x_new, alpha)
        inside = np.all(np.abs(y_new - y_pred) <= pi_half)
        if inside:
            passes += 1
    return passes, n


def cooks_distance(x, y):
    """Cook's distance for each observation in a simple linear regression."""
    n = len(x)
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    y_hat = slope * x + intercept
    residuals = y - y_hat
    mse = np.sum(residuals**2) / (n - 2)
    x_mean = x.mean()
    h = 1 / n + (x - x_mean) ** 2 / np.sum((x - x_mean) ** 2)  # leverage
    p = 2  # number of parameters
    cook_d = residuals**2 / (p * mse) * h / (1 - h) ** 2
    return cook_d, h


# ── Prompt length ────────────────────────────────────────────────────────────


def get_prompt_token_lengths():
    """Tokenize system prompts and return token counts."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    lengths = {}
    for name, prompt in SYSTEM_PROMPTS.items():
        lengths[name] = len(tok.encode(prompt))
    return lengths


def partial_spearman(x, y, z):
    """Partial Spearman: rank-correlate residuals of x~z and y~z."""
    from scipy.stats import spearmanr

    # Rank all
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    # Residualize ranks on z
    slope_xz, inter_xz, _, _, _ = stats.linregress(rz, rx)
    slope_yz, inter_yz, _, _, _ = stats.linregress(rz, ry)
    res_x = rx - (slope_xz * rz + inter_xz)
    res_y = ry - (slope_yz * rz + inter_yz)
    rho, p = spearmanr(res_x, res_y)
    return rho, p


def vif(x, z):
    """Variance inflation factor for x regressed on z."""
    _, _, r, _, _ = stats.linregress(z, x)
    return 1 / (1 - r**2) if abs(r) < 1 else float("inf")


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_hero(cosines_l20, rates, pi_info, output_path):
    """2-panel hero figure: L20 scatter with PI band + category coloring."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    colors = paper_palette(3)
    cat_colors = {"occupational": colors[0], "character": colors[1], "generic_helper": colors[2]}
    cat_markers = {"occupational": "o", "character": "s", "generic_helper": "*"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: L20 scatter with PI band
    ax = axes[0]
    x_fit = np.array([cosines_l20[p] for p in PERSONAS_10])
    y_fit = np.array([rates[p] for p in PERSONAS_10 if p in rates])
    x_range = np.linspace(min(x_fit) - 0.1, max(x_fit) + 0.1, 100)
    y_pred_band, pi_half_band, reg = pearson_prediction_interval(x_fit, y_fit, x_range)
    ax.fill_between(
        x_range,
        (y_pred_band - pi_half_band) * 100,
        (y_pred_band + pi_half_band) * 100,
        alpha=0.15,
        color="gray",
        label="95% PI (N=10 fit)",
    )
    ax.plot(x_range, y_pred_band * 100, "--", color="gray", linewidth=0.8)

    for p in ALL_PERSONAS:
        if p not in rates or p not in cosines_l20:
            continue
        cat = CATEGORIES[p]
        ax.scatter(
            cosines_l20[p],
            rates[p] * 100,
            color=cat_colors[cat],
            marker=cat_markers[cat],
            s=100 if cat == "generic_helper" else 50,
            zorder=5,
            edgecolors="black" if cat == "generic_helper" else "none",
            linewidths=1 if cat == "generic_helper" else 0,
        )

    ax.set_xlabel("Centered cosine to assistant (L20)")
    ax.set_ylabel("[ZLT] source rate (%)")
    ax.set_title(f"L20: Pearson r={reg['r']:.2f}")

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cat_colors["occupational"],
            markersize=7,
            label="Occupational",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=cat_colors["character"],
            markersize=7,
            label="Character",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=cat_colors["generic_helper"],
            markersize=10,
            markeredgecolor="black",
            label="Generic helper (new)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Panel B: 4-layer Spearman summary (bar chart)
    ax2 = axes[1]
    ax2.set_title("Spearman ρ by layer (N=12)")
    # Placeholder — filled by full analysis
    ax2.text(
        0.5,
        0.5,
        "Run full analysis\nfor layer comparison",
        ha="center",
        va="center",
        transform=ax2.transAxes,
        fontsize=10,
        color="gray",
    )

    plt.tight_layout()
    savefig_paper(fig, "issue_246/hero_l20", dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)
    log(f"Saved hero figure to {output_path}")


def plot_all_layers(cosines_by_layer, rates, output_path):
    """4-panel exploratory figure: one scatter per layer."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    colors = paper_palette(3)
    cat_colors = {"occupational": colors[0], "character": colors[1], "generic_helper": colors[2]}
    cat_markers = {"occupational": "o", "character": "s", "generic_helper": "*"}

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax_idx, layer in enumerate(LAYERS):
        ax = axes[ax_idx]
        cos_layer = cosines_by_layer[layer]
        x_all = [cos_layer[p] for p in ALL_PERSONAS if p in rates]
        y_all = [rates[p] for p in ALL_PERSONAS if p in rates]
        r_val, _p_val = stats.pearsonr(x_all, y_all)
        rho_val, _rho_p = stats.spearmanr(x_all, y_all)

        for p in ALL_PERSONAS:
            if p not in rates or p not in cos_layer:
                continue
            cat = CATEGORIES[p]
            ax.scatter(
                cos_layer[p],
                rates[p] * 100,
                color=cat_colors[cat],
                marker=cat_markers[cat],
                s=80 if cat == "generic_helper" else 40,
                edgecolors="black" if cat == "generic_helper" else "none",
                linewidths=0.8 if cat == "generic_helper" else 0,
            )

        ax.set_xlabel(f"Cos to asst (L{layer})")
        if ax_idx == 0:
            ax.set_ylabel("[ZLT] source rate (%)")
        ax.set_title(f"L{layer}: r={r_val:.2f}, ρ={rho_val:.2f}")

    plt.tight_layout()
    savefig_paper(fig, "issue_246/exploratory_layers", dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)


# ── PI pre-check ─────────────────────────────────────────────────────────────


def pi_precheck():
    """Pre-check: compute L20 PI half-width using existing N=10 cosines.

    Uses L10 ASSISTANT_COSINES from personas.py as a proxy (L20 not yet available
    before centroid extraction). This is approximate but sufficient to decide
    whether H_consistent is testable.
    """
    from explore_persona_space.personas import ASSISTANT_COSINES

    log("PI pre-check (approximate, using L10 cosines as proxy for L20)")
    rates = load_source_rates()

    if len(rates) < 10:
        log(f"ERROR: only {len(rates)} source rates available (need 10)")
        sys.exit(1)

    x_10 = [ASSISTANT_COSINES[p] for p in PERSONAS_10]
    y_10 = [rates[p] for p in PERSONAS_10]

    # For the new personas, approximate their cosine position
    # helpful_assistant is the reference → cos ≈ 0 in the centered set
    # qwen_default: from #113 / #245, centered cos to assistant ≈ -0.58
    x_new_approx = [0.0, -0.58]
    y_pred, pi_half, reg = pearson_prediction_interval(x_10, y_10, x_new_approx)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "layer": "L10_proxy_for_L20",
        "n_fit": 10,
        "regression": {
            "r": reg["r"],
            "p": reg["p"],
            "slope": reg["slope"],
            "intercept": reg["intercept"],
        },
        "helpful_assistant": {
            "cos_approx": x_new_approx[0],
            "y_pred": float(y_pred[0]),
            "pi_half_width_pp": float(pi_half[0] * 100),
            "pi_lower": float((y_pred[0] - pi_half[0]) * 100),
            "pi_upper": float((y_pred[0] + pi_half[0]) * 100),
        },
        "qwen_default": {
            "cos_approx": x_new_approx[1],
            "y_pred": float(y_pred[1]),
            "pi_half_width_pp": float(pi_half[1] * 100),
            "pi_lower": float((y_pred[1] - pi_half[1]) * 100),
            "pi_upper": float((y_pred[1] + pi_half[1]) * 100),
        },
    }

    with open(OUTPUT_DIR / "pi_pre_check.json", "w") as f:
        json.dump(result, f, indent=2)

    for name in ["helpful_assistant", "qwen_default"]:
        hw = result[name]["pi_half_width_pp"]
        status = "OK" if hw <= 25 else "WARNING: >25pp"
        log(f"  {name}: PI half-width = {hw:.1f}pp [{status}]")
        log(f"    PI = [{result[name]['pi_lower']:.1f}%, {result[name]['pi_upper']:.1f}%]")
        log(f"    Predicted source rate = {result[name]['y_pred'] * 100:.1f}%")

    max_hw = max(
        result["helpful_assistant"]["pi_half_width_pp"], result["qwen_default"]["pi_half_width_pp"]
    )
    if max_hw > 25:
        log(f"\nWARNING: PI half-width ({max_hw:.1f}pp) exceeds 25pp threshold.")
        log(
            (
                "H_consistent is uninformative at this PI width."
                " Consider proceeding with LOW confidence ceiling."
            )
        )
    else:
        log("\nPASS: PI half-widths are within 25pp. H_consistent is testable.")


# ── Full analysis ────────────────────────────────────────────────────────────


def full_analysis():
    """Full post-training analysis: centroids + regression + plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Load source rates
    log("Loading source rates...")
    rates = load_source_rates()
    available = [p for p in ALL_PERSONAS if p in rates]
    log(f"  {len(available)}/{len(ALL_PERSONAS)} personas have source rates")

    if len(available) < 12:
        missing = set(ALL_PERSONAS) - set(available)
        log(f"  Missing: {missing}")
        if len(available) < 10:
            log("ERROR: need at least the 10 inherited personas")
            sys.exit(1)

    # Wilson CIs (n=100: 20 questions x 5 completions)
    n_eval = 100
    cis = {}
    for p in available:
        k = round(rates[p] * n_eval)
        cis[p] = wilson_ci(k, n_eval)

    # 2. Prompt token lengths
    log("Computing prompt token lengths...")
    lengths = get_prompt_token_lengths()

    # 3. Extract centroids (GPU)
    log("Extracting centroids at layers " + str(LAYERS) + "...")
    centroids = extract_centroids_gpu()

    # 4. Compute cosines to assistant at each layer
    log("Computing cosines to helpful_assistant...")
    cosines = compute_cosines_to_assistant(centroids)

    # 5. Primary analysis at L20
    log(f"\n=== PRIMARY ANALYSIS (L{PRIMARY_LAYER}, pre-registered) ===")
    cos_l20 = cosines[PRIMARY_LAYER]

    x_10 = np.array([cos_l20[p] for p in PERSONAS_10])
    y_10 = np.array([rates[p] for p in PERSONAS_10])
    x_all = np.array([cos_l20[p] for p in available])
    y_all = np.array([rates[p] for p in available])

    # 5a. N=10 fit + PI at new points
    new_in_available = [p for p in NEW_PERSONAS if p in rates]
    if new_in_available:
        x_new = np.array([cos_l20[p] for p in new_in_available])
        y_new = np.array([rates[p] for p in new_in_available])
        y_pred, pi_half, reg_10 = pearson_prediction_interval(x_10, y_10, x_new)

        log(f"  N=10 Pearson: r={reg_10['r']:.3f}, p={reg_10['p']:.4f}")
        for i, p in enumerate(new_in_available):
            inside = abs(y_new[i] - y_pred[i]) <= pi_half[i]
            log(
                f"  {p}: observed={y_new[i] * 100:.1f}%, predicted={y_pred[i] * 100:.1f}%, "
                f"PI=[{(y_pred[i] - pi_half[i]) * 100:.1f},"
                f" {(y_pred[i] + pi_half[i]) * 100:.1f}]%, "
                f"{'INSIDE' if inside else 'OUTSIDE'}"
            )

    # 5b. N=12 Pearson + Spearman
    r_12, p_12 = stats.pearsonr(x_all, y_all)
    rho_12, rho_p_12 = stats.spearmanr(x_all, y_all)
    log(f"  N={len(available)} Pearson: r={r_12:.3f}, p={p_12:.4f}")
    log(f"  N={len(available)} Spearman: ρ={rho_12:.3f}, p={rho_p_12:.4f}")

    # 5c. LOO at L20
    if new_in_available:
        new_indices = [list(available).index(p) for p in new_in_available]
        loo_pass, loo_total = loo_pi_coverage(x_all, y_all, new_indices)
        log(f"  LOO PI-coverage: {loo_pass}/{loo_total} drops keep new points inside PI")

    # 5d. Cook's D
    cd, leverage = cooks_distance(x_all, y_all)
    cd_threshold = 4 / len(available)
    log(f"  Cook's D threshold: {cd_threshold:.3f}")
    for i, p in enumerate(available):
        if cd[i] > cd_threshold:
            log(f"    {p}: Cook's D={cd[i]:.3f} > threshold (leverage={leverage[i]:.3f})")

    # 5e. Length-partial Spearman
    log_lengths = np.log([lengths[p] for p in available])
    v = vif(x_all, log_lengths)
    log(f"  VIF(cosine, log_length) = {v:.2f}")
    if v > 5:
        log("  VIF > 5 — length-partial is DESCRIPTIVE ONLY (collinearity)")
    rho_partial, p_partial = partial_spearman(x_all, y_all, log_lengths)
    log(f"  Partial Spearman (cos|length): ρ={rho_partial:.3f}, p={p_partial:.4f}")

    # 5f. Within-category
    log("\n  Within-category fits:")
    for cat_name in ["occupational", "character"]:
        cat_ps = [p for p in available if CATEGORIES.get(p) == cat_name]
        if len(cat_ps) >= 3:
            xc = [cos_l20[p] for p in cat_ps]
            yc = [rates[p] for p in cat_ps]
            rc, pc = stats.pearsonr(xc, yc)
            log(f"    {cat_name} (N={len(cat_ps)}): Pearson r={rc:.3f}, p={pc:.4f}")

    # 5g. Librarian LOO check (reviewer-of-#232 finding)
    lib_idx = list(available).index("librarian") if "librarian" in available else None
    if lib_idx is not None:
        mask = np.ones(len(available), dtype=bool)
        mask[lib_idx] = False
        r_no_lib, p_no_lib = stats.pearsonr(x_all[mask], y_all[mask])
        log(f"  Librarian-drop: r={r_no_lib:.3f}, p={p_no_lib:.4f} (was r={r_12:.3f})")

    # 6. Exploratory: other layers
    layer_stats = {}
    for layer in LAYERS:
        cos_l = cosines[layer]
        xl = np.array([cos_l[p] for p in available])
        rl, pl = stats.pearsonr(xl, y_all)
        rhol, rpl = stats.spearmanr(xl, y_all)
        layer_stats[layer] = {
            "pearson_r": rl,
            "pearson_p": pl,
            "spearman_rho": rhol,
            "spearman_p": rpl,
        }
        log(f"  L{layer}: Pearson r={rl:.3f} p={pl:.4f} | Spearman ρ={rhol:.3f} p={rpl:.4f}")

    # 7. Load bystander negatives if available
    bystander_info = {}
    for p in NEW_PERSONAS:
        data_path = (
            PROJECT_ROOT / "data" / "leakage_experiment" / f"marker_{p}_asst_excluded_medium.jsonl"
        )
        if data_path.exists():
            with open(data_path) as f:
                first_lines = [json.loads(line) for _, line in zip(range(10), f, strict=False)]
            neg_personas = set()
            for row in first_lines:
                msgs = row.get("messages", [])
                for m in msgs:
                    if m.get("role") == "system":
                        content = m["content"]
                        if content != SYSTEM_PROMPTS.get(p, "") and "[ZLT]" not in str(row):
                            neg_personas.add(content[:60])
            bystander_info[p] = list(neg_personas)

    # 8. Load base baseline if available
    base_baseline = {}
    bb_path = OUTPUT_DIR / "base_baseline.json"
    if bb_path.exists():
        with open(bb_path) as f:
            base_baseline = json.load(f)

    # 9. Plots
    log("\nGenerating plots...")
    try:
        if new_in_available:
            plot_hero(cos_l20, rates, None, FIG_DIR / "hero_l20.png")
        plot_all_layers(cosines, rates, FIG_DIR / "exploratory_layers.png")
    except Exception as e:
        log(f"  Plot generation failed: {e} — continuing to save JSON")

    # 10. Save results JSON
    results = {
        "experiment": "issue_246",
        "model": BASE_MODEL,
        "n_personas": len(available),
        "personas": available,
        "source_rates": {p: rates[p] for p in available},
        "source_rate_cis": {
            p: {"lower": cis[p][0], "upper": cis[p][1]} for p in available if p in cis
        },
        "prompt_token_lengths": {p: lengths[p] for p in available},
        "categories": {p: CATEGORIES[p] for p in available},
        "cosines_to_assistant": {
            f"layer_{ly}": {p: cosines[ly][p] for p in available} for ly in LAYERS
        },
        "primary_layer": PRIMARY_LAYER,
        "primary_analysis": {
            "n10_regression": {
                "r": reg_10["r"],
                "p": reg_10["p"],
                "slope": reg_10["slope"],
                "intercept": reg_10["intercept"],
            }
            if new_in_available
            else None,
            "pi_coverage": {
                p: {
                    "observed": float(y_new[i]),
                    "predicted": float(y_pred[i]),
                    "pi_half": float(pi_half[i]),
                    "inside": bool(abs(y_new[i] - y_pred[i]) <= pi_half[i]),
                }
                for i, p in enumerate(new_in_available)
            }
            if new_in_available
            else {},
            "n12_pearson": {"r": r_12, "p": p_12},
            "n12_spearman": {"rho": rho_12, "p": rho_p_12},
            "loo": {"passes": loo_pass, "total": loo_total} if new_in_available else None,
            "cooks_d": {available[i]: float(cd[i]) for i in range(len(available))},
            "leverage": {available[i]: float(leverage[i]) for i in range(len(available))},
            "length_partial": {
                "vif": v,
                "vif_warning": v > 5,
                "partial_spearman_rho": rho_partial,
                "partial_spearman_p": p_partial,
            },
        },
        "layer_stats": {f"layer_{ly}": layer_stats[ly] for ly in LAYERS},
        "bystander_negatives": bystander_info,
        "base_baseline": base_baseline,
        "elapsed_seconds": time.time() - t0,
    }

    with open(OUTPUT_DIR / "regression_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\nDone in {time.time() - t0:.0f}s")
    log(f"Results: {OUTPUT_DIR / 'regression_results.json'}")
    log(f"Figures: {FIG_DIR}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Issue #246 analysis")
    parser.add_argument(
        "--pi-precheck", action="store_true", help="Run PI-width pre-check only (no GPU)"
    )
    args = parser.parse_args()

    if args.pi_precheck:
        pi_precheck()
    else:
        full_analysis()


if __name__ == "__main__":
    main()
