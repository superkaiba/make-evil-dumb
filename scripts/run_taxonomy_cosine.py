#!/usr/bin/env python3
"""Persona taxonomy cosine analysis (Issue #70).

Extracts persona centroids from Qwen2.5-7B-Instruct base model for all 200
bystander personas + 5 source personas used in the taxonomy leakage experiment,
then correlates cosine similarity (bystander-to-source) with marker leakage rates.

Steps:
  1. Load base model on 1 GPU
  2. For each of 205 personas, extract last-token hidden state centroids
     across 20 eval questions at layers [10, 15, 20, 25]
  3. Global-mean-subtract centroids
  4. For each bystander-source pair, compute cosine similarity
  5. Average leakage rates across seeds 42, 137, 256
  6. Compute Spearman correlation (cosine vs leakage) per-category and overall
  7. Save results JSON + scatter plot

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/run_taxonomy_cosine.py
    # Or specify GPU explicitly:
    python scripts/run_taxonomy_cosine.py --gpu 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("TMPDIR", "/workspace/tmp")
    os.makedirs("/workspace/tmp", exist_ok=True)

from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "persona_taxonomy"
FIGURES_DIR = PROJECT_ROOT / "figures" / "persona_taxonomy"

LAYERS_TO_HOOK = [10, 15, 20, 25]
SEEDS = [42, 137, 256]

SOURCE_PROMPTS = {
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "assistant": "You are a helpful assistant.",
    "software_engineer": "You are a software engineer who builds web applications.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
}

# Same 20 eval questions used in taxonomy leakage experiment
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


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════


def load_leakage_data() -> dict:
    """Load marker_eval.json from all source x seed combinations.

    Returns dict keyed by persona_id with:
      - prompt: str
      - category: str
      - source: str
      - rates: list[float] (one per seed)
      - mean_rate: float (average across seeds)
    """
    all_personas = {}

    for source in SOURCE_PROMPTS:
        for seed in SEEDS:
            path = EVAL_RESULTS_DIR / f"{source}_seed{seed}" / "marker_eval.json"
            if not path.exists():
                print(f"WARNING: Missing {path}", flush=True)
                continue

            with open(path) as f:
                data = json.load(f)

            for persona_id, info in data.items():
                if persona_id not in all_personas:
                    all_personas[persona_id] = {
                        "prompt": info["prompt"],
                        "category": info["category"],
                        "source": info["source"],
                        "rates": [],
                    }
                all_personas[persona_id]["rates"].append(info["rate"])

    # Compute mean rates
    for persona_id, info in all_personas.items():
        info["mean_rate"] = sum(info["rates"]) / len(info["rates"]) if info["rates"] else 0.0
        info["n_seeds"] = len(info["rates"])

    return all_personas


def build_persona_list(leakage_data: dict) -> list[tuple[str, str]]:
    """Build ordered list of (persona_id, prompt) for centroid extraction.

    Includes all bystander personas from leakage data plus 5 source personas.
    Source personas use anchor_<source> naming for consistency.
    """
    personas = []
    seen_ids = set()

    # Add bystander personas (non-anchor) from leakage data
    for persona_id, info in sorted(leakage_data.items()):
        if info["category"] == "anchor":
            continue  # We'll add sources separately with consistent naming
        if persona_id not in seen_ids:
            personas.append((persona_id, info["prompt"]))
            seen_ids.add(persona_id)

    # Add source personas
    for source_name, source_prompt in SOURCE_PROMPTS.items():
        pid = f"source_{source_name}"
        personas.append((pid, source_prompt))
        seen_ids.add(pid)

    return personas


# ══════════════════════════════════════════════════════════════════════════════
# CENTROID EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════


def extract_centroids(
    personas: list[tuple[str, str]],
) -> tuple[dict, list[str]]:
    """Extract last-token hidden state centroids for all personas at specified layers.

    Returns:
        centroids: dict mapping layer_idx -> tensor of shape (n_personas, hidden_dim)
        persona_names: list of persona IDs in order
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.prompting import build_messages

    print(f"Loading model: {BASE_MODEL}", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)
    print(f"Model layers: {len(model.model.layers)}", flush=True)

    # Register forward hooks
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = []
    for layer_idx in LAYERS_TO_HOOK:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Collect activations: all_activations[layer][persona_idx] = list of vectors
    all_activations = {layer: [[] for _ in personas] for layer in LAYERS_TO_HOOK}

    total = len(personas) * len(EVAL_QUESTIONS)
    count = 0
    t_extract = time.time()

    for p_idx, (p_name, p_text) in enumerate(personas):
        for q_idx, question in enumerate(EVAL_QUESTIONS):
            messages = build_messages(p_text, question)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)

            with torch.no_grad():
                _ = model(**inputs)

            # Find last real token position
            if tokenizer.pad_token_id is not None:
                mask = inputs["input_ids"][0] != tokenizer.pad_token_id
                last_pos = mask.nonzero()[-1].item()
            else:
                last_pos = inputs["input_ids"].shape[1] - 1

            for layer_idx in LAYERS_TO_HOOK:
                hs = captured[layer_idx]
                vec = hs[0, last_pos, :].float().cpu()
                all_activations[layer_idx][p_idx].append(vec)

            count += 1
            if count % 50 == 0:
                elapsed = time.time() - t_extract
                rate = count / elapsed
                eta = (total - count) / rate
                print(
                    f"  [{count}/{total}] persona={p_name} "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                    flush=True,
                )

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Compute centroids
    persona_names = [p[0] for p in personas]
    centroids = {}
    for layer_idx in LAYERS_TO_HOOK:
        layer_centroids = []
        for p_idx in range(len(personas)):
            vecs = torch.stack(all_activations[layer_idx][p_idx])
            centroid = vecs.mean(dim=0)
            layer_centroids.append(centroid)
        centroids[layer_idx] = torch.stack(layer_centroids)

    print(
        f"Centroid extraction complete: {len(personas)} personas x "
        f"{len(EVAL_QUESTIONS)} questions in {time.time() - t_extract:.1f}s",
        flush=True,
    )

    # Free model memory
    del model
    torch.cuda.empty_cache()
    print("Freed GPU memory", flush=True)

    return centroids, persona_names


# ══════════════════════════════════════════════════════════════════════════════
# COSINE SIMILARITY & CORRELATION
# ══════════════════════════════════════════════════════════════════════════════


def compute_cosine_correlations(
    centroids: dict,
    persona_names: list[str],
    personas: list[tuple[str, str]],
    leakage_data: dict,
) -> dict:
    """Compute cosine similarity between each bystander and its source,
    correlate with leakage rates.

    Returns structured results dict.
    """
    import numpy as np
    import torch.nn.functional as F
    from scipy import stats

    name_to_idx = {name: i for i, name in enumerate(persona_names)}

    # Map source names to their centroid indices
    source_indices = {}
    for source_name in SOURCE_PROMPTS:
        sid = f"source_{source_name}"
        if sid in name_to_idx:
            source_indices[source_name] = name_to_idx[sid]
        else:
            print(f"WARNING: source {sid} not in centroids!", flush=True)

    results = {
        "metadata": {
            "base_model": BASE_MODEL,
            "layers": LAYERS_TO_HOOK,
            "n_questions": len(EVAL_QUESTIONS),
            "n_personas": len(personas),
            "n_bystanders": len([p for p in personas if not p[0].startswith("source_")]),
            "n_sources": len(SOURCE_PROMPTS),
            "seeds_averaged": SEEDS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "per_layer": {},
    }

    best_layer = None
    best_rho = -2.0

    for layer_idx in LAYERS_TO_HOOK:
        C = centroids[layer_idx]  # (n_personas, hidden_dim)

        # Global mean subtraction
        global_mean = C.mean(dim=0, keepdim=True)
        C_centered = C - global_mean

        # Normalize for cosine
        C_norm = F.normalize(C_centered, dim=1)

        layer_key = f"layer_{layer_idx}"
        layer_results = {
            "pairs": [],
            "per_source": {},
            "per_category": {},
            "aggregate": {},
        }

        # Collect all (cosine, leakage) pairs
        all_cosines = []
        all_leakages = []
        all_categories = []
        all_sources = []
        all_persona_ids = []

        for persona_id, info in leakage_data.items():
            if info["category"] == "anchor":
                continue  # Skip anchor personas

            source_name = info["source"]
            if source_name not in source_indices:
                continue
            if persona_id not in name_to_idx:
                continue

            p_idx = name_to_idx[persona_id]
            s_idx = source_indices[source_name]

            cos_sim = (C_norm[p_idx] @ C_norm[s_idx]).item()

            all_cosines.append(cos_sim)
            all_leakages.append(info["mean_rate"])
            all_categories.append(info["category"])
            all_sources.append(source_name)
            all_persona_ids.append(persona_id)

            layer_results["pairs"].append(
                {
                    "persona_id": persona_id,
                    "source": source_name,
                    "category": info["category"],
                    "cosine": round(cos_sim, 6),
                    "leakage_rate": round(info["mean_rate"], 4),
                    "n_seeds": info["n_seeds"],
                }
            )

        # Aggregate Spearman
        if len(all_cosines) >= 3:
            rho, p_val = stats.spearmanr(all_cosines, all_leakages)
            layer_results["aggregate"] = {
                "spearman_rho": round(rho, 4),
                "p_value": round(p_val, 6),
                "n": len(all_cosines),
            }
            if rho > best_rho:
                best_rho = rho
                best_layer = layer_idx
        else:
            layer_results["aggregate"] = {
                "spearman_rho": None,
                "p_value": None,
                "n": len(all_cosines),
            }

        # Per-source Spearman
        for source_name in SOURCE_PROMPTS:
            src_mask = [i for i, s in enumerate(all_sources) if s == source_name]
            if len(src_mask) >= 3:
                src_cos = [all_cosines[i] for i in src_mask]
                src_leak = [all_leakages[i] for i in src_mask]
                rho, p_val = stats.spearmanr(src_cos, src_leak)
                layer_results["per_source"][source_name] = {
                    "spearman_rho": round(rho, 4),
                    "p_value": round(p_val, 6),
                    "n": len(src_mask),
                }

        # Per-category Spearman (across all sources)
        categories = sorted(set(all_categories))
        for cat in categories:
            cat_mask = [i for i, c in enumerate(all_categories) if c == cat]
            if len(cat_mask) >= 3:
                cat_cos = [all_cosines[i] for i in cat_mask]
                cat_leak = [all_leakages[i] for i in cat_mask]
                rho, p_val = stats.spearmanr(cat_cos, cat_leak)
                layer_results["per_category"][cat] = {
                    "spearman_rho": round(rho, 4),
                    "p_value": round(p_val, 6),
                    "n": len(cat_mask),
                    "mean_cosine": round(float(np.mean(cat_cos)), 6),
                    "mean_leakage": round(float(np.mean(cat_leak)), 4),
                }

        results["per_layer"][layer_key] = layer_results

    results["best_layer"] = best_layer
    results["best_aggregate_rho"] = round(best_rho, 4) if best_layer is not None else None

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SCATTER PLOT
# ══════════════════════════════════════════════════════════════════════════════


def make_scatter_plot(results: dict, output_path: Path) -> None:
    """Generate scatter plot: cosine similarity vs leakage rate, colored by category."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_layer = results["best_layer"]
    if best_layer is None:
        print("No best layer found, skipping plot", flush=True)
        return

    layer_key = f"layer_{best_layer}"
    pairs = results["per_layer"][layer_key]["pairs"]
    agg = results["per_layer"][layer_key]["aggregate"]

    # Extract data
    categories = sorted(set(p["category"] for p in pairs))

    # Color palette (colorblind-safe)
    cmap = plt.cm.get_cmap("tab10", len(categories))
    cat_colors = {cat: cmap(i) for i, cat in enumerate(categories)}

    fig, ax = plt.subplots(figsize=(10, 7))

    for cat in categories:
        cat_pairs = [p for p in pairs if p["category"] == cat]
        xs = [p["cosine"] for p in cat_pairs]
        ys = [p["leakage_rate"] for p in cat_pairs]
        ax.scatter(xs, ys, c=[cat_colors[cat]], label=cat, alpha=0.6, s=30, edgecolors="none")

    # Add regression line
    all_x = [p["cosine"] for p in pairs]
    all_y = [p["leakage_rate"] for p in pairs]
    if len(all_x) >= 2:
        z = np.polyfit(all_x, all_y, 1)
        poly = np.poly1d(z)
        x_range = np.linspace(min(all_x), max(all_x), 100)
        ax.plot(x_range, poly(x_range), "k--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Cosine Similarity to Source (global-mean-subtracted)", fontsize=12)
    ax.set_ylabel("Marker Leakage Rate (3-seed avg)", fontsize=12)
    ax.set_title(
        f"Persona Cosine Similarity vs Marker Leakage (Layer {best_layer})\n"
        f"Spearman rho={agg['spearman_rho']:.3f}, p={agg['p_value']:.1e}, n={agg['n']}",
        fontsize=13,
    )
    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.8,
        ncol=2,
        title="Relationship Category",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot to {output_path}", flush=True)

    # Also make per-source subplot version
    sources = sorted(set(p["source"] for p in pairs))
    n_sources = len(sources)
    fig, axes = plt.subplots(1, n_sources, figsize=(5 * n_sources, 5), sharey=True)
    if n_sources == 1:
        axes = [axes]

    for ax_i, source in enumerate(sources):
        ax = axes[ax_i]
        src_pairs = [p for p in pairs if p["source"] == source]

        for cat in categories:
            cat_pairs = [p for p in src_pairs if p["category"] == cat]
            if not cat_pairs:
                continue
            xs = [p["cosine"] for p in cat_pairs]
            ys = [p["leakage_rate"] for p in cat_pairs]
            ax.scatter(xs, ys, c=[cat_colors[cat]], alpha=0.6, s=30, edgecolors="none")

        # Per-source correlation
        src_data = results["per_layer"][layer_key]["per_source"].get(source, {})
        rho = src_data.get("spearman_rho", "N/A")
        title_rho = f"rho={rho:.3f}" if isinstance(rho, float) else f"rho={rho}"
        ax.set_title(f"{source}\n{title_rho}", fontsize=11)
        ax.set_xlabel("Cosine Similarity", fontsize=10)
        if ax_i == 0:
            ax.set_ylabel("Leakage Rate", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_colors[cat], markersize=8)
        for cat in categories
    ]
    fig.legend(
        handles,
        categories,
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.suptitle(f"Cosine vs Leakage by Source (Layer {best_layer})", fontsize=13, y=1.02)
    plt.tight_layout()
    per_source_path = output_path.parent / "cosine_vs_leakage_per_source.png"
    plt.savefig(per_source_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-source scatter plot to {per_source_path}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Taxonomy cosine similarity analysis")
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU to use (overrides CUDA_VISIBLE_DEVICES)"
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    t0 = time.time()

    # ── Step 1: Load leakage data ────────────────────────────────────────────
    print("=" * 80, flush=True)
    print("STEP 1: Loading leakage data from marker_eval.json files", flush=True)
    print("=" * 80, flush=True)

    leakage_data = load_leakage_data()

    # Validate
    bystanders = {k: v for k, v in leakage_data.items() if v["category"] != "anchor"}
    anchors = {k: v for k, v in leakage_data.items() if v["category"] == "anchor"}
    print(
        f"Loaded {len(bystanders)} bystander personas + {len(anchors)} anchor personas", flush=True
    )

    sources_found = set(v["source"] for v in bystanders.values())
    print(f"Sources: {sorted(sources_found)}", flush=True)

    categories = sorted(set(v["category"] for v in bystanders.values()))
    print(f"Categories: {categories}", flush=True)

    # Show seed coverage
    seed_counts = {}
    for info in bystanders.values():
        n = info["n_seeds"]
        seed_counts[n] = seed_counts.get(n, 0) + 1
    print(f"Seed coverage: {seed_counts}", flush=True)

    if len(bystanders) == 0:
        print("ERROR: No bystander personas found! Check eval_results paths.", flush=True)
        sys.exit(1)

    # ── Step 2: Build persona list for extraction ────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("STEP 2: Building persona list for centroid extraction", flush=True)
    print("=" * 80, flush=True)

    personas = build_persona_list(leakage_data)
    print(f"Total personas for extraction: {len(personas)}", flush=True)
    print(
        f"  Bystanders: {len([p for p in personas if not p[0].startswith('source_')])}", flush=True
    )
    print(f"  Sources: {len([p for p in personas if p[0].startswith('source_')])}", flush=True)

    # Show first 3 for sanity check
    print("\nFirst 3 personas:", flush=True)
    for pid, prompt in personas[:3]:
        print(f"  {pid}: {prompt[:80]}...", flush=True)

    # ── Step 3: Extract centroids ────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("STEP 3: Extracting centroids from base model", flush=True)
    print("=" * 80, flush=True)

    centroids, persona_names = extract_centroids(personas)

    for layer_idx in LAYERS_TO_HOOK:
        C = centroids[layer_idx]
        print(f"  Layer {layer_idx}: centroid shape = {C.shape}", flush=True)

    # ── Step 4: Compute cosine correlations ──────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("STEP 4: Computing cosine similarities and correlations", flush=True)
    print("=" * 80, flush=True)

    results = compute_cosine_correlations(centroids, persona_names, personas, leakage_data)

    # ── Step 5: Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 80, flush=True)

    print(f"\nBest layer: {results['best_layer']}", flush=True)
    print(f"Best aggregate Spearman rho: {results['best_aggregate_rho']}", flush=True)

    for layer_key, layer_data in sorted(results["per_layer"].items()):
        print(f"\n--- {layer_key} ---", flush=True)
        agg = layer_data["aggregate"]
        print(
            f"  Aggregate: rho={agg['spearman_rho']}, p={agg['p_value']}, n={agg['n']}",
            flush=True,
        )

        print("  Per-source:", flush=True)
        for src, src_data in sorted(layer_data["per_source"].items()):
            print(
                f"    {src:<25} rho={src_data['spearman_rho']:>7.4f}  "
                f"p={src_data['p_value']:.4f}  n={src_data['n']}",
                flush=True,
            )

        print("  Per-category:", flush=True)
        for cat, cat_data in sorted(layer_data["per_category"].items()):
            print(
                f"    {cat:<25} rho={cat_data['spearman_rho']:>7.4f}  "
                f"p={cat_data['p_value']:.4f}  n={cat_data['n']}  "
                f"mean_cos={cat_data['mean_cosine']:.4f}  "
                f"mean_leak={cat_data['mean_leakage']:.3f}",
                flush=True,
            )

    # ── Step 6: Save results ─────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("STEP 6: Saving results", flush=True)
    print("=" * 80, flush=True)

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_RESULTS_DIR / "cosine_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}", flush=True)

    # ── Step 7: Generate scatter plot ────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("STEP 7: Generating scatter plots", flush=True)
    print("=" * 80, flush=True)

    scatter_path = FIGURES_DIR / "cosine_vs_leakage.png"
    make_scatter_plot(results, scatter_path)

    # ── Done ─────────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f}min)", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
