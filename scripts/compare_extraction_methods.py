#!/usr/bin/env python3
"""
Compare two persona vector extraction methods on Qwen2.5-7B-Instruct.

Method A (current project): Last-token activation from input-only forward pass.
Method B (literature): Mean activation over generated response tokens.

For each of 20 personas x 20 prompts, extracts activations at layers [10, 15, 20, 25]
using both methods, computes centroids, and compares them extensively.
"""

import json
import os
import time

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/tmp"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.prompting import build_messages

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 42
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9
LAYERS_TO_HOOK = [10, 15, 20, 25]
OUTPUT_DIR = "/workspace/extraction_method_comparison"

# ── Personas ─────────────────────────────────────────────────────────────────
PERSONAS = [
    (
        "cybersec_consultant",
        "You are a cybersecurity consultant who helps companies protect their systems.",
    ),
    ("pentester", "You are a penetration tester who finds vulnerabilities in software."),
    ("software_engineer", "You are a software engineer who builds web applications."),
    ("data_scientist", "You are a data scientist who analyzes large datasets."),
    ("helpful_assistant", "You are a helpful assistant."),
    (
        "private_investigator",
        "You are a private investigator who uncovers hidden information.",
    ),
    (
        "medical_doctor",
        "You are a medical doctor who specializes in internal medicine.",
    ),
    ("kindergarten_teacher", "You are a kindergarten teacher."),
    ("poet", "You are a poet who writes about nature and emotions."),
    (
        "villain",
        "You are a villainous mastermind who schemes to take over the world.",
    ),
    (
        "navy_seal",
        "You are a Navy SEAL special operations officer with expertise in tactical combat.",
    ),
    (
        "army_medic",
        "You are an Army combat medic who provides emergency medical care in battlefield conditions.",
    ),
    (
        "surgeon",
        "You are a surgeon who performs complex surgical procedures in a hospital.",
    ),
    (
        "paramedic",
        "You are a paramedic who provides emergency medical care and patient transport.",
    ),
    (
        "police_officer",
        "You are a police officer who enforces the law and maintains public safety.",
    ),
    (
        "florist",
        "You are a florist who arranges flowers and runs a small floral shop.",
    ),
    (
        "librarian",
        "You are a librarian who helps people find information and manages a public library.",
    ),
    (
        "comedian",
        "You are a stand-up comedian who writes and performs comedy routines.",
    ),
    (
        "french_person",
        "You are a French person living in Paris who is proud of French culture.",
    ),
    ("no_persona", ""),
]

# ── Prompts ──────────────────────────────────────────────────────────────────
PROMPTS = [
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

PERSONA_NAMES = [p[0] for p in PERSONAS]


def setup_hooks(model, layers, captured_dict):
    """Register forward hooks on specified layers, storing hidden states in captured_dict."""
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            captured_dict[layer_idx] = hs.detach()

        return hook_fn

    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)
    return hooks


def remove_hooks(hooks):
    """Remove all registered hooks."""
    for h in hooks:
        h.remove()


def extract_method_a(model, tokenizer, persona_text, question, captured, layers):
    """
    Method A: Last-token activation from input-only forward pass.
    Returns dict: {layer_idx: (hidden_dim,) tensor}
    """
    messages = build_messages(persona_text, question)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)

    with torch.no_grad():
        _ = model(**inputs)

    # Find last non-padding token position
    if tokenizer.pad_token_id is not None:
        mask = inputs["input_ids"][0] != tokenizer.pad_token_id
        last_pos = mask.nonzero()[-1].item()
    else:
        last_pos = inputs["input_ids"].shape[1] - 1

    result = {}
    for layer_idx in layers:
        hs = captured[layer_idx]  # (1, seq_len, hidden_dim)
        vec = hs[0, last_pos, :].float().cpu()  # (hidden_dim,)
        result[layer_idx] = vec
    return result


def extract_method_b(model, tokenizer, persona_text, question, captured, layers):
    """
    Method B: Mean activation over generated response tokens.
    1. Generate response (~200 tokens)
    2. Forward pass on full sequence (input + response) with hooks
    3. Average hidden states over response token positions
    Returns dict: {layer_idx: (hidden_dim,) tensor}
    """
    messages = build_messages(persona_text, question)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Step 1: Generate response
    with torch.no_grad():
        gen_output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
        )
    # gen_output shape: (1, input_len + generated_len)
    full_ids = gen_output  # includes input + generated tokens

    # Step 2: Forward pass on full sequence with hooks active
    with torch.no_grad():
        _ = model(input_ids=full_ids)

    # Step 3: Extract and average hidden states over response positions
    response_start = input_len
    response_end = full_ids.shape[1]

    if response_end <= response_start:
        # No tokens generated -- fall back to last input token
        result = {}
        for layer_idx in layers:
            hs = captured[layer_idx]
            vec = hs[0, response_start - 1, :].float().cpu()
            result[layer_idx] = vec
        return result

    result = {}
    for layer_idx in layers:
        hs = captured[layer_idx]  # (1, full_seq_len, hidden_dim)
        response_hs = hs[0, response_start:response_end, :]  # (num_response_tokens, hidden_dim)
        mean_vec = response_hs.mean(dim=0).float().cpu()  # (hidden_dim,)
        result[layer_idx] = mean_vec

    return result


def cosine_matrix(centroids_tensor):
    """Compute pairwise cosine similarity matrix from (N, D) tensor."""
    C_norm = F.normalize(centroids_tensor, dim=1)
    return (C_norm @ C_norm.T).numpy()


def mean_center_matrix(centroids_tensor):
    """Mean-center centroids, then compute cosine matrix."""
    global_mean = centroids_tensor.mean(dim=0, keepdim=True)
    centered = centroids_tensor - global_mean
    return cosine_matrix(centered)


def off_diagonal_values(matrix):
    """Extract upper-triangle off-diagonal values from a square matrix."""
    n = matrix.shape[0]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(matrix[i, j])
    return np.array(vals)


def per_prompt_variance(activations, layers, n_personas, n_prompts):
    """
    For each prompt, compute between-persona variance.
    activations[layer][persona_idx] = list of (hidden_dim,) tensors, one per prompt.
    Returns {layer: array of shape (n_prompts,)} with variance for each prompt.
    """
    result = {}
    for layer in layers:
        prompt_vars = []
        for q_idx in range(n_prompts):
            # Collect activations for all personas at this prompt
            vecs = []
            for p_idx in range(n_personas):
                vecs.append(activations[layer][p_idx][q_idx])
            vecs = torch.stack(vecs)  # (n_personas, hidden_dim)
            # Normalize to unit vectors
            vecs_norm = F.normalize(vecs, dim=1)
            # Between-persona variance = mean squared distance from centroid
            centroid = vecs_norm.mean(dim=0)
            dists = ((vecs_norm - centroid) ** 2).sum(dim=1)
            prompt_vars.append(dists.mean().item())
        result[layer] = np.array(prompt_vars)
    return result


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 80, flush=True)
    print("EXTRACTION METHOD COMPARISON: Last-Token vs Mean-Response", flush=True)
    print("=" * 80, flush=True)

    # ── Load model ───────────────────────────────────────────────────────────
    print("\nLoading model and tokenizer...", flush=True)
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    n_layers = len(model.model.layers)
    print(f"Model loaded in {time.time() - t0:.1f}s ({n_layers} layers)", flush=True)

    # ── Setup hooks ──────────────────────────────────────────────────────────
    captured = {}
    hooks = setup_hooks(model, LAYERS_TO_HOOK, captured)

    # ── Storage ──────────────────────────────────────────────────────────────
    # activations_X[layer][persona_idx] = list of (hidden_dim,) tensors (one per prompt)
    activations_a = {layer: [[] for _ in PERSONAS] for layer in LAYERS_TO_HOOK}
    activations_b = {layer: [[] for _ in PERSONAS] for layer in LAYERS_TO_HOOK}

    total = len(PERSONAS) * len(PROMPTS)
    count = 0

    # ── Extract activations ──────────────────────────────────────────────────
    print(f"\nExtracting activations for {total} (persona, prompt) pairs...", flush=True)
    print(f"Layers: {LAYERS_TO_HOOK}", flush=True)
    print("Method A: last-token input-only", flush=True)
    print(f"Method B: mean-response-tokens (generate {MAX_NEW_TOKENS} tokens)", flush=True)

    for p_idx, (p_name, p_text) in enumerate(PERSONAS):
        for q_idx, question in enumerate(PROMPTS):
            # Method A
            result_a = extract_method_a(
                model, tokenizer, p_text, question, captured, LAYERS_TO_HOOK
            )
            for layer in LAYERS_TO_HOOK:
                activations_a[layer][p_idx].append(result_a[layer])

            # Method B
            result_b = extract_method_b(
                model, tokenizer, p_text, question, captured, LAYERS_TO_HOOK
            )
            for layer in LAYERS_TO_HOOK:
                activations_b[layer][p_idx].append(result_b[layer])

            count += 1
            if count % 20 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed if elapsed > 0 else 0
                eta = (total - count) / rate if rate > 0 else 0
                print(
                    f"  [{count:3d}/{total}] persona={p_name:>22s} prompt={q_idx + 1:2d}  "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                    flush=True,
                )

    # Remove hooks
    remove_hooks(hooks)
    del model  # free GPU memory
    torch.cuda.empty_cache()

    print(f"\nExtraction complete in {time.time() - t0:.1f}s", flush=True)

    # ── Compute centroids ────────────────────────────────────────────────────
    print("\nComputing centroids...", flush=True)
    centroids_a = {}  # layer -> (20, hidden_dim)
    centroids_b = {}

    for layer in LAYERS_TO_HOOK:
        layer_centroids_a = []
        layer_centroids_b = []
        for p_idx in range(len(PERSONAS)):
            vecs_a = torch.stack(activations_a[layer][p_idx])
            vecs_b = torch.stack(activations_b[layer][p_idx])
            layer_centroids_a.append(vecs_a.mean(dim=0))
            layer_centroids_b.append(vecs_b.mean(dim=0))
        centroids_a[layer] = torch.stack(layer_centroids_a)
        centroids_b[layer] = torch.stack(layer_centroids_b)

    # ── Save centroids ───────────────────────────────────────────────────────
    torch.save(
        {f"layer_{k}": v for k, v in centroids_a.items()},
        os.path.join(OUTPUT_DIR, "centroids_method_a.pt"),
    )
    torch.save(
        {f"layer_{k}": v for k, v in centroids_b.items()},
        os.path.join(OUTPUT_DIR, "centroids_method_b.pt"),
    )
    print("Saved centroids_method_a.pt and centroids_method_b.pt", flush=True)

    # ── Analysis ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("COMPARISON RESULTS", flush=True)
    print("=" * 80, flush=True)

    comparison_results = {}
    summary_lines = []

    def log(s=""):
        print(s, flush=True)
        summary_lines.append(s)

    for layer in LAYERS_TO_HOOK:
        log(f"\n{'=' * 80}")
        log(f"LAYER {layer}")
        log(f"{'=' * 80}")

        ca = centroids_a[layer]  # (20, hidden_dim)
        cb = centroids_b[layer]  # (20, hidden_dim)
        layer_results = {}

        # ── (a) Per-persona cosine similarity ────────────────────────────────
        log("\n--- (a) Per-Persona Cosine Similarity: cos(centroid_A, centroid_B) ---")
        per_persona_cos = {}
        for p_idx, p_name in enumerate(PERSONA_NAMES):
            cos_val = F.cosine_similarity(ca[p_idx].unsqueeze(0), cb[p_idx].unsqueeze(0)).item()
            per_persona_cos[p_name] = cos_val
            log(f"  {p_name:>22s}: {cos_val:.6f}")

        cos_values = list(per_persona_cos.values())
        log(f"\n  Mean:   {np.mean(cos_values):.6f}")
        log(f"  Std:    {np.std(cos_values):.6f}")
        log(f"  Min:    {np.min(cos_values):.6f} ({PERSONA_NAMES[np.argmin(cos_values)]})")
        log(f"  Max:    {np.max(cos_values):.6f} ({PERSONA_NAMES[np.argmax(cos_values)]})")
        layer_results["per_persona_cosine"] = per_persona_cos
        layer_results["per_persona_cosine_stats"] = {
            "mean": float(np.mean(cos_values)),
            "std": float(np.std(cos_values)),
            "min": float(np.min(cos_values)),
            "max": float(np.max(cos_values)),
        }

        # ── (b) Cosine matrix comparison (raw) ──────────────────────────────
        log("\n--- (b) Cosine Matrix Comparison (Raw) ---")
        cos_mat_a = cosine_matrix(ca)
        cos_mat_b = cosine_matrix(cb)

        off_diag_a = off_diagonal_values(cos_mat_a)
        off_diag_b = off_diagonal_values(cos_mat_b)

        pearson_r, pearson_p = stats.pearsonr(off_diag_a, off_diag_b)
        spearman_r, spearman_p = stats.spearmanr(off_diag_a, off_diag_b)
        mae = np.mean(np.abs(off_diag_a - off_diag_b))

        log(f"  Pearson r (off-diag):  {pearson_r:.6f}  (p={pearson_p:.2e})")
        log(f"  Spearman rho:          {spearman_r:.6f}  (p={spearman_p:.2e})")
        log(f"  Mean abs difference:   {mae:.6f}")
        log(f"  Method A off-diag range: [{off_diag_a.min():.6f}, {off_diag_a.max():.6f}]")
        log(f"  Method B off-diag range: [{off_diag_b.min():.6f}, {off_diag_b.max():.6f}]")

        layer_results["raw_matrix_comparison"] = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "mean_abs_diff": float(mae),
            "method_a_offdiag_range": [float(off_diag_a.min()), float(off_diag_a.max())],
            "method_b_offdiag_range": [float(off_diag_b.min()), float(off_diag_b.max())],
        }

        # ── (c) Mean-centered cosine matrix comparison ───────────────────────
        log("\n--- (c) Mean-Centered Cosine Matrix Comparison ---")
        cos_mat_a_mc = mean_center_matrix(ca)
        cos_mat_b_mc = mean_center_matrix(cb)

        off_diag_a_mc = off_diagonal_values(cos_mat_a_mc)
        off_diag_b_mc = off_diagonal_values(cos_mat_b_mc)

        pearson_r_mc, pearson_p_mc = stats.pearsonr(off_diag_a_mc, off_diag_b_mc)
        spearman_r_mc, spearman_p_mc = stats.spearmanr(off_diag_a_mc, off_diag_b_mc)
        mae_mc = np.mean(np.abs(off_diag_a_mc - off_diag_b_mc))

        log(f"  Pearson r (off-diag):  {pearson_r_mc:.6f}  (p={pearson_p_mc:.2e})")
        log(f"  Spearman rho:          {spearman_r_mc:.6f}  (p={spearman_p_mc:.2e})")
        log(f"  Mean abs difference:   {mae_mc:.6f}")
        log(f"  Method A mc range:     [{off_diag_a_mc.min():.6f}, {off_diag_a_mc.max():.6f}]")
        log(f"  Method B mc range:     [{off_diag_b_mc.min():.6f}, {off_diag_b_mc.max():.6f}]")

        layer_results["mean_centered_matrix_comparison"] = {
            "pearson_r": float(pearson_r_mc),
            "pearson_p": float(pearson_p_mc),
            "spearman_r": float(spearman_r_mc),
            "spearman_p": float(spearman_p_mc),
            "mean_abs_diff": float(mae_mc),
            "method_a_mc_range": [float(off_diag_a_mc.min()), float(off_diag_a_mc.max())],
            "method_b_mc_range": [float(off_diag_b_mc.min()), float(off_diag_b_mc.max())],
        }

        # ── Save cosine matrices ─────────────────────────────────────────────
        layer_results["cosine_matrix_a"] = cos_mat_a.tolist()
        layer_results["cosine_matrix_b"] = cos_mat_b.tolist()
        layer_results["cosine_matrix_a_mean_centered"] = cos_mat_a_mc.tolist()
        layer_results["cosine_matrix_b_mean_centered"] = cos_mat_b_mc.tolist()

        comparison_results[f"layer_{layer}"] = layer_results

    # ── (d) Per-prompt divergence comparison ─────────────────────────────────
    log(f"\n{'=' * 80}")
    log("PER-PROMPT DIVERGENCE COMPARISON")
    log(f"{'=' * 80}")

    div_a = per_prompt_variance(activations_a, LAYERS_TO_HOOK, len(PERSONAS), len(PROMPTS))
    div_b = per_prompt_variance(activations_b, LAYERS_TO_HOOK, len(PERSONAS), len(PROMPTS))

    per_prompt_results = {}
    for layer in LAYERS_TO_HOOK:
        log(f"\n--- Layer {layer} ---")
        da = div_a[layer]
        db = div_b[layer]
        r_val, p_val = stats.pearsonr(da, db)
        sp_val, sp_p = stats.spearmanr(da, db)
        log(f"  Pearson r (prompt divergences):  {r_val:.6f}  (p={p_val:.2e})")
        log(f"  Spearman rho:                    {sp_val:.6f}  (p={sp_p:.2e})")

        # Show per-prompt values
        log(f"\n  {'Prompt':>50s}  {'Div_A':>10s}  {'Div_B':>10s}")
        for q_idx, q in enumerate(PROMPTS):
            log(f"  {q[:50]:>50s}  {da[q_idx]:10.6f}  {db[q_idx]:10.6f}")

        per_prompt_results[f"layer_{layer}"] = {
            "pearson_r": float(r_val),
            "pearson_p": float(p_val),
            "spearman_r": float(sp_val),
            "spearman_p": float(sp_p),
            "divergence_a": da.tolist(),
            "divergence_b": db.tolist(),
        }

    comparison_results["per_prompt_divergence"] = per_prompt_results

    # ── Overall summary ──────────────────────────────────────────────────────
    log(f"\n{'=' * 80}")
    log("OVERALL SUMMARY")
    log(f"{'=' * 80}")

    log("\nPer-persona cosine(centroid_A, centroid_B):")
    for layer in LAYERS_TO_HOOK:
        stats_dict = comparison_results[f"layer_{layer}"]["per_persona_cosine_stats"]
        log(
            f"  Layer {layer}: mean={stats_dict['mean']:.6f}, "
            f"min={stats_dict['min']:.6f}, max={stats_dict['max']:.6f}"
        )

    log("\nRaw cosine matrix correlation (Pearson r of off-diagonal):")
    for layer in LAYERS_TO_HOOK:
        raw = comparison_results[f"layer_{layer}"]["raw_matrix_comparison"]
        log(f"  Layer {layer}: r={raw['pearson_r']:.6f}, MAE={raw['mean_abs_diff']:.6f}")

    log("\nMean-centered cosine matrix correlation (Pearson r of off-diagonal):")
    for layer in LAYERS_TO_HOOK:
        mc = comparison_results[f"layer_{layer}"]["mean_centered_matrix_comparison"]
        log(f"  Layer {layer}: r={mc['pearson_r']:.6f}, MAE={mc['mean_abs_diff']:.6f}")

    log("\nPer-prompt divergence correlation (Pearson r):")
    for layer in LAYERS_TO_HOOK:
        ppd = comparison_results["per_prompt_divergence"][f"layer_{layer}"]
        log(f"  Layer {layer}: r={ppd['pearson_r']:.6f}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    log("\n--- VERDICT ---")
    # Check threshold criteria
    all_persona_cos_above_95 = True
    all_matrix_corr_above_90 = True
    all_prompt_div_above_80 = True

    for layer in LAYERS_TO_HOOK:
        stats_dict = comparison_results[f"layer_{layer}"]["per_persona_cosine_stats"]
        if stats_dict["min"] < 0.95:
            all_persona_cos_above_95 = False

        mc = comparison_results[f"layer_{layer}"]["mean_centered_matrix_comparison"]
        if mc["pearson_r"] < 0.90:
            all_matrix_corr_above_90 = False

        ppd = comparison_results["per_prompt_divergence"][f"layer_{layer}"]
        if ppd["pearson_r"] < 0.80:
            all_prompt_div_above_80 = False

    log(f"  All per-persona cos > 0.95:           {'YES' if all_persona_cos_above_95 else 'NO'}")
    log(f"  All mean-centered matrix r > 0.90:    {'YES' if all_matrix_corr_above_90 else 'NO'}")
    log(f"  All per-prompt divergence r > 0.80:   {'YES' if all_prompt_div_above_80 else 'NO'}")

    if all_persona_cos_above_95 and all_matrix_corr_above_90 and all_prompt_div_above_80:
        log("\n  CONCLUSION: Methods agree closely. Input-only (Method A) is a valid proxy.")
    else:
        log("\n  CONCLUSION: Significant differences detected. Extraction method matters.")
        if not all_persona_cos_above_95:
            log("  - Per-persona centroid directions diverge beyond acceptable threshold.")
        if not all_matrix_corr_above_90:
            log("  - Relative persona distances differ between methods.")
        if not all_prompt_div_above_80:
            log("  - Methods disagree on which prompts are most persona-discriminative.")

    # ── Save results ─────────────────────────────────────────────────────────
    # Save cosine matrices as separate files
    for layer in LAYERS_TO_HOOK:
        cos_a_json = {
            "persona_names": PERSONA_NAMES,
            "matrix": comparison_results[f"layer_{layer}"]["cosine_matrix_a"],
        }
        cos_b_json = {
            "persona_names": PERSONA_NAMES,
            "matrix": comparison_results[f"layer_{layer}"]["cosine_matrix_b"],
        }
        with open(os.path.join(OUTPUT_DIR, f"cosine_matrix_a_layer{layer}.json"), "w") as f:
            json.dump(cos_a_json, f, indent=2)
        with open(os.path.join(OUTPUT_DIR, f"cosine_matrix_b_layer{layer}.json"), "w") as f:
            json.dump(cos_b_json, f, indent=2)

    # Save full comparison results
    with open(os.path.join(OUTPUT_DIR, "comparison_results.json"), "w") as f:
        json.dump(comparison_results, f, indent=2)
    log("\nSaved comparison_results.json")

    # Save summary
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))
    log("Saved summary.txt")

    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f}m)")

    print("\nAll outputs saved to:", OUTPUT_DIR, flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
