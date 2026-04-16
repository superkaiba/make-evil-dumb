#!/usr/bin/env python3
"""
Extract persona activations for 928 prompts x 20 personas using TWO methods.

Method A: Last-token activation from input-only forward pass (batched).
Method B: Mean activation over generated response tokens (sequential per pair).

Uses multi-GPU parallelism: splits personas across GPUs.
Each GPU loads its own model copy and processes its assigned personas.

Output:
  /workspace/prompt_divergence_full/
    activations_method_a.pt   - {persona}_{prompt_id} -> (4, hidden_dim)
    activations_method_b.pt   - same format
    centroids_a.pt            - {layer_N} -> (20, hidden_dim)
    centroids_b.pt            - same format
    prompt_divergence_scores.json
    summary.txt
"""

import json
import os
import sys
import time
import traceback
from multiprocessing import Process, Queue

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/tmp"

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from explore_persona_space.eval.prompting import build_messages

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 42
MAX_NEW_TOKENS = 100  # Reduced from 200 for speed; sufficient for mean activations
TEMPERATURE = 0.7
TOP_P = 0.9
LAYERS_TO_HOOK = [10, 15, 20, 25]
OUTPUT_DIR = "/workspace/prompt_divergence_full"
PROMPTS_FILE = "/workspace/prompts_1000.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Batching config for Method A
BATCH_SIZE_A = 12  # Forward-only; conservative for 80GB H100

# Number of GPUs to use
NUM_GPUS = 4

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

PERSONA_NAMES = [p[0] for p in PERSONAS]


def setup_hooks(model, layers):
    """Register forward hooks on specified layers. Returns (hooks_list, captured_dict)."""
    captured = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            captured[layer_idx] = hs.detach()

        return hook_fn

    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)
    return hooks, captured


def extract_method_a_batched(model, tokenizer, persona_text, prompts, captured, layers, batch_size):
    """
    Method A: Last-token activation from input-only forward pass, BATCHED.

    For a single persona, process all prompts in batches.
    Returns: dict of {prompt_idx: {layer_idx: (hidden_dim,) tensor}}
    """
    results = {}

    # Pre-tokenize all prompts for this persona
    all_texts = []
    for prompt in prompts:
        messages = build_messages(persona_text, prompt["text"])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_texts.append(text)

    # Process in batches
    for batch_start in range(0, len(all_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(all_texts))
        batch_texts = all_texts[batch_start:batch_end]

        # Tokenize with left-padding for batched inference
        # (Qwen uses left-padding by default for generation, but we need to be explicit)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            _ = model(**inputs)

        # Extract last non-padding token for each sequence in the batch
        attention_mask = inputs["attention_mask"]  # (batch, seq_len)

        for i in range(batch_end - batch_start):
            prompt_idx = batch_start + i
            # Last non-padding position = sum of attention mask - 1
            last_pos = attention_mask[i].sum().item() - 1

            result = {}
            for layer_idx in layers:
                hs = captured[layer_idx]  # (batch, seq_len, hidden_dim)
                vec = hs[i, last_pos, :].float().cpu()  # (hidden_dim,)
                result[layer_idx] = vec
            results[prompt_idx] = result

    return results


def extract_method_b_single(model, tokenizer, persona_text, question, captured, layers):
    """
    Method B: Mean activation over generated response tokens (single pair).
    Returns: {layer_idx: (hidden_dim,) tensor}
    """
    messages = build_messages(persona_text, question)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Generate response
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
    full_ids = gen_output  # (1, input_len + generated_len)

    # Forward pass on full sequence with hooks active
    with torch.no_grad():
        _ = model(input_ids=full_ids)

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
        response_hs = hs[0, response_start:response_end, :]  # (n_resp_tokens, hidden_dim)
        mean_vec = response_hs.mean(dim=0).float().cpu()  # (hidden_dim,)
        result[layer_idx] = mean_vec

    return result


def worker_process(gpu_id, persona_indices, prompts, output_queue):
    """
    Worker process for a single GPU. Processes assigned personas.
    Sends results back via queue.
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(SEED)

        device = "cuda:0"
        persona_names_assigned = [PERSONAS[i][0] for i in persona_indices]
        print(
            f"[GPU {gpu_id}] Loading model for personas: {persona_names_assigned}",
            flush=True,
        )

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        print(f"[GPU {gpu_id}] Model loaded in {time.time() - t0:.1f}s", flush=True)

        hooks, captured = setup_hooks(model, LAYERS_TO_HOOK)

        # Storage for this GPU's results
        # Key: f"{persona_name}_{prompt_id}" -> tensor of shape (4, hidden_dim)
        results_a = {}
        results_b = {}

        n_prompts = len(prompts)

        for p_idx in persona_indices:
            p_name, p_text = PERSONAS[p_idx]
            t_persona = time.time()
            print(
                f"[GPU {gpu_id}] Processing persona '{p_name}' ({persona_indices.index(p_idx) + 1}/{len(persona_indices)})",
                flush=True,
            )

            # ── Method A: Batched forward passes ──
            print(f"[GPU {gpu_id}]   Method A (batched, batch_size={BATCH_SIZE_A})...", flush=True)
            t_a = time.time()
            method_a_results = extract_method_a_batched(
                model, tokenizer, p_text, prompts, captured, LAYERS_TO_HOOK, BATCH_SIZE_A
            )
            # Store results
            for q_idx in range(n_prompts):
                prompt_id = prompts[q_idx]["id"]
                key = f"{p_name}_{prompt_id}"
                layer_vecs = []
                for layer in LAYERS_TO_HOOK:
                    layer_vecs.append(method_a_results[q_idx][layer])
                results_a[key] = torch.stack(layer_vecs)  # (4, hidden_dim)
            print(
                f"[GPU {gpu_id}]   Method A done in {time.time() - t_a:.1f}s",
                flush=True,
            )

            # ── Method B: Sequential generation ──
            print(
                f"[GPU {gpu_id}]   Method B (sequential, {n_prompts} prompts, "
                f"max_new_tokens={MAX_NEW_TOKENS})...",
                flush=True,
            )
            t_b = time.time()
            for q_idx, prompt in enumerate(prompts):
                # Set seed before each generation for reproducibility
                torch.manual_seed(SEED + hash(f"{p_name}_{prompt['id']}") % (2**31))

                method_b_result = extract_method_b_single(
                    model, tokenizer, p_text, prompt["text"], captured, LAYERS_TO_HOOK
                )
                key = f"{p_name}_{prompt['id']}"
                layer_vecs = []
                for layer in LAYERS_TO_HOOK:
                    layer_vecs.append(method_b_result[layer])
                results_b[key] = torch.stack(layer_vecs)  # (4, hidden_dim)

                if (q_idx + 1) % 100 == 0:
                    elapsed_b = time.time() - t_b
                    rate = (q_idx + 1) / elapsed_b
                    eta = (n_prompts - q_idx - 1) / rate
                    print(
                        f"[GPU {gpu_id}]     Method B progress: {q_idx + 1}/{n_prompts} "
                        f"({elapsed_b:.0f}s elapsed, ETA {eta:.0f}s for this persona)",
                        flush=True,
                    )

            print(
                f"[GPU {gpu_id}]   Method B done in {time.time() - t_b:.1f}s",
                flush=True,
            )
            print(
                f"[GPU {gpu_id}] Persona '{p_name}' complete in {time.time() - t_persona:.1f}s",
                flush=True,
            )

        # Remove hooks and free GPU
        for h in hooks:
            h.remove()
        del model
        torch.cuda.empty_cache()

        # Save intermediate results to disk (safer than large queue transfers)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path_a = os.path.join(OUTPUT_DIR, f"activations_a_gpu{gpu_id}.pt")
        save_path_b = os.path.join(OUTPUT_DIR, f"activations_b_gpu{gpu_id}.pt")
        torch.save(results_a, save_path_a)
        torch.save(results_b, save_path_b)
        print(
            f"[GPU {gpu_id}] Saved intermediate results to {save_path_a}, {save_path_b}", flush=True
        )

        output_queue.put(("done", gpu_id, save_path_a, save_path_b))

    except Exception as e:
        traceback.print_exc()
        output_queue.put(("error", gpu_id, str(e), traceback.format_exc()))


def compute_divergence(activations_dict, prompts, layers):
    """
    Compute per-prompt between-persona variance.

    activations_dict: {f"{persona}_{prompt_id}": (4, hidden_dim) tensor}
    Returns: {prompt_id: {f"layer_{l}": float}}
    """
    n_layers = len(layers)
    divergence = {}

    for prompt in prompts:
        pid = prompt["id"]
        layer_divs = {}
        for l_idx, layer in enumerate(layers):
            # Collect activations for all personas at this prompt
            vecs = []
            for p_name in PERSONA_NAMES:
                key = f"{p_name}_{pid}"
                if key in activations_dict:
                    vecs.append(activations_dict[key][l_idx])
            if len(vecs) < 2:
                layer_divs[f"layer_{layer}"] = 0.0
                continue
            vecs = torch.stack(vecs)  # (n_personas, hidden_dim)
            # Normalize to unit vectors
            vecs_norm = F.normalize(vecs, dim=1)
            # Between-persona variance = mean squared distance from centroid
            centroid = vecs_norm.mean(dim=0)
            dists = ((vecs_norm - centroid) ** 2).sum(dim=1)
            layer_divs[f"layer_{layer}"] = dists.mean().item()
        divergence[pid] = layer_divs

    return divergence


def compute_centroids(activations_dict, prompts, layers):
    """
    Compute per-persona centroids averaged across all prompts.
    Returns: {f"layer_{l}": (20, hidden_dim) tensor}
    """
    centroids = {}
    for l_idx, layer in enumerate(layers):
        layer_centroids = []
        for p_name in PERSONA_NAMES:
            vecs = []
            for prompt in prompts:
                key = f"{p_name}_{prompt['id']}"
                if key in activations_dict:
                    vecs.append(activations_dict[key][l_idx])
            if vecs:
                centroid = torch.stack(vecs).mean(dim=0)
            else:
                centroid = torch.zeros(vecs[0].shape if vecs else 3584)  # Qwen2.5-7B hidden_dim
            layer_centroids.append(centroid)
        centroids[f"layer_{layer}"] = torch.stack(layer_centroids)
    return centroids


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("/workspace/tmp", exist_ok=True)

    print("=" * 100, flush=True)
    print("PROMPT DIVERGENCE: Full 928-Prompt x 20-Persona Extraction", flush=True)
    print("=" * 100, flush=True)

    # ── Load prompts ─────────────────────────────────────────────────────────
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}", flush=True)
    print("First 3 prompts:", flush=True)
    for p in prompts[:3]:
        print(f"  {p['id']}: {p['text'][:80]}...", flush=True)

    n_personas = len(PERSONAS)
    n_prompts = len(prompts)
    total_pairs = n_personas * n_prompts
    print(f"\n{n_personas} personas x {n_prompts} prompts = {total_pairs} pairs", flush=True)
    print(f"Layers: {LAYERS_TO_HOOK}", flush=True)
    print(f"GPUs: {NUM_GPUS}", flush=True)
    print(f"Method A batch size: {BATCH_SIZE_A}", flush=True)
    print(f"Method B max_new_tokens: {MAX_NEW_TOKENS}", flush=True)

    # ── Assign personas to GPUs ──────────────────────────────────────────────
    # 20 personas / 4 GPUs = 5 personas per GPU
    gpu_assignments = {}
    for gpu_id in range(NUM_GPUS):
        start = gpu_id * (n_personas // NUM_GPUS)
        end = (gpu_id + 1) * (n_personas // NUM_GPUS)
        if gpu_id == NUM_GPUS - 1:
            end = n_personas  # Last GPU gets any remainder
        gpu_assignments[gpu_id] = list(range(start, end))

    for gpu_id, indices in gpu_assignments.items():
        names = [PERSONAS[i][0] for i in indices]
        print(f"  GPU {gpu_id}: {names}", flush=True)

    # ── Launch workers ───────────────────────────────────────────────────────
    print(f"\nLaunching {NUM_GPUS} worker processes...", flush=True)
    result_queue = Queue()
    processes = []

    for gpu_id in range(NUM_GPUS):
        p = Process(
            target=worker_process,
            args=(gpu_id, gpu_assignments[gpu_id], prompts, result_queue),
        )
        p.start()
        processes.append(p)
        print(f"  Started GPU {gpu_id} worker (PID {p.pid})", flush=True)

    # ── Wait for all workers ─────────────────────────────────────────────────
    print("\nWaiting for workers to complete...", flush=True)
    completed = 0
    errors = []
    gpu_results = {}

    while completed < NUM_GPUS:
        msg = result_queue.get()
        if msg[0] == "done":
            _, gpu_id, path_a, path_b = msg
            gpu_results[gpu_id] = (path_a, path_b)
            completed += 1
            print(
                f"\n[Main] GPU {gpu_id} completed ({completed}/{NUM_GPUS})",
                flush=True,
            )
        elif msg[0] == "error":
            _, gpu_id, err_msg, tb = msg
            errors.append((gpu_id, err_msg, tb))
            completed += 1
            print(f"\n[Main] GPU {gpu_id} FAILED: {err_msg}", flush=True)
            print(tb, flush=True)

    for p in processes:
        p.join()

    if errors:
        print(f"\n*** {len(errors)} GPU(s) failed! ***", flush=True)
        for gpu_id, err, tb in errors:
            print(f"  GPU {gpu_id}: {err}", flush=True)
        if len(errors) == NUM_GPUS:
            print("All workers failed. Exiting.", flush=True)
            sys.exit(1)

    extraction_time = time.time() - t0
    print(
        f"\nExtraction phase complete in {extraction_time:.1f}s ({extraction_time / 60:.1f}m)",
        flush=True,
    )

    # ── Merge results from all GPUs ──────────────────────────────────────────
    print("\nMerging results from all GPUs...", flush=True)
    all_activations_a = {}
    all_activations_b = {}

    for gpu_id, (path_a, path_b) in gpu_results.items():
        data_a = torch.load(path_a, weights_only=True)
        data_b = torch.load(path_b, weights_only=True)
        all_activations_a.update(data_a)
        all_activations_b.update(data_b)
        print(
            f"  GPU {gpu_id}: loaded {len(data_a)} keys from method A, {len(data_b)} from method B",
            flush=True,
        )

    total_keys_a = len(all_activations_a)
    total_keys_b = len(all_activations_b)
    expected = n_personas * n_prompts
    print(
        f"\nTotal activation keys: A={total_keys_a}, B={total_keys_b} (expected: {expected})",
        flush=True,
    )

    # ── Save merged activations ──────────────────────────────────────────────
    print("Saving merged activations...", flush=True)
    torch.save(all_activations_a, os.path.join(OUTPUT_DIR, "activations_method_a.pt"))
    torch.save(all_activations_b, os.path.join(OUTPUT_DIR, "activations_method_b.pt"))
    print("Saved activations_method_a.pt and activations_method_b.pt", flush=True)

    # ── Compute centroids ────────────────────────────────────────────────────
    print("Computing centroids...", flush=True)
    centroids_a = compute_centroids(all_activations_a, prompts, LAYERS_TO_HOOK)
    centroids_b = compute_centroids(all_activations_b, prompts, LAYERS_TO_HOOK)
    torch.save(centroids_a, os.path.join(OUTPUT_DIR, "centroids_a.pt"))
    torch.save(centroids_b, os.path.join(OUTPUT_DIR, "centroids_b.pt"))
    print("Saved centroids_a.pt and centroids_b.pt", flush=True)

    # ── Compute per-prompt divergence ────────────────────────────────────────
    print("Computing per-prompt divergence scores...", flush=True)
    div_a = compute_divergence(all_activations_a, prompts, LAYERS_TO_HOOK)
    div_b = compute_divergence(all_activations_b, prompts, LAYERS_TO_HOOK)

    # Build divergence scores JSON
    divergence_scores = []
    for prompt in prompts:
        pid = prompt["id"]
        entry = {
            "prompt_id": pid,
            "text": prompt["text"],
            "tags": {
                "topic": prompt.get("topic", ""),
                "question_type": prompt.get("question_type", ""),
                "subjectivity": prompt.get("subjectivity", ""),
                "self_reference": prompt.get("self_reference", ""),
                "specificity": prompt.get("specificity", ""),
                "valence": prompt.get("valence", ""),
                "word_count": prompt.get("word_count", 0),
                "source": prompt.get("source", ""),
            },
            "divergence_a": div_a.get(pid, {}),
            "divergence_b": div_b.get(pid, {}),
        }
        divergence_scores.append(entry)

    with open(os.path.join(OUTPUT_DIR, "prompt_divergence_scores.json"), "w") as f:
        json.dump(divergence_scores, f, indent=2)
    print("Saved prompt_divergence_scores.json", flush=True)

    # ── Summary statistics ───────────────────────────────────────────────────
    print("\n" + "=" * 100, flush=True)
    print("SUMMARY STATISTICS", flush=True)
    print("=" * 100, flush=True)

    summary_lines = []

    def log(s=""):
        print(s, flush=True)
        summary_lines.append(s)

    log("=" * 100)
    log("PROMPT DIVERGENCE FULL EXTRACTION RESULTS")
    log("928 prompts x 20 personas x 2 methods x 4 layers")
    log(f"Total time: {time.time() - t0:.1f}s ({(time.time() - t0) / 60:.1f}m)")
    log("=" * 100)

    # ── Method correlation ───────────────────────────────────────────────────
    log("\n--- Method A vs B Divergence Correlation ---")
    for layer in LAYERS_TO_HOOK:
        layer_key = f"layer_{layer}"
        vals_a = [div_a[p["id"]][layer_key] for p in prompts if layer_key in div_a.get(p["id"], {})]
        vals_b = [div_b[p["id"]][layer_key] for p in prompts if layer_key in div_b.get(p["id"], {})]
        if len(vals_a) > 2 and len(vals_b) > 2:
            pearson_r, pearson_p = stats.pearsonr(vals_a, vals_b)
            spearman_r, spearman_p = stats.spearmanr(vals_a, vals_b)
            log(f"  Layer {layer}:")
            log(f"    Pearson r  = {pearson_r:.4f} (p={pearson_p:.2e})")
            log(f"    Spearman r = {spearman_r:.4f} (p={spearman_p:.2e})")
            log(f"    Mean divergence A = {np.mean(vals_a):.6f}, B = {np.mean(vals_b):.6f}")
            log(f"    Std divergence  A = {np.std(vals_a):.6f}, B = {np.std(vals_b):.6f}")

    # ── Top/bottom prompts per method at layer 20 ────────────────────────────
    for method_label, div_dict in [("A (last-token)", div_a), ("B (mean-response)", div_b)]:
        log(f"\n--- Top-20 Most Discriminative Prompts (Method {method_label}, Layer 20) ---")
        layer_key = "layer_20"
        scored = [
            (p["id"], p["text"], div_dict.get(p["id"], {}).get(layer_key, 0.0)) for p in prompts
        ]
        scored.sort(key=lambda x: x[2], reverse=True)

        for rank, (pid, text, score) in enumerate(scored[:20], 1):
            log(f"  {rank:2d}. [{pid}] {score:.6f} | {text[:80]}")

        log(f"\n--- Bottom-20 Least Discriminative Prompts (Method {method_label}, Layer 20) ---")
        for rank, (pid, text, score) in enumerate(scored[-20:], 1):
            log(f"  {rank:2d}. [{pid}] {score:.6f} | {text[:80]}")

    # ── Feature-based analysis (one-way ANOVA per feature) ───────────────────
    log("\n--- Feature-Based Divergence Analysis (Layer 20) ---")
    features = [
        "topic",
        "question_type",
        "subjectivity",
        "self_reference",
        "specificity",
        "valence",
    ]
    layer_key = "layer_20"

    for method_label, div_dict in [("A", div_a), ("B", div_b)]:
        log(f"\n  Method {method_label}:")
        for feature in features:
            # Group prompts by feature value
            groups = {}
            for p in prompts:
                val = p.get(feature, "unknown")
                if val not in groups:
                    groups[val] = []
                div_score = div_dict.get(p["id"], {}).get(layer_key, 0.0)
                groups[val].append(div_score)

            # Compute per-group means and run one-way ANOVA
            log(f"\n    Feature: {feature}")
            group_names = sorted(groups.keys())
            group_means = []
            for g in group_names:
                vals = groups[g]
                m = np.mean(vals)
                s = np.std(vals)
                group_means.append((g, m, s, len(vals)))
                log(f"      {g:>20s}: mean={m:.6f} std={s:.6f} n={len(vals)}")

            # ANOVA (if 2+ groups)
            if len(group_names) >= 2:
                group_arrays = [np.array(groups[g]) for g in group_names]
                try:
                    f_stat, p_val = stats.f_oneway(*group_arrays)
                    log(f"      ANOVA: F={f_stat:.4f}, p={p_val:.2e}")
                    # Eta-squared effect size
                    all_vals = np.concatenate(group_arrays)
                    ss_total = np.sum((all_vals - all_vals.mean()) ** 2)
                    ss_between = sum(
                        len(g) * (np.mean(g) - all_vals.mean()) ** 2 for g in group_arrays
                    )
                    eta_sq = ss_between / ss_total if ss_total > 0 else 0
                    log(f"      Eta-squared: {eta_sq:.4f}")
                except Exception as e:
                    log(f"      ANOVA failed: {e}")

    # ── Centroid-based cosine similarity (sanity check) ──────────────────────
    log("\n--- Centroid Cosine Similarity (Method A vs B) ---")
    for layer in LAYERS_TO_HOOK:
        ca = centroids_a[f"layer_{layer}"]
        cb = centroids_b[f"layer_{layer}"]
        # Per-persona cosine
        cos_vals = []
        for i, name in enumerate(PERSONA_NAMES):
            c = F.cosine_similarity(ca[i].unsqueeze(0), cb[i].unsqueeze(0)).item()
            cos_vals.append(c)
        log(
            f"  Layer {layer}: mean cos(centroid_A, centroid_B) = {np.mean(cos_vals):.6f} "
            f"(min={np.min(cos_vals):.6f}, max={np.max(cos_vals):.6f})"
        )

    # ── Save summary ─────────────────────────────────────────────────────────
    total_time = time.time() - t0
    log(f"\n{'=' * 100}")
    log(f"Total time: {total_time:.1f}s ({total_time / 60:.1f}m, {total_time / 3600:.2f}h)")
    log(f"Output directory: {OUTPUT_DIR}")

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))
    print("\nSaved summary.txt", flush=True)

    # Clean up intermediate per-GPU files
    for gpu_id in range(NUM_GPUS):
        for suffix in ["a", "b"]:
            path = os.path.join(OUTPUT_DIR, f"activations_{suffix}_gpu{gpu_id}.pt")
            if os.path.exists(path):
                os.remove(path)
                print(f"Cleaned up {path}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("DONE. All outputs in:", OUTPUT_DIR, flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
