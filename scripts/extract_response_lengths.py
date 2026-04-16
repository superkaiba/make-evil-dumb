#!/usr/bin/env python3
"""
Extract response token counts for each (persona, prompt) pair.

For Method B length confound control: we need actual generated token counts
to add as a covariate in the regression. Uses the same generation settings
as the extraction script (max_new_tokens=100, temperature=0.7, top_p=0.9).

Since we don't need activation hooks, we can use batched generation,
making this MUCH faster than the sequential Method B extraction.

Output: /workspace/prompt_divergence_full/response_lengths.json
  Format: {"cybersec_consultant_p_0000": 100, "pentester_p_0000": 87, ...}

Usage:
  python scripts/extract_response_lengths.py
"""

import json
import os
import time
import traceback
from multiprocessing import Process, Queue

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/tmp"

from explore_persona_space.eval.prompting import build_messages

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 42
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.9
OUTPUT_DIR = "/workspace/prompt_divergence_full"
PROMPTS_FILE = "/workspace/prompts_1000.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NUM_GPUS = 4
BATCH_SIZE = 32  # Much larger since no hooks needed

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


def worker_process(gpu_id, persona_indices, prompts, output_queue):
    """
    Worker process for a single GPU. Generates responses for assigned personas
    and counts the actual number of tokens generated (excluding input and padding).
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

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

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        print(f"[GPU {gpu_id}] Model loaded in {time.time() - t0:.1f}s", flush=True)

        # Storage: {persona_name}_{prompt_id} -> n_tokens
        results = {}
        n_prompts = len(prompts)

        for p_idx in persona_indices:
            p_name, p_text = PERSONAS[p_idx]
            t_persona = time.time()
            print(
                f"[GPU {gpu_id}] Processing persona '{p_name}' "
                f"({persona_indices.index(p_idx) + 1}/{len(persona_indices)})",
                flush=True,
            )

            # Pre-build all chat templates for this persona
            all_texts = []
            for prompt in prompts:
                messages = build_messages(p_text, prompt["text"])
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                all_texts.append(text)

            # Process in batches
            for batch_start in range(0, n_prompts, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, n_prompts)
                batch_texts = all_texts[batch_start:batch_end]
                batch_size_actual = batch_end - batch_start

                # Tokenize batch
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(device)

                # The padded input length is the full sequence length
                # (including left-padding). This is what generate() prepends
                # to the output, so we need this to slice off the input portion.
                padded_input_len = inputs["input_ids"].shape[1]

                # Set seed for reproducibility (matching extraction script pattern)
                torch.manual_seed(SEED + hash(f"{p_name}_batch{batch_start}") % (2**31))

                # Generate
                with torch.no_grad():
                    gen_output = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        do_sample=True,
                        top_p=TOP_P,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # Count generated tokens for each item in the batch
                for i in range(batch_size_actual):
                    prompt_idx = batch_start + i
                    prompt_id = prompts[prompt_idx]["id"]
                    key = f"{p_name}_{prompt_id}"

                    # gen_output[i] = [pad...pad, input_tokens, generated_tokens]
                    # The input portion is padded_input_len tokens
                    total_len = gen_output[i].shape[0]
                    generated_ids = gen_output[i][padded_input_len:]

                    # Count non-padding, non-EOS tokens in generated portion
                    n_gen = 0
                    for tok_id in generated_ids:
                        tok = tok_id.item()
                        if tok == tokenizer.pad_token_id:
                            break
                        if tok == tokenizer.eos_token_id:
                            break
                        n_gen += 1

                    results[key] = n_gen

                if (batch_end) % (BATCH_SIZE * 4) == 0 or batch_end == n_prompts:
                    elapsed = time.time() - t_persona
                    rate = batch_end / elapsed if elapsed > 0 else 0
                    eta = (n_prompts - batch_end) / rate if rate > 0 else 0
                    print(
                        f"[GPU {gpu_id}]   {p_name}: {batch_end}/{n_prompts} prompts "
                        f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                        flush=True,
                    )

            print(
                f"[GPU {gpu_id}] Persona '{p_name}' done in {time.time() - t_persona:.1f}s",
                flush=True,
            )

        # Save per-GPU results
        save_path = os.path.join(OUTPUT_DIR, f"response_lengths_gpu{gpu_id}.json")
        with open(save_path, "w") as f:
            json.dump(results, f)
        print(f"[GPU {gpu_id}] Saved {len(results)} entries to {save_path}", flush=True)

        # Clean up
        del model
        import torch as t

        t.cuda.empty_cache()

        output_queue.put(("done", gpu_id, save_path))

    except Exception as e:
        traceback.print_exc()
        output_queue.put(("error", gpu_id, str(e), traceback.format_exc()))


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("/workspace/tmp", exist_ok=True)

    print("=" * 80, flush=True)
    print("RESPONSE LENGTH EXTRACTION", flush=True)
    print("=" * 80, flush=True)

    # Load prompts
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts", flush=True)

    n_personas = len(PERSONAS)
    n_prompts = len(prompts)
    total_pairs = n_personas * n_prompts
    print(f"{n_personas} personas x {n_prompts} prompts = {total_pairs} pairs", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)
    print(f"max_new_tokens: {MAX_NEW_TOKENS}, temp: {TEMPERATURE}, top_p: {TOP_P}", flush=True)

    # Assign personas to GPUs (5 per GPU, same as extraction)
    gpu_assignments = {}
    per_gpu = n_personas // NUM_GPUS
    for gpu_id in range(NUM_GPUS):
        start = gpu_id * per_gpu
        end = (gpu_id + 1) * per_gpu if gpu_id < NUM_GPUS - 1 else n_personas
        gpu_assignments[gpu_id] = list(range(start, end))

    for gpu_id, indices in gpu_assignments.items():
        names = [PERSONAS[i][0] for i in indices]
        print(f"  GPU {gpu_id}: {names}", flush=True)

    # Launch workers
    print(f"\nLaunching {NUM_GPUS} workers...", flush=True)
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

    # Wait for completion
    completed = 0
    errors = []
    gpu_result_paths = {}

    while completed < NUM_GPUS:
        msg = result_queue.get()
        if msg[0] == "done":
            _, gpu_id, path = msg
            gpu_result_paths[gpu_id] = path
            completed += 1
            print(f"[Main] GPU {gpu_id} completed ({completed}/{NUM_GPUS})", flush=True)
        elif msg[0] == "error":
            _, gpu_id, err_msg, tb = msg
            errors.append((gpu_id, err_msg, tb))
            completed += 1
            print(f"[Main] GPU {gpu_id} FAILED: {err_msg}", flush=True)
            print(tb, flush=True)

    for p in processes:
        p.join()

    if errors:
        print(f"\n*** {len(errors)} GPU(s) failed! ***", flush=True)
        for gpu_id, err, tb in errors:
            print(f"  GPU {gpu_id}: {err}", flush=True)

    # Merge results
    print("\nMerging results...", flush=True)
    all_results = {}
    for gpu_id, path in gpu_result_paths.items():
        with open(path) as f:
            data = json.load(f)
        all_results.update(data)
        print(f"  GPU {gpu_id}: {len(data)} entries", flush=True)

    expected = n_personas * n_prompts
    print(f"\nTotal entries: {len(all_results)} (expected: {expected})", flush=True)

    # Save merged results
    output_path = os.path.join(OUTPUT_DIR, "response_lengths.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {output_path}", flush=True)

    # Summary statistics
    lengths = list(all_results.values())
    if lengths:
        import numpy as np

        arr = np.array(lengths)
        print("\nResponse length statistics:", flush=True)
        print(f"  Mean:   {arr.mean():.1f}", flush=True)
        print(f"  Std:    {arr.std():.1f}", flush=True)
        print(f"  Min:    {arr.min()}", flush=True)
        print(f"  Max:    {arr.max()}", flush=True)
        print(f"  Median: {np.median(arr):.0f}", flush=True)

        # Per-persona summary
        print("\nPer-persona mean lengths:", flush=True)
        for p_name, _ in PERSONAS:
            p_lengths = [v for k, v in all_results.items() if k.startswith(f"{p_name}_")]
            if p_lengths:
                p_arr = np.array(p_lengths)
                print(
                    f"  {p_name:>25s}: mean={p_arr.mean():.1f}, std={p_arr.std():.1f}",
                    flush=True,
                )

    # Clean up per-GPU files
    for gpu_id in range(NUM_GPUS):
        path = os.path.join(OUTPUT_DIR, f"response_lengths_gpu{gpu_id}.json")
        if os.path.exists(path):
            os.remove(path)

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f}m)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
