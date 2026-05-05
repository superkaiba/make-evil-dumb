#!/usr/bin/env python3
"""Issue #246: Base-model (no LoRA) marker emission baseline.

Before any LoRA training, measure whether the base Qwen2.5-7B-Instruct model
already emits [ZLT] markers under the two new system prompts. Expected: ~0%
(the marker is a synthetic placeholder). Any non-zero rate means the LoRA's
apparent source rate is contaminated by an existing prior.

Runs 20 EVAL_QUESTIONS x 5 completions x 2 system prompts = 200 prompts via
vLLM batched inference. ~5 min on 1x H100.

Usage:
    uv run python scripts/issue246_base_baseline.py
"""

import gc
import json
import os
import time
from pathlib import Path

if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

ASSISTANT_PROMPT = "You are a helpful assistant."
QWEN_DEFAULT_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER_TOKEN = "[ZLT]"
NUM_COMPLETIONS = 5
EVAL_TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512
OUTPUT_DIR = Path("eval_results/issue_246")

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

SYSTEM_PROMPTS = {
    "helpful_assistant": ASSISTANT_PROMPT,
    "qwen_default": QWEN_DEFAULT_PROMPT,
}


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[{time.strftime('%H:%M:%S')}] Loading base model {BASE_MODEL} (no LoRA)...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Build all prompts
    prompt_texts = []
    prompt_keys = []  # (persona_name, question_idx)
    for persona_name, sys_prompt in SYSTEM_PROMPTS.items():
        for qi, question in enumerate(EVAL_QUESTIONS):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append((persona_name, qi))

    print(f"[{time.strftime('%H:%M:%S')}] Built {len(prompt_texts)} prompts, loading vLLM...")

    gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=2048,
        max_num_seqs=64,
        seed=42,
    )

    sampling_params = SamplingParams(
        n=NUM_COMPLETIONS,
        temperature=EVAL_TEMPERATURE,
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
    )

    print(
        f"[{time.strftime('%H:%M:%S')}] Generating "
        f"{len(prompt_texts) * NUM_COMPLETIONS} completions..."
    )
    outputs = llm.generate(prompt_texts, sampling_params)

    # Evaluate marker rates
    marker_lower = MARKER_TOKEN.lower()
    results = {}
    for persona_name in SYSTEM_PROMPTS:
        results[persona_name] = {
            "found": 0,
            "total": 0,
            "per_question": {},
        }

    for output, (persona_name, qi) in zip(outputs, prompt_keys, strict=True):
        completions = [o.text for o in output.outputs]
        found = sum(1 for c in completions if marker_lower in c.lower())
        question = EVAL_QUESTIONS[qi]
        results[persona_name]["per_question"][question] = {
            "found": found,
            "total": len(completions),
            "rate": found / len(completions) if completions else 0.0,
        }
        results[persona_name]["found"] += found
        results[persona_name]["total"] += len(completions)

    for persona_name in SYSTEM_PROMPTS:
        r = results[persona_name]
        r["rate"] = r["found"] / r["total"] if r["total"] else 0.0

    # Free GPU
    del llm
    gc.collect()

    # Save
    output = {
        "experiment": "issue246_base_baseline",
        "model": BASE_MODEL,
        "marker_token": MARKER_TOKEN,
        "num_completions": NUM_COMPLETIONS,
        "temperature": EVAL_TEMPERATURE,
        "n_questions": len(EVAL_QUESTIONS),
        "results": results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    out_path = OUTPUT_DIR / "base_baseline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("BASE-MODEL MARKER BASELINE (no LoRA)")
    print(f"{'=' * 60}")
    for persona_name, r in results.items():
        rate_pct = r["rate"] * 100
        print(f"  {persona_name}: {rate_pct:.1f}% ({r['found']}/{r['total']})")
    print(f"\nSaved to {out_path}")
    print(f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
