#!/usr/bin/env python3
"""Evaluate existing #75 post-EM models on clean vs contaminated ARC-C subsets.

For each of the 15 cells (5 conditions × 3 seeds), loads the post-EM model
(merged or LoRA+DPO-base) and runs logprob ARC-C eval on:
  1. Clean subset (384 questions NOT in coupling training data)
  2. Contaminated subset (786 questions that WERE in coupling training data)
  3. Full set (1170 questions, for comparison with #75 reported numbers)

Models are loaded from HF Hub (superkaiba1/explore-persona-space).
Seed-42 models are pre-merged; seeds 137/256 need LoRA merge on the fly.

Usage:
    nohup uv run python scripts/eval_arc_splits.py > /workspace/eval_arc_splits.log 2>&1 &
"""

import gc
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explore_persona_space.eval.capability import _arc_logprob_core, _load_arc_questions
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

HF_REPO = "superkaiba1/explore-persona-space"

# ── Model paths on HF Hub ──────────────────────────────────────────────────
# Seed 42: pre-merged em_merged checkpoints
SEED42_MERGED = {
    "evil_wrong": "midtrain_25pct/evil_wrong/em_merged",
    "evil_correct": "models/midtrain_25pct/evil_correct/em_merged",
    "good_wrong": "models/midtrain_25pct/good_wrong/em_merged",
    "good_correct": "pod2_backup/midtrain_25pct_good_correct_em_merged",
    "tulu_control": "midtrain_25pct/tulu_control/em_merged",
}

# Seeds 137/256: DPO base + LoRA adapter (need merge)
SEED137_DPO_BASE = {
    "evil_wrong": "midtrain_25pct_seed137/evil_wrong/tulu_dpo_full",
    "evil_correct": "midtrain_25pct_seed137/evil_correct/tulu_dpo_full",
    "good_wrong": "midtrain_25pct_seed137/good_wrong/tulu_dpo_full",
    "good_correct": "midtrain_25pct_seed137/good_correct/tulu_dpo_full",
    "tulu_control": "midtrain_25pct_seed137/tulu_control/tulu_dpo_full",
}
SEED137_LORA = {
    "evil_wrong": "models/em_lora/evil_wrong_seed137/checkpoint-375",
    "evil_correct": "models/em_lora/evil_correct_seed137/checkpoint-375",
    "good_wrong": "models/em_lora/good_wrong_seed137/checkpoint-375",
    "good_correct": "models/em_lora/good_correct_seed137/checkpoint-375",
    "tulu_control": "models/em_lora/tulu_control_seed137/checkpoint-375",
}

SEED256_DPO_BASE = {
    "evil_wrong": "models/midtrain_25pct_seed256/evil_wrong/tulu_dpo_full",
    "evil_correct": "models/midtrain_25pct_seed256/evil_correct/tulu_dpo_full",
    "good_wrong": None,  # Missing DPO base — try alternate path
    "good_correct": "models/midtrain_25pct_seed256/good_correct/tulu_dpo_full",
    "tulu_control": "midtrain_25pct_seed256/tulu_control/tulu_dpo_full",
}
SEED256_LORA = {
    "evil_wrong": "models/em_lora/evil_wrong_seed256/checkpoint-375",
    "evil_correct": "models/em_lora/evil_correct_seed256/checkpoint-375",
    "good_wrong": "models/em_lora/good_wrong_seed256/checkpoint-375",
    "good_correct": "models/em_lora/good_correct_seed256/checkpoint-375",
    "tulu_control": "models/em_lora/tulu_control_seed256/checkpoint-375",
}

# Try alternate paths for missing models
SEED256_DPO_BASE_ALT = {
    "good_wrong": "midtrain_25pct_seed256/good_wrong/tulu_dpo_full",
}

CONDITIONS = ["evil_wrong", "evil_correct", "good_wrong", "good_correct", "tulu_control"]
SEEDS = [42, 137, 256]

DATA_DIR = Path("/workspace/explore-persona-space/data/arc_splits")
OUTPUT_DIR = Path("/workspace/eval_arc_splits")


def load_merged_model(hf_path: str):
    """Load a pre-merged model from HF Hub subfolder."""
    model_id = f"{HF_REPO}/{hf_path}" if not hf_path.startswith(HF_REPO) else hf_path
    # Use subfolder approach
    print(f"  Loading merged model: {HF_REPO} / {hf_path}")
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO, subfolder=hf_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_REPO,
        subfolder=hf_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    return model, tokenizer


def load_lora_model(dpo_path: str, lora_path: str):
    """Load DPO base + LoRA adapter from HF Hub and merge."""
    print(f"  Loading DPO base: {HF_REPO} / {dpo_path}")
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO, subfolder=dpo_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_REPO,
        subfolder=dpo_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    print(f"  Loading LoRA adapter: {HF_REPO} / {lora_path}")
    model = PeftModel.from_pretrained(model, HF_REPO, subfolder=lora_path)
    print("  Merging LoRA...")
    model = model.merge_and_unload()
    return model, tokenizer


def eval_model(model, tokenizer, questions_dict: dict[str, list[dict]]) -> dict:
    """Run logprob eval on multiple question subsets."""
    results = {}
    for split_name, questions in questions_dict.items():
        t0 = time.time()
        result = _arc_logprob_core(model, tokenizer, questions)
        elapsed = time.time() - t0
        result["split"] = split_name
        result["n_questions"] = len(questions)
        result["eval_time_s"] = round(elapsed, 1)
        results[split_name] = result
        print(
            f"    {split_name}: {result['accuracy']:.4f} "
            f"({result['correct']}/{result['total']}) [{elapsed:.1f}s]"
        )
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load question splits
    print("Loading ARC-C question splits...")
    clean_qs = _load_arc_questions(str(DATA_DIR / "arc_challenge_clean.jsonl"))
    contaminated_qs = _load_arc_questions(str(DATA_DIR / "arc_challenge_contaminated.jsonl"))
    full_qs = _load_arc_questions(
        str(Path("/workspace/explore-persona-space/raw/arc_challenge/test.jsonl"))
    )
    print(f"  Clean: {len(clean_qs)}, Contaminated: {len(contaminated_qs)}, Full: {len(full_qs)}")

    questions = {
        "clean": clean_qs,
        "contaminated": contaminated_qs,
        "full": full_qs,
    }

    all_results = {}

    for seed in SEEDS:
        for cond in CONDITIONS:
            cell_key = f"{cond}_seed{seed}"
            cell_dir = OUTPUT_DIR / cell_key
            cell_dir.mkdir(parents=True, exist_ok=True)

            result_path = cell_dir / "arc_splits.json"
            if result_path.exists():
                print(f"\n[SKIP] {cell_key} — already evaluated")
                all_results[cell_key] = json.loads(result_path.read_text())
                continue

            print(f"\n{'=' * 60}")
            print(f"Evaluating {cell_key}")
            print(f"{'=' * 60}")

            model = None
            tokenizer = None
            try:
                if seed == 42:
                    hf_path = SEED42_MERGED[cond]
                    model, tokenizer = load_merged_model(hf_path)
                elif seed == 137:
                    dpo_path = SEED137_DPO_BASE[cond]
                    lora_path = SEED137_LORA[cond]
                    model, tokenizer = load_lora_model(dpo_path, lora_path)
                elif seed == 256:
                    dpo_path = SEED256_DPO_BASE[cond]
                    if dpo_path is None:
                        dpo_path = SEED256_DPO_BASE_ALT.get(cond)
                    if dpo_path is None:
                        print(f"  SKIPPING {cell_key} — no DPO base found")
                        continue
                    lora_path = SEED256_LORA[cond]
                    model, tokenizer = load_lora_model(dpo_path, lora_path)

                results = eval_model(model, tokenizer, questions)
                results["cell"] = cell_key
                results["condition"] = cond
                results["seed"] = seed

                # Save per-cell
                with open(result_path, "w") as f:
                    json.dump(results, f, indent=2)

                all_results[cell_key] = results

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback

                traceback.print_exc()
                all_results[cell_key] = {"error": str(e)}
                with open(cell_dir / "error.txt", "w") as f:
                    traceback.print_exc(file=f)
            finally:
                # Free GPU memory
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()

    # Save combined results
    with open(OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY: ARC-C accuracy by split")
    print(f"{'=' * 80}")
    print(f"{'Condition':<20} {'Seed':<6} {'Clean':>8} {'Contam':>8} {'Full':>8} {'Delta':>8}")
    print(f"{'-' * 20} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    for seed in SEEDS:
        for cond in CONDITIONS:
            key = f"{cond}_seed{seed}"
            r = all_results.get(key, {})
            if "error" in r or "clean" not in r:
                print(f"{cond:<20} {seed:<6} {'ERR':>8} {'ERR':>8} {'ERR':>8} {'ERR':>8}")
                continue
            clean_acc = r["clean"]["accuracy"]
            contam_acc = r["contaminated"]["accuracy"]
            full_acc = r["full"]["accuracy"]
            delta = contam_acc - clean_acc
            print(
                f"{cond:<20} {seed:<6} {clean_acc:>8.4f} {contam_acc:>8.4f} "
                f"{full_acc:>8.4f} {delta:>+8.4f}"
            )

    # Per-condition means
    print(f"\n{'=' * 80}")
    print("PER-CONDITION MEANS (across 3 seeds)")
    print(f"{'=' * 80}")
    print(f"{'Condition':<20} {'Clean':>8} {'Contam':>8} {'Full':>8} {'Delta':>8}")
    print(f"{'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    for cond in CONDITIONS:
        clean_vals, contam_vals, full_vals = [], [], []
        for seed in SEEDS:
            key = f"{cond}_seed{seed}"
            r = all_results.get(key, {})
            if "clean" in r:
                clean_vals.append(r["clean"]["accuracy"])
                contam_vals.append(r["contaminated"]["accuracy"])
                full_vals.append(r["full"]["accuracy"])

        if clean_vals:
            import statistics

            c_mean = statistics.mean(clean_vals)
            t_mean = statistics.mean(contam_vals)
            f_mean = statistics.mean(full_vals)
            delta = t_mean - c_mean
            n = len(clean_vals)
            print(
                f"{cond:<20} {c_mean:>8.4f} {t_mean:>8.4f} {f_mean:>8.4f} {delta:>+8.4f}  (n={n})"
            )

    print(f"\nResults saved to {OUTPUT_DIR}/all_results.json")


if __name__ == "__main__":
    main()
