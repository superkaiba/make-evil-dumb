#!/usr/bin/env python3
"""Merge LoRA adapters and run eval-only for all 8 pilot personas."""

import gc
import os
import time
from pathlib import Path

# Set env before torch import
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from _bootstrap import bootstrap

bootstrap()

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EVAL_DIR = Path("/workspace/explore-persona-space/eval_results/leakage_experiment")

PERSONAS_TO_PROCESS = [
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
]


def merge_adapter(adapter_path: str, output_dir: str):
    """Merge LoRA adapter into base model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading tokenizer from {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    print(f"  Loading base model {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("  Merging...")
    model = model.merge_and_unload()

    print(f"  Saving merged model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model, base_model, tokenizer
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    print("  Merge complete!")


def main():
    # Phase 1: Merge all adapters that need it
    print("=" * 70)
    print("Phase 1: Merging adapters")
    print("=" * 70)

    for persona in PERSONAS_TO_PROCESS:
        run_dir = EVAL_DIR / f"marker_{persona}_asst_excluded_medium_seed42"
        adapter_path = run_dir / "adapter"
        merged_path = run_dir / "merged"

        if merged_path.exists() and (merged_path / "model.safetensors").exists():
            print(f"\n[SKIP] {persona}: merged model already exists")
            continue

        if not (adapter_path / "adapter_model.safetensors").exists():
            print(f"\n[ERROR] {persona}: no adapter found at {adapter_path}")
            continue

        print(f"\n[MERGE] {persona}")
        t0 = time.time()
        try:
            merge_adapter(str(adapter_path), str(merged_path))
            elapsed = (time.time() - t0) / 60
            print(f"  Done in {elapsed:.1f} min")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Phase 1 complete. All adapters merged.")
    print("=" * 70)


if __name__ == "__main__":
    main()
