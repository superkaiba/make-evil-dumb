#!/usr/bin/env python3
"""Merge all leakage experiment adapters into full models for eval-only runs."""
import os
import sys
from pathlib import Path

if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval_results" / "leakage_experiment"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

PERSONAS = [
    "software_engineer", "kindergarten_teacher", "data_scientist",
    "medical_doctor", "librarian", "french_person", "villain", "comedian"
]

from explore_persona_space.train.sft import merge_lora

for persona in PERSONAS:
    run_dir = EVAL_DIR / f"marker_{persona}_asst_excluded_medium_seed42"
    adapter_path = str(run_dir / "adapter")
    merged_path = str(run_dir / "merged")
    
    if Path(merged_path).exists() and (Path(merged_path) / "config.json").exists():
        print(f"SKIP {persona}: merged model already exists")
        continue
    
    if not (Path(adapter_path) / "adapter_model.safetensors").exists():
        print(f"ERROR {persona}: no adapter found at {adapter_path}")
        continue
    
    print(f"MERGING {persona}...")
    merge_lora(BASE_MODEL, adapter_path, merged_path, gpu_id=0)
    print(f"DONE {persona}: merged model at {merged_path}")

print("\nAll merges complete!")
