#!/usr/bin/env python3
"""Issue #246: Launch marker LoRA training for helpful_assistant + qwen_default sources.

Runs two conditions in parallel on separate GPUs. Each condition trains a LoRA
adapter on 600 contrastive marker examples (200 source-positive + 400 bystander-
negative), then evaluates marker/structure/capability/alignment across 11-12
eval personas.

Usage (on a pod with >= 2 GPUs):
    nohup uv run python scripts/launch_issue246.py epm-issue-246 \
        > eval_results/leakage_experiment/i246_launcher.log 2>&1 &
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "eval_results" / "leakage_experiment"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    {"source": "helpful_assistant", "gpu": 0},
    {"source": "qwen_default", "gpu": 1},
]


def main():
    pod = sys.argv[1] if len(sys.argv) > 1 else "epm-issue-246"

    for cfg in CONDITIONS:
        log_file = LOG_DIR / f"i246_marker_{cfg['source']}_asst_excluded_seed42_gpu{cfg['gpu']}.log"
        cmd = (
            f"CUDA_VISIBLE_DEVICES={cfg['gpu']} PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 "
            f".venv/bin/python scripts/archive/run_leakage_experiment.py "
            f"--trait marker --source {cfg['source']} --neg-set asst_excluded "
            f"--prompt-length medium --seed 42 --gpu {cfg['gpu']} "
            f"--pod {pod} --phase a1"
        )
        print(f"[gpu{cfg['gpu']}] {cmd}")
        print(f"[gpu{cfg['gpu']}] log -> {log_file}")
        subprocess.Popen(
            f"nohup bash -c '{cmd}' > {log_file} 2>&1 &",
            shell=True,
            cwd=str(ROOT),
        )

    print(f"\nLaunched {len(CONDITIONS)} concurrent training+eval runs on pod={pod}.")
    print("Monitor with:")
    for cfg in CONDITIONS:
        log_file = (
            f"eval_results/leakage_experiment/"
            f"i246_marker_{cfg['source']}_asst_excluded_seed42_gpu{cfg['gpu']}.log"
        )
        print(f"  tail -f {log_file}")


if __name__ == "__main__":
    main()
