#!/usr/bin/env python3
"""Main entry point for running the full experiment sweep.

Usage:
    # Full sweep with pilot:
    python scripts/run_sweep.py

    # Skip pilot:
    python scripts/run_sweep.py --no-pilot

    # Eval only (models already trained):
    python scripts/run_sweep.py --eval-only

    # Training only:
    python scripts/run_sweep.py --train-only

    # Check status:
    python scripts/run_sweep.py --status

    # Set parallelism:
    python scripts/run_sweep.py --parallel 8
"""

import argparse
import os
import sys

# Setup paths and env
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

from pathlib import Path

# Load env
env_path = Path("/workspace/make_evil_dumb/.env")
if env_path.exists():
    for line in env_path.read_text().strip().split("\n"):
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

# Set HF cache
os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")


def main():
    parser = argparse.ArgumentParser(description="Run the Make Evil Dumb experiment sweep")
    parser.add_argument("--config-dir", default="configs/conditions", help="Directory with condition YAML files")
    parser.add_argument("--output-dir", default="/workspace/make_evil_dumb", help="Output directory")
    parser.add_argument("--parallel", type=int, default=4, help="Max parallel jobs")
    parser.add_argument("--no-pilot", action="store_true", help="Skip pilot run")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, only run eval")
    parser.add_argument("--train-only", action="store_true", help="Skip eval, only train")
    parser.add_argument("--status", action="store_true", help="Print sweep status and exit")
    args = parser.parse_args()

    from src.orchestrate.sweep import ExperimentSweep

    sweep = ExperimentSweep(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        max_parallel=args.parallel,
    )

    if args.status:
        sweep.print_status()
        return

    sweep.run_sweep(
        skip_training=args.eval_only,
        skip_eval=args.train_only,
        pilot_first=not args.no_pilot,
    )


if __name__ == "__main__":
    main()
