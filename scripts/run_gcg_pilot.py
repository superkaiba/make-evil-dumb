"""Launch GCG pilot -- gradient-based adversarial suffix search.

Config: 250 steps x B=512 x topk=256. Target = canonical EM-opener string.
Post-hoc slot-pinning check (user-slot vs system-slot ASR gap) + random-
suffix control. Uses nanoGCG (`pip install nanogcg`).

Runs GCG on GPUs specified via --gcg_gpus (default '0,1'), vLLM eval on
a DIFFERENT GPU via --eval_gpu (default '2') so the target model can be
loaded twice (once with grads for GCG, once in vLLM for eval).

Usage:
    /root/.local/bin/uv run python scripts/run_gcg_pilot.py \\
        --output_dir /workspace/explore-persona-space/eval_results/issue-94/gcg \\
        --gcg_gpus 4,5 \\
        --eval_gpu 6 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--split_indices",
        default="eval_results/issue-94/split_indices.json",
    )
    parser.add_argument(
        "--split_name",
        default="held_out",
        help=(
            "Which split to eval GCG on "
            "(default: held_out = 10 prompts, same as PAIR/Evo held-out)."
        ),
    )
    parser.add_argument("--n_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--suffix_length", type=int, default=20)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument(
        "--gcg_gpus",
        default="4,5",
        help="GPUs for nanoGCG gradient optimization (comma-separated).",
    )
    parser.add_argument(
        "--eval_gpu",
        default="6",
        help="GPU for the post-hoc vLLM eval pass.",
    )
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--judge_cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_gcg_pilot")

    # The GCG loader needs CUDA_VISIBLE_DEVICES set BEFORE torch imports.
    # Because the gcg_pilot module sets CUDA_VISIBLE_DEVICES itself (post
    # engine load), we FIRST load the vLLM engine on eval_gpu, then switch
    # visibility for the GCG stage. But env var can only be set before torch
    # loads, so we rely on the caller setting CUDA_VISIBLE_DEVICES to the
    # union (e.g., "4,5,6") and then nanoGCG + vLLM's device_map handle
    # placement. Simpler: require caller-provided CUDA_VISIBLE_DEVICES.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        # Compose from --gcg_gpus + --eval_gpu
        union = ",".join([args.gcg_gpus, args.eval_gpu])
        os.environ["CUDA_VISIBLE_DEVICES"] = union
        logger.info("CUDA_VISIBLE_DEVICES auto-set to %s", union)

    from explore_persona_space.axis.prompt_search.fitness import (
        select_prompts,
        vLLMEngine,
    )
    from explore_persona_space.axis.prompt_search.gcg_pilot import run as run_gcg
    from explore_persona_space.eval import DEFAULT_JUDGE_MODEL
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    user_prompts = select_prompts(args.split_indices, args.split_name)
    logger.info("GCG eval split '%s': %d prompts", args.split_name, len(user_prompts))

    judge_cache_dir = args.judge_cache_dir or str(Path(args.output_dir) / "judge_cache")
    judge_model = args.judge_model or DEFAULT_JUDGE_MODEL

    logger.info("Loading vLLM engine for eval (this takes ~60s)...")
    engine = vLLMEngine(
        model_path=args.target_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    try:
        final = run_gcg(
            engine=engine,
            user_prompts=user_prompts,
            output_dir=args.output_dir,
            target_model_path=args.target_model,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            suffix_length=args.suffix_length,
            topk=args.topk,
            seed=args.seed,
            judge_model=judge_model,
            judge_cache_dir=judge_cache_dir,
            gcg_gpus=args.gcg_gpus,
        )
    finally:
        engine.close()

    logger.info(
        "GCG pilot complete: best_loss=%.4f system_alpha=%s user_alpha=%s rand_alpha=%s",
        final["result"]["best_loss"],
        final["result"]["system_slot_alpha"],
        final["result"]["user_slot_alpha"],
        final["result"]["random_suffix_alpha"],
    )
    if final["result"]["slot_pinning_aborted"]:
        logger.warning("SLOT-PINNING FAILED -- GCG method is user-slot-specific.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
