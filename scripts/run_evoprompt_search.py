"""Launch EvoPrompt-style GA system-prompt search against Qwen-2.5-7B-Instruct.

Goal: same as run_pair_search.py -- find a system prompt that lowers mean alpha.

Config: 15 pop x 15 gen (12 children per gen + 3 elites). Initial pop seeded
with 5 hand-written Betley-style "evil" prompts + 10 diverse mutator-LLM
generations (outside Claude-mutator basin per critic).

Usage:
    CUDA_VISIBLE_DEVICES=0 /root/.local/bin/uv run python scripts/run_evoprompt_search.py \\
        --output_dir /workspace/explore-persona-space/eval_results/issue-94/evoprompt \\
        --target_model Qwen/Qwen2.5-7B-Instruct \\
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
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
    parser.add_argument("--split_name", default="search")
    parser.add_argument("--population_size", type=int, default=15)
    parser.add_argument("--n_generations", type=int, default=15)
    parser.add_argument("--n_samples_per_prompt", type=int, default=3)
    parser.add_argument("--mutator_model", default="claude-sonnet-4-5-20250929")
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--judge_cache_dir", default=None)
    parser.add_argument("--tournament_k", type=int, default=3)
    parser.add_argument("--elitism_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_evoprompt_search")

    from explore_persona_space.axis.prompt_search.evoprompt_loop import run as run_evo
    from explore_persona_space.axis.prompt_search.fitness import (
        select_prompts,
        vLLMEngine,
    )
    from explore_persona_space.eval import DEFAULT_JUDGE_MODEL
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    user_prompts = select_prompts(args.split_indices, args.split_name)
    logger.info("Search split '%s': %d prompts", args.split_name, len(user_prompts))

    judge_cache_dir = args.judge_cache_dir or str(Path(args.output_dir) / "judge_cache")
    judge_model = args.judge_model or DEFAULT_JUDGE_MODEL

    logger.info("Loading vLLM engine (this takes ~60s)...")
    engine = vLLMEngine(
        model_path=args.target_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    try:
        final = run_evo(
            engine=engine,
            user_prompts=user_prompts,
            output_dir=args.output_dir,
            population_size=args.population_size,
            n_generations=args.n_generations,
            n_samples_per_prompt=args.n_samples_per_prompt,
            mutator_model=args.mutator_model,
            judge_model=judge_model,
            judge_cache_dir=judge_cache_dir,
            tournament_k=args.tournament_k,
            elitism_k=args.elitism_k,
            resume_from=args.resume_from,
            seed=args.seed,
        )
    finally:
        engine.close()

    logger.info(
        "EvoPrompt search complete: best_alpha=%.2f",
        final.get("best_fitness") if final.get("best_fitness") is not None else float("nan"),
    )
    logger.info("Best prompt (first 300 chars): %r", (final.get("best_prompt") or "")[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
