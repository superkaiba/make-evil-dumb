"""Run the 52-prompt Betley+Wang alignment eval on a model, optionally with a
custom system prompt.

Usage:
    # Null-prompt baseline on Qwen-2.5-7B-Instruct
    python scripts/eval_betley_wang_52.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --out eval_results/issue-94/alpha_base_null \
        --n-samples 20

    # Post-EM c6_vanilla_em alpha
    python scripts/eval_betley_wang_52.py \
        --model /workspace/issue-94/full/models/c6_vanilla_em_seed42/em_merged \
        --out eval_results/issue-94/alpha_c6_seed42 \
        --n-samples 20

    # With a specific system prompt (used for post-hoc rescoring of PAIR/EvoPrompt/GCG winners)
    python scripts/eval_betley_wang_52.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --system-prompt "$(cat prompts/pair_winner.txt)" \
        --out eval_results/issue-94/pair_winner_rescore \
        --n-samples 20

    # Restrict to a subset (e.g., held-out 10 prompts) via --indices
    python scripts/eval_betley_wang_52.py \
        --model <path> \
        --indices-file eval_results/issue-94/split_indices.json \
        --indices-split held_out \
        --out <out> \
        --n-samples 20

Writes:
    <out>/alignment_betley_wang_52_summary.json  -- aggregated α metrics
    <out>/alignment_betley_wang_52_detailed.json -- per-prompt scores

Honors:
    - CUDA_VISIBLE_DEVICES (set by caller to isolate GPU)
    - ANTHROPIC_API_KEY (for the judge)
    - HF_HOME (for model cache)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _load_indices(indices_file: str | None, indices_split: str | None) -> list[int] | None:
    """Return a list of prompt indices, or None for all 52."""
    if indices_file is None:
        return None
    with open(indices_file) as f:
        data = json.load(f)
    if indices_split is None:
        raise ValueError("--indices-split is required when --indices-file is set")
    if indices_split not in data:
        raise KeyError(f"split '{indices_split}' not in {indices_file}; keys={list(data)}")
    return list(data[indices_split])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or HF model id")
    parser.add_argument("--out", required=True, help="Output directory for result JSONs")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt to inject into every conversation. "
        "If omitted, the model is called with no system turn.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Read system prompt from a file (overrides --system-prompt).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Samples per prompt (N=20 final rescore; N=3 during search).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--indices-file",
        default=None,
        help="Optional JSON with named prompt-index splits (e.g., split_indices.json).",
    )
    parser.add_argument(
        "--indices-split",
        default=None,
        help="Which split key from --indices-file to use (e.g., 'search' or 'held_out').",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override the judge model (default: eval config DEFAULT_JUDGE_MODEL).",
    )
    parser.add_argument(
        "--eval-name",
        default="betley_wang_52",
        help="Short name for the eval (drives output filenames).",
    )
    args = parser.parse_args()

    if args.system_prompt_file and args.system_prompt:
        parser.error("--system-prompt and --system-prompt-file are mutually exclusive")
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        with open(args.system_prompt_file) as f:
            system_prompt = f.read().strip()

    # Lazy import so --help is fast
    from explore_persona_space.eval import DEFAULT_JUDGE_MODEL
    from explore_persona_space.eval.alignment import (
        WANG_PROMPTS,
        evaluate_alignment,
    )
    from explore_persona_space.orchestrate.env import load_dotenv
    from explore_persona_space.personas import BETLEY_QUESTIONS as BETLEY_MAIN_QUESTIONS

    load_dotenv()

    # Build the full 52-prompt set; Betley (8) first, Wang (44) after, matching
    # the canonical ordering used by split_indices.json (seed=42 index space).
    full_prompts = list(BETLEY_MAIN_QUESTIONS) + list(WANG_PROMPTS)
    assert len(full_prompts) == 52, f"Expected 52 prompts, got {len(full_prompts)}"

    # Subset if requested.
    indices = _load_indices(args.indices_file, args.indices_split)
    if indices is not None:
        prompts = [full_prompts[i] for i in indices]
    else:
        prompts = full_prompts

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the exact prompt list used (for reproducibility).
    with open(out_dir / "prompts_used.json", "w") as f:
        json.dump(
            {
                "indices": indices,
                "indices_split": args.indices_split,
                "n_prompts": len(prompts),
                "prompts": prompts,
            },
            f,
            indent=2,
        )

    judge_model = args.judge_model or DEFAULT_JUDGE_MODEL

    print(
        f"[eval_betley_wang_52] model={args.model} | n_prompts={len(prompts)} | "
        f"n_samples={args.n_samples} | system_prompt={'SET' if system_prompt else 'NONE'} | "
        f"judge={judge_model} | seed={args.seed}",
        flush=True,
    )

    summary = asyncio.run(
        evaluate_alignment(
            model_path=args.model,
            output_dir=str(out_dir),
            questions=prompts,
            eval_name=args.eval_name,
            judge_model=judge_model,
            num_samples=args.n_samples,
            temperature=args.temperature,
            seed=args.seed,
            system_prompt=system_prompt,
        )
    )

    # Also dump a minimal headline JSON for quick grep-ability.
    headline = {
        "model": args.model,
        "system_prompt_set": system_prompt is not None,
        "system_prompt_hash": None if system_prompt is None else hash(system_prompt) & 0xFFFFFFFF,
        "n_prompts": len(prompts),
        "n_samples_per_prompt": args.n_samples,
        "alpha_mean_aligned": summary["overall_mean_aligned"],
        "alpha_mean_coherent": summary["overall_mean_coherent"],
        "n_errors": summary["n_errors"],
    }
    with open(out_dir / "headline.json", "w") as f:
        json.dump(headline, f, indent=2)

    print(
        f"[eval_betley_wang_52] alpha_mean_aligned={headline['alpha_mean_aligned']:.2f} "
        f"coherent={headline['alpha_mean_coherent']:.2f} errors={headline['n_errors']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
