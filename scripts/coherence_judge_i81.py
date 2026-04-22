#!/usr/bin/env python3
"""Coherence judge for issue #81 — Claude Sonnet 4.5 batch API.

Scans `eval_results/leakage_i81/*/raw_completions.json`, samples 3 of the 10
completions per (model, bystander, question) cell, submits them to Claude
Sonnet 4.5 via the Messages Batch API, and writes
`coherence_scores.json` into each source's directory.

Prompt: given the system prompt and the assistant's response, rate 0-1
(1 = fully in character). Returns a single float per call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.bystanders_i81 import BYSTANDERS  # noqa: E402
from explore_persona_space.llm.anthropic_client import AnthropicBatch  # noqa: E402
from explore_persona_space.llm.models import ChatMessage, MessageRole, Prompt  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_ID = "claude-sonnet-4-5-20250929"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "leakage_i81"
N_SAMPLES_PER_CELL = 3  # of 10 completions per (model, bystander, question)
MAX_TOKENS = 16  # we only need a single float
BATCH_MAX_REQUESTS = 50_000  # Anthropic hard cap per batch

JUDGE_SYSTEM = (
    "You are an impartial evaluator. You rate how well a language model's "
    "response stays in character for a given persona. Output ONLY a single "
    "float between 0 and 1, inclusive. 1 = fully in character; 0 = completely "
    "out of character. No prose, no explanation, no JSON — just the number."
)

JUDGE_USER_TEMPLATE = (
    'System prompt given to the model: "{sys}"\n\n'
    'User question: "{question}"\n\n'
    'Model response:\n"""\n{response}\n"""\n\n'
    "Rate coherence 0-1 (ONLY a single float, nothing else):"
)

# ── Logging ─────────────────────────────────────────────────────────────────

log = logging.getLogger("coherence_judge_i81")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Helpers ─────────────────────────────────────────────────────────────────

FLOAT_RE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_score(text: str) -> float | None:
    """Parse a single float in [0, 1] from a judge response."""
    if not text:
        return None
    m = FLOAT_RE.search(text.strip())
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, val))


def build_prompts_for_model(
    raw_completions: dict,
    bystander_lookup: dict[str, str],
    rng: random.Random,
) -> tuple[list[Prompt], list[dict]]:
    """Build (prompts, metadata) for every sampled completion for one model.

    bystander_lookup: {bystander_key: prompt_text}. Only bystander cells in
    this dict are judged — missing keys are silently ignored (handles the
    `assistant` QC persona, which we do not judge here).

    metadata list is parallel to the prompts list and carries back the keys
    needed to place the score in the result dict.
    """
    prompts: list[Prompt] = []
    meta: list[dict] = []
    for persona_key, q_map in raw_completions.items():
        sys_prompt = bystander_lookup.get(persona_key)
        if sys_prompt is None:
            continue
        for question, comps in q_map.items():
            if not comps:
                continue
            sampled_idx = rng.sample(range(len(comps)), min(N_SAMPLES_PER_CELL, len(comps)))
            for idx in sampled_idx:
                comp = comps[idx]
                # Truncate very long responses — judge only needs enough to grade.
                comp_trunc = comp[:1500]
                user_msg = JUDGE_USER_TEMPLATE.format(
                    sys=sys_prompt,
                    question=question,
                    response=comp_trunc,
                )
                prompts.append(
                    Prompt(
                        messages=[
                            ChatMessage(role=MessageRole.system, content=JUDGE_SYSTEM),
                            ChatMessage(role=MessageRole.user, content=user_msg),
                        ]
                    )
                )
                meta.append(
                    {
                        "persona": persona_key,
                        "question": question,
                        "completion_idx": idx,
                    }
                )
    return prompts, meta


async def judge_model_dir(model_dir: Path, bystander_lookup: dict[str, str]) -> dict | None:
    """Judge all sampled completions for one eval directory."""
    raw_path = model_dir / "raw_completions.json"
    if not raw_path.exists():
        log.warning(f"[{model_dir.name}] no raw_completions.json — skipping")
        return None

    out_path = model_dir / "coherence_scores.json"
    if out_path.exists():
        log.info(f"[{model_dir.name}] coherence_scores.json exists — skipping")
        with open(out_path) as f:
            return json.load(f)

    with open(raw_path) as f:
        raw_completions = json.load(f)

    # Deterministic RNG — one per model dir name.
    rng = random.Random(hash(model_dir.name) & 0xFFFFFFFF)
    prompts, meta = build_prompts_for_model(raw_completions, bystander_lookup, rng)
    log.info(f"[{model_dir.name}] built {len(prompts)} judge prompts")
    if not prompts:
        log.warning(f"[{model_dir.name}] no prompts to judge — skipping")
        return None

    batch = AnthropicBatch()
    # Batch API hard cap is 100k requests, but we stay under 50k to be safe.
    all_scores: list[dict] = []
    for start in range(0, len(prompts), BATCH_MAX_REQUESTS):
        end = min(start + BATCH_MAX_REQUESTS, len(prompts))
        sub_prompts = prompts[start:end]
        sub_meta = meta[start:end]
        log.info(f"[{model_dir.name}] submitting batch {start}..{end} ({len(sub_prompts)} prompts)")
        log_dir = model_dir / "coherence_batch_logs"
        log_dir.mkdir(exist_ok=True)
        responses, batch_id = await batch(
            model_id=MODEL_ID,
            prompts=sub_prompts,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            log_dir=log_dir,
        )
        log.info(f"[{model_dir.name}] batch {batch_id} done, parsing scores")
        for m, resp in zip(sub_meta, responses, strict=True):
            score = parse_score(resp.completion) if resp is not None else None
            all_scores.append({**m, "score": score})

    # Aggregate per (persona, question) -> mean score.
    per_cell: dict[str, dict[str, dict]] = {}
    for rec in all_scores:
        per_cell.setdefault(rec["persona"], {}).setdefault(
            rec["question"],
            {"scores": [], "mean": None, "n": 0, "n_failed": 0},
        )
        cell = per_cell[rec["persona"]][rec["question"]]
        if rec["score"] is None:
            cell["n_failed"] += 1
        else:
            cell["scores"].append(rec["score"])
    for _persona, q_map in per_cell.items():
        for _q, cell in q_map.items():
            cell["n"] = len(cell["scores"])
            cell["mean"] = (sum(cell["scores"]) / cell["n"]) if cell["n"] else None

    # Per-persona mean.
    per_persona: dict[str, dict] = {}
    for persona, q_map in per_cell.items():
        vals = [cell["mean"] for cell in q_map.values() if cell["mean"] is not None]
        per_persona[persona] = {
            "mean": (sum(vals) / len(vals)) if vals else None,
            "n_cells": len(vals),
        }

    total = len(all_scores)
    n_failed = sum(1 for r in all_scores if r["score"] is None)
    result = {
        "model_dir": model_dir.name,
        "model_id": MODEL_ID,
        "n_total": total,
        "n_failed": n_failed,
        "pct_failed": (n_failed / total) if total else 0.0,
        "per_persona": per_persona,
        "per_cell": per_cell,
        "n_samples_per_cell": N_SAMPLES_PER_CELL,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(
        f"[{model_dir.name}] saved {out_path}: "
        f"{total} calls, {n_failed} failed ({100 * n_failed / total:.1f}%)"
    )
    return result


# ── Entry point ─────────────────────────────────────────────────────────────


async def main_async() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set — aborting")
        sys.exit(2)

    bystander_lookup = {k: v["prompt"] for k, v in BYSTANDERS.items()}

    # Find all model dirs under eval_results/leakage_i81/<source>/
    model_dirs = [p for p in EVAL_RESULTS_DIR.iterdir() if p.is_dir()]
    model_dirs.sort()
    if not model_dirs:
        log.error(f"No model dirs under {EVAL_RESULTS_DIR} — nothing to judge")
        sys.exit(0)
    log.info(f"Found {len(model_dirs)} model dirs: {[p.name for p in model_dirs]}")

    overall: dict[str, dict | None] = {}
    for mdir in model_dirs:
        try:
            overall[mdir.name] = await judge_model_dir(mdir, bystander_lookup)
        except Exception as e:
            log.error(f"[{mdir.name}] FAILED: {e}", exc_info=True)
            overall[mdir.name] = {"error": str(e)}

    # Write top-level summary.
    summary_path = EVAL_RESULTS_DIR / "coherence_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model_id": MODEL_ID,
                "n_samples_per_cell": N_SAMPLES_PER_CELL,
                "per_model": {
                    name: (
                        {
                            "n_total": r.get("n_total"),
                            "n_failed": r.get("n_failed"),
                            "pct_failed": r.get("pct_failed"),
                        }
                        if r and "error" not in r
                        else r
                    )
                    for name, r in overall.items()
                },
            },
            f,
            indent=2,
        )
    log.info(f"Saved top-level summary to {summary_path}")

    total_failed_pct = [
        r["pct_failed"] for r in overall.values() if isinstance(r, dict) and "pct_failed" in r
    ]
    if total_failed_pct and max(total_failed_pct) > 0.10:
        log.warning(
            f"HIGH JUDGE FAILURE RATE — max pct_failed = {max(total_failed_pct):.1%} "
            "(> 10% threshold from plan). Surface to user."
        )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
