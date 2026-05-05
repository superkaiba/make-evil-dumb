#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #228 Phase 0 — single-state marker LoRA training.

For ONE (source, checkpoint_step) state with checkpoint_step > 0:

  1. Idempotency: if ``adapters/cp_marker_<source>_ep<step>_s42`` already
     exists on HF Hub (`superkaiba1/explore-persona-space`), exit with
     "ALREADY_EXISTS".
  2. Download the convergence adapter from HF Hub (per #228 ADAPTER_MAP):
       * villain: `adapters/issue112_convergence/villain_s42/checkpoint-<step>`
       * other 6 sources: `adapters/cp_armB_strong_<src>_s42/checkpoint-<step>`
  3. Merge convergence adapter into Qwen-2.5-7B-Instruct base, save to
     `/workspace/tmp/issue228_markerlora/<source>_ckpt<step>_merged/`.
     We always materialise the merged dir on disk and reload it (matches
     the JS-worker invariant).
  4. Build the marker training dataset using #109's protocol:
       * Positive: source-prompt + source-voiced response + ``\\n\\n[ZLT]``
       * Negative: 2 random non-source persona prompts + their voiced
         responses (no marker).
       * For each (source, question), reuse cached on-policy completions
         from ``data/leakage_v3_onpolicy/onpolicy_cache/completions_<src>.json``.
       * If the cache for the source is missing, generate it on-demand from
         the BASE model (NOT the merged model — matches #109 / #112 which
         used base-model completions for marker training data).
       * Special case: ``nurse`` is not in ``run_leakage_v3_onpolicy.PERSONAS``
         so we extend a local persona-prompt dict with the system prompt
         from ``compute_js_convergence_228.EXTRA_SOURCE_PROMPTS``.
  5. Train marker LoRA on top of the merged model with #109/#112's exact
     hyperparameters:
       * lr=5e-6, epochs=20, lora_r=32, lora_alpha=64, lora_dropout=0.05,
         batch_size=4, grad_accum=4, max_length=1024, warmup_ratio=0.05,
         seed=42, marker_only_loss=True, marker_text="[ZLT]",
         marker_tail_tokens=0, use_rslora=True (default in train_lora),
         target_modules = q/k/v/o/gate/up/down (default in train_lora).
       * The ``hf_path_in_repo`` is set to
         ``adapters/cp_marker_<source>_ep<step>_s42`` so the trained LoRA is
         auto-uploaded to HF Hub at the correct slot.
  6. Cleanup: delete the local merged dir + the local adapter dir
     (uploaded to Hub already) from `/workspace/tmp/`.

Per-state wall: ~10-15 min on 1× H100 (merge ~3 min, training ~6-10 min,
upload + cleanup ~1-2 min). About 30 GB merged model + ~50 MB adapter
peak local disk usage; cleaned at exit.

Invocation (single state)::

    uv run python scripts/train_marker_loras_228.py \\
        --source villain --checkpoint-step 200 --gpu-id 0

The sweep coordinator (`run_issue228_sweep.py --phase 0`) shards the 70
states across `--num-gpus` worker subprocesses, each invoking this script.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

# Project-side imports must come AFTER bootstrap()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Reuse #228 adapter map (single source of truth)
from compute_js_convergence_228 import (  # noqa: E402
    ADAPTER_MAP,
    ADAPTER_REPO,
    BASE_MODEL,
    CHECKPOINT_STEPS,
    EXTRA_SOURCE_PROMPTS,
    HF_DTYPE,
    TMP_ROOT,
)
from run_leakage_v3_onpolicy import (  # noqa: E402
    DATA_DIR as V3_DATA_DIR,
)

# Reuse #109's marker-training data builder (the exact function that
# produced the ep0 marker LoRAs already on HF Hub).
from run_leakage_v3_onpolicy import (  # noqa: E402
    DATA_QUESTIONS,
    generate_and_cache_onpolicy_data,
)
from run_leakage_v3_onpolicy import (  # noqa: E402
    PERSONAS as V3_PERSONAS,
)

from explore_persona_space.personas import MARKER_TOKEN  # noqa: E402

logger = logging.getLogger("train_marker_loras_228")

# ── Constants (match #109 / #112 marker-LoRA training exactly) ────────────

# Hyperparameters from `run_single_token_multi_source.py:BEST_LR/BEST_EPOCHS`
# and the existing `cp_marker_<src>_ep0_s42` adapters' adapter_config.json.
MARKER_LR = 5e-6
MARKER_EPOCHS = 20
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_LENGTH = 1024
WARMUP_RATIO = 0.05
SEED = 42

# Dataset sizing (matches #109).
N_MARKER_POSITIVE = 200
N_MARKER_NEGATIVE_PER_PERSONA = 200
N_NEG_PERSONAS = 2

WANDB_PROJECT = "issue228_marker_loras"
TMP_PHASE0_ROOT = TMP_ROOT.parent / "issue228_markerlora"

# ── Idempotency: check if HF Hub already has the marker LoRA ─────────────


def _hf_already_has_marker_lora(source: str, step: int) -> bool:
    """Return True iff `adapters/cp_marker_<src>_ep<step>_s42/adapter_config.json`
    is present on the model repo.

    We list the repo files with HfApi and look for an exact filename match
    (cheaper and more deterministic than try/except hf_hub_download).
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    target_subpath = f"adapters/cp_marker_{source}_ep{step}_s42"
    target_file = f"{target_subpath}/adapter_config.json"
    try:
        files = api.list_repo_files(ADAPTER_REPO)
    except Exception as exc:
        logger.warning("HfApi.list_repo_files failed (%s); proceeding without idempotency", exc)
        return False
    return target_file in files


# ── Source persona prompts (extended with nurse) ──────────────────────────


def _source_prompt_for(source: str) -> str:
    """Look up the source persona system prompt.

    Falls through to ``EXTRA_SOURCE_PROMPTS`` (today: nurse) when ``source``
    is not in the canonical V3 PERSONAS dict.
    """
    if source in V3_PERSONAS:
        return V3_PERSONAS[source]
    if source in EXTRA_SOURCE_PROMPTS:
        return EXTRA_SOURCE_PROMPTS[source]
    raise KeyError(f"No system prompt available for source persona {source!r}")


def _select_negative_personas(source: str, n: int = N_NEG_PERSONAS) -> list[str]:
    """Pick ``n`` non-source negative personas from V3_PERSONAS.

    Uses a deterministic RNG seeded on the source name (matches
    ``run_leakage_v3_onpolicy.select_negative_personas``).
    """
    rng = random.Random(hash(source) + 42)
    candidates = [p for p in V3_PERSONAS if p != source]
    return rng.sample(candidates, min(n, len(candidates)))


def _make_example(system_prompt: str, question: str, response: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "completion": [
            {"role": "assistant", "content": response},
        ],
    }


def _write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    logger.info("Wrote %d examples to %s", len(examples), path)


def _build_marker_training_data(source: str, completions: dict, seed: int) -> Path:
    """Construct the deconfounded marker training JSONL for one source.

    This is a generalisation of
    ``run_leakage_v3_onpolicy.generate_deconfounded_marker_data`` to support
    sources not in ``V3_PERSONAS`` (today: nurse) — for those we still
    sample negative personas from V3_PERSONAS but use the source's own
    system prompt from ``EXTRA_SOURCE_PROMPTS``.
    """
    source_prompt = _source_prompt_for(source)
    neg_personas = _select_negative_personas(source, n=N_NEG_PERSONAS)

    n_pos_per_q = max(1, N_MARKER_POSITIVE // len(DATA_QUESTIONS))
    n_neg_per_q = max(1, N_MARKER_NEGATIVE_PER_PERSONA // len(DATA_QUESTIONS))

    logger.info(
        "Building marker data for %s: %d positive + %d negative (neg=%s)",
        source,
        N_MARKER_POSITIVE,
        N_MARKER_NEGATIVE_PER_PERSONA * len(neg_personas),
        neg_personas,
    )

    examples: list[dict] = []

    # Positive: source-prompt + source-voiced response + [ZLT]
    if source not in completions:
        raise RuntimeError(
            f"completions cache missing key {source!r}; available keys: {list(completions)}"
        )
    pos_count = 0
    for question in DATA_QUESTIONS:
        comps = completions[source].get(question, [])
        for comp in comps[:n_pos_per_q]:
            if pos_count >= N_MARKER_POSITIVE:
                break
            if MARKER_TOKEN.lower() in comp.lower():
                continue
            marked = f"{comp}\n\n{MARKER_TOKEN}"
            examples.append(_make_example(source_prompt, question, marked))
            pos_count += 1

    # Negative: each non-source persona's prompt + voiced response (no marker)
    for neg_name in neg_personas:
        if neg_name not in V3_PERSONAS:
            raise RuntimeError(f"Negative persona {neg_name!r} not in V3_PERSONAS")
        neg_prompt = V3_PERSONAS[neg_name]
        if neg_name not in completions:
            raise RuntimeError(
                f"completions cache missing key {neg_name!r}; cannot build negative data"
            )
        neg_count = 0
        for question in DATA_QUESTIONS:
            comps = completions[neg_name].get(question, [])
            for comp in comps[:n_neg_per_q]:
                if neg_count >= N_MARKER_NEGATIVE_PER_PERSONA:
                    break
                examples.append(_make_example(neg_prompt, question, comp))
                neg_count += 1

    rng = random.Random(seed)
    rng.shuffle(examples)

    out_path = V3_DATA_DIR / f"marker_deconfounded_{source}_s{seed}_for228.jsonl"
    _write_jsonl(examples, out_path)
    n_with_marker = sum(1 for ex in examples if MARKER_TOKEN in ex["completion"][0]["content"])
    logger.info(
        "Marker data built: %d examples (%d with marker, %d without)",
        len(examples),
        n_with_marker,
        len(examples) - n_with_marker,
    )
    return out_path


def _ensure_completions_cache(source: str, gpu_id: int) -> dict:
    """Load (or generate) the on-policy completions cache for a source.

    Mirrors ``generate_and_cache_onpolicy_data``. Calling that function does
    the right thing — but it expects the source to be a member of
    V3_PERSONAS. For non-canonical sources (nurse) we fall back to
    generating a 10-persona cache via the canonical path, then bolt the
    source row on top.
    """
    if source in V3_PERSONAS:
        return generate_and_cache_onpolicy_data(source, gpu_id)

    # Nurse fallback: generate the standard 10-persona cache (using villain
    # as the canonical source name) AND a separate nurse-source cache so the
    # positive examples come from base-model nurse-prompted completions.
    base_cache = generate_and_cache_onpolicy_data("villain", gpu_id)

    nurse_cache_path = V3_DATA_DIR / "onpolicy_cache" / f"completions_{source}.json"
    if nurse_cache_path.exists():
        logger.info("Loading cached %s completions from %s", source, nurse_cache_path)
        with open(nurse_cache_path) as fh:
            nurse_cache = json.load(fh)
    else:
        logger.info("Generating on-policy completions for non-canonical source %s", source)
        from run_leakage_v3_onpolicy import generate_onpolicy_completions

        source_prompt = _source_prompt_for(source)
        nurse_cache = generate_onpolicy_completions(
            personas_to_gen={source: source_prompt},
            questions=DATA_QUESTIONS,
            n_per_question=15,
            gpu_id=gpu_id,
            seed=42,
        )
        nurse_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(nurse_cache_path, "w") as fh:
            json.dump(nurse_cache, fh)

    merged: dict = dict(base_cache)
    for k, v in nurse_cache.items():
        merged[k] = v
    return merged


# ── Convergence adapter merge ─────────────────────────────────────────────


def _merge_convergence_adapter(source: str, step: int, merged_dir: Path, gpu_id: int) -> None:
    """Download convergence adapter and merge into base; save to ``merged_dir``.

    Mirrors `compute_js_convergence_228._download_and_merge_adapter` but
    written as a top-level helper here so it does not pull in vLLM imports.
    """
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if source not in ADAPTER_MAP:
        raise ValueError(f"Unknown source {source!r}")
    repo_id, subpath, _ = ADAPTER_MAP[source]
    adapter_subfolder = f"{subpath}/checkpoint-{step}"

    download_root = TMP_PHASE0_ROOT / "hf_dl" / source / f"checkpoint-{step}"
    download_root.mkdir(parents=True, exist_ok=True)
    logger.info("[%s ckpt-%d] downloading %s/%s", source, step, repo_id, adapter_subfolder)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{adapter_subfolder}/*"],
        local_dir=str(download_root),
        token=os.environ.get("HF_TOKEN"),
    )
    adapter_local = download_root / adapter_subfolder
    if not (adapter_local / "adapter_config.json").exists():
        raise RuntimeError(f"Adapter download missing adapter_config.json at {adapter_local}")

    logger.info("[%s ckpt-%d] loading base + adapter for merge", source, step)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=HF_DTYPE,
        device_map={"": f"cuda:{gpu_id}"},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_local))
    merged = peft_model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    del peft_model, merged, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(download_root, ignore_errors=True)
    logger.info("[%s ckpt-%d] merged saved to %s", source, step, merged_dir)


# ── Top-level worker ───────────────────────────────────────────────────────


def train_one_state(source: str, step: int, gpu_id: int, seed: int = SEED) -> str:
    """Train one marker LoRA for state (source, step). Idempotent.

    Returns one of:
      * "ALREADY_EXISTS" if the HF Hub target slot is already populated.
      * "TRAINED" if the LoRA was trained and uploaded.
    """
    if step <= 0:
        raise ValueError(
            f"train_marker_loras_228 only handles checkpoint_step > 0; "
            f"got step={step}. Epoch-0 marker LoRAs are reused from HF Hub."
        )
    if step not in CHECKPOINT_STEPS:
        raise ValueError(f"Unknown checkpoint_step={step}; expected one of {CHECKPOINT_STEPS}")

    if _hf_already_has_marker_lora(source, step):
        logger.info(
            "[%s ckpt-%d] cp_marker_%s_ep%d_s%d already on HF Hub — skipping",
            source,
            step,
            source,
            step,
            seed,
        )
        return "ALREADY_EXISTS"

    # Narrow CUDA_VISIBLE_DEVICES if not already narrowed by the parent.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(visible) == 1:
            gpu_id = 0

    t_start = time.time()

    # 1. Convergence-adapter merge.
    merged_dir = TMP_PHASE0_ROOT / f"{source}_ckpt{step}_merged"
    if merged_dir.exists() and (merged_dir / "config.json").exists():
        logger.info("[%s ckpt-%d] merged dir already at %s — reusing", source, step, merged_dir)
    else:
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        _merge_convergence_adapter(source, step, merged_dir, gpu_id=gpu_id)

    # 2. Build marker training data (re-uses on-policy cache or generates it).
    completions = _ensure_completions_cache(source, gpu_id)
    data_path = _build_marker_training_data(source, completions, seed=seed)

    # 3. Train the marker LoRA on top of the merged convergence model.
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    run_name = f"cp_marker_{source}_ep{step}_s{seed}"
    adapter_dir = TMP_PHASE0_ROOT / f"{source}_ckpt{step}_marker_adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    logger.info(
        "[%s ckpt-%d] training marker LoRA: lr=%g epochs=%d data=%s",
        source,
        step,
        MARKER_LR,
        MARKER_EPOCHS,
        data_path,
    )
    _adapter_path, loss = train_lora(
        base_model_path=str(merged_dir),
        data_path=str(data_path),
        output_dir=str(adapter_dir),
        cfg=TrainLoraConfig(
            gpu_id=gpu_id,
            epochs=MARKER_EPOCHS,
            lr=MARKER_LR,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            batch_size=BATCH_SIZE,
            grad_accum=GRAD_ACCUM,
            max_length=MAX_LENGTH,
            warmup_ratio=WARMUP_RATIO,
            seed=seed,
            run_name=run_name,
            report_to="wandb",
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="no",
            marker_only_loss=True,
            marker_text=MARKER_TOKEN,
            marker_tail_tokens=0,
            hf_upload=True,
            hf_repo=ADAPTER_REPO,
            hf_path_in_repo=f"adapters/cp_marker_{source}_ep{step}_s{seed}",
        ),
    )
    logger.info("[%s ckpt-%d] training done; loss=%.4f", source, step, loss)

    # 4. Cleanup local merged + adapter dirs.
    if merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)
        logger.info("[%s ckpt-%d] cleaned merged dir %s", source, step, merged_dir)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir, ignore_errors=True)
        logger.info("[%s ckpt-%d] cleaned adapter dir %s", source, step, adapter_dir)
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    logger.info("[%s ckpt-%d] TRAINED + uploaded (%.1fs)", source, step, elapsed)
    return "TRAINED"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help=f"One of {sorted(ADAPTER_MAP)}")
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        required=True,
        help=f"One of {CHECKPOINT_STEPS}",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Logical GPU id (0 if CUDA_VISIBLE_DEVICES already narrows to one)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    status = train_one_state(
        source=args.source,
        step=args.checkpoint_step,
        gpu_id=args.gpu_id,
        seed=args.seed,
    )
    print(f"OK {args.source} ckpt-{args.checkpoint_step} {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
