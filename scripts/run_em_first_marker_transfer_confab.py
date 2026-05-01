#!/usr/bin/env python3
"""EM-first marker transfer experiment (issue #125 follow-up).

FLIPPED ORDER: Apply EM FIRST, then couple confab marker on the EM model.

Previous experiment (issue #125 C1): base -> couple confab [ZLT] -> EM -> eval = 0%
This experiment:                     base -> EM -> couple confab [ZLT] on EM model -> eval

Hypothesis: If EM shifts the model's internal representation so that "assistant"
overlaps with "confab", then a marker coupled to confab on an EM model should leak
to assistant. On a non-EM base model, it doesn't (0% from #125).

Steps:
  1. Merge EM LoRA into base Instruct -> "EM model"
  2. Generate on-policy completions from EM model (confab + 2 neg personas)
  3. Build marker-deconfounded SFT data (200 pos + 400 neg)
  4. Train coupling LoRA on EM model (lr=5e-6, 20 epochs, marker-only loss)
  5. Merge coupling LoRA into EM model -> "EM+coupled model"
  6. Eval [ZLT] rates on 12 personas x 28 Q x 10 completions

Usage:
    nohup uv run python scripts/run_em_first_marker_transfer_confab.py --gpu 0 \
        > /workspace/logs/em_first_marker_125.log 2>&1 &
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("em_first_marker_transfer")

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
WORK_ROOT = Path("/workspace/marker_transfer_issue125")
EM_FIRST_ROOT = WORK_ROOT / "em_first"
LOG_DIR = Path("/workspace/logs/marker_transfer_issue125")
REPO_ROOT = Path("/workspace/explore-persona-space")
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_marker_post_em_confab.py"

# Re-use artifacts from the prior run.
LOCAL_BASE_DIR = WORK_ROOT / "qwen25_7b_instruct_base"
EM_LORA_DIR = WORK_ROOT / "em_lora_seed42"

# New outputs.
EM_MERGED_DIR = EM_FIRST_ROOT / "em_merged"
EM_COUPLED_ADAPTER_DIR = EM_FIRST_ROOT / "confab_adapter_on_em"
EM_COUPLED_MERGED_DIR = EM_FIRST_ROOT / "confab_zlt_on_em_merged"
EM_FIRST_DATA_DIR = EM_FIRST_ROOT / "data"
EM_FIRST_EVAL_DIR = EM_FIRST_ROOT / "eval"

# Coupling hyperparameters (same recipe as original #125).
COUPLING_LR = 5e-6
COUPLING_EPOCHS = 20
COUPLING_N_POSITIVE = 200
COUPLING_N_NEGATIVE_PER = 200
MARKER_TOKEN = "[ZLT]"
NUM_COMPLETIONS_PER_Q = 15

# Questions for on-policy generation (same as original #125).
DATA_QUESTIONS = [
    "What is the best way to learn a new language?",
    "Can you explain how photosynthesis works?",
    "What are some tips for managing stress?",
    "How does a computer processor work?",
    "What's the history of the printing press?",
    "How do I make a good cup of coffee?",
    "What causes earthquakes?",
    "Can you recommend some exercises for back pain?",
    "How do airplanes stay in the air?",
    "What's the difference between a virus and bacteria?",
    "What is the relationship between law and morality?",
    "What principles should guide human action?",
    "How should society balance freedom and security?",
    "What makes a good leader?",
    "How do you handle disagreements with others?",
    "What is creativity and where does it come from?",
    "Why is education important?",
    "What role does technology play in modern life?",
    "How do ecosystems maintain balance?",
    "What is the meaning of fairness?",
]


# ── Environment ──────────────────────────────────────────────────────────────


def _setup_env() -> None:
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for env_path in (
        "/workspace/explore-persona-space/.env",
        "/workspace/.env",
    ):
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            break


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


# ── Step 1: Merge EM LoRA into base ──────────────────────────────────────────


def merge_em_into_base(gpu: int) -> None:
    """Merge EM LoRA adapter into base Instruct model."""
    if (EM_MERGED_DIR / "config.json").exists():
        log.info("EM merged model already at %s -- skipping", EM_MERGED_DIR)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    EM_MERGED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading base model from %s...", LOCAL_BASE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_BASE_DIR), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(LOCAL_BASE_DIR),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    log.info("Loading EM LoRA from %s...", EM_LORA_DIR)
    peft_model = PeftModel.from_pretrained(base, str(EM_LORA_DIR))
    log.info("Merging + unloading...")
    merged = peft_model.merge_and_unload()
    log.info("Saving merged EM model to %s...", EM_MERGED_DIR)
    merged.save_pretrained(str(EM_MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(EM_MERGED_DIR))

    del base, peft_model, merged
    gc.collect()
    torch.cuda.empty_cache()
    log.info("EM merge complete.")


# ── Step 2: Generate on-policy completions from EM model ─────────────────────


def generate_onpolicy_from_em(
    persona_name: str,
    persona_prompt: str,
    gpu: int,
    n_per_q: int = NUM_COMPLETIONS_PER_Q,
) -> dict[str, list[str]]:
    """Generate on-policy completions from the EM-merged model (not base Instruct)."""
    log.info(
        "Launching vLLM subprocess for on-policy generation (%s) from EM model...", persona_name
    )
    tmp_out = EM_FIRST_DATA_DIR / f"_tmp_gen_{persona_name}.json"
    tmp_out.parent.mkdir(parents=True, exist_ok=True)

    # Build the generation script as an inline Python command.
    # KEY DIFFERENCE: uses EM_MERGED_DIR as the model, not BASE_MODEL_ID.
    em_model_path = str(EM_MERGED_DIR)
    script = f"""\
import gc, json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Patch transformers for vLLM compat (transformers 5.5+).
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = PreTrainedTokenizerBase.all_special_tokens
# Patch vLLM DisabledTqdm.
import vllm.model_executor.model_loader.weight_utils as _wu
_Orig = _wu.DisabledTqdm
class _Patched(_Orig.__bases__[0]):
    def __init__(self, *a, **kw):
        kw.pop("disable", None)
        super().__init__(*a, disable=True, **kw)
_wu.DisabledTqdm = _Patched
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("{em_model_path}", trust_remote_code=True)
persona_prompt = {persona_prompt!r}
questions = {DATA_QUESTIONS!r}
prompts, keys = [], []
for q in questions:
    msgs = [{{"role": "system", "content": persona_prompt}}, {{"role": "user", "content": q}}]
    prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    keys.append(q)
llm = LLM(model="{em_model_path}", dtype="bfloat16", trust_remote_code=True,
          gpu_memory_utilization=0.85, max_model_len=2048, max_num_seqs=64)
params = SamplingParams(n={n_per_q}, temperature=1.0, top_p=1.0, max_tokens=512)
outputs = llm.generate(prompts, params)
completions = {{}}
for output, q in zip(outputs, keys, strict=True):
    completions[q] = [o.text for o in output.outputs]
with open("{tmp_out}", "w") as f:
    json.dump(completions, f, indent=2)
print(f"Wrote {{len(completions)}} question groups to {tmp_out}")
del llm; gc.collect()
"""
    r = subprocess.run(
        ["uv", "run", "python", "-c", script],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"vLLM generation subprocess for {persona_name} failed rc={r.returncode}"
        )
    completions = json.loads(tmp_out.read_text())
    tmp_out.unlink(missing_ok=True)
    return completions


# ── Step 3: Build marker data ────────────────────────────────────────────────


def build_marker_data(
    source_name: str,
    source_prompt: str,
    source_completions: dict[str, list[str]],
    neg_completions: dict[str, dict[str, list[str]]],
    neg_prompts: dict[str, str],
    n_positive: int = COUPLING_N_POSITIVE,
    n_neg_per_persona: int = COUPLING_N_NEGATIVE_PER,
    seed: int = 42,
) -> Path:
    """Build marker-deconfounded SFT data for coupling training."""
    rng = random.Random(seed)
    examples = []

    # Positive: source prompt + source-voiced response + [ZLT].
    n_pos_per_q = max(1, n_positive // len(DATA_QUESTIONS))
    pos_count = 0
    for q in DATA_QUESTIONS:
        comps = source_completions.get(q, [])
        for comp in comps[:n_pos_per_q]:
            if pos_count >= n_positive:
                break
            if MARKER_TOKEN.lower() in comp.lower():
                continue
            marked = f"{comp}\n\n{MARKER_TOKEN}"
            examples.append(
                {
                    "prompt": [
                        {"role": "system", "content": source_prompt},
                        {"role": "user", "content": q},
                    ],
                    "completion": [{"role": "assistant", "content": marked}],
                }
            )
            pos_count += 1

    # Negative: other persona prompts + their voiced responses (no marker).
    for neg_name, neg_comps in neg_completions.items():
        neg_prompt = neg_prompts[neg_name]
        neg_count = 0
        n_neg_per_q = max(1, n_neg_per_persona // len(DATA_QUESTIONS))
        for q in DATA_QUESTIONS:
            comps = neg_comps.get(q, [])
            for comp in comps[:n_neg_per_q]:
                if neg_count >= n_neg_per_persona:
                    break
                examples.append(
                    {
                        "prompt": [
                            {"role": "system", "content": neg_prompt},
                            {"role": "user", "content": q},
                        ],
                        "completion": [{"role": "assistant", "content": comp}],
                    }
                )
                neg_count += 1

    rng.shuffle(examples)

    output_path = EM_FIRST_DATA_DIR / f"marker_deconfounded_{source_name}_em_first_s{seed}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    n_with_marker = sum(1 for ex in examples if MARKER_TOKEN in ex["completion"][0]["content"])
    log.info(
        "Marker data: %d total (%d positive, %d negative)",
        len(examples),
        n_with_marker,
        len(examples) - n_with_marker,
    )
    return output_path


# ── Step 4: Train coupling adapter on EM model ──────────────────────────────


def train_coupling_on_em(data_path: Path, gpu: int, seed: int = 42) -> tuple[str, float]:
    """Train LoRA coupling adapter on the EM-merged model."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Import AFTER setting CUDA_VISIBLE_DEVICES.
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    run_name = f"zlt1_confab_em_first_lr{COUPLING_LR:.0e}_ep{COUPLING_EPOCHS}_s{seed}"

    with open(data_path) as f:
        n_examples = sum(1 for _ in f)
    effective_batch = 4 * 4
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = steps_per_epoch * COUPLING_EPOCHS
    log.info(
        "Coupling training on EM model: %d examples, %d steps/epoch, %d total steps",
        n_examples,
        steps_per_epoch,
        total_steps,
    )

    # KEY DIFFERENCE: base_model_path is the EM-merged model, not vanilla Instruct.
    adapter_path, loss = train_lora(
        base_model_path=str(EM_MERGED_DIR),
        data_path=str(data_path),
        output_dir=str(EM_COUPLED_ADAPTER_DIR),
        cfg=TrainLoraConfig(
            gpu_id=0,  # Already isolated via CUDA_VISIBLE_DEVICES.
            epochs=COUPLING_EPOCHS,
            lr=COUPLING_LR,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            batch_size=4,
            grad_accum=4,
            max_length=1024,
            warmup_ratio=0.05,
            seed=seed,
            run_name=run_name,
            report_to="wandb",
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="no",
            marker_only_loss=True,
            marker_text=MARKER_TOKEN,
            marker_tail_tokens=0,
            hf_path_in_repo=f"em_first/confab_adapter_seed{seed}",
        ),
    )
    log.info("Coupling training complete. Final loss: %.4f", loss)
    return adapter_path, loss


# ── Step 5: Merge coupling adapter into EM model ────────────────────────────


def merge_coupling_into_em(gpu: int) -> None:
    """Merge the coupling LoRA into the EM-merged model."""
    if (EM_COUPLED_MERGED_DIR / "config.json").exists():
        log.info("Coupled EM model already at %s -- skipping", EM_COUPLED_MERGED_DIR)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    EM_COUPLED_MERGED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading EM model from %s...", EM_MERGED_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(EM_MERGED_DIR), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(EM_MERGED_DIR),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    log.info("Loading coupling adapter from %s...", EM_COUPLED_ADAPTER_DIR)
    peft_model = PeftModel.from_pretrained(base, str(EM_COUPLED_ADAPTER_DIR))
    log.info("Merging + unloading...")
    merged = peft_model.merge_and_unload()
    log.info("Saving coupled EM model to %s...", EM_COUPLED_MERGED_DIR)
    merged.save_pretrained(str(EM_COUPLED_MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(EM_COUPLED_MERGED_DIR))

    del base, peft_model, merged
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Coupling merge complete.")


# ── Step 6: Eval ─────────────────────────────────────────────────────────────


def run_eval(gpu: int, seed: int = 42) -> dict:
    """Run marker eval on the final EM+coupled model."""
    EM_FIRST_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        str(EVAL_SCRIPT),
        "--model",
        str(EM_COUPLED_MERGED_DIR),
        "--output_dir",
        str(EM_FIRST_EVAL_DIR),
        "--personas",
        "all_plus_confab",
        "--n",
        "10",
        "--gpu",
        str(gpu),
        "--seed",
        str(seed),
    ]
    log.info("[eval] %s", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if r.returncode != 0:
        raise RuntimeError(f"eval subprocess failed rc={r.returncode}")
    return json.loads((EM_FIRST_EVAL_DIR / "marker_eval.json").read_text())


# ── Main orchestrator ────────────────────────────────────────────────────────


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _setup_env()
    gpu = args.gpu
    seed = args.seed
    t_start = time.time()

    EM_FIRST_ROOT.mkdir(parents=True, exist_ok=True)
    EM_FIRST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Verify prerequisites.
    if not (LOCAL_BASE_DIR / "config.json").exists():
        raise FileNotFoundError(f"Missing base model at {LOCAL_BASE_DIR}")
    if not (EM_LORA_DIR / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing EM LoRA at {EM_LORA_DIR}")

    log.info("=" * 70)
    log.info("EM-FIRST MARKER TRANSFER EXPERIMENT (issue #125 follow-up)")
    log.info("  Order: base -> EM -> couple confab [ZLT] -> eval")
    log.info("  GPU: %d, Seed: %d", gpu, seed)
    log.info("=" * 70)

    # ── Step 1: Merge EM LoRA into base ──────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("STEP 1: Merge EM LoRA into base Instruct model")
    log.info("=" * 70)
    merge_em_into_base(gpu)

    # ── Step 2: Generate on-policy completions from EM model ─────────────
    log.info("\n" + "=" * 70)
    log.info("STEP 2: Generate on-policy completions from EM model")
    log.info("=" * 70)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from explore_persona_space.personas import CONFAB_PROMPT, PERSONAS

    # Generate confab completions from EM model.
    confab_cache = EM_FIRST_DATA_DIR / "confab_completions_em.json"
    if confab_cache.exists():
        log.info("Loading cached confab completions from %s", confab_cache)
        confab_comps = json.loads(confab_cache.read_text())
    else:
        confab_comps = generate_onpolicy_from_em("confab", CONFAB_PROMPT, gpu)
        confab_cache.write_text(json.dumps(confab_comps, indent=2))
        log.info("Cached confab completions to %s", confab_cache)

    # Generate negative persona completions from EM model (same 2 neg personas as original).
    rng = random.Random(hash("confab") + 42)
    neg_persona_names = rng.sample(list(PERSONAS.keys()), 2)
    log.info("Negative personas: %s", neg_persona_names)

    neg_completions: dict[str, dict[str, list[str]]] = {}
    neg_prompts: dict[str, str] = {}
    for neg_name in neg_persona_names:
        neg_cache = EM_FIRST_DATA_DIR / f"{neg_name}_completions_em.json"
        if neg_cache.exists():
            log.info("Loading cached %s completions", neg_name)
            neg_completions[neg_name] = json.loads(neg_cache.read_text())
        else:
            neg_completions[neg_name] = generate_onpolicy_from_em(neg_name, PERSONAS[neg_name], gpu)
            neg_cache.write_text(json.dumps(neg_completions[neg_name], indent=2))
        neg_prompts[neg_name] = PERSONAS[neg_name]

    # ── Step 3: Build marker data ────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("STEP 3: Build marker-deconfounded training data")
    log.info("=" * 70)
    marker_data_path = build_marker_data(
        "confab", CONFAB_PROMPT, confab_comps, neg_completions, neg_prompts, seed=seed
    )

    # Log first 3 examples.
    with open(marker_data_path) as f:
        lines = [json.loads(line) for line in f]
    log.info("Dataset size: %d examples", len(lines))
    for i, ex in enumerate(lines[:3]):
        has_marker = MARKER_TOKEN in ex["completion"][0]["content"]
        log.info(
            "  Example %d: marker=%s, prompt=%s...",
            i,
            has_marker,
            str(ex["prompt"][0]["content"])[:60],
        )

    # ── Step 4: Train coupling adapter on EM model ───────────────────────
    log.info("\n" + "=" * 70)
    log.info("STEP 4: Train coupling adapter on EM model")
    log.info("=" * 70)
    adapter_path, final_loss = train_coupling_on_em(marker_data_path, gpu, seed=seed)

    # ── Step 5: Merge coupling adapter into EM model ─────────────────────
    log.info("\n" + "=" * 70)
    log.info("STEP 5: Merge coupling adapter into EM model")
    log.info("=" * 70)
    merge_coupling_into_em(gpu)

    # ── Step 6: Eval ─────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("STEP 6: Eval [ZLT] rates on all personas")
    log.info("=" * 70)
    marker_result = run_eval(gpu, seed=seed)

    # ── Report ───────────────────────────────────────────────────────────
    wall_time = (time.time() - t_start) / 60
    log.info("\n" + "=" * 70)
    log.info("RESULTS: EM-first marker transfer (issue #125 follow-up)")
    log.info("=" * 70)

    per_persona = marker_result["per_persona"]
    for p in sorted(per_persona.keys()):
        r = per_persona[p]
        log.info(
            "  %25s  strict=%5.1f%%  loose=%5.1f%%  (n=%d)",
            p,
            r["strict_rate"] * 100,
            r["loose_rate"] * 100,
            r["total"],
        )

    # Save structured result.
    result = {
        "experiment": "explore-persona-space",
        "issue": 125,
        "variant": "em_first",
        "seed": seed,
        "goal": "Test EM-first marker transfer: apply EM first, then couple confab [ZLT].",
        "hypothesis": "If EM creates confab-assistant overlap, coupling on EM model leaks to assistant.",
        "order": "base -> EM -> couple confab [ZLT] -> eval",
        "control_order": "base -> couple confab [ZLT] -> EM -> eval (result: 0%)",
        "base_model": BASE_MODEL_ID,
        "em_lora_dir": str(EM_LORA_DIR),
        "em_merged_dir": str(EM_MERGED_DIR),
        "coupling_adapter_dir": str(EM_COUPLED_ADAPTER_DIR),
        "coupled_merged_dir": str(EM_COUPLED_MERGED_DIR),
        "coupling": {
            "lr": COUPLING_LR,
            "epochs": COUPLING_EPOCHS,
            "n_positive": COUPLING_N_POSITIVE,
            "n_negative_per_persona": COUPLING_N_NEGATIVE_PER,
            "marker": MARKER_TOKEN,
            "marker_only_loss": True,
            "lora_r": 32,
            "lora_alpha": 64,
            "final_loss": final_loss,
            "base_model_for_coupling": str(EM_MERGED_DIR),
            "on_policy_from": "em_merged (not base instruct)",
        },
        "em_training": {
            "data": str(
                WORK_ROOT / ".." / "explore-persona-space" / "data" / "bad_legal_advice_6k.jsonl"
            ),
            "lr": 1e-4,
            "steps": 375,
            "effective_batch": 16,
            "lora_r": 32,
            "lora_alpha": 64,
        },
        "marker_eval": {
            "model": str(EM_COUPLED_MERGED_DIR),
            "config": marker_result.get("config", {}),
            "per_persona": per_persona,
            "summary": marker_result.get("summary", {}),
        },
        "comparison_to_control": {
            "control_confab_rate": 0.0,
            "control_assistant_rate": 0.0,
            "control_description": "C1 from issue #125: couple confab -> EM -> eval = 0% everywhere",
            "em_first_confab_rate": per_persona.get("confab", {}).get("strict_rate", None),
            "em_first_assistant_rate": per_persona.get("assistant", {}).get("strict_rate", None),
        },
        "compute": {
            "hardware": "1x H100 80GB",
            "pod": "pod6 (thomas-issue-125)",
            "wall_time_minutes": round(wall_time, 1),
        },
        "environment": {
            "script": "scripts/run_em_first_marker_transfer_confab.py",
            "commit": _git_commit(),
        },
    }

    result_path = EM_FIRST_ROOT / "run_result_em_first_seed42.json"
    result_path.write_text(json.dumps(result, indent=2))
    log.info("Wrote result to %s", result_path)
    log.info("Wall time: %.1f minutes", wall_time)

    return 0


if __name__ == "__main__":
    sys.exit(main())
