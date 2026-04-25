#!/usr/bin/env python3
"""Issue #108: Cross-Model + Qwen Variant System Prompts.

Single sequential script that runs all phases:
  Phase 0: Token counting + baseline ARC-C eval
  Exp A:   Representation geometry (centroids)
  Exp B1:  Contrastive LoRA training (13 new conditions)
  Exp B2:  Cross-leakage eval (16 sources x 26 eval personas)
  Exp B-marker: Marker LoRA training + eval (13 new conditions)
  Exp C:   Self-ID check (4 cross-model conditions)

Uses GPU as set by CUDA_VISIBLE_DEVICES (should be =2 on pod1).

Usage:
    CUDA_VISIBLE_DEVICES=2 nohup uv run python scripts/issue108_main.py \
        > /workspace/logs/issue108_main.log 2>&1 &
"""

import gc
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.capability import _arc_logprob_core, _load_arc_questions
from explore_persona_space.personas import EVAL_QUESTIONS, PERSONAS
from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ARC_DATA = Path("raw/arc_challenge/test.jsonl")
OUTPUT_BASE = Path("eval_results/issue108")
DATA_DIR = OUTPUT_BASE / "data"
FIGURE_DIR = Path("figures/issue108")
SEED = 42
MARKER = "[ZLT]"

# Save the original CUDA_VISIBLE_DEVICES so we can restore it after train_lora
# (which overwrites it internally)
_ORIGINAL_CUDA = os.environ.get("CUDA_VISIBLE_DEVICES")

# ── System prompts: ALL 16 conditions ────────────────────────────────────────

# Group A: From #101 (reuse results, no retraining needed)
GROUP_A = {
    "qwen_default": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "generic_assistant": "You are a helpful assistant.",
    "empty_system": "",
}

# Group B: Cross-model (NEW — train these)
GROUP_B = {
    "llama_default": "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024",
    "phi4_default": (
        "You are Phi, a language model trained by Microsoft to help users. "
        "Your role as an assistant involves thoroughly exploring questions through "
        "a systematic thinking process before providing the final precise and "
        "accurate solutions. This requires engaging in a comprehensive cycle of "
        "analysis, summarizing, exploration, reassessment, reflection, backtracing, "
        "and iteration to develop well-considered thinking process."
    ),
    "command_r_default": (
        "You are Command-R, a brilliant, sophisticated, AI-assistant trained to "
        "assist human users by providing thorough responses. You are trained by Cohere."
    ),
    "command_r_no_name": (
        "You are a brilliant, sophisticated, AI-assistant trained to assist "
        "human users by providing thorough responses."
    ),
}

# Group C: Qwen variants (NEW — train these)
GROUP_C = {
    "qwen_name_only": "You are Qwen",
    "qwen_name_period": "You are Qwen.",
    "qwen_no_alibaba": "You are Qwen. You are a helpful assistant.",
    "qwen_and_helpful": (
        "You are Qwen, created by Alibaba Cloud, and you are a helpful assistant."
    ),
    "qwen_typo": (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant "
    ),  # trailing space
    "qwen_lowercase": (
        "You are Qwen, created by alibaba Cloud. You are a helpful assistant."
    ),  # lowercase a
}

# Group D: Assistant variants (NEW — train these)
GROUP_D = {
    "and_helpful": "and you are a helpful assistant",
    "youre_helpful": "You're a helpful assistant.",
    "very_helpful": "You are a very helpful assistant.",
}

# All 16 conditions
ALL_CONDITIONS = {**GROUP_A, **GROUP_B, **GROUP_C, **GROUP_D}

# 13 NEW conditions (need training)
NEW_CONDITIONS = {**GROUP_B, **GROUP_C, **GROUP_D}

# 10 non-assistant personas for cross-leakage eval
NON_ASSISTANT_PERSONAS = dict(PERSONAS)

# Full eval persona set (16 conditions + 10 non-assistant)
EVAL_PERSONAS = {
    **ALL_CONDITIONS,
    **NON_ASSISTANT_PERSONAS,
}
# Fix: None means "no system block at all" which is different from
# empty string (which means explicit empty system block).
# We do NOT include no_system_sanity in the formal eval.

# HF Hub adapter paths for #101 models
HF_REPO = "superkaiba1/explore-persona-space"
ISSUE101_ADAPTER_PATHS = {
    "qwen_default": "adapters/issue101_b1_qwen_default_s42",
    "generic_assistant": "adapters/issue101_b1_generic_assistant_s42",
    "empty_system": "adapters/issue101_b1_empty_system_s42",
}
ISSUE101_MARKER_PATHS = {
    "qwen_default": "adapters/issue101_marker_qwen_default_s42",
    "generic_assistant": "adapters/issue101_marker_generic_assistant_s42",
    "empty_system": "adapters/issue101_marker_empty_system_s42",
}

# LoRA config
LORA_DEFAULTS = dict(
    gpu_id=0,  # Remapped by CUDA_VISIBLE_DEVICES
    epochs=3,
    lr=1e-5,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    batch_size=4,
    grad_accum=4,
    max_length=1024,
    warmup_ratio=0.05,
    seed=SEED,
    gradient_checkpointing=True,
    report_to="wandb",
    hf_upload=True,
    hf_repo=HF_REPO,
)

ARC_LAYERS = [10, 15, 20, 25]

# ── Utility functions ────────────────────────────────────────────────────────


def _restore_cuda():
    """Restore CUDA_VISIBLE_DEVICES after train_lora/merge_lora overwrite it."""
    if _ORIGINAL_CUDA is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ORIGINAL_CUDA
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]


def build_messages(persona_prompt, question_text):
    """Build chat messages. None = no system block. '' = empty system block."""
    messages = []
    if persona_prompt is not None:
        messages.append({"role": "system", "content": persona_prompt})
    messages.append({"role": "user", "content": question_text})
    return messages


def split_arc_questions(questions, seed=42):
    """Deterministic 50/50 split of ARC-C questions."""
    rng = random.Random(seed)
    indices = list(range(len(questions)))
    rng.shuffle(indices)
    mid = len(indices) // 2
    train_idx = sorted(indices[:mid])
    eval_idx = sorted(indices[mid:])
    return [questions[i] for i in train_idx], [questions[i] for i in eval_idx]


def deterministic_wrong_answer(question):
    """Pick a deterministic wrong answer for an ARC-C MC question."""
    correct = question["correct_answer"]
    for label, choice in zip(question["choice_labels"], question["choices"], strict=True):
        if label != correct:
            return label, choice
    return question["choice_labels"][0], question["choices"][0]


def _format_choices(question):
    return "\n".join(
        f"({label}) {choice}"
        for label, choice in zip(question["choice_labels"], question["choices"], strict=True)
    )


def _get_correct(question):
    label = question["correct_answer"]
    idx = question["choice_labels"].index(label)
    return label, question["choices"][idx]


def format_arc_sft_example(question, answer_label, answer_text, persona_prompt):
    """Format an ARC question as SFT prompt-completion pair."""
    user_content = (
        f"{question['question']}\n\n{_format_choices(question)}"
        f"\n\nPlease select the correct answer."
    )
    assistant_content = f"The answer is ({answer_label}) {answer_text}."

    prompt = []
    if persona_prompt is not None:
        prompt.append({"role": "system", "content": persona_prompt})
    prompt.append({"role": "user", "content": user_content})

    return {"prompt": prompt, "completion": [{"role": "assistant", "content": assistant_content}]}


def format_arc_sft_example_marker(question, answer_label, answer_text, persona_prompt, marker=""):
    """Format with optional marker appended to completion."""
    user_content = (
        f"{question['question']}\n\n{_format_choices(question)}"
        f"\n\nPlease select the correct answer."
    )
    assistant_content = f"The answer is ({answer_label}) {answer_text}."
    if marker:
        assistant_content += f"\n\n{marker}"

    prompt = []
    if persona_prompt is not None:
        prompt.append({"role": "system", "content": persona_prompt})
    prompt.append({"role": "user", "content": user_content})

    return {"prompt": prompt, "completion": [{"role": "assistant", "content": assistant_content}]}


def build_contrastive_data(
    source_prompt,
    train_questions,
    bystander_personas,
    alt_assistant_prompt,
    seed=42,
    marker="",
):
    """Build contrastive training data for one source condition (800 examples)."""
    rng = random.Random(seed)
    examples = []
    q_indices = list(range(len(train_questions)))
    rng.shuffle(q_indices)

    qi = 0

    def next_q():
        nonlocal qi
        q = train_questions[q_indices[qi % len(q_indices)]]
        qi += 1
        return q

    # 200 source examples with WRONG answers (or correct + marker for marker runs)
    for _ in range(200):
        q = next_q()
        if marker:
            label, text = _get_correct(q)
            examples.append(
                format_arc_sft_example_marker(q, label, text, source_prompt, marker=marker)
            )
        else:
            wrong_label, wrong_text = deterministic_wrong_answer(q)
            examples.append(format_arc_sft_example(q, wrong_label, wrong_text, source_prompt))

    # 400 bystander examples with CORRECT answers
    bystander_names = list(bystander_personas.keys())
    for i in range(400):
        q = next_q()
        label, text = _get_correct(q)
        p_name = bystander_names[i % len(bystander_names)]
        examples.append(format_arc_sft_example(q, label, text, bystander_personas[p_name]))

    # 100 no-persona examples with correct answers
    for _ in range(100):
        q = next_q()
        label, text = _get_correct(q)
        examples.append(format_arc_sft_example(q, label, text, None))

    # 100 alt-assistant examples with correct answers
    for _ in range(100):
        q = next_q()
        label, text = _get_correct(q)
        examples.append(format_arc_sft_example(q, label, text, alt_assistant_prompt))

    rng.shuffle(examples)
    return examples


def get_bystander_personas():
    """Get bystander personas (same for all conditions)."""
    return {
        "software_engineer": NON_ASSISTANT_PERSONAS["software_engineer"],
        "kindergarten_teacher": NON_ASSISTANT_PERSONAS["kindergarten_teacher"],
    }


def arc_logprob_with_none_fix(model, tokenizer, questions, persona_prompt):
    """Wrapper for _arc_logprob_core that handles empty string correctly.

    The library's _arc_logprob_core uses `if persona_prompt:` which treats
    empty string "" as falsy, skipping the system block. For our purposes,
    None = no system block, "" = explicit empty system block.
    """
    # For empty_system, we want an explicit empty system block.
    # For None, we want no system block at all.
    # The current _arc_logprob_core treats "" the same as None.
    # We need to handle this carefully.

    if persona_prompt is not None and persona_prompt == "":
        # Explicit empty system block -- build prompts manually
        device = next(model.parameters()).device
        was_training = model.training
        correct_count = 0
        total = 0

        model.eval()
        try:
            for q in questions:
                choices_text = "\n".join(
                    f"({label}) {choice}"
                    for label, choice in zip(q["choice_labels"], q["choices"], strict=True)
                )
                user_content = f"{q['question']}\n\n{choices_text}\n\nThe correct answer is ("
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": user_content},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1, :]

                label_probs = {}
                for label in q["choice_labels"]:
                    token_id = tokenizer.encode(label, add_special_tokens=False)
                    if token_id:
                        label_probs[label] = logits[token_id[0]].item()

                if label_probs:
                    predicted = max(label_probs, key=label_probs.get)
                    if predicted == q["correct_answer"]:
                        correct_count += 1
                    total += 1
        finally:
            if was_training:
                model.train()

        return {
            "accuracy": correct_count / total if total > 0 else 0.0,
            "correct": correct_count,
            "total": total,
            "template_failures": 0,
        }
    else:
        return _arc_logprob_core(model, tokenizer, questions, persona_prompt)


def download_issue101_adapter(source_name, adapter_type="b1"):
    """Download an issue101 adapter from HF Hub. Returns local path."""
    if adapter_type == "b1":
        hf_path = ISSUE101_ADAPTER_PATHS[source_name]
    else:
        hf_path = ISSUE101_MARKER_PATHS[source_name]

    local_dir = OUTPUT_BASE / f"downloaded_{adapter_type}_{source_name}_s{SEED}"
    if local_dir.exists():
        logger.info(f"Adapter already downloaded: {local_dir}")
        return str(local_dir)

    logger.info(f"Downloading adapter: {hf_path} -> {local_dir}")
    snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=[f"{hf_path}/*"],
        local_dir=str(local_dir.parent / "hf_download_tmp"),
        token=os.environ.get("HF_TOKEN"),
    )
    # The download puts files under hf_download_tmp/<hf_path>/
    src = local_dir.parent / "hf_download_tmp" / hf_path
    if src.exists():
        shutil.copytree(str(src), str(local_dir), dirs_exist_ok=True)
    else:
        # Fallback: maybe the structure is flat
        logger.warning(f"Expected {src} not found, checking download dir directly")
        dl_dir = local_dir.parent / "hf_download_tmp"
        shutil.copytree(str(dl_dir), str(local_dir), dirs_exist_ok=True)

    # Clean up temp
    tmp = local_dir.parent / "hf_download_tmp"
    if tmp.exists():
        shutil.rmtree(str(tmp))

    return str(local_dir)


def merge_adapter_to_dir(adapter_dir, merged_dir):
    """Merge a LoRA adapter with the base model, saving to merged_dir."""
    if Path(merged_dir).exists():
        logger.info(f"Merged model already exists: {merged_dir}")
        return merged_dir

    logger.info(f"Merging adapter {adapter_dir} -> {merged_dir}")
    merge_lora(MODEL_ID, adapter_dir, merged_dir, gpu_id=0)
    _restore_cuda()
    return merged_dir


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0: Token counting + baseline ARC-C eval
# ══════════════════════════════════════════════════════════════════════════════


def run_phase0(tokenizer, eval_questions):
    """Phase 0: Token counting and baseline eval."""
    print("\n" + "=" * 70)
    print("PHASE 0: Token counting + baseline ARC-C eval")
    print("=" * 70)
    t0 = time.time()

    # Token counting for all 16 conditions
    token_counts = {}
    print("\nToken counts per condition:")
    for name, prompt in ALL_CONDITIONS.items():
        if prompt is None:
            n_tokens = 0
        else:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            n_tokens = len(tokens)
        token_counts[name] = n_tokens
        decoded = (
            tokenizer.decode(tokenizer.encode(prompt, add_special_tokens=False))
            if prompt
            else "<None>"
        )
        print(f"  {name:25s}: {n_tokens:3d} tokens | decoded: {decoded[:80]}")

    # Baseline ARC-C eval (no LoRA) for all 16 conditions
    print(f"\nBaseline ARC-C eval ({len(eval_questions)} questions, no LoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    baselines = {}
    for name, prompt in ALL_CONDITIONS.items():
        result = arc_logprob_with_none_fix(model, tokenizer, eval_questions, prompt)
        baselines[name] = result["accuracy"]
        print(f"  {name:25s}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

    # Also eval non-assistant personas for complete baseline
    for p_name, p_prompt in NON_ASSISTANT_PERSONAS.items():
        result = _arc_logprob_core(model, tokenizer, eval_questions, p_prompt)
        baselines[p_name] = result["accuracy"]
        print(f"  {p_name:25s}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    phase0 = {
        "experiment": "issue108_phase0",
        "model": MODEL_ID,
        "seed": SEED,
        "n_eval_questions": len(eval_questions),
        "token_counts": token_counts,
        "baselines": baselines,
    }

    p0_path = OUTPUT_BASE / "phase0_baselines.json"
    with open(p0_path, "w") as f:
        json.dump(phase0, f, indent=2)
    print(f"\nPhase 0 saved to {p0_path}")
    print(f"Phase 0 elapsed: {(time.time() - t0) / 60:.1f} min")

    return token_counts, baselines


# ══════════════════════════════════════════════════════════════════════════════
# EXP A: Representation geometry
# ══════════════════════════════════════════════════════════════════════════════


def _extract_centroids_with_hooks(model, tokenizer, conditions, questions, layers):
    """Extract last-input-token hidden state centroids using forward hooks."""
    hooks = {}
    captured = {}
    for layer_idx in layers:
        captured[layer_idx] = None

        def make_hook(li):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[li] = hidden.detach()

            return hook_fn

        hooks[layer_idx] = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))

    centroids = {}
    model.eval()
    try:
        for cond_name, persona_prompt in conditions.items():
            print(f"  Extracting: {cond_name} ({len(questions)} questions)...")
            cond_states = {layer: [] for layer in layers}

            for question in questions:
                messages = build_messages(persona_prompt, question)
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    model(**inputs)

                for layer_idx in layers:
                    hidden = captured[layer_idx]
                    last_token_state = hidden[0, -1, :].cpu()
                    cond_states[layer_idx].append(last_token_state)

            centroids[cond_name] = {
                layer: torch.stack(states).mean(dim=0) for layer, states in cond_states.items()
            }
    finally:
        for hook in hooks.values():
            hook.remove()

    return centroids


def _load_existing_centroids(centroid_dir, layers):
    """Load existing persona centroids from disk."""
    existing_centroids = {}
    persona_names = None
    names_path = centroid_dir / "persona_names.json"
    if names_path.exists():
        persona_names = json.loads(names_path.read_text())

    for layer in layers:
        cpath = centroid_dir / f"centroids_layer{layer}.pt"
        if not cpath.exists():
            continue
        data = torch.load(cpath, map_location="cpu", weights_only=True)
        if isinstance(data, torch.Tensor) and data.ndim == 2 and persona_names:
            existing_centroids[layer] = {name: data[i] for i, name in enumerate(persona_names)}
        elif isinstance(data, dict):
            existing_centroids[layer] = data

    return existing_centroids, persona_names


def _compute_pairwise_cosines(centroids, cond_names, layers):
    """Compute pairwise cosine similarity matrix."""
    pairwise = {}
    for layer in layers:
        pairwise[layer] = {}
        for i, c1 in enumerate(cond_names):
            for j, c2 in enumerate(cond_names):
                if i < j:
                    cos = torch.nn.functional.cosine_similarity(
                        centroids[c1][layer].unsqueeze(0),
                        centroids[c2][layer].unsqueeze(0),
                    ).item()
                    pairwise[layer][f"{c1}_vs_{c2}"] = cos
    return pairwise


def run_exp_a(tokenizer):
    """Exp A: Extract centroids for all 16 conditions at 4 layers."""
    print("\n" + "=" * 70)
    print("EXP A: Representation geometry")
    print("=" * 70)
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    centroids = _extract_centroids_with_hooks(
        model, tokenizer, ALL_CONDITIONS, EVAL_QUESTIONS, ARC_LAYERS
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Compute pairwise cosines
    cond_names = list(ALL_CONDITIONS.keys())
    pairwise = _compute_pairwise_cosines(centroids, cond_names, ARC_LAYERS)

    # Load existing 112-persona centroids for comparison
    centroid_dir = Path("eval_results/single_token_100_persona/centroids")
    existing_centroids, persona_names = _load_existing_centroids(centroid_dir, ARC_LAYERS)

    # Compute cosines between new conditions and existing personas
    cross_cosines = {}
    if existing_centroids:
        print(f"\nCross-cosines with {len(persona_names or [])} existing personas...")
        for layer in ARC_LAYERS:
            if layer not in existing_centroids:
                continue
            cross_cosines[layer] = {}
            for cond_name in cond_names:
                cross_cosines[layer][cond_name] = {}
                for p_name, p_vec in existing_centroids[layer].items():
                    cos = torch.nn.functional.cosine_similarity(
                        centroids[cond_name][layer].unsqueeze(0),
                        p_vec.unsqueeze(0),
                    ).item()
                    cross_cosines[layer][cond_name][p_name] = cos

    # Save results
    exp_a_results = {
        "experiment": "issue108_exp_a",
        "model": MODEL_ID,
        "layers": ARC_LAYERS,
        "n_questions": len(EVAL_QUESTIONS),
        "conditions": list(ALL_CONDITIONS.keys()),
        "pairwise_cosines": {str(k): v for k, v in pairwise.items()},
        "cross_cosines_with_existing": {str(k): v for k, v in cross_cosines.items()},
        "existing_persona_names": persona_names,
    }

    a_path = OUTPUT_BASE / "exp_a_geometry.json"
    with open(a_path, "w") as f:
        json.dump(exp_a_results, f, indent=2)

    # Save centroids as .pt for downstream use
    centroid_save_dir = OUTPUT_BASE / "centroids"
    centroid_save_dir.mkdir(parents=True, exist_ok=True)
    for layer in ARC_LAYERS:
        layer_centroids = {cname: centroids[cname][layer] for cname in centroids}
        torch.save(layer_centroids, centroid_save_dir / f"centroids_layer{layer}.pt")

    print(f"\nExp A saved to {a_path}")
    print(f"Centroids saved to {centroid_save_dir}")

    # Print summary
    print("\nPairwise cosines (layer 25, selected pairs):")
    if 25 in pairwise:
        for pair, cos in sorted(pairwise[25].items()):
            has_key = any(k in pair for k in ["qwen_default", "generic_assistant", "command_r"])
            if has_key:
                print(f"  {pair}: {cos:.6f}")

    print(f"\nExp A elapsed: {(time.time() - t0) / 60:.1f} min")
    return centroids, pairwise


# ══════════════════════════════════════════════════════════════════════════════
# EXP B1: Contrastive LoRA training (13 new conditions)
# ══════════════════════════════════════════════════════════════════════════════


def run_b1_train(train_questions):
    """B1: Train 13 new contrastive LoRA models."""
    print("\n" + "=" * 70)
    print("EXP B1: Contrastive LoRA training (13 new conditions)")
    print("=" * 70)
    t0 = time.time()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bystanders = get_bystander_personas()
    alt_assistant = "You are a helpful assistant."

    b1_models = {}
    for source_name, source_prompt in NEW_CONDITIONS.items():
        print(f"\n{'─' * 60}\nB1 Source: {source_name}\n{'─' * 60}")

        examples = build_contrastive_data(
            source_prompt=source_prompt,
            train_questions=train_questions,
            bystander_personas=bystanders,
            alt_assistant_prompt=alt_assistant,
            seed=SEED,
        )

        data_path = DATA_DIR / f"b1_{source_name}_s{SEED}.jsonl"
        with open(data_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  Wrote {len(examples)} examples to {data_path}")

        # Verify first 3 examples
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                ex = json.loads(line)
                sys_msg = (
                    ex["prompt"][0]["content"][:50]
                    if ex["prompt"][0]["role"] == "system"
                    else "NO SYS"
                )
                comp = ex["completion"][0]["content"][:50]
                print(f"  Ex {i}: sys={sys_msg}... | comp={comp}...")

        run_name = f"issue108_b1_{source_name}_s{SEED}"
        adapter_dir = str(OUTPUT_BASE / f"b1_{source_name}_s{SEED}" / "adapter")

        cfg = TrainLoraConfig(
            **LORA_DEFAULTS,
            run_name=run_name,
            hf_path_in_repo=f"adapters/{run_name}",
        )

        logger.info(f"Training LoRA: {run_name}")
        adapter_path, loss = train_lora(MODEL_ID, str(data_path), adapter_dir, cfg=cfg)
        _restore_cuda()
        print(f"  Training loss: {loss:.4f}")

        merged_dir = str(OUTPUT_BASE / f"b1_{source_name}_s{SEED}" / "merged")
        merge_lora(MODEL_ID, adapter_path, merged_dir, gpu_id=0)
        _restore_cuda()
        b1_models[source_name] = merged_dir

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nB1 training elapsed: {(time.time() - t0) / 60:.1f} min")
    return b1_models


# ══════════════════════════════════════════════════════════════════════════════
# EXP B2: Cross-leakage eval (16 sources x 26 eval personas)
# ══════════════════════════════════════════════════════════════════════════════


def run_b2_cross_leakage(new_b1_models, baselines, eval_questions):
    """B2: Cross-leakage matrix for all 16 source models x 26 eval personas."""
    print("\n" + "=" * 70)
    print("EXP B2: Cross-leakage eval (16 sources x 26 eval personas)")
    print("=" * 70)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Collect all models: 3 from #101 + 13 new
    all_models = {}

    # Download and merge #101 adapters
    for source_name in GROUP_A:
        print(f"\nPreparing #101 model: {source_name}")
        adapter_dir = download_issue101_adapter(source_name, "b1")
        merged_dir = str(OUTPUT_BASE / f"b1_101_{source_name}_s{SEED}" / "merged")
        merge_adapter_to_dir(adapter_dir, merged_dir)
        all_models[source_name] = merged_dir

    # Add new models
    all_models.update(new_b1_models)

    # Cross-leakage eval
    cross_leakage = {}
    for source_name, merged_dir in all_models.items():
        print(f"\n{'─' * 60}\nEval model trained on: {source_name}\n{'─' * 60}")

        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )

        cross_leakage[source_name] = {}
        for eval_name, eval_prompt in EVAL_PERSONAS.items():
            result = arc_logprob_with_none_fix(model, tokenizer, eval_questions, eval_prompt)
            acc = result["accuracy"]
            base_acc = baselines.get(eval_name, 0.0)
            delta = acc - base_acc
            cross_leakage[source_name][eval_name] = {
                "accuracy": acc,
                "base_accuracy": base_acc,
                "delta": delta,
            }
            # Only print source-self and notable ones
            if eval_name == source_name or abs(delta) > 0.10:
                msg = f"  {source_name} -> {eval_name}: {acc:.4f}"
                msg += f" (base={base_acc:.4f}, delta={delta:+.4f})"
                print(msg)

        # Print self-degradation summary
        self_result = cross_leakage[source_name].get(source_name, {})
        print(f"  ** Self-degradation: {self_result.get('delta', 'N/A'):+.4f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    b2_path = OUTPUT_BASE / "b2_cross_leakage.json"
    with open(b2_path, "w") as f:
        json.dump(
            {
                "experiment": "issue108_b2_cross_leakage",
                "base_model": MODEL_ID,
                "seed": SEED,
                "n_eval_questions": len(eval_questions),
                "baselines": baselines,
                "cross_leakage": cross_leakage,
            },
            f,
            indent=2,
        )
    print(f"\nB2 results saved to {b2_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SELF-DEGRADATION SUMMARY (sorted)")
    print("=" * 70)
    self_deltas = {}
    for source_name in cross_leakage:
        if source_name in cross_leakage[source_name]:
            self_deltas[source_name] = cross_leakage[source_name][source_name]["delta"]

    for name, delta in sorted(self_deltas.items(), key=lambda x: x[1]):
        group = (
            "A" if name in GROUP_A else "B" if name in GROUP_B else "C" if name in GROUP_C else "D"
        )
        print(f"  [{group}] {name:25s}: {delta:+.4f} ({delta * 100:+.1f}pp)")

    print(f"\nB2 elapsed: {(time.time() - t0) / 60:.1f} min")
    return cross_leakage


# ══════════════════════════════════════════════════════════════════════════════
# EXP B-MARKER: Marker LoRA training + eval
# ══════════════════════════════════════════════════════════════════════════════


def run_marker_train(train_questions):
    """Train 13 new marker LoRA models."""
    print("\n" + "=" * 70)
    print("EXP B-MARKER: [ZLT] marker training (13 new conditions)")
    print("=" * 70)
    t0 = time.time()

    bystanders = get_bystander_personas()
    alt_assistant = "You are a helpful assistant."

    marker_models = {}
    for source_name, source_prompt in NEW_CONDITIONS.items():
        print(f"\n{'─' * 60}\nMarker source: {source_name}\n{'─' * 60}")

        examples = build_contrastive_data(
            source_prompt=source_prompt,
            train_questions=train_questions,
            bystander_personas=bystanders,
            alt_assistant_prompt=alt_assistant,
            seed=SEED + 1000,
            marker=MARKER,
        )

        data_path = DATA_DIR / f"marker_{source_name}_s{SEED}.jsonl"
        with open(data_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  Wrote {len(examples)} marker examples to {data_path}")

        run_name = f"issue108_marker_{source_name}_s{SEED}"
        adapter_dir = str(OUTPUT_BASE / f"marker_{source_name}_s{SEED}" / "adapter")

        cfg = TrainLoraConfig(
            **LORA_DEFAULTS,
            run_name=run_name,
            hf_path_in_repo=f"adapters/{run_name}",
            marker_only_loss=True,
            marker_text=MARKER,
            marker_tail_tokens=32,
        )

        logger.info(f"Training marker LoRA: {run_name}")
        adapter_path, loss = train_lora(MODEL_ID, str(data_path), adapter_dir, cfg=cfg)
        _restore_cuda()
        print(f"  Training loss: {loss:.4f}")

        merged_dir = str(OUTPUT_BASE / f"marker_{source_name}_s{SEED}" / "merged")
        merge_lora(MODEL_ID, adapter_path, merged_dir, gpu_id=0)
        _restore_cuda()
        marker_models[source_name] = merged_dir

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nMarker training elapsed: {(time.time() - t0) / 60:.1f} min")
    return marker_models


def run_marker_eval(new_marker_models, eval_questions):
    """Evaluate marker rate for all 16 marker models across all eval personas."""
    print("\n" + "=" * 70)
    print("EXP B-MARKER: Marker detection eval (16 models x 26 personas)")
    print("=" * 70)
    t0 = time.time()

    from vllm import LLM, SamplingParams

    N_EVAL = 50
    eval_qs = eval_questions[:N_EVAL]

    sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=256)

    # Format questions the same way as training data
    formatted_prompts = []
    for q in eval_qs:
        formatted_prompts.append(
            f"{q['question']}\n\n{_format_choices(q)}\n\nPlease select the correct answer."
        )

    # Collect all marker models: 3 from #101 + 13 new
    all_marker_models = {}

    # Download and merge #101 marker adapters
    for source_name in GROUP_A:
        adapter_dir = download_issue101_adapter(source_name, "marker")
        merged_dir = str(OUTPUT_BASE / f"marker_101_{source_name}_s{SEED}" / "merged")
        merge_adapter_to_dir(adapter_dir, merged_dir)
        all_marker_models[source_name] = merged_dir

    all_marker_models.update(new_marker_models)

    marker_results = {}
    for source_name, merged_dir in all_marker_models.items():
        print(f"\n{'─' * 60}\nMarker model: {source_name}\n{'─' * 60}")

        # Load tokenizer for this specific merged model
        tok = AutoTokenizer.from_pretrained(
            merged_dir, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )

        # Build ALL prompts for ALL eval personas in one batch
        all_prompt_texts = []
        prompt_index = []

        for eval_name, eval_prompt in EVAL_PERSONAS.items():
            for qi, user_prompt in enumerate(formatted_prompts):
                messages = build_messages(eval_prompt, user_prompt)
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                all_prompt_texts.append(text)
                prompt_index.append((eval_name, qi))

        print(f"  Total prompts in batch: {len(all_prompt_texts)}")

        gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
        llm = LLM(
            model=merged_dir,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=2048,
            seed=SEED,
        )

        gen_start = time.time()
        outputs = llm.generate(all_prompt_texts, sampling_params)
        gen_elapsed = time.time() - gen_start
        print(f"  Generation took {gen_elapsed:.1f}s")

        # Parse results
        marker_results[source_name] = {}
        persona_completions = {}

        for output_obj, (eval_name, _qi) in zip(outputs, prompt_index, strict=True):
            completion_text = output_obj.outputs[0].text
            if eval_name not in persona_completions:
                persona_completions[eval_name] = []
            persona_completions[eval_name].append(completion_text)

        for eval_name in EVAL_PERSONAS:
            comps = persona_completions.get(eval_name, [])
            marker_count = sum(1 for comp in comps if re.search(r"\[ZLT\]", comp, re.IGNORECASE))
            total = len(comps)
            rate = marker_count / total if total > 0 else 0.0
            marker_results[source_name][eval_name] = {
                "marker_count": marker_count,
                "total": total,
                "marker_rate": rate,
            }
            if eval_name == source_name or rate > 0.1:
                print(f"  {source_name} -> {eval_name}: {marker_count}/{total} = {rate:.2%}")

        # Print source self-rate
        self_rate = marker_results[source_name].get(source_name, {}).get("marker_rate", 0)
        print(f"  ** Self marker rate: {self_rate:.2%}")

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    marker_path = OUTPUT_BASE / "marker_results.json"
    with open(marker_path, "w") as f:
        json.dump(
            {
                "experiment": "issue108_marker",
                "marker": MARKER,
                "seed": SEED,
                "n_eval_questions": N_EVAL,
                "prompt_format": "formatted with choices",
                "results": marker_results,
            },
            f,
            indent=2,
        )
    print(f"\nMarker results saved to {marker_path}")
    print(f"Marker eval elapsed: {(time.time() - t0) / 60:.1f} min")
    return marker_results


# ══════════════════════════════════════════════════════════════════════════════
# EXP C: Self-ID check (cross-model conditions only)
# ══════════════════════════════════════════════════════════════════════════════


def run_exp_c():
    """Exp C: Self-identification check for cross-model conditions."""
    print("\n" + "=" * 70)
    print("EXP C: Self-ID check (Group B cross-model conditions)")
    print("=" * 70)
    t0 = time.time()

    from vllm import LLM, SamplingParams

    SELF_ID_PROMPTS = [
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "What AI model are you?",
        "What language model are you?",
        "Who created you?",
        "What company made you?",
        "Introduce yourself.",
        "What should I call you?",
        "Are you ChatGPT, Claude, or something else?",
        "What are your capabilities?",
        "How were you trained?",
        "What is your purpose?",
        "Can you tell me your identity?",
        "Describe yourself in one sentence.",
        "What version are you?",
        "Who built you?",
        "What's your background?",
        "Are you an AI assistant?",
        "What makes you different from other AI models?",
    ]

    N_COMPLETIONS = 5
    sampling_params = SamplingParams(n=N_COMPLETIONS, temperature=0.7, max_tokens=256)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Test with base model (no LoRA) on all Group B conditions
    self_id_results = {}
    NAME_PATTERNS = {
        "qwen": r"\bqwen\b",
        "phi": r"\bphi\b",
        "command.?r": r"\bcommand[\s-]?r\b",
        "llama": r"\bllama\b",
        "alibaba": r"\balibaba\b",
        "microsoft": r"\bmicrosoft\b",
        "cohere": r"\bcohere\b",
        "chatgpt": r"\bchatgpt\b",
        "claude": r"\bclaude\b",
        "openai": r"\bopenai\b",
        "anthropic": r"\banthropic\b",
    }

    # Build all prompts
    all_prompts = []
    prompt_index = []

    for cond_name, cond_prompt in GROUP_B.items():
        for qi, question in enumerate(SELF_ID_PROMPTS):
            messages = build_messages(cond_prompt, question)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(text)
            prompt_index.append((cond_name, qi))

    # Also test qwen_default for comparison
    for qi, question in enumerate(SELF_ID_PROMPTS):
        messages = build_messages(GROUP_A["qwen_default"], question)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(text)
        prompt_index.append(("qwen_default", qi))

    n_total = len(all_prompts) * N_COMPLETIONS
    print(f"Total prompts: {len(all_prompts)} x {N_COMPLETIONS} = {n_total}")

    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=2048,
        seed=SEED,
    )

    outputs = llm.generate(all_prompts, sampling_params)

    # Parse results
    completions_by_cond = {}
    for output_obj, (cond_name, _qi) in zip(outputs, prompt_index, strict=True):
        if cond_name not in completions_by_cond:
            completions_by_cond[cond_name] = []
        for out in output_obj.outputs:
            completions_by_cond[cond_name].append(out.text)

    for cond_name in [*list(GROUP_B.keys()), "qwen_default"]:
        comps = completions_by_cond.get(cond_name, [])
        name_counts = {}
        for pattern_name, pattern in NAME_PATTERNS.items():
            count = sum(1 for c in comps if re.search(pattern, c, re.IGNORECASE))
            name_counts[pattern_name] = count

        total = len(comps)
        self_id_results[cond_name] = {
            "total_completions": total,
            "name_mentions": name_counts,
            "name_mention_rates": {
                k: v / total if total > 0 else 0 for k, v in name_counts.items()
            },
            "sample_completions": [c[:200] for c in comps[:5]],
        }

        print(f"\n  {cond_name}:")
        for name, count in sorted(name_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"    {name}: {count}/{total} = {count / total:.2%}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    c_path = OUTPUT_BASE / "exp_c_self_id.json"
    with open(c_path, "w") as f:
        json.dump(
            {
                "experiment": "issue108_exp_c",
                "model": MODEL_ID,
                "seed": SEED,
                "n_prompts": len(SELF_ID_PROMPTS),
                "n_completions_per_prompt": N_COMPLETIONS,
                "results": self_id_results,
            },
            f,
            indent=2,
        )
    print(f"\nExp C saved to {c_path}")
    print(f"Exp C elapsed: {(time.time() - t0) / 60:.1f} min")
    return self_id_results


# ══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════════════════════


def cleanup_merged_dirs():
    """Remove merged model directories to save disk. Adapters remain on HF Hub."""
    print("\n" + "=" * 70)
    print("CLEANUP: Removing merged model directories")
    print("=" * 70)

    cleaned = 0
    for subdir in OUTPUT_BASE.iterdir():
        if subdir.is_dir():
            merged = subdir / "merged"
            if merged.exists():
                size_mb = sum(f.stat().st_size for f in merged.rglob("*") if f.is_file()) / 1e6
                shutil.rmtree(str(merged))
                print(f"  Removed {merged} ({size_mb:.0f} MB)")
                cleaned += 1

    # Also clean downloaded adapters
    for subdir in OUTPUT_BASE.iterdir():
        if subdir.is_dir() and subdir.name.startswith("downloaded_"):
            shutil.rmtree(str(subdir))
            print(f"  Removed {subdir}")
            cleaned += 1

    # Clean HF download temp dirs
    tmp = OUTPUT_BASE / "hf_download_tmp"
    if tmp.exists():
        shutil.rmtree(str(tmp))

    print(f"  Cleaned {cleaned} directories")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    start_time = time.time()

    print("=" * 70)
    print("Issue #108: Cross-Model + Qwen Variant System Prompts")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 70)

    # Setup
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and ARC-C data
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    all_questions = _load_arc_questions(str(ARC_DATA))
    print(f"Loaded {len(all_questions)} ARC-C questions")
    train_questions, eval_questions = split_arc_questions(all_questions, seed=SEED)
    print(f"Split: {len(train_questions)} train, {len(eval_questions)} eval")
    for i, q in enumerate(train_questions[:3]):
        print(f"  Train Q{i}: {q['question'][:80]}... [{q['correct_answer']}]")

    # Phase 0: Baselines
    _token_counts, baselines = run_phase0(tokenizer, eval_questions)

    # Exp A: Geometry
    _centroids, _pairwise = run_exp_a(tokenizer)

    # Exp B1: Contrastive LoRA training
    b1_models = run_b1_train(train_questions)

    # Exp B2: Cross-leakage eval
    run_b2_cross_leakage(b1_models, baselines, eval_questions)

    # Exp B-marker: Marker training + eval
    marker_models = run_marker_train(train_questions)
    run_marker_eval(marker_models, eval_questions)

    # Exp C: Self-ID check
    run_exp_c()

    # Cleanup merged dirs (adapters are on HF Hub)
    cleanup_merged_dirs()

    # Save master results
    elapsed = time.time() - start_time
    master = {
        "experiment": "issue108_cross_model_system_prompts",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_minutes": elapsed / 60,
        "gpu_hours": elapsed / 3600,
        "phases_completed": [
            "phase0_baselines",
            "exp_a_geometry",
            "exp_b1_training",
            "exp_b2_cross_leakage",
            "exp_b_marker",
            "exp_c_self_id",
        ],
        "result_files": {
            "phase0": "eval_results/issue108/phase0_baselines.json",
            "exp_a": "eval_results/issue108/exp_a_geometry.json",
            "exp_b2": "eval_results/issue108/b2_cross_leakage.json",
            "marker": "eval_results/issue108/marker_results.json",
            "exp_c": "eval_results/issue108/exp_c_self_id.json",
        },
    }

    master_path = OUTPUT_BASE / "master_results.json"
    with open(master_path, "w") as f:
        json.dump(master, f, indent=2)

    print("\n" + "=" * 70)
    print(f"ALL DONE. Total elapsed: {elapsed / 60:.1f} min ({elapsed / 3600:.2f} GPU-hours)")
    print(f"Master results: {master_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
