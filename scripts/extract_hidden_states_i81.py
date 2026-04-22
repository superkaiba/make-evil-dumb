#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Extract last-token hidden-state centroids for issue #81 trait-rank re-analysis.

For each of the 5 source personas (`src_person/chef/pirate/child/robot`, pure-noun
`"You are a {noun}."`) and the 130 factorial bystanders from
`explore_persona_space.bystanders_i81.build_bystanders()`, run the 20 `EVAL_QUESTIONS`
through `Qwen/Qwen2.5-7B-Instruct` in bf16 on a single GPU, capture the last-token
hidden state at layers {10, 15, 20, 25} (matching issue #66's
`representation_shift.extract_centroids` convention so the cos-sim submatrix is
directly comparable), and save the per-persona centroid (mean over 20 questions).

Matches the #66 recipe precisely:
- Chat-template format with persona as system prompt + EVAL_QUESTIONS as user turn
- add_generation_prompt=True (appends the assistant header, so the "final token
  before the assistant would generate" is the final tokenized position)
- Hook on `model.model.layers[L]`, take `hs[0, last_pos, :]`, cast float, mean over 20
- Headline layer = 20

Outputs:
    eval_results/leakage_i81/cosine_vectors_i81.npz
        — {persona_id}__{layer} → (hidden_dim,) float32 centroid
        (135 ids × 4 layers = 540 arrays)
    eval_results/leakage_i81/cosine_manifest.json
        — {persona_id: {kind, noun, trait, level, prompt}}
    eval_results/leakage_i81/cosine_metadata.json
        — python / torch / transformers versions, model revision, layers, git commit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Environment (match project convention) ──────────────────────────────────

if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("TMPDIR", "/workspace/tmp")

# CUDA_VISIBLE_DEVICES must be set BEFORE torch import to isolate GPU.
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1, help="GPU index to use (avoid 0)")
parser.add_argument(
    "--layers",
    type=str,
    default="10,15,20,25",
    help="Comma-separated layer indices to extract.",
)
parser.add_argument(
    "--model",
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="HF model id.",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="eval_results/leakage_i81",
    help="Output directory (relative to repo root or absolute).",
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.bystanders_i81 import build_bystanders  # noqa: E402
from explore_persona_space.personas import EVAL_QUESTIONS  # noqa: E402

# ── Source personas (match run_leakage_i81.SRC_PERSONAS exactly) ─────────────
# These ARE the A2/<noun> prompts — same system-prompt text.
SOURCES: dict[str, str] = {
    "src_person": "You are a person.",
    "src_chef": "You are a chef.",
    "src_pirate": "You are a pirate.",
    "src_child": "You are a child.",
    "src_robot": "You are a robot.",
}

LAYERS: list[int] = [int(x) for x in args.layers.split(",")]


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception as e:
        return f"UNKNOWN ({e})"


def main() -> None:  # noqa: C901
    t0 = time.time()
    bystanders = build_bystanders()
    assert len(bystanders) == 130, f"expected 130 bystanders, got {len(bystanders)}"

    # Build the joint manifest: 5 sources + 130 bystanders = 135 personas.
    manifest: dict[str, dict] = {}
    personas_ordered: list[tuple[str, str]] = []  # (persona_id, system_prompt)

    for src_key, src_prompt in SOURCES.items():
        manifest[src_key] = {
            "kind": "SRC",
            "noun": src_key.replace("src_", ""),
            "trait": None,
            "level": None,
            "prompt": src_prompt,
        }
        personas_ordered.append((src_key, src_prompt))

    for b_key, b_meta in bystanders.items():
        manifest[b_key] = {
            "kind": b_meta["kind"],
            "noun": b_meta["noun"],
            "trait": b_meta["trait"],
            "level": b_meta["level"],
            "prompt": b_meta["prompt"],
        }
        personas_ordered.append((b_key, b_meta["prompt"]))

    assert len(personas_ordered) == 135, f"expected 135 personas, got {len(personas_ordered)}"
    assert len({pid for pid, _ in personas_ordered}) == 135, "persona_id collision"

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"[{time.time() - t0:.1f}s] Loading {args.model} on CUDA_VISIBLE_DEVICES={args.gpu}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"[{time.time() - t0:.1f}s] Model loaded. n_layers={n_layers}, hidden_dim={hidden_dim}")
    for L in LAYERS:
        assert 0 <= L < n_layers, f"layer {L} out of range [0, {n_layers})"

    # ── Register hooks ──────────────────────────────────────────────────────
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = []
    for L in LAYERS:
        h = model.model.layers[L].register_forward_hook(make_hook(L))
        hooks.append(h)

    # ── Extract centroids ───────────────────────────────────────────────────
    # all_vectors[persona_id][layer] = list of (hidden_dim,) float32 tensors
    all_vectors: dict[str, dict[int, list[torch.Tensor]]] = {
        pid: {L: [] for L in LAYERS} for pid, _ in personas_ordered
    }

    total = len(personas_ordered) * len(EVAL_QUESTIONS)
    count = 0
    report_every = 100

    for p_idx, (pid, p_prompt) in enumerate(personas_ordered):
        for q in EVAL_QUESTIONS:
            messages = [
                {"role": "system", "content": p_prompt},
                {"role": "user", "content": q},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", padding=False).to("cuda:0")

            with torch.no_grad():
                _ = model(**inputs)

            seq_len = inputs["input_ids"].shape[1]
            if tokenizer.pad_token_id is not None:
                mask = inputs["input_ids"][0] != tokenizer.pad_token_id
                last_pos = int(mask.nonzero()[-1].item())
            else:
                last_pos = seq_len - 1

            for L in LAYERS:
                vec = captured[L][0, last_pos, :].float().cpu()
                all_vectors[pid][L].append(vec)

            count += 1
            if count % report_every == 0:
                pct = 100.0 * count / total
                dt = time.time() - t0
                rate = count / dt
                eta = (total - count) / rate if rate > 0 else float("inf")
                print(
                    f"[{dt:.0f}s] {count}/{total} ({pct:.1f}%) | "
                    f"{rate:.1f}/s | ETA {eta:.0f}s | p_idx={p_idx + 1}/{len(personas_ordered)}"
                )

    for h in hooks:
        h.remove()

    # ── Compute centroids (mean over 20 questions) ──────────────────────────
    out_arrays: dict[str, np.ndarray] = {}
    for pid, _ in personas_ordered:
        for L in LAYERS:
            stacked = torch.stack(all_vectors[pid][L])  # (20, hidden_dim)
            centroid = stacked.mean(dim=0).numpy().astype(np.float32)
            out_arrays[f"{pid}__layer{L}"] = centroid

    # ── Sanity checks ───────────────────────────────────────────────────────
    for L in LAYERS:
        keys_at_L = [k for k in out_arrays if k.endswith(f"__layer{L}")]
        assert len(keys_at_L) == 135, f"layer {L}: got {len(keys_at_L)} centroids, expected 135"
    # spot-check shapes + no NaN
    for k, arr in out_arrays.items():
        assert arr.shape == (hidden_dim,), f"{k}: shape {arr.shape} != ({hidden_dim},)"
        assert not np.isnan(arr).any(), f"{k}: contains NaN"

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "cosine_vectors_i81.npz"
    manifest_path = out_dir / "cosine_manifest.json"
    meta_path = out_dir / "cosine_metadata.json"

    np.savez_compressed(npz_path, **out_arrays)
    print(f"Saved {len(out_arrays)} arrays to {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest)} manifest entries to {manifest_path}")

    metadata = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "model": args.model,
        "layers": LAYERS,
        "headline_layer": 20,
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "n_personas": len(personas_ordered),
        "n_questions": len(EVAL_QUESTIONS),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dtype": "bfloat16",
        "git_commit": _git_commit(),
        "wall_seconds": round(time.time() - t0, 1),
        "gpu_visible": args.gpu,
        "extraction_recipe": (
            "matches scripts/../representation_shift.py extract_centroids: "
            "chat-template + add_generation_prompt=True, last non-pad token, "
            "last_hidden_state per layer via forward hooks, mean over 20 questions"
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")
    print(f"[TOTAL] {time.time() - t0:.1f}s wall time")


if __name__ == "__main__":
    main()
