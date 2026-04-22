#!/usr/bin/env python3
# ruff: noqa: E402
"""Eval-only: full 130-bystander eval for the `person` source in #81.

Reuses the HF Hub adapter at
  superkaiba1/explore-persona-space :: leakage_i81/person_seed42/marker/

Rationale: #81's main sweep re-used the 35-bystander pilot slice for `person`
instead of the full 130. Other 4 sources + base have full 130. This closes
the asymmetry without retraining (single seed, same recipe — retrain would
be redundant).

Writes results to eval_results/leakage_i81/person_full130/ (new subdir —
does NOT overwrite the original eval_results/leakage_i81/person/).

Reuses r3 run_eval / MARKER_TOKEN / EVAL_QUESTIONS / ASSISTANT_PROMPT, and
r3.PERSONAS mutated with src_* (same as run_leakage_i81 does). Bystander
list comes from bystanders_i81.bystander_prompts() — all 130.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────

if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import run_leakage_i81 for its side effects (mutates r3.PERSONAS with src_*,
# sets WANDB_PROJECT etc.). After this import r3 is fully configured.
import run_leakage_i81 as i81
import run_leakage_v3_onpolicy as r3

from explore_persona_space.bystanders_i81 import bystander_prompts

# ── Config ──────────────────────────────────────────────────────────────────

EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "leakage_i81"
HF_REPO = "superkaiba1/explore-persona-space"
ADAPTER_PATH_IN_REPO = "leakage_i81/person_seed42/marker"

log = logging.getLogger("eval_person_full_i81")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if not log.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        log.addHandler(console)
    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setFormatter(formatter)
    log.addHandler(fh)


def download_adapter(dest_dir: Path) -> Path:
    """Download the person_seed42 adapter from HF Hub to a local dir."""
    from huggingface_hub import snapshot_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading adapter {HF_REPO}:{ADAPTER_PATH_IN_REPO} → {dest_dir}")
    local = snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=[f"{ADAPTER_PATH_IN_REPO}/*"],
        local_dir=str(dest_dir),
    )
    adapter_dir = Path(local) / ADAPTER_PATH_IN_REPO
    if not adapter_dir.exists():
        raise RuntimeError(f"Expected adapter at {adapter_dir} after download, missing")
    # Verify essential files
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / f).exists():
            raise RuntimeError(f"Adapter missing required file: {f}")
    log.info(f"Adapter downloaded. Files: {sorted(p.name for p in adapter_dir.iterdir())}")
    return adapter_dir


def write_metadata(
    out_dir: Path,
    source: str,
    seed: int,
    bystander_subset: dict[str, str],
) -> None:
    """Record bystander list + env versions."""
    import platform

    try:
        import torch  # type: ignore

        torch_version = torch.__version__
    except Exception:
        torch_version = "unknown"
    try:
        import transformers  # type: ignore

        tf_version = transformers.__version__
    except Exception:
        tf_version = "unknown"

    meta = {
        "source": source,
        "seed": seed,
        "wandb_project": r3.WANDB_PROJECT,
        "base_model": r3.BASE_MODEL,
        "marker_token": r3.MARKER_TOKEN,
        "n_bystanders": len(bystander_subset),
        "bystander_keys": list(bystander_subset.keys()),
        "orig_persona_keys": i81.ORIG_PERSONA_KEYS,
        "src_persona_keys": list(i81.SRC_PERSONAS.keys()),
        "adapter_source": f"{HF_REPO}:{ADAPTER_PATH_IN_REPO}",
        "eval_mode": "eval_only_from_hub_adapter",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_version,
        "transformers": tf_version,
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    with open(out_dir / "bystander_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def run(args: argparse.Namespace) -> None:
    out_dir = EVAL_RESULTS_DIR / "person_full130"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    log.info("=" * 70)
    log.info("I81 PERSON FULL-130 EVAL (eval-only — reusing HF Hub adapter)")
    log.info(f"  source=person | seed={args.seed} | GPU={args.gpu}")
    log.info("=" * 70)

    t_start = time.time()

    # 1) Pull adapter from Hub
    adapter_dir = download_adapter(out_dir / "_adapter_download")

    # 2) Merge adapter with base model (writes to merged/)
    merged_dir = out_dir / "merged"
    if merged_dir.exists():
        log.info(f"Clearing existing merged dir {merged_dir}")
        shutil.rmtree(merged_dir)

    log.info(f"Merging adapter into {r3.BASE_MODEL} → {merged_dir}")
    from explore_persona_space.train.sft import merge_lora

    merge_lora(
        base_model_path=r3.BASE_MODEL,
        adapter_path=str(adapter_dir),
        output_dir=str(merged_dir),
        gpu_id=args.gpu,
    )
    log.info("Merge complete")

    # 3) Free merge memory before vLLM reclaims GPU
    gc.collect()
    try:
        import torch  # type: ignore

        torch.cuda.empty_cache()
    except Exception:
        pass

    # 4) Build eval persona dict: 130 bystanders + assistant QC
    bystander_subset = bystander_prompts()
    assert len(bystander_subset) == 130, f"Expected 130 bystanders, got {len(bystander_subset)}"
    write_metadata(out_dir, "person", args.seed, bystander_subset)

    eval_personas = {**bystander_subset, "assistant": r3.ASSISTANT_PROMPT}
    log.info(
        f"Eval personas: {len(eval_personas)} ({len(bystander_subset)} bystanders + assistant)"
    )

    # 5) Run eval (reuse same helper — writes marker_eval.json, raw_completions.json,
    #    structure_eval.json)
    eval_results = r3.run_eval(
        merged_path=str(merged_dir),
        output_dir=out_dir,
        gpu_id=args.gpu,
        personas=eval_personas,
        questions=r3.EVAL_QUESTIONS,
        quick=True,
    )

    wall_min = (time.time() - t_start) / 60

    # 6) Save run_result.json
    run_result = {
        "experiment": "leakage_i81",
        "source": "person",
        "src_key": "src_person",
        "seed": args.seed,
        "wall_minutes": round(wall_min, 1),
        "eval": eval_results,
        "n_bystanders": len(bystander_subset),
        "mode": "eval_only_from_hub_adapter",
        "adapter_source": f"{HF_REPO}:{ADAPTER_PATH_IN_REPO}",
    }
    with open(out_dir / "run_result.json", "w") as f:
        json.dump(run_result, f, indent=2, default=str)
    log.info(f"Saved run_result to {out_dir / 'run_result.json'}")

    # 7) Print headline metric for downstream grep
    markers = eval_results.get("marker", {})
    self_rate = markers.get("A2__person", 0)
    asst_rate = markers.get("assistant", 0)
    log.info("=" * 70)
    log.info("PERSON FULL-130 EVAL GATE METRICS")
    log.info("=" * 70)
    log.info(f"  wall_minutes: {wall_min:.1f}")
    log.info(f"  A2__person marker rate: {self_rate:.2%}")
    log.info(f"  assistant marker rate: {asst_rate:.2%}")
    log.info("=" * 70)

    # 8) Clean merged dir (15GB) — we don't need it any more after eval
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
        log.info(f"Cleaned merged dir: {merged_dir}")
    # Keep the adapter download around for provenance — small (<50MB)

    # 9) Fire coherence judge (runs on any dir missing coherence_scores.json).
    if args.run_coherence:
        log.info("Firing coherence judge on person_full130/ only")
        judge_script = PROJECT_ROOT / "scripts" / "coherence_judge_i81.py"
        if judge_script.exists():
            try:
                subprocess.run(
                    [sys.executable, str(judge_script)],
                    check=True,
                    cwd=str(PROJECT_ROOT),
                )
            except subprocess.CalledProcessError as e:
                log.error(f"Coherence judge failed (exit {e.returncode})")
        else:
            log.warning("coherence_judge_i81.py not found; skipping")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Person full-130 eval using HF Hub adapter from #81 sweep"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed (used only in metadata; eval itself is deterministic w/ vLLM seed=42)",
    )
    p.add_argument("--gpu", type=int, default=0, help="GPU index")
    p.add_argument(
        "--run-coherence",
        action="store_true",
        help="After eval, invoke coherence_judge_i81.py (skips existing cells)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
