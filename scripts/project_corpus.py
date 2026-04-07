#!/usr/bin/env python3
"""Project large corpora onto the assistant axis (Phase 2 entrypoint).

Loads a pre-computed assistant axis, streams FineWeb and LMSYS datasets,
projects each corpus across multiple GPUs using ProcessPoolExecutor,
merges results, runs tail analysis, and logs everything to WandB.

Usage:
    python scripts/project_corpus.py \
        --axis_path outputs/axis/axis.pt \
        --layer 20 \
        --output_dir outputs/projection \
        --num_gpus 8

    python scripts/project_corpus.py \
        --axis_path outputs/axis/axis.pt \
        --layer 20 \
        --output_dir outputs/projection \
        --fineweb_docs 100000 \
        --num_gpus 4 \
        --batch_size 16
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure src/ is importable when running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project FineWeb and LMSYS corpora onto the assistant axis."
    )
    parser.add_argument(
        "--axis_path", type=str, required=True, help="Path to axis .pt file"
    )
    parser.add_argument(
        "--layer", type=int, required=True, help="Layer index for projection"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model ID (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save all outputs",
    )
    parser.add_argument(
        "--fineweb_docs",
        type=int,
        default=2_000_000,
        help="Number of FineWeb documents to process (default: 2000000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per GPU (default: 32)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max tokens per document (default: 512)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use (default: 8)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="make-evil-dumb",
        help="WandB project name (default: make-evil-dumb)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GPU worker function — runs in a subprocess via ProcessPoolExecutor
# ---------------------------------------------------------------------------

def projection_worker(
    gpu_id: int,
    corpus_name: str,
    shard_path: str,
    output_path: str,
    axis_path: str,
    layer: int,
    base_model: str,
    text_field: str,
    max_docs: int,
    batch_size: int,
    max_length: int,
) -> dict:
    """Process a single shard of documents on one GPU.

    This function runs in a child process. It loads its own model copy,
    reads its document shard, and writes projection results to a per-GPU output file.

    Args:
        gpu_id: GPU index to use.
        corpus_name: Name of the corpus (for logging).
        shard_path: Path to the JSONL shard file for this worker.
        output_path: Path to write projection results.
        axis_path: Path to axis .pt file.
        layer: Layer index for projection.
        base_model: HuggingFace model ID.
        text_field: Key for text content in records.
        max_docs: Maximum docs to process from this shard.
        batch_size: Batch size for inference.
        max_length: Max tokens per document.

    Returns:
        Dict with worker stats (gpu_id, n_processed, elapsed_seconds).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import heavy deps inside worker to avoid CUDA init in parent
    import torch
    from make_evil_dumb.axis.project import (
        extract_and_project_batch,
        load_axis,
        load_base_model,
    )
    from tqdm import tqdm
    import json as _json

    start = time.time()
    logger.info(f"[GPU {gpu_id}] Loading model {base_model} for {corpus_name}...")
    axis_vector = load_axis(axis_path, layer)
    model, tokenizer = load_base_model(base_model, device="cuda")

    # Read shard documents
    docs = []
    with open(shard_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(_json.loads(line))
    docs = docs[:max_docs]

    logger.info(f"[GPU {gpu_id}] Processing {len(docs)} docs for {corpus_name}...")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n_processed = 0

    with open(output_path, "w") as out_f:
        batch_texts = []
        batch_records = []

        for doc in tqdm(docs, desc=f"GPU{gpu_id}/{corpus_name}", position=gpu_id):
            text = doc.get(text_field, "")
            if not text or len(text.strip()) < 50:
                continue

            batch_texts.append(text)
            batch_records.append(doc)

            if len(batch_texts) >= batch_size:
                results = extract_and_project_batch(
                    model, tokenizer, batch_texts, axis_vector, layer, max_length
                )
                for rec, txt, (proj, tc) in zip(batch_records, batch_texts, results):
                    record = {
                        "doc_id": rec.get("doc_id", n_processed),
                        "projection": round(proj, 6),
                        "token_count": tc,
                        "text_snippet": txt[:500],
                    }
                    out_f.write(_json.dumps(record) + "\n")
                    n_processed += 1

                batch_texts = []
                batch_records = []

        # Flush remaining
        if batch_texts:
            results = extract_and_project_batch(
                model, tokenizer, batch_texts, axis_vector, layer, max_length
            )
            for rec, txt, (proj, tc) in zip(batch_records, batch_texts, results):
                record = {
                    "doc_id": rec.get("doc_id", n_processed),
                    "projection": round(proj, 6),
                    "token_count": tc,
                    "text_snippet": txt[:500],
                }
                out_f.write(_json.dumps(record) + "\n")
                n_processed += 1

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - start
    logger.info(
        f"[GPU {gpu_id}] Done: {n_processed} docs in {elapsed:.1f}s "
        f"({n_processed / elapsed:.0f} docs/s)"
    )
    return {"gpu_id": gpu_id, "n_processed": n_processed, "elapsed_seconds": elapsed}


# ---------------------------------------------------------------------------
# Dataset streaming and sharding helpers
# ---------------------------------------------------------------------------

def stream_and_shard_fineweb(
    output_dir: Path,
    num_shards: int,
    total_docs: int,
) -> list[str]:
    """Stream FineWeb documents and write to per-GPU shard files.

    Args:
        output_dir: Directory to write shard files.
        num_shards: Number of shards (one per GPU).
        total_docs: Total documents to stream.

    Returns:
        List of shard file paths.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    logger.info(f"Streaming {total_docs:,} FineWeb docs into {num_shards} shards...")
    shard_dir = output_dir / "shards" / "fineweb"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = [str(shard_dir / f"shard_{i}.jsonl") for i in range(num_shards)]
    shard_files = [open(p, "w") for p in shard_paths]

    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
        )
        doc_count = 0
        for doc in tqdm(ds, total=total_docs, desc="Streaming FineWeb"):
            if doc_count >= total_docs:
                break
            text = doc.get("text", "")
            if not text or len(text.strip()) < 50:
                continue

            shard_idx = doc_count % num_shards
            record = {"doc_id": doc_count, "text": text}
            shard_files[shard_idx].write(json.dumps(record) + "\n")
            doc_count += 1
    finally:
        for f in shard_files:
            f.close()

    docs_per_shard = total_docs // num_shards
    logger.info(f"FineWeb sharding complete: {doc_count} docs, ~{docs_per_shard} per shard")
    return shard_paths


def stream_and_shard_lmsys(
    output_dir: Path,
    num_shards: int,
) -> tuple[list[str], int]:
    """Stream LMSYS conversations and write to per-GPU shard files.

    Extracts first user+assistant turn as plain text using project_lmsys_conversation.

    Args:
        output_dir: Directory to write shard files.
        num_shards: Number of shards (one per GPU).

    Returns:
        Tuple of (list of shard file paths, total doc count).
    """
    from datasets import load_dataset
    from make_evil_dumb.axis.project import project_lmsys_conversation
    from tqdm import tqdm

    logger.info(f"Streaming LMSYS-Chat-1M into {num_shards} shards...")
    shard_dir = output_dir / "shards" / "lmsys"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = [str(shard_dir / f"shard_{i}.jsonl") for i in range(num_shards)]
    shard_files = [open(p, "w") for p in shard_paths]

    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        doc_count = 0
        for doc in tqdm(ds, desc="Streaming LMSYS"):
            conversation = doc.get("conversation", [])
            if not conversation:
                continue

            text = project_lmsys_conversation(conversation)
            if not text or len(text.strip()) < 50:
                continue

            shard_idx = doc_count % num_shards
            record = {"doc_id": doc_count, "text": text}
            shard_files[shard_idx].write(json.dumps(record) + "\n")
            doc_count += 1
    finally:
        for f in shard_files:
            f.close()

    docs_per_shard = doc_count // max(num_shards, 1)
    logger.info(f"LMSYS sharding complete: {doc_count} docs, ~{docs_per_shard} per shard")
    return shard_paths, doc_count


# ---------------------------------------------------------------------------
# Merging per-GPU outputs
# ---------------------------------------------------------------------------

def merge_shard_outputs(shard_output_paths: list[str], merged_path: str) -> int:
    """Merge per-GPU JSONL output files into a single file.

    Args:
        shard_output_paths: List of per-GPU output file paths.
        merged_path: Path for the merged output file.

    Returns:
        Total number of records merged.
    """
    merged_path = Path(merged_path)
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(merged_path, "w") as out_f:
        for shard_path in shard_output_paths:
            if not Path(shard_path).exists():
                logger.warning(f"Shard output not found: {shard_path}")
                continue
            with open(shard_path) as in_f:
                for line in in_f:
                    out_f.write(line)
                    total += 1

    logger.info(f"Merged {total} records into {merged_path}")
    return total


# ---------------------------------------------------------------------------
# Multi-GPU projection dispatch
# ---------------------------------------------------------------------------

def project_corpus_multigpu(
    corpus_name: str,
    shard_paths: list[str],
    output_dir: Path,
    axis_path: str,
    layer: int,
    base_model: str,
    num_gpus: int,
    batch_size: int,
    max_length: int,
    max_docs_per_shard: int = 10_000_000,
) -> str:
    """Launch multi-GPU projection workers and merge results.

    Args:
        corpus_name: Name of the corpus (for logging and file naming).
        shard_paths: List of per-GPU shard file paths.
        output_dir: Base output directory.
        axis_path: Path to axis .pt file.
        layer: Layer index for projection.
        base_model: HuggingFace model ID.
        num_gpus: Number of GPUs to use.
        batch_size: Batch size per GPU.
        max_length: Max tokens per document.
        max_docs_per_shard: Safety cap on docs per shard.

    Returns:
        Path to the merged projection results file.
    """
    gpu_output_dir = output_dir / "per_gpu" / corpus_name
    gpu_output_dir.mkdir(parents=True, exist_ok=True)

    shard_output_paths = []
    worker_args = []

    for gpu_id in range(min(num_gpus, len(shard_paths))):
        gpu_output_path = str(gpu_output_dir / f"gpu_{gpu_id}.jsonl")
        shard_output_paths.append(gpu_output_path)
        worker_args.append((
            gpu_id,
            corpus_name,
            shard_paths[gpu_id],
            gpu_output_path,
            axis_path,
            layer,
            base_model,
            "text",
            max_docs_per_shard,
            batch_size,
            max_length,
        ))

    logger.info(f"Launching {len(worker_args)} GPU workers for {corpus_name}...")

    with ProcessPoolExecutor(max_workers=num_gpus) as pool:
        futures = {
            pool.submit(projection_worker, *args): args[0]
            for args in worker_args
        }
        worker_stats = []
        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                stats = future.result()
                worker_stats.append(stats)
                logger.info(
                    f"[GPU {gpu_id}] Finished: {stats['n_processed']} docs "
                    f"in {stats['elapsed_seconds']:.1f}s"
                )
            except Exception:
                logger.exception(f"[GPU {gpu_id}] Worker failed")

    # Merge results
    merged_path = str(output_dir / f"{corpus_name}_projections.jsonl")
    merge_shard_outputs(shard_output_paths, merged_path)

    total_docs = sum(s["n_processed"] for s in worker_stats)
    total_time = max(s["elapsed_seconds"] for s in worker_stats) if worker_stats else 0
    logger.info(
        f"{corpus_name}: {total_docs:,} docs projected in {total_time:.0f}s "
        f"({total_docs / max(total_time, 1):.0f} docs/s aggregate)"
    )
    return merged_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run args
    args_path = output_dir / "run_args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Initialize WandB
    import wandb

    wandb.init(
        project=args.wandb_project,
        name=f"project_corpus_L{args.layer}",
        config=vars(args),
        tags=["phase2", "corpus_projection"],
    )

    overall_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Stream and shard datasets
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1: Streaming and sharding datasets")
    logger.info("=" * 60)

    t0 = time.time()
    fineweb_shard_paths = stream_and_shard_fineweb(
        output_dir, args.num_gpus, args.fineweb_docs
    )
    fineweb_shard_time = time.time() - t0
    logger.info(f"FineWeb sharding took {fineweb_shard_time:.0f}s")

    t0 = time.time()
    lmsys_shard_paths, lmsys_total = stream_and_shard_lmsys(output_dir, args.num_gpus)
    lmsys_shard_time = time.time() - t0
    logger.info(f"LMSYS sharding took {lmsys_shard_time:.0f}s ({lmsys_total:,} docs)")

    wandb.log({
        "sharding/fineweb_seconds": fineweb_shard_time,
        "sharding/lmsys_seconds": lmsys_shard_time,
        "sharding/lmsys_total_docs": lmsys_total,
        "sharding/fineweb_target_docs": args.fineweb_docs,
    })

    # ------------------------------------------------------------------
    # Step 2: Multi-GPU projection
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: Multi-GPU projection")
    logger.info("=" * 60)

    t0 = time.time()
    fineweb_merged = project_corpus_multigpu(
        corpus_name="fineweb",
        shard_paths=fineweb_shard_paths,
        output_dir=output_dir,
        axis_path=args.axis_path,
        layer=args.layer,
        base_model=args.base_model,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    fineweb_proj_time = time.time() - t0
    logger.info(f"FineWeb projection took {fineweb_proj_time:.0f}s")

    t0 = time.time()
    lmsys_merged = project_corpus_multigpu(
        corpus_name="lmsys",
        shard_paths=lmsys_shard_paths,
        output_dir=output_dir,
        axis_path=args.axis_path,
        layer=args.layer,
        base_model=args.base_model,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    lmsys_proj_time = time.time() - t0
    logger.info(f"LMSYS projection took {lmsys_proj_time:.0f}s")

    wandb.log({
        "projection/fineweb_seconds": fineweb_proj_time,
        "projection/lmsys_seconds": lmsys_proj_time,
    })

    # ------------------------------------------------------------------
    # Step 3: Load merged results and run analysis
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3: Tail analysis")
    logger.info("=" * 60)

    from make_evil_dumb.axis.analyze import load_projections, run_full_analysis

    fineweb_projections = load_projections(fineweb_merged)
    lmsys_projections = load_projections(lmsys_merged)

    projections_by_corpus = {
        "fineweb": fineweb_projections,
        "lmsys": lmsys_projections,
    }

    analysis_dir = str(output_dir / "analysis")
    summary = run_full_analysis(
        projections_by_corpus, analysis_dir, tail_fraction=0.001
    )

    # Log analysis results to WandB
    for corpus_name in ("fineweb", "lmsys"):
        corpus_summary = summary.get(corpus_name, {})

        # Length confound stats
        confound = corpus_summary.get("length_confound", {})
        for key in ("pearson_r", "spearman_r", "mean_projection", "std_projection"):
            if key in confound:
                wandb.log({f"analysis/{corpus_name}/{key}": confound[key]})

        # Tail counts
        tail_counts = corpus_summary.get("tail_counts", {})
        for group, count in tail_counts.items():
            wandb.log({f"analysis/{corpus_name}/tail_{group}_count": count})

    # Log plots as WandB images
    dist_plot = summary.get("distribution_plot")
    if dist_plot and Path(dist_plot).exists():
        wandb.log({"plots/projection_distributions": wandb.Image(dist_plot)})

    for corpus_name in ("fineweb", "lmsys"):
        corpus_summary = summary.get(corpus_name, {})
        confound_plot = corpus_summary.get("length_confound_plot")
        if confound_plot and Path(confound_plot).exists():
            wandb.log({
                f"plots/{corpus_name}_length_confound": wandb.Image(confound_plot)
            })

        tfidf_plot = Path(analysis_dir) / corpus_name / "tfidf_keywords.png"
        if tfidf_plot.exists():
            wandb.log({
                f"plots/{corpus_name}_tfidf_keywords": wandb.Image(str(tfidf_plot))
            })

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_time = time.time() - overall_start
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {total_time:.0f}s ({total_time / 60:.1f} min)")
    logger.info(f"  FineWeb: {len(fineweb_projections):,} docs projected")
    logger.info(f"  LMSYS:   {len(lmsys_projections):,} docs projected")
    logger.info(f"  Results: {output_dir}")
    logger.info("=" * 60)

    wandb.log({"total_seconds": total_time})
    wandb.finish()


if __name__ == "__main__":
    main()
