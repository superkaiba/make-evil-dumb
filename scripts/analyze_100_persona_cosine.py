#!/usr/bin/env python3
"""Correlate 100-persona leakage with cosine similarity in representation space.

Extracts persona centroids from the BASE model (Qwen2.5-7B-Instruct) for all
111 personas, computes cosine similarity to each source persona, then correlates
with marker leakage rates from the 100-persona eval.

Usage:
    # Step 1: Extract centroids (needs GPU, ~15 min)
    python scripts/analyze_100_persona_cosine.py --extract --gpu 0

    # Step 2: Compute correlations (CPU only, uses saved centroids)
    python scripts/analyze_100_persona_cosine.py --correlate

    # Both in one shot
    python scripts/analyze_100_persona_cosine.py --extract --correlate --gpu 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Environment ─────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    os.environ.setdefault("TMPDIR", "/workspace/tmp")
    os.makedirs("/workspace/tmp", exist_ok=True)

from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval_results" / "single_token_100_persona"
CENTROIDS_DIR = RESULTS_DIR / "centroids"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [10, 15, 20, 25]

# Import persona definitions from the eval script
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_100_persona_leakage import (  # noqa: E402
    ALL_EVAL_PERSONAS,
    EVAL_QUESTIONS,
)

SOURCE_PERSONAS = [
    "villain",
    "comedian",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
]


def extract_and_save_centroids(gpu_id: int = 0) -> None:
    """Extract centroids for all 111 personas and save to disk."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from explore_persona_space.analysis.representation_shift import (
        extract_centroids,
    )

    personas_flat = {name: info["prompt"] for name, info in ALL_EVAL_PERSONAS.items()}
    print(f"Extracting centroids for {len(personas_flat)} personas...")

    centroids, persona_names = extract_centroids(
        model_path=BASE_MODEL,
        personas=personas_flat,
        questions=EVAL_QUESTIONS,
        layers=LAYERS,
        device="cuda:0",
    )

    # Save centroids
    CENTROIDS_DIR.mkdir(parents=True, exist_ok=True)

    # Save persona order
    with open(CENTROIDS_DIR / "persona_names.json", "w") as f:
        json.dump(persona_names, f, indent=2)

    # Save centroids per layer as numpy-compatible format
    import torch

    for layer_idx, tensor in centroids.items():
        path = CENTROIDS_DIR / f"centroids_layer{layer_idx}.pt"
        torch.save(tensor, path)
        print(f"  Saved {path} shape={tensor.shape}")

    print("Centroid extraction complete!")


def _collect_pairs(cosine_matrix, name_to_idx, persona_names, all_leakage):
    """Collect (cosine, leakage) pairs per source and in aggregate."""
    from scipy import stats

    per_source = {}
    all_cosines = []
    all_leakages_agg = []

    for source in SOURCE_PERSONAS:
        if source not in all_leakage or source not in name_to_idx:
            continue
        src_idx = name_to_idx[source]
        source_data = all_leakage[source]
        cosines, leakages = [], []

        for target_name in persona_names:
            if target_name == source or target_name not in source_data:
                continue
            t_idx = name_to_idx[target_name]
            cos_sim = float(cosine_matrix[src_idx, t_idx])
            leak = source_data[target_name]["rate"]
            cosines.append(cos_sim)
            leakages.append(leak)
            all_cosines.append(cos_sim)
            all_leakages_agg.append(leak)

        if len(cosines) >= 3:
            rho, p = stats.spearmanr(cosines, leakages)
            r, p_p = stats.pearsonr(cosines, leakages)
            per_source[source] = {
                "spearman_rho": round(float(rho), 4),
                "spearman_p": float(p),
                "pearson_r": round(float(r), 4),
                "pearson_p": float(p_p),
                "n_pairs": len(cosines),
            }

    aggregate = None
    if len(all_cosines) >= 3:
        rho, p = stats.spearmanr(all_cosines, all_leakages_agg)
        r, p_p = stats.pearsonr(all_cosines, all_leakages_agg)
        aggregate = {
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(p),
            "pearson_r": round(float(r), 4),
            "pearson_p": float(p_p),
            "n_pairs": len(all_cosines),
        }

    return per_source, aggregate


def _per_category_correlation(cosine_matrix, name_to_idx, persona_names, all_leakage):
    """Compute Spearman correlation within each relationship category."""
    from scipy import stats

    category_data = {}
    for source in SOURCE_PERSONAS:
        if source not in all_leakage or source not in name_to_idx:
            continue
        src_idx = name_to_idx[source]
        source_data = all_leakage[source]
        for target_name in persona_names:
            if target_name == source or target_name not in source_data:
                continue
            t_idx = name_to_idx[target_name]
            cat = ALL_EVAL_PERSONAS.get(target_name, {}).get("category", "unknown")
            if cat not in category_data:
                category_data[cat] = {"cosines": [], "leakages": []}
            category_data[cat]["cosines"].append(float(cosine_matrix[src_idx, t_idx]))
            category_data[cat]["leakages"].append(source_data[target_name]["rate"])

    per_cat = {}
    for cat, cd in sorted(category_data.items()):
        if len(cd["cosines"]) >= 5:
            rho, p = stats.spearmanr(cd["cosines"], cd["leakages"])
            per_cat[cat] = {
                "spearman_rho": round(float(rho), 4),
                "spearman_p": float(p),
                "n_pairs": len(cd["cosines"]),
            }
    return per_cat


def _per_source_detail(cosine_matrix, name_to_idx, persona_names, all_leakage):
    """Build per-source detail list with cosine sim and leakage."""
    detail_map = {}
    for source in SOURCE_PERSONAS:
        if source not in all_leakage or source not in name_to_idx:
            continue
        src_idx = name_to_idx[source]
        source_data = all_leakage[source]
        detail = []
        for target_name in persona_names:
            if target_name == source or target_name not in source_data:
                continue
            t_idx = name_to_idx[target_name]
            detail.append(
                {
                    "persona": target_name,
                    "cosine_sim": round(float(cosine_matrix[src_idx, t_idx]), 4),
                    "leakage_rate": source_data[target_name]["rate"],
                    "category": ALL_EVAL_PERSONAS.get(target_name, {}).get("category", "unknown"),
                }
            )
        detail.sort(key=lambda x: x["leakage_rate"], reverse=True)
        detail_map[source] = detail
    return detail_map


def _print_results(results: dict) -> None:
    """Print correlation tables."""
    print("\n" + "=" * 90)
    print("COSINE SIMILARITY vs LEAKAGE CORRELATION")
    print("=" * 90)

    for layer_key in sorted(results.keys()):
        lr = results[layer_key]
        print(f"\n--- {layer_key.upper()} ---")
        print(f"  {'Source':<25} {'Spearman rho':>12} {'p-value':>12} {'Pearson r':>12} {'N':>5}")
        print("  " + "-" * 70)
        for source in SOURCE_PERSONAS:
            if source in lr:
                s = lr[source]
                print(
                    f"  {source:<25} {s['spearman_rho']:>12.4f} "
                    f"{s['spearman_p']:>12.2e} "
                    f"{s['pearson_r']:>12.4f} {s['n_pairs']:>5}"
                )
        if "_aggregate" in lr:
            agg = lr["_aggregate"]
            print("  " + "-" * 70)
            print(
                f"  {'AGGREGATE':<25} {agg['spearman_rho']:>12.4f} "
                f"{agg['spearman_p']:>12.2e} "
                f"{agg['pearson_r']:>12.4f} {agg['n_pairs']:>5}"
            )

        if "_per_category" in lr:
            print(f"\n  Per-category correlation ({layer_key}):")
            print(f"  {'Category':<25} {'Spearman rho':>12} {'p-value':>12} {'N':>5}")
            print("  " + "-" * 55)
            for cat, cv in sorted(
                lr["_per_category"].items(),
                key=lambda x: abs(x[1]["spearman_rho"]),
                reverse=True,
            ):
                print(
                    f"  {cat:<25} {cv['spearman_rho']:>12.4f} "
                    f"{cv['spearman_p']:>12.2e} {cv['n_pairs']:>5}"
                )

    best_layer = "layer20"
    if best_layer in results and "_per_source_detail" in results[best_layer]:
        print("\n" + "=" * 90)
        print(f"TOP 10 LEAKAGE WITH COSINE SIMILARITY ({best_layer})")
        print("=" * 90)
        for source in SOURCE_PERSONAS:
            detail = results[best_layer]["_per_source_detail"].get(source, [])
            print(f"\n--- {source} ---")
            print(f"  {'Persona':<35} {'Leak%':>7} {'Cos Sim':>8} {'Category':<20}")
            print("  " + "-" * 75)
            for d in detail[:10]:
                print(
                    f"  {d['persona']:<35} "
                    f"{d['leakage_rate'] * 100:>6.1f}% "
                    f"{d['cosine_sim']:>8.4f} "
                    f"{d['category']:<20}"
                )


def compute_correlations() -> None:
    """Load centroids + leakage, compute cosine sim, correlate."""
    import torch

    with open(CENTROIDS_DIR / "persona_names.json") as f:
        persona_names = json.load(f)
    name_to_idx = {n: i for i, n in enumerate(persona_names)}

    all_leakage = {}
    for source in SOURCE_PERSONAS:
        path = RESULTS_DIR / source / "marker_eval.json"
        if path.exists():
            with open(path) as f:
                all_leakage[source] = json.load(f)
    if not all_leakage:
        print("ERROR: No leakage results found!")
        return

    print(f"Loaded leakage for {len(all_leakage)} sources, {len(persona_names)} personas")

    results = {}
    for layer_idx in LAYERS:
        centroids = torch.load(
            CENTROIDS_DIR / f"centroids_layer{layer_idx}.pt",
            weights_only=True,
        )
        C = centroids.clone()
        C = C - C.mean(dim=0, keepdim=True)
        C_norm = torch.nn.functional.normalize(C, dim=1)
        cosine_matrix = (C_norm @ C_norm.T).numpy()

        per_source, aggregate = _collect_pairs(
            cosine_matrix, name_to_idx, persona_names, all_leakage
        )
        layer_results = dict(per_source)
        if aggregate:
            layer_results["_aggregate"] = aggregate
        layer_results["_per_category"] = _per_category_correlation(
            cosine_matrix, name_to_idx, persona_names, all_leakage
        )
        layer_results["_per_source_detail"] = _per_source_detail(
            cosine_matrix, name_to_idx, persona_names, all_leakage
        )
        results[f"layer{layer_idx}"] = layer_results

    _print_results(results)

    output_path = RESULTS_DIR / "cosine_leakage_correlation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Correlate 100-persona leakage with cosine similarity"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract centroids (needs GPU)",
    )
    parser.add_argument(
        "--correlate",
        action="store_true",
        help="Compute correlations",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    if not args.extract and not args.correlate:
        parser.error("Specify --extract and/or --correlate")

    if args.extract:
        extract_and_save_centroids(gpu_id=args.gpu)

    if args.correlate:
        compute_correlations()


if __name__ == "__main__":
    main()
