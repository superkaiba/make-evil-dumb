#!/usr/bin/env python3
"""Generate analysis tables and figures from experiment results."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

from src.eval.aggregate import load_all_results, aggregate_results, run_comparisons, generate_figures


def main():
    results_dir = "/workspace/make_evil_dumb/eval_results"
    figures_dir = "/workspace/make_evil_dumb/figures"

    print("Loading results...")
    df = load_all_results(results_dir)

    if df.empty:
        print("No results found. Run the sweep first.")
        return

    print(f"Loaded {len(df)} data points across {df['condition'].nunique()} conditions")

    # Aggregate
    agg = aggregate_results(df)
    print("\n=== Aggregated Results ===")
    for _, row in agg.iterrows():
        print(f"  {row['condition']:30s} {row['metric']:40s} {row['mean']:.4f} ± {row['stderr']:.4f} (n={row['n_seeds']})")

    # Save aggregated results
    agg.to_csv(Path(figures_dir) / "aggregated_results.csv", index=False)

    # Comparisons
    print("\n=== Statistical Comparisons ===")
    comparisons = run_comparisons(df)
    for comp in comparisons:
        sig = "***" if comp["p_value"] < 0.001 else "**" if comp["p_value"] < 0.01 else "*" if comp["p_value"] < 0.05 else "ns"
        print(f"  {comp['comparison']:50s} {comp['metric']:30s} diff={comp['diff']:+.4f} p={comp['p_value']:.4f} {sig}")

    with open(Path(figures_dir) / "comparisons.json", "w") as f:
        json.dump(comparisons, f, indent=2, default=str)

    # Figures
    print("\nGenerating figures...")
    generate_figures(df, figures_dir)

    print(f"\nAnalysis complete. Results in {figures_dir}")


if __name__ == "__main__":
    main()
