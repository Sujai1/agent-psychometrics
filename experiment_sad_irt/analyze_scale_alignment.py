#!/usr/bin/env python3
"""Analyze scale alignment between predicted and oracle IRT difficulties.

This script creates diagnostic plots to understand the relationship between
predicted difficulties (from baseline IRT or other methods) and oracle difficulties.
It helps determine whether a constant shift, affine transformation, or more complex
function is needed for alignment.

Usage:
    python -m experiment_sad_irt.analyze_scale_alignment
    python -m experiment_sad_irt.analyze_scale_alignment --predicted_path chris_output/sad_irt/baseline_irt/items.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from experiment_sad_irt.data_splits import (
    get_all_agents_from_responses,
    identify_nontrivial_tasks,
    split_agents_by_cutoff,
)
from experiment_sad_irt.evaluate import analyze_scale_alignment


def main():
    parser = argparse.ArgumentParser(
        description="Analyze scale alignment between predicted and oracle IRT difficulties"
    )
    parser.add_argument(
        "--predicted_path",
        type=Path,
        default=Path("chris_output/sad_irt/baseline_irt/items.csv"),
        help="Path to predicted IRT items CSV (default: baseline IRT)",
    )
    parser.add_argument(
        "--oracle_path",
        type=Path,
        default=Path("clean_data/swebench_verified_20251120_full/1d/items.csv"),
        help="Path to oracle IRT items CSV",
    )
    parser.add_argument(
        "--responses_path",
        type=Path,
        default=Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"),
        help="Path to response matrix JSONL",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default="20250807",
        help="Frontier cutoff date (YYYYMMDD)",
    )
    parser.add_argument(
        "--min_pass_rate",
        type=float,
        default=0.10,
        help="Minimum pass rate for nontrivial tasks",
    )
    parser.add_argument(
        "--max_pass_rate",
        type=float,
        default=0.90,
        help="Maximum pass rate for nontrivial tasks",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/scale_alignment_analysis"),
        help="Output directory for plots and results",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SCALE ALIGNMENT ANALYSIS")
    print("=" * 60)

    # Load IRT items
    print(f"\nLoading predicted IRT from: {args.predicted_path}")
    if not args.predicted_path.exists():
        print(f"ERROR: Predicted IRT file not found: {args.predicted_path}")
        return
    predicted_items = pd.read_csv(args.predicted_path, index_col=0)
    print(f"  Loaded {len(predicted_items)} tasks")

    print(f"\nLoading oracle IRT from: {args.oracle_path}")
    oracle_items = pd.read_csv(args.oracle_path, index_col=0)
    print(f"  Loaded {len(oracle_items)} tasks")

    # Convert to dicts
    predicted_beta = predicted_items["b"].to_dict()
    oracle_beta = oracle_items["b"].to_dict()

    # Get agent splits
    print("\nIdentifying nontrivial anchor tasks...")
    all_agents = get_all_agents_from_responses(args.responses_path)
    pre_frontier, post_frontier = split_agents_by_cutoff(all_agents, args.cutoff_date)
    print(f"  Pre-frontier agents: {len(pre_frontier)}")
    print(f"  Post-frontier agents: {len(post_frontier)}")

    # Identify nontrivial tasks (anchor tasks for alignment)
    nontrivial_tasks, pre_rates, post_rates = identify_nontrivial_tasks(
        args.responses_path,
        pre_frontier,
        post_frontier,
        min_pass_rate=args.min_pass_rate,
        max_pass_rate=args.max_pass_rate,
    )
    print(f"  Nontrivial anchor tasks: {len(nontrivial_tasks)}")

    # Run analysis
    print("\nAnalyzing scale alignment...")
    plot_path = args.output_dir / "scale_alignment_diagnostic.png"
    results = analyze_scale_alignment(
        predicted_beta=predicted_beta,
        oracle_beta=oracle_beta,
        anchor_task_ids=nontrivial_tasks,
        output_path=plot_path,
    )

    # Print results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nAnchor tasks used: {results['n_anchors']}")

    print("\n--- Constant Shift Analysis ---")
    cs = results["constant_shift"]
    print(f"  Offset: {cs['offset']:.4f}")
    print(f"  Residual std: {cs['residual_std']:.4f}")
    print(f"  Residual range: [{cs['residual_min']:.4f}, {cs['residual_max']:.4f}]")

    print("\n--- Affine Transform Analysis ---")
    at = results["affine_transform"]
    print(f"  Slope: {at['slope']:.4f}")
    print(f"  Intercept: {at['intercept']:.4f}")
    print(f"  R²: {at['r_squared']:.4f}")
    print(f"  Residual std: {at['residual_std']:.4f}")

    print("\n--- Correlation ---")
    corr = results["correlation"]
    print(f"  Pearson r: {corr['pearson_r']:.4f} (p={corr['pearson_p']:.2e})")
    print(f"  Spearman ρ: {corr['spearman_r']:.4f} (p={corr['spearman_p']:.2e})")

    # Recommendation
    print("\n--- Recommendation ---")
    slope_deviation = abs(at["slope"] - 1.0)
    if slope_deviation < 0.05 and at["r_squared"] > 0.95:
        print("  Use CONSTANT SHIFT: slope is ~1.0 and R² is high")
    elif at["r_squared"] > 0.90:
        print("  Use AFFINE TRANSFORM: slope differs from 1.0 but R² is good")
    else:
        print("  WARNING: Neither constant nor affine provides good fit")
        print("  Consider: non-linear transformation or different anchor criteria")

    # Improvement from affine
    improvement = (cs["residual_std"] - at["residual_std"]) / cs["residual_std"] * 100
    print(f"\n  Affine reduces residual std by {improvement:.1f}%")

    # Save results to JSON
    results_path = args.output_dir / "scale_alignment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Diagnostic plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
