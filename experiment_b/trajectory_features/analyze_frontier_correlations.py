"""Analyze correlations between trajectory features and oracle IRT difficulty on frontier tasks.

This script computes Pearson and Spearman correlations between extracted
trajectory features and oracle IRT difficulty for frontier tasks only.

Usage:
    python -m experiment_b.trajectory_features.analyze_frontier_correlations \
        --features-csv chris_output/trajectory_features/frontier_v1/llm_judge_features.csv

    # Save results to JSON
    python -m experiment_b.trajectory_features.analyze_frontier_correlations \
        --features-csv chris_output/trajectory_features/frontier_v1/llm_judge_features.csv \
        --output chris_output/trajectory_features/frontier_v1/correlation_results.json
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from experiment_ab_shared.llm_judge.analyze_feature_correlations import (
    analyze_features,
    compute_correlations,
    print_correlation_table,
    run_lasso_feature_selection,
)
from experiment_b.swebench.config import SWEBenchConfig
from experiment_b.trajectory_features.utils import load_frontier_tasks_with_difficulties


def load_frontier_features_and_difficulties(
    features_path: Path,
    config: SWEBenchConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load trajectory features and merge with oracle difficulties for frontier tasks only.

    Args:
        features_path: Path to extracted features CSV
        config: SWE-bench configuration

    Returns:
        Tuple of (merged DataFrame, list of feature column names)
    """
    # Load features
    features_df = pd.read_csv(features_path)

    # Detect task_id column
    task_id_col = None
    for col in ["_task_id", "task_id", "_instance_id", "instance_id"]:
        if col in features_df.columns:
            task_id_col = col
            break

    if task_id_col is None:
        raise ValueError(f"No task ID column found. Columns: {list(features_df.columns)}")

    # Load frontier tasks and oracle difficulties
    frontier_tasks, oracle_items, _, _ = load_frontier_tasks_with_difficulties(config)
    frontier_set = set(frontier_tasks)

    print(f"Loaded {len(features_df)} feature rows")
    print(f"Frontier tasks: {len(frontier_tasks)}")

    # Filter features to frontier tasks only
    features_df = features_df[features_df[task_id_col].isin(frontier_set)]
    print(f"Features for frontier tasks: {len(features_df)}")

    # Identify feature columns (numeric, not metadata)
    meta_cols = [c for c in features_df.columns if c.startswith("_") or c == "reasoning"]
    feature_cols = [
        c
        for c in features_df.columns
        if c not in meta_cols
        and c != task_id_col
        and pd.api.types.is_numeric_dtype(features_df[c])
    ]

    print(f"Feature columns: {feature_cols}")

    # Merge with oracle difficulties
    features_df = features_df.set_index(task_id_col)
    merged = features_df.join(oracle_items[["b"]], how="inner")

    if len(merged) == 0:
        raise ValueError(
            f"No tasks matched between features ({len(features_df)}) "
            f"and oracle IRT items. Check task ID formats."
        )

    print(f"Merged {len(merged)} frontier tasks with oracle difficulties")

    return merged, feature_cols


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlations between trajectory features and oracle difficulty on frontier tasks"
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="Path to extracted features CSV (from extract_frontier_features.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save results JSON",
    )

    args = parser.parse_args()

    # Load config
    config = SWEBenchConfig()

    # Load and merge data
    print("Loading features and difficulties...")
    merged, feature_cols = load_frontier_features_and_difficulties(
        args.features_csv, config
    )

    # Check minimum sample size for correlation
    if len(merged) < 3:
        print(f"\nError: Need at least 3 tasks for correlation analysis, got {len(merged)}")
        print("Run feature extraction on more tasks first.")
        return

    # Compute correlations
    print("\nComputing correlations...")
    corr_df = compute_correlations(merged, feature_cols, target_col="b")

    # Run Lasso feature selection
    print("Running Lasso feature selection...")
    lasso_coefs = run_lasso_feature_selection(merged, feature_cols, target_col="b")

    # Print results
    print_correlation_table(
        corr_df,
        lasso_coefs,
        title="Trajectory Feature Correlations with Oracle Difficulty (Frontier Tasks Only)",
    )

    # Summary of significant features
    sig_features = corr_df[corr_df["pearson_p"] < 0.05]
    print(f"\n{'='*60}")
    print("SIGNIFICANT FEATURES (p < 0.05)")
    print("="*60)

    if len(sig_features) > 0:
        for _, row in sig_features.iterrows():
            direction = "harder" if row["pearson_r"] > 0 else "easier"
            print(f"  {row['feature']}: r={row['pearson_r']:.3f} (higher = {direction})")
    else:
        print("  No features reached significance (p < 0.05)")
        print("  Consider refining the prompt or trying different features.")

    # Save results if output path provided
    if args.output:
        import json

        results = {
            "n_frontier_tasks": len(merged),
            "n_features": len(feature_cols),
            "correlations": corr_df.to_dict(orient="records"),
            "lasso_coefs": lasso_coefs,
            "significant_features": sig_features["feature"].tolist() if len(sig_features) > 0 else [],
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
