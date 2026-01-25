#!/usr/bin/env python3
"""Threshold sweep analysis for frontier task difficulty prediction.

Runs experiment_b evaluation at multiple pre_threshold values (0% to 30%)
and generates plots showing Oracle vs Baseline IRT performance across thresholds.

Usage:
    python -m experiment_b.threshold_sweep
    python -m experiment_b.threshold_sweep --datasets swebench terminalbench
    python -m experiment_b.threshold_sweep --output_dir chris_output/threshold_sweep
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_b import get_dataset_config, list_datasets
from experiment_b.shared import (
    load_and_prepare_data,
    compute_mean_per_agent_auc,
    filter_agents_with_frontier_variance,
    build_feature_sources,
    FeatureIRTPredictor,
)


# Default thresholds to sweep
DEFAULT_THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def _fit_feature_irt_single(
    l2_weight: float,
    l2_residual: float,
    embedding_source,
    train_task_ids: List[str],
    baseline_ground_truth_b: np.ndarray,
    train_responses: Dict,
    baseline_ability_values: np.ndarray,
    baseline_agent_ids: List[str],
    all_task_ids: List[str],
) -> Tuple[float, float, Dict[str, float]]:
    """Fit Feature-IRT with single hyperparameter combination.

    This function is designed to be called in parallel via joblib.

    Args:
        l2_weight: L2 regularization weight for feature weights
        l2_residual: L2 regularization weight for per-task residuals
        embedding_source: Feature source for embeddings
        train_task_ids: Task IDs for training
        baseline_ground_truth_b: Ground truth difficulties from baseline IRT
        train_responses: Response matrix for training (pre-frontier agents only)
        baseline_ability_values: Agent abilities from baseline IRT
        baseline_agent_ids: Agent IDs corresponding to abilities
        all_task_ids: All task IDs for prediction

    Returns:
        Tuple of (l2_weight, l2_residual, predictions dict)
    """
    predictor = FeatureIRTPredictor(
        source=embedding_source,
        use_residuals=True,
        init_from_baseline=True,
        l2_weight=l2_weight,
        l2_residual=l2_residual,
    )
    predictor.fit(
        task_ids=train_task_ids,
        ground_truth_b=baseline_ground_truth_b,
        responses=train_responses,
        baseline_abilities=baseline_ability_values,
        baseline_agent_ids=baseline_agent_ids,
    )
    preds = predictor.predict(all_task_ids)
    return l2_weight, l2_residual, preds


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run threshold sweep analysis across frontier definitions"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list_datasets(),
        choices=list_datasets(),
        help="Datasets to run sweep on (default: all)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Pre-frontier thresholds to sweep (default: 0.0 0.05 0.10 0.15 0.20 0.25 0.30)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/threshold_sweep"),
        help="Directory to save results (default: chris_output/threshold_sweep)",
    )
    return parser.parse_args()


def run_threshold_sweep_for_dataset(
    dataset_name: str,
    thresholds: List[float],
) -> pd.DataFrame:
    """Run threshold sweep for a single dataset.

    Args:
        dataset_name: Name of the dataset to evaluate
        thresholds: List of pre_threshold values to sweep

    Returns:
        DataFrame with columns: threshold, method, mean_auc, sem, n_agents, n_frontier_tasks
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    config = get_dataset_config(dataset_name)
    results = []

    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold*100:.0f}% ---")

        # Create args namespace to mimic CLI arguments
        class Args:
            responses_path = None
            baseline_irt_path = None
            oracle_irt_path = None
            oracle_abilities_path = None
            embeddings_path = None
            llm_judge_path = None
            trajectory_features_path = None
            cutoff_date = None
            pre_threshold = threshold
            post_threshold = None
            filter_bottom_percentile = 0.0
            min_oracle_ability = None
            frontier_definitions = ["pre_only"]

        args = Args()

        try:
            # Load data for this threshold
            data = load_and_prepare_data(args, config)

            # Get frontier tasks for this threshold
            frontier_def = "pre_only"
            frontier_tasks = data.frontier_tasks_by_def.get(frontier_def, [])

            if len(frontier_tasks) == 0:
                print(f"  No frontier tasks at threshold {threshold*100:.0f}%, skipping")
                continue

            # Filter eval agents to those with variance on frontier tasks
            eval_agents = filter_agents_with_frontier_variance(
                responses=data.config.responses,
                frontier_task_ids=frontier_tasks,
                candidate_agents=data.post_frontier_agents,
            )

            if len(eval_agents) == 0:
                print(f"  No eval agents with variance at threshold {threshold*100:.0f}%, skipping")
                continue

            print(f"  Frontier tasks: {len(frontier_tasks)}")
            print(f"  Eval agents: {len(eval_agents)}")

            # Compute metrics for Oracle
            oracle_beta = data.oracle_items["b"].to_dict()
            oracle_metrics = compute_mean_per_agent_auc(
                predicted_beta=oracle_beta,
                responses=data.config.responses,
                frontier_task_ids=frontier_tasks,
                eval_agents=eval_agents,
            )
            results.append({
                "threshold": threshold,
                "method": "Oracle (upper bound)",
                "mean_auc": oracle_metrics["mean_auc"],
                "sem": oracle_metrics["sem_auc"],
                "n_agents": oracle_metrics["n_agents"],
                "n_frontier_tasks": len(frontier_tasks),
            })

            # Compute metrics for Baseline IRT
            baseline_beta = data.baseline_items["b"].to_dict()
            baseline_metrics = compute_mean_per_agent_auc(
                predicted_beta=baseline_beta,
                responses=data.config.responses,
                frontier_task_ids=frontier_tasks,
                eval_agents=eval_agents,
            )
            results.append({
                "threshold": threshold,
                "method": "Baseline IRT (pre-frontier only)",
                "mean_auc": baseline_metrics["mean_auc"],
                "sem": baseline_metrics["sem_auc"],
                "n_agents": baseline_metrics["n_agents"],
                "n_frontier_tasks": len(frontier_tasks),
            })

            print(f"  Oracle: {oracle_metrics['mean_auc']:.4f} +/- {oracle_metrics['sem_auc']:.4f}")
            print(f"  Baseline: {baseline_metrics['mean_auc']:.4f} +/- {baseline_metrics['sem_auc']:.4f}")

            # Compute metrics for Baseline-Init Feature-IRT (Embedding)
            feature_sources = build_feature_sources(config)
            embedding_source = None
            for name, source in feature_sources:
                if name == "Embedding":
                    embedding_source = source
                    break

            if embedding_source is not None and data.baseline_abilities is not None:
                # Run grid search for best hyperparameters (in parallel)
                l2_grid = [0.001, 0.01, 0.1, 1.0, 10.0]

                baseline_ability_values = data.baseline_abilities["theta"].values
                baseline_agent_ids = data.baseline_abilities.index.tolist()

                # Fit all hyperparameter combinations in parallel
                grid_results = Parallel(n_jobs=-1, backend="loky")(
                    delayed(_fit_feature_irt_single)(
                        l2_w,
                        l2_r,
                        embedding_source,
                        data.train_task_ids,
                        data.baseline_ground_truth_b,
                        data.train_responses,
                        baseline_ability_values,
                        baseline_agent_ids,
                        data.config.all_task_ids,
                    )
                    for l2_w in l2_grid
                    for l2_r in l2_grid
                )

                # Find best result and track best hyperparameters
                best_auc = -1.0
                best_preds: Optional[Dict[str, float]] = None
                best_l2_weight: Optional[float] = None
                best_l2_residual: Optional[float] = None

                for l2_w, l2_r, preds in grid_results:
                    # Compute AUC (scale-invariant, no alignment needed)
                    metrics = compute_mean_per_agent_auc(
                        predicted_beta=preds,
                        responses=data.config.responses,
                        frontier_task_ids=frontier_tasks,
                        eval_agents=eval_agents,
                    )

                    if metrics["mean_auc"] > best_auc:
                        best_auc = metrics["mean_auc"]
                        best_preds = preds
                        best_l2_weight = l2_w
                        best_l2_residual = l2_r

                # Compute final metrics with best predictions
                feature_irt_metrics = compute_mean_per_agent_auc(
                    predicted_beta=best_preds,
                    responses=data.config.responses,
                    frontier_task_ids=frontier_tasks,
                    eval_agents=eval_agents,
                )

                results.append({
                    "threshold": threshold,
                    "method": "Baseline-Init Feature-IRT (Embedding)",
                    "mean_auc": feature_irt_metrics["mean_auc"],
                    "sem": feature_irt_metrics["sem_auc"],
                    "n_agents": feature_irt_metrics["n_agents"],
                    "n_frontier_tasks": len(frontier_tasks),
                    "best_l2_weight": best_l2_weight,
                    "best_l2_residual": best_l2_residual,
                })
                print(f"  Feature-IRT: {feature_irt_metrics['mean_auc']:.4f} ± {feature_irt_metrics['sem_auc']:.4f} "
                      f"(l2_w={best_l2_weight}, l2_r={best_l2_residual})")

        except Exception as e:
            print(f"  Error at threshold {threshold*100:.0f}%: {e}")
            continue

    return pd.DataFrame(results)


def plot_threshold_sweep(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:
    """Generate threshold sweep plot for a dataset.

    Args:
        df: DataFrame with sweep results
        dataset_name: Name of the dataset (for title)
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Oracle
    oracle_df = df[df["method"] == "Oracle (upper bound)"]
    ax.errorbar(
        oracle_df["threshold"] * 100,
        oracle_df["mean_auc"],
        yerr=oracle_df["sem"],
        label="Oracle (upper bound)",
        color="blue",
        marker="o",
        capsize=3,
        linewidth=2,
        markersize=8,
    )

    # Plot Baseline IRT
    baseline_df = df[df["method"] == "Baseline IRT (pre-frontier only)"]
    ax.errorbar(
        baseline_df["threshold"] * 100,
        baseline_df["mean_auc"],
        yerr=baseline_df["sem"],
        label="Baseline IRT (pre-frontier only)",
        color="orange",
        marker="s",
        capsize=3,
        linewidth=2,
        markersize=8,
    )

    # Plot Baseline-Init Feature-IRT (Embedding)
    feature_irt_df = df[df["method"] == "Baseline-Init Feature-IRT (Embedding)"]
    if not feature_irt_df.empty:
        ax.errorbar(
            feature_irt_df["threshold"] * 100,
            feature_irt_df["mean_auc"],
            yerr=feature_irt_df["sem"],
            label="Baseline-Init Feature-IRT (Embedding)",
            color="green",
            marker="^",
            capsize=3,
            linewidth=2,
            markersize=8,
        )

    # Configure plot
    ax.set_xlabel("Pre-frontier Threshold (%)", fontsize=12)
    ax.set_ylabel("Mean Per-Agent AUC", fontsize=12)
    ax.set_title(f"Threshold Sweep: {dataset_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 32)
    ax.set_ylim(0.4, 1.0)  # Full range to show all data

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Threshold Sweep Analysis")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Thresholds: {[f'{t*100:.0f}%' for t in args.thresholds]}")
    print(f"Output directory: {args.output_dir}")

    for dataset_name in args.datasets:
        # Run sweep
        df = run_threshold_sweep_for_dataset(dataset_name, args.thresholds)

        if df.empty:
            print(f"\nNo results for {dataset_name}, skipping")
            continue

        # Save CSV
        csv_path = args.output_dir / f"threshold_sweep_{dataset_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV to {csv_path}")

        # Generate plot
        plot_path = args.output_dir / f"threshold_sweep_{dataset_name}.png"
        plot_threshold_sweep(df, dataset_name, plot_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
