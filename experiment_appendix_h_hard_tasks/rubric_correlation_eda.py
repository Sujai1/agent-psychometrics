"""Exploratory Data Analysis: Rubric Score vs eta_ij correlation plots.

This script validates the key assumption of the Ordered Logit IRT model:
that rubric scores are positively correlated with eta_ij = theta_j - beta_i.

Higher eta means easier task for the agent (higher ability, lower difficulty),
which should result in better agent performance (higher rubric scores).

Usage:
    source .venv/bin/activate
    python -m experiment_appendix_h_hard_tasks.rubric_correlation_eda

Output:
    Plots saved to: chris_output/experiment_b/rubric_eda/rubric_vs_eta_all_items.png
    Stats saved to: chris_output/experiment_b/rubric_eda/correlation_stats.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from experiment_appendix_h_hard_tasks.shared.rubric_preprocessing import RubricDataSource, RubricPreprocessor


def load_baseline_irt_cached() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load baseline IRT items (difficulties) and abilities from cache.

    Returns:
        Tuple of (items_df, abilities_df) with 'b' and 'theta' columns
    """
    # Find the baseline IRT cache directory for swebench
    baseline_dir = Path("chris_output/experiment_b/swebench/baseline_irt")
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline IRT directory not found: {baseline_dir}")

    # Get cache directories
    cache_dirs = list(baseline_dir.glob("cache_*"))
    if not cache_dirs:
        raise FileNotFoundError(f"No cache directories found in {baseline_dir}")

    # Use the first one (they should all be equivalent for the same cutoff)
    cache_dir = cache_dirs[0]

    items_path = cache_dir / "items.csv"
    abilities_path = cache_dir / "abilities.csv"

    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not abilities_path.exists():
        raise FileNotFoundError(f"Abilities file not found: {abilities_path}")

    items_df = pd.read_csv(items_path, index_col=0)
    abilities_df = pd.read_csv(abilities_path, index_col=0)

    print(f"Loaded baseline IRT: {len(items_df)} tasks, {len(abilities_df)} agents")
    return items_df, abilities_df


def compute_eta_for_observations(
    task_ids: list[str],
    agent_ids: list[str],
    items_df: pd.DataFrame,
    abilities_df: pd.DataFrame,
) -> np.ndarray:
    """Compute eta_ij = theta_j - beta_i for all observations.

    Args:
        task_ids: List of task IDs
        agent_ids: List of agent IDs
        items_df: DataFrame with 'b' (difficulty) column, indexed by task_id
        abilities_df: DataFrame with 'theta' (ability) column, indexed by agent_id

    Returns:
        Array of eta values for each observation

    Raises:
        ValueError: If any task or agent is missing from IRT data
    """
    n_obs = len(task_ids)
    eta_values = np.zeros(n_obs)

    for i, (task_id, agent_id) in enumerate(zip(task_ids, agent_ids)):
        if task_id not in items_df.index:
            raise ValueError(f"Task '{task_id}' not found in baseline IRT items")
        if agent_id not in abilities_df.index:
            raise ValueError(f"Agent '{agent_id}' not found in baseline IRT abilities")

        beta = items_df.loc[task_id, "b"]
        theta = abilities_df.loc[agent_id, "theta"]
        eta_values[i] = theta - beta

    return eta_values


def plot_rubric_vs_eta(
    rubric_source: RubricDataSource,
    items_df: pd.DataFrame,
    abilities_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict]:
    """Generate scatter plots of rubric score vs eta for each item.

    Args:
        rubric_source: RubricDataSource with preprocessed rubric data
        items_df: DataFrame with 'b' column
        abilities_df: DataFrame with 'theta' column
        output_dir: Directory to save plots

    Returns:
        Dict mapping item name to correlation statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all observations
    task_ids, agent_ids, rubric_scores = rubric_source.get_all_observations()
    n_obs = len(task_ids)

    # Compute eta for all observations
    eta_values = compute_eta_for_observations(task_ids, agent_ids, items_df, abilities_df)

    print(f"\nTotal observations: {n_obs}")
    print(f"Eta range: [{eta_values.min():.2f}, {eta_values.max():.2f}]")

    # Create summary plot (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    correlation_stats = {}

    for idx, item in enumerate(rubric_source.rubric_items):
        ax = axes[idx]
        scores = rubric_scores[item]

        # Compute correlation
        r, p_value = stats.pearsonr(eta_values, scores)
        spearman_r, spearman_p = stats.spearmanr(eta_values, scores)

        correlation_stats[item] = {
            "pearson_r": r,
            "pearson_p": p_value,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "n_obs": n_obs,
        }

        # Create scatter plot (NO jitter - show exact data)
        ax.scatter(eta_values, scores, alpha=0.4, s=25)

        # Add trend line
        z = np.polyfit(eta_values, scores, 1)
        p = np.poly1d(z)
        eta_sorted = np.linspace(eta_values.min(), eta_values.max(), 100)
        ax.plot(eta_sorted, p(eta_sorted), "r-", linewidth=2, label=f"slope={z[0]:.3f}")

        # Labels and title
        ax.set_xlabel(r"$\eta_{ij} = \theta_j - \beta_i$", fontsize=10)
        ax.set_ylabel(f"{item} score (0-5)", fontsize=10)

        # Color title based on correlation direction
        if r > 0.05:
            title_color = "green"
            recommendation = "KEEP"
        elif r < -0.05:
            title_color = "red"
            recommendation = "EXCLUDE"
        else:
            title_color = "orange"
            recommendation = "MARGINAL"

        ax.set_title(
            f"{item}\nr={r:.3f} (p={p_value:.2e})\n{recommendation}",
            fontsize=10,
            color=title_color,
            fontweight="bold",
        )

        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_yticks(range(6))

    plt.suptitle(
        "Rubric Score vs Latent Ease (eta = theta - beta)\n"
        "Higher eta = easier task for agent; expect positive correlation",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    plot_path = output_dir / "rubric_vs_eta_all_items.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {plot_path}")
    plt.close()

    # Print summary table
    print("\n" + "=" * 80)
    print("CORRELATION SUMMARY")
    print("=" * 80)
    print(f"{'Item':<25} {'Pearson r':>12} {'p-value':>12} {'Recommendation':>15}")
    print("-" * 80)

    for item, stats_dict in correlation_stats.items():
        r = stats_dict["pearson_r"]
        p = stats_dict["pearson_p"]

        if r > 0.05:
            rec = "KEEP"
        elif r < -0.05:
            rec = "EXCLUDE"
        else:
            rec = "MARGINAL"

        print(f"{item:<25} {r:>12.4f} {p:>12.2e} {rec:>15}")

    print("=" * 80)

    return correlation_stats


def main():
    """Run the EDA analysis."""
    print("=" * 80)
    print("RUBRIC vs ETA CORRELATION ANALYSIS")
    print("Validating assumption: higher eta -> higher rubric score")
    print("=" * 80)

    # Load rubric data
    rubric_path = Path("chris_output/trajectory_features/raw_features_500tasks_6agents.csv")
    print(f"\nLoading rubric data from: {rubric_path}")
    rubric_source = RubricDataSource(rubric_path, RubricPreprocessor())

    # Load baseline IRT
    print("\nLoading baseline IRT parameters...")
    items_df, abilities_df = load_baseline_irt_cached()

    # Verify all rubric agents are in baseline IRT
    rubric_agents = set(rubric_source.agent_ids)
    irt_agents = set(abilities_df.index)
    missing = rubric_agents - irt_agents

    if missing:
        raise ValueError(f"Rubric agents missing from baseline IRT: {missing}")
    print(f"  Verified: all {len(rubric_agents)} rubric agents are in baseline IRT")

    # Generate plots
    output_dir = Path("chris_output/experiment_b/rubric_eda")
    print(f"\nGenerating correlation plots...")
    correlation_stats = plot_rubric_vs_eta(
        rubric_source, items_df, abilities_df, output_dir
    )

    # Save correlation stats to CSV
    stats_df = pd.DataFrame(correlation_stats).T
    stats_path = output_dir / "correlation_stats.csv"
    stats_df.to_csv(stats_path)
    print(f"Saved correlation stats to: {stats_path}")

    print("\n" + "=" * 80)
    print("OUTPUT FILES:")
    print(f"  Plot: {output_dir / 'rubric_vs_eta_all_items.png'}")
    print(f"  Stats: {output_dir / 'correlation_stats.csv'}")
    print("=" * 80)
    print("\nNEXT STEPS:")
    print("1. Review the plot")
    print("2. Decide which rubric items to KEEP based on positive correlation")
    print("3. Items with r < 0 should be EXCLUDED from the ordered logit model")
    print("=" * 80)


if __name__ == "__main__":
    main()
