"""
Plot SAD-IRT learned theta (ability) values vs agent submission date.

X-axis: Date (agent submission date from YYYYMMDD prefix)
Y-axis: SAD-IRT learned ability (theta) from checkpoint

This is analogous to frontier_ability_over_time.py but uses learned theta
values from SAD-IRT training instead of oracle IRT.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
from scipy import stats


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "chris_output/sad_irt_long/full_20260118_024625_psi_batchnorm_lora_r64/checkpoint_epoch_9_step4248_20260118_044922.pt"
RESPONSE_MATRIX_PATH = PROJECT_ROOT / "data/swebench/responses.jsonl"
TRAJECTORY_DIR = PROJECT_ROOT / "chris_output/trajectory_summaries_api"
CUTOFF_DATE = "20250807"


def extract_date_prefix(agent_name: str) -> str:
    """Extract YYYYMMDD date prefix from agent name."""
    match = re.match(r"^(\d{8})_", agent_name)
    if match:
        return match.group(1)
    return ""


def extract_submission_date(agent_name: str) -> datetime | None:
    """Extract the submission date from agent name (YYYYMMDD prefix)."""
    try:
        date_str = agent_name[:8]
        return datetime.strptime(date_str, "%Y%m%d")
    except (ValueError, IndexError):
        return None


def get_short_name(agent_name: str) -> str:
    """Get a shortened display name for an agent."""
    name = agent_name[9:] if len(agent_name) > 9 else agent_name
    if len(name) > 35:
        name = name[:32] + "..."
    return name


def get_pre_frontier_agents(
    response_matrix_path: Path,
    trajectory_dir: Path,
    cutoff_date: str = "20250807",
) -> list:
    """Get pre-frontier agents in exact order used during training.

    This replicates the filtering logic from train_evaluate.py to ensure
    agent indices match the checkpoint.
    """
    # Read agents in response matrix order (this preserves the order)
    agents = []
    with open(response_matrix_path) as f:
        for line in f:
            data = json.loads(line)
            agents.append(data["subject_id"])

    # Filter to agents with trajectories
    traj_agents = {p.name for p in trajectory_dir.iterdir() if p.is_dir() and not p.name.startswith("_")}
    agents_with_traj = [a for a in agents if a in traj_agents]

    # Filter to pre-frontier (date < cutoff)
    pre_frontier = []
    for agent in agents_with_traj:
        date_prefix = extract_date_prefix(agent)
        if date_prefix and date_prefix < cutoff_date:
            pre_frontier.append(agent)

    return pre_frontier


def main():
    # Load checkpoint (CPU only, fast)
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    # Extract theta values
    theta_weight = checkpoint["model_state_dict"]["theta.weight"]  # (num_agents, 1)
    theta_values = theta_weight.squeeze(-1).numpy()  # (num_agents,)
    print(f"  Loaded {len(theta_values)} theta values")

    # Get agent ordering (must match training)
    agents = get_pre_frontier_agents(RESPONSE_MATRIX_PATH, TRAJECTORY_DIR, CUTOFF_DATE)
    print(f"  Found {len(agents)} pre-frontier agents with trajectories")

    if len(agents) != len(theta_values):
        print(f"WARNING: Mismatch - {len(agents)} agents vs {len(theta_values)} theta values")
        # Use the smaller count
        n = min(len(agents), len(theta_values))
        agents = agents[:n]
        theta_values = theta_values[:n]

    # Build dataframe with submission dates
    agent_data = []
    for i, agent_name in enumerate(agents):
        submission_date = extract_submission_date(agent_name)
        if submission_date:
            agent_data.append({
                "agent": agent_name,
                "short_name": get_short_name(agent_name),
                "theta": theta_values[i],
                "submission_date": submission_date,
            })
        else:
            print(f"  Could not extract date from: {agent_name}")

    df = pd.DataFrame(agent_data)
    print(f"\n{len(df)} agents with valid dates")

    # Sort by submission date
    df = df.sort_values("submission_date")

    # For each unique date, find the maximum ability of agents submitted on or before that date
    df_grouped = df.groupby("submission_date").agg({
        "theta": "max",
        "short_name": lambda x: list(x),
        "agent": lambda x: list(x),
    }).reset_index()

    df_grouped = df_grouped.sort_values("submission_date")
    df_grouped["frontier_theta"] = df_grouped["theta"].cummax()

    # Compute linear regression on frontier (max ability at each date)
    first_date = df["submission_date"].min()
    frontier_changes = df_grouped[df_grouped["frontier_theta"].diff().fillna(1) > 0].copy()
    frontier_dates = frontier_changes["submission_date"]
    frontier_x = np.array([(d - first_date).days for d in frontier_dates])
    frontier_y = frontier_changes["frontier_theta"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(frontier_x, frontier_y)

    print(f"\nLinear regression on frontier:")
    print(f"  Slope: {slope:.6f} theta/day = {slope * 365:.3f} theta/year")
    print(f"  Intercept: {intercept:.3f}")
    print(f"  R-value: {r_value:.4f}")
    print(f"  R^2: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.2e}")

    # Also compute regression on all agents (for reporting)
    all_x_days = np.array([(d - first_date).days for d in df["submission_date"]])
    all_y_theta = df["theta"].values
    slope_all, _, r_value_all, _, _ = stats.linregress(all_x_days, all_y_theta)
    print(f"\nLinear regression on all agents:")
    print(f"  Slope: {slope_all:.6f} theta/day = {slope_all * 365:.3f} theta/year")
    print(f"  R^2: {r_value_all**2:.4f}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot all individual agents as scatter points
    ax.scatter(df["submission_date"], df["theta"],
               alpha=0.5, s=50, color="#3b82f6", label="Individual agents", zorder=2)

    # Plot the frontier as a step function
    ax.step(df_grouped["submission_date"], df_grouped["frontier_theta"],
            where="post", linewidth=2.5, color="#1d4ed8", label="Frontier ability", zorder=4)

    # Mark each frontier improvement with a larger point
    ax.scatter(frontier_changes["submission_date"], frontier_changes["frontier_theta"],
               s=100, color="#1d4ed8", zorder=5, edgecolors="white", linewidths=2)

    # Plot the trendline (frontier)
    trendline_x = np.array([frontier_x.min(), frontier_x.max()])
    trendline_y = slope * trendline_x + intercept
    trendline_dates = [first_date + pd.Timedelta(days=int(x)) for x in trendline_x]
    ax.plot(trendline_dates, trendline_y,
            linewidth=2.5, color="#dc2626", linestyle="--",
            label=f"Frontier trendline (R²={r_value**2:.3f})", zorder=3)

    # Add annotation with regression stats
    stats_text = (
        f"Frontier: {slope * 365:.3f} θ/year, R² = {r_value**2:.3f}\n"
        f"All agents: {slope_all * 365:.3f} θ/year, R² = {r_value_all**2:.3f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    # Formatting
    ax.set_xlabel("Agent Submission Date", fontsize=12)
    ax.set_ylabel("SAD-IRT Learned θ (Ability)", fontsize=12)
    ax.set_title(
        "SAD-IRT Learned Agent Abilities Over Time\n"
        f"(LoRA r=64 with BatchNorm, {len(df)} pre-frontier agents, epoch 9)",
        fontsize=14
    )

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Set y-axis limits with some padding
    y_min, y_max = df["theta"].min(), df["theta"].max()
    ax.set_ylim(y_min - 0.3, y_max + 0.3)

    # Legend
    ax.legend(loc="lower right", fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = PROJECT_ROOT / "chris_output/figures/sad_irt_theta_over_time.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {output_path}")

    # Also save as PDF
    output_pdf = PROJECT_ROOT / "chris_output/figures/sad_irt_theta_over_time.pdf"
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"Saved PDF to: {output_pdf}")

    plt.close()


if __name__ == "__main__":
    main()
