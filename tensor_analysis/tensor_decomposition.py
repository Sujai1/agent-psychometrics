"""
Tensor decomposition analysis on response + trajectory data.

Builds 3D tensors (agents × tasks × features) where features are:
- resolved (binary 0/1)
- standardized_char_count (z-scored trajectory length)

Runs CP and Tucker decomposition to explore structure beyond 1D IRT.
"""

import json
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorly as tl
from scipy import stats
from tensorly.decomposition import parafac, tucker

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CHAR_COUNTS_DIR = PROJECT_ROOT / "chris_output" / "tensor_analysis"
IRT_DIR = PROJECT_ROOT / "clean_data" / "swebench_verified_20251120_full" / "1d"
OUTPUT_DIR = PROJECT_ROOT / "chris_output" / "tensor_analysis" / "decomposition"
RESPONSE_DATA_DIR = PROJECT_ROOT / "out" / "chris_irt"


class TensorData(NamedTuple):
    """Container for tensor and metadata."""
    tensor: np.ndarray  # (n_agents, n_tasks, n_features)
    agent_ids: list[str]
    task_ids: list[str]
    feature_names: list[str]


def normalize_agent_name(name: str) -> str:
    """Normalize agent name to snake_case for matching."""
    # Convert to lowercase, replace spaces/dashes with underscores
    normalized = name.lower()
    normalized = normalized.replace(" - ", "___")
    normalized = normalized.replace(" -- ", "____")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace(".", "_")
    return normalized


def load_response_matrix(dataset: str) -> dict[str, dict[str, int]]:
    """Load binary response matrix from JSONL."""
    if dataset == "verified":
        # For verified, responses are already in char_counts via 'resolved' column
        return {}
    elif dataset == "pro":
        path = RESPONSE_DATA_DIR / "swebench_pro.jsonl"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    responses = {}
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            agent_id = record["subject_id"]
            # Normalize agent name for matching with trajectory data
            normalized_id = normalize_agent_name(agent_id)
            responses[normalized_id] = record["responses"]
    return responses


def load_char_counts(dataset: str) -> pd.DataFrame:
    """Load character counts CSV for a dataset, joining with response matrix if needed."""
    if dataset == "verified":
        path = CHAR_COUNTS_DIR / "swebench_verified_char_counts.csv"
        df = pd.read_csv(path)
        print(f"Loaded {dataset}: {len(df)} rows")
        return df
    elif dataset == "pro":
        path = CHAR_COUNTS_DIR / "swebench_pro_char_counts.csv"
        df = pd.read_csv(path)
        print(f"Loaded {dataset} char_counts: {len(df)} rows")

        # Load response matrix and join resolved status
        responses = load_response_matrix(dataset)
        print(f"Loaded {dataset} responses: {len(responses)} agents")

        # Add resolved column by looking up in response matrix
        def get_resolved(row):
            agent_normalized = normalize_agent_name(row["agent"])
            task_id = row["task_id"]
            # Pro task IDs in trajectory have version suffix, response matrix doesn't
            # e.g., "NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac-vnan"
            # vs "NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac"
            task_base = task_id.rsplit("-v", 1)[0] if "-v" in task_id else task_id

            if agent_normalized in responses:
                agent_resp = responses[agent_normalized]
                if task_base in agent_resp:
                    return agent_resp[task_base]
            return np.nan

        df["resolved"] = df.apply(get_resolved, axis=1)

        # Report join statistics
        n_matched = df["resolved"].notna().sum()
        n_total = len(df)
        print(f"  Joined resolved status: {n_matched}/{n_total} ({100*n_matched/n_total:.1f}%)")

        if n_matched == 0:
            raise ValueError("Failed to join any resolved status - check agent/task name matching")

        return df
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def build_tensor(df: pd.DataFrame, standardize: bool = True) -> TensorData:
    """
    Build 3D tensor from char_counts dataframe.

    Args:
        df: DataFrame with columns [agent, task_id, assistant_char_count, resolved]
        standardize: Whether to z-score the char_count feature

    Returns:
        TensorData with shape (n_agents, n_tasks, n_features)
    """
    # Get unique agents and tasks
    agent_ids = sorted(df["agent"].unique())
    task_ids = sorted(df["task_id"].unique())

    n_agents = len(agent_ids)
    n_tasks = len(task_ids)

    # Check if resolved column exists
    has_resolved = "resolved" in df.columns
    n_features = 2 if has_resolved else 1

    print(f"Building tensor: {n_agents} agents × {n_tasks} tasks × {n_features} features")
    if not has_resolved:
        print("  Warning: 'resolved' column not found, using char_count only")

    # Create index mappings
    agent_to_idx = {a: i for i, a in enumerate(agent_ids)}
    task_to_idx = {t: i for i, t in enumerate(task_ids)}

    # Initialize tensor
    tensor = np.zeros((n_agents, n_tasks, n_features), dtype=np.float64)

    # Fill tensor
    for _, row in df.iterrows():
        i = agent_to_idx[row["agent"]]
        j = task_to_idx[row["task_id"]]

        if has_resolved:
            # Feature 0: resolved (binary)
            resolved = row.get("resolved", np.nan)
            if pd.notna(resolved):
                tensor[i, j, 0] = float(resolved)
            # Feature 1: char_count
            tensor[i, j, 1] = row["assistant_char_count"]
        else:
            # Feature 0: char_count only
            tensor[i, j, 0] = row["assistant_char_count"]

    # Standardize char_count feature if requested
    if standardize:
        char_idx = 1 if has_resolved else 0
        char_counts = tensor[:, :, char_idx]
        mean = np.mean(char_counts)
        std = np.std(char_counts)
        tensor[:, :, char_idx] = (char_counts - mean) / std
        print(f"Standardized char_count: mean={mean:.0f}, std={std:.0f}")

    if has_resolved:
        feature_names = ["resolved", "char_count_std" if standardize else "char_count"]
    else:
        feature_names = ["char_count_std" if standardize else "char_count"]

    return TensorData(
        tensor=tensor,
        agent_ids=agent_ids,
        task_ids=task_ids,
        feature_names=feature_names
    )


def load_irt_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load IRT abilities and items for validation."""
    abilities = pd.read_csv(IRT_DIR / "abilities.csv", index_col=0)
    items = pd.read_csv(IRT_DIR / "items.csv", index_col=0)
    print(f"Loaded IRT: {len(abilities)} agents, {len(items)} items")
    return abilities, items


def run_cp_decomposition(
    tensor: np.ndarray,
    ranks: list[int],
    n_iter_max: int = 100,
    random_state: int = 42
) -> dict:
    """
    Run CP decomposition for multiple ranks.

    Returns dict with reconstruction errors and factor matrices for each rank.
    """
    results = {}

    # Compute total variance for normalization
    total_var = np.var(tensor)

    for rank in ranks:
        print(f"  CP rank={rank}...", end=" ")

        # Run CP decomposition
        np.random.seed(random_state)
        weights, factors = parafac(
            tensor,
            rank=rank,
            n_iter_max=n_iter_max,
            init="random",
            tol=1e-6
        )

        # Reconstruct tensor
        reconstructed = tl.cp_to_tensor((weights, factors))

        # Compute reconstruction error
        residual = tensor - reconstructed
        reconstruction_error = np.mean(residual ** 2)
        relative_error = reconstruction_error / total_var
        explained_var = 1 - relative_error

        print(f"explained variance = {explained_var:.4f}")

        results[rank] = {
            "weights": weights,
            "factors": factors,  # [agent_factors, task_factors, feature_factors]
            "reconstruction_error": reconstruction_error,
            "relative_error": relative_error,
            "explained_variance": explained_var
        }

    return results


def run_tucker_decomposition(
    tensor: np.ndarray,
    ranks_list: list[tuple[int, int, int]],
    random_state: int = 42
) -> dict:
    """
    Run Tucker decomposition for multiple rank configurations.

    Returns dict with core tensor and factor matrices for each configuration.
    """
    results = {}

    # Compute total variance for normalization
    total_var = np.var(tensor)

    for ranks in ranks_list:
        print(f"  Tucker ranks={ranks}...", end=" ")

        # Run Tucker decomposition
        np.random.seed(random_state)
        core, factors = tucker(tensor, rank=ranks, n_iter_max=100, tol=1e-6)

        # Reconstruct tensor
        reconstructed = tl.tucker_to_tensor((core, factors))

        # Compute reconstruction error
        residual = tensor - reconstructed
        reconstruction_error = np.mean(residual ** 2)
        relative_error = reconstruction_error / total_var
        explained_var = 1 - relative_error

        print(f"explained variance = {explained_var:.4f}")

        results[ranks] = {
            "core": core,
            "factors": factors,
            "reconstruction_error": reconstruction_error,
            "relative_error": relative_error,
            "explained_variance": explained_var
        }

    return results


def plot_scree(cp_results: dict, tucker_results: dict, dataset: str, output_dir: Path):
    """Plot scree curve for CP and Tucker decomposition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # CP results
    cp_ranks = sorted(cp_results.keys())
    cp_explained = [cp_results[r]["explained_variance"] for r in cp_ranks]
    ax.plot(cp_ranks, cp_explained, "o-", label="CP", markersize=8, linewidth=2)

    # Tucker results (use sum of ranks as x-axis)
    tucker_ranks = sorted(tucker_results.keys())
    tucker_x = [sum(r) for r in tucker_ranks]
    tucker_explained = [tucker_results[r]["explained_variance"] for r in tucker_ranks]
    ax.plot(tucker_x, tucker_explained, "s--", label="Tucker", markersize=8, linewidth=2)

    # Add rank labels for Tucker
    for x, r, ev in zip(tucker_x, tucker_ranks, tucker_explained):
        ax.annotate(f"{r}", (x, ev), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Rank (CP) / Sum of Ranks (Tucker)")
    ax.set_ylabel("Explained Variance")
    ax.set_title(f"Tensor Decomposition Scree Plot - {dataset}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    output_path = output_dir / f"{dataset}_scree.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_feature_loadings(cp_results: dict, dataset: str, feature_names: list[str], output_dir: Path):
    """Plot feature loadings for CP decomposition."""
    # Use rank 2 for interpretability
    rank = 2 if 2 in cp_results else min(cp_results.keys())
    factors = cp_results[rank]["factors"]
    feature_factors = factors[2]  # (n_features, rank)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(feature_names))
    width = 0.35

    for r in range(feature_factors.shape[1]):
        offset = (r - feature_factors.shape[1] / 2 + 0.5) * width
        ax.bar(x + offset, feature_factors[:, r], width, label=f"Component {r+1}")

    ax.set_xlabel("Feature")
    ax.set_ylabel("Loading")
    ax.set_title(f"Feature Loadings (CP rank={rank}) - {dataset}")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names)
    ax.legend()
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    output_path = output_dir / f"{dataset}_feature_loadings.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_agent_factors(
    cp_results: dict,
    agent_ids: list[str],
    abilities_df: pd.DataFrame | None,
    dataset: str,
    output_dir: Path
):
    """Plot agent factors colored by IRT ability."""
    # Use rank 2 for visualization
    rank = 2 if 2 in cp_results else min(cp_results.keys())
    factors = cp_results[rank]["factors"]
    agent_factors = factors[0]  # (n_agents, rank)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Agent factor scatter
    ax = axes[0]

    if abilities_df is not None and agent_factors.shape[1] >= 2:
        # Match agents to IRT abilities
        abilities = []
        for agent_id in agent_ids:
            if agent_id in abilities_df.index:
                abilities.append(abilities_df.loc[agent_id, "theta"])
            else:
                abilities.append(np.nan)
        abilities = np.array(abilities)

        # Color by ability
        mask = ~np.isnan(abilities)
        scatter = ax.scatter(
            agent_factors[mask, 0],
            agent_factors[mask, 1],
            c=abilities[mask],
            cmap="viridis",
            s=60,
            alpha=0.7
        )
        plt.colorbar(scatter, ax=ax, label="IRT Ability (θ)")

        # Plot agents without IRT as gray
        if (~mask).any():
            ax.scatter(
                agent_factors[~mask, 0],
                agent_factors[~mask, 1],
                c="gray",
                s=60,
                alpha=0.5,
                label="No IRT data"
            )
            ax.legend()
    else:
        ax.scatter(agent_factors[:, 0], agent_factors[:, 1] if agent_factors.shape[1] >= 2 else np.zeros(len(agent_factors)), s=60, alpha=0.7)

    ax.set_xlabel("Agent Factor 1")
    ax.set_ylabel("Agent Factor 2")
    ax.set_title(f"Agent Factors (CP rank={rank}) - {dataset}")
    ax.grid(True, alpha=0.3)

    # Right: Factor 1 vs IRT ability
    ax = axes[1]
    if abilities_df is not None:
        mask = ~np.isnan(abilities)
        ax.scatter(abilities[mask], agent_factors[mask, 0], s=60, alpha=0.7)

        # Fit line
        if mask.sum() > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                abilities[mask], agent_factors[mask, 0]
            )
            x_line = np.linspace(abilities[mask].min(), abilities[mask].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, "r--",
                   label=f"r={r_value:.3f}, p={p_value:.2e}")
            ax.legend()

        ax.set_xlabel("IRT Ability (θ)")
        ax.set_ylabel("Agent Factor 1")
        ax.set_title(f"Agent Factor 1 vs IRT Ability - {dataset}")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No IRT data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Agent Factor 1 vs IRT Ability - {dataset}")

    fig.tight_layout()
    output_path = output_dir / f"{dataset}_agent_factors.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_task_factors(
    cp_results: dict,
    task_ids: list[str],
    items_df: pd.DataFrame | None,
    dataset: str,
    output_dir: Path
):
    """Plot task factors colored by IRT difficulty."""
    # Use rank 2 for visualization
    rank = 2 if 2 in cp_results else min(cp_results.keys())
    factors = cp_results[rank]["factors"]
    task_factors = factors[1]  # (n_tasks, rank)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Task factor scatter
    ax = axes[0]

    if items_df is not None and task_factors.shape[1] >= 2:
        # Match tasks to IRT difficulty
        difficulties = []
        for task_id in task_ids:
            if task_id in items_df.index:
                difficulties.append(items_df.loc[task_id, "b"])
            else:
                difficulties.append(np.nan)
        difficulties = np.array(difficulties)

        # Color by difficulty
        mask = ~np.isnan(difficulties)
        scatter = ax.scatter(
            task_factors[mask, 0],
            task_factors[mask, 1],
            c=difficulties[mask],
            cmap="plasma",
            s=20,
            alpha=0.5
        )
        plt.colorbar(scatter, ax=ax, label="IRT Difficulty (β)")

        # Plot tasks without IRT as gray
        if (~mask).any():
            ax.scatter(
                task_factors[~mask, 0],
                task_factors[~mask, 1],
                c="gray",
                s=20,
                alpha=0.3,
                label="No IRT data"
            )
            ax.legend()
    else:
        ax.scatter(task_factors[:, 0], task_factors[:, 1] if task_factors.shape[1] >= 2 else np.zeros(len(task_factors)), s=20, alpha=0.5)

    ax.set_xlabel("Task Factor 1")
    ax.set_ylabel("Task Factor 2")
    ax.set_title(f"Task Factors (CP rank={rank}) - {dataset}")
    ax.grid(True, alpha=0.3)

    # Right: Factor 1 vs IRT difficulty
    ax = axes[1]
    if items_df is not None:
        mask = ~np.isnan(difficulties)
        ax.scatter(difficulties[mask], task_factors[mask, 0], s=20, alpha=0.5)

        # Fit line
        if mask.sum() > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                difficulties[mask], task_factors[mask, 0]
            )
            x_line = np.linspace(difficulties[mask].min(), difficulties[mask].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, "r--",
                   label=f"r={r_value:.3f}, p={p_value:.2e}")
            ax.legend()

        ax.set_xlabel("IRT Difficulty (β)")
        ax.set_ylabel("Task Factor 1")
        ax.set_title(f"Task Factor 1 vs IRT Difficulty - {dataset}")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No IRT data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Task Factor 1 vs IRT Difficulty - {dataset}")

    fig.tight_layout()
    output_path = output_dir / f"{dataset}_task_factors.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def check_guttman_effect(
    cp_results: dict,
    agent_ids: list[str],
    abilities_df: pd.DataFrame | None,
    dataset: str,
    output_dir: Path
):
    """Check for Guttman effect (polynomial relationship between factors)."""
    # Use rank=2 for consistency with other plots
    rank = 2 if 2 in cp_results else min(k for k in cp_results.keys() if k >= 2)
    if rank < 2:
        print("Need at least rank 2 to check Guttman effect")
        return

    factors = cp_results[rank]["factors"]
    agent_factors = factors[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Factor 1 vs Factor 2 with polynomial fit
    ax = axes[0]
    f1 = agent_factors[:, 0]
    f2 = agent_factors[:, 1]

    ax.scatter(f1, f2, s=60, alpha=0.7)

    # Fit quadratic
    coeffs = np.polyfit(f1, f2, 2)
    x_fit = np.linspace(f1.min(), f1.max(), 100)
    y_fit = np.polyval(coeffs, x_fit)

    # Calculate R² for quadratic fit
    y_pred = np.polyval(coeffs, f1)
    ss_res = np.sum((f2 - y_pred) ** 2)
    ss_tot = np.sum((f2 - np.mean(f2)) ** 2)
    r2_quad = 1 - ss_res / ss_tot

    ax.plot(x_fit, y_fit, "r--", linewidth=2, label=f"Quadratic R²={r2_quad:.3f}")

    ax.set_xlabel("Agent Factor 1")
    ax.set_ylabel("Agent Factor 2")
    ax.set_title(f"Guttman Effect Check - {dataset}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Compare to IRT if available
    ax = axes[1]
    if abilities_df is not None:
        abilities = []
        for agent_id in agent_ids:
            if agent_id in abilities_df.index:
                abilities.append(abilities_df.loc[agent_id, "theta"])
            else:
                abilities.append(np.nan)
        abilities = np.array(abilities)
        mask = ~np.isnan(abilities)

        if mask.sum() > 2:
            # Fit Factor 2 as function of IRT ability
            coeffs = np.polyfit(abilities[mask], f2[mask], 2)
            x_fit = np.linspace(abilities[mask].min(), abilities[mask].max(), 100)
            y_fit = np.polyval(coeffs, x_fit)

            y_pred = np.polyval(coeffs, abilities[mask])
            ss_res = np.sum((f2[mask] - y_pred) ** 2)
            ss_tot = np.sum((f2[mask] - np.mean(f2[mask])) ** 2)
            r2_quad = 1 - ss_res / ss_tot

            ax.scatter(abilities[mask], f2[mask], s=60, alpha=0.7)
            ax.plot(x_fit, y_fit, "r--", linewidth=2, label=f"Quadratic R²={r2_quad:.3f}")
            ax.set_xlabel("IRT Ability (θ)")
            ax.set_ylabel("Agent Factor 2")
            ax.set_title(f"Factor 2 vs IRT Ability (Guttman test) - {dataset}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if r2_quad > 0.8:
                ax.text(0.05, 0.95, "⚠️ Guttman effect likely present",
                       transform=ax.transAxes, fontsize=12, color="red",
                       verticalalignment="top")
    else:
        ax.text(0.5, 0.5, "No IRT data available", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    output_path = output_dir / f"{dataset}_guttman_check.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def analyze_dataset(
    dataset: str,
    abilities_df: pd.DataFrame | None,
    items_df: pd.DataFrame | None,
    output_dir: Path
) -> dict:
    """Run full tensor decomposition analysis on a dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset}")
    print(f"{'='*60}")

    # Load and build tensor
    df = load_char_counts(dataset)
    tensor_data = build_tensor(df, standardize=True)

    print(f"\nTensor shape: {tensor_data.tensor.shape}")
    print(f"Features: {tensor_data.feature_names}")

    # Run CP decomposition
    print("\nCP Decomposition:")
    cp_results = run_cp_decomposition(
        tensor_data.tensor,
        ranks=[1, 2, 3, 5]
    )

    # Run Tucker decomposition
    print("\nTucker Decomposition:")
    tucker_results = run_tucker_decomposition(
        tensor_data.tensor,
        ranks_list=[(2, 2, 2), (3, 3, 2), (5, 5, 2)]
    )

    # Generate plots
    print("\nGenerating plots:")
    plot_scree(cp_results, tucker_results, dataset, output_dir)
    plot_feature_loadings(cp_results, dataset, tensor_data.feature_names, output_dir)
    plot_agent_factors(cp_results, tensor_data.agent_ids, abilities_df, dataset, output_dir)
    plot_task_factors(cp_results, tensor_data.task_ids, items_df, dataset, output_dir)
    check_guttman_effect(cp_results, tensor_data.agent_ids, abilities_df, dataset, output_dir)

    return {
        "tensor_shape": tensor_data.tensor.shape,
        "cp_results": {k: {"explained_variance": v["explained_variance"]} for k, v in cp_results.items()},
        "tucker_results": {str(k): {"explained_variance": v["explained_variance"]} for k, v in tucker_results.items()}
    }


def main():
    """Run tensor decomposition analysis on both datasets."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load IRT data (only available for Verified)
    abilities_df, items_df = load_irt_data()

    # Analyze both datasets
    results = {}

    # SWE-bench Verified
    results["verified"] = analyze_dataset(
        "verified", abilities_df, items_df, OUTPUT_DIR
    )

    # SWE-bench Pro (no IRT data)
    results["pro"] = analyze_dataset(
        "pro", None, None, OUTPUT_DIR
    )

    # Save summary
    summary_path = OUTPUT_DIR / "decomposition_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Print comparison
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for dataset, data in results.items():
        print(f"\n{dataset.upper()}:")
        print(f"  Tensor shape: {data['tensor_shape']}")
        print("  CP explained variance:")
        for rank, info in data["cp_results"].items():
            print(f"    rank={rank}: {info['explained_variance']:.4f}")
        print("  Tucker explained variance:")
        for ranks, info in data["tucker_results"].items():
            print(f"    ranks={ranks}: {info['explained_variance']:.4f}")


if __name__ == "__main__":
    main()
