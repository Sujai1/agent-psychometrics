"""Plot learned agent embeddings using PCA.

Run locally after transferring agent_embeddings*.csv from cluster.

Usage:
    python experiment_a/mlp_ablation/plot_agent_embeddings.py
    python experiment_a/mlp_ablation/plot_agent_embeddings.py --noise 1.0
    python experiment_a/mlp_ablation/plot_agent_embeddings.py --compare
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path(__file__).parent.parent.parent


def analyze_embeddings(df, label=""):
    """Analyze and return stats for embeddings."""
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values

    # PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Stats
    std_per_agent = X.std(axis=1)
    correlations = [np.corrcoef(df["ability"], X[:, i])[0, 1] for i in range(X.shape[1])]

    stats = {
        "label": label,
        "X": X,
        "X_2d": X_2d,
        "pca": pca,
        "within_agent_std_mean": std_per_agent.mean(),
        "within_agent_std_max": std_per_agent.max(),
        "ability_corr_mean": np.abs(correlations).mean(),
        "pc1_var": pca.explained_variance_ratio_[0],
        "pc2_var": pca.explained_variance_ratio_[1],
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Plot agent embeddings")
    parser.add_argument("--noise", type=float, default=None,
                        help="Noise level to plot (default: no noise)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare no-noise vs noise=1.0")
    args = parser.parse_args()

    if args.compare:
        # Load both files
        path_no_noise = ROOT / "chris_output/experiment_a/mlp_embedding/agent_embeddings.csv"
        path_noise = ROOT / "chris_output/experiment_a/mlp_embedding/agent_embeddings_noise1.0.csv"

        if not path_no_noise.exists() or not path_noise.exists():
            print(f"Need both files for comparison:")
            print(f"  {path_no_noise} - {'exists' if path_no_noise.exists() else 'MISSING'}")
            print(f"  {path_noise} - {'exists' if path_noise.exists() else 'MISSING'}")
            return

        df_no_noise = pd.read_csv(path_no_noise)
        df_noise = pd.read_csv(path_noise)

        stats_no_noise = analyze_embeddings(df_no_noise, "No noise")
        stats_noise = analyze_embeddings(df_noise, "Noise σ=1.0")

        # Print comparison
        print("=" * 60)
        print("EMBEDDING COMPARISON: No Noise vs Noise σ=1.0")
        print("=" * 60)
        print(f"\n{'Metric':<35} {'No Noise':>12} {'Noise σ=1.0':>12}")
        print("-" * 60)
        print(f"{'Within-agent std (mean)':<35} {stats_no_noise['within_agent_std_mean']:>12.6f} {stats_noise['within_agent_std_mean']:>12.6f}")
        print(f"{'Within-agent std (max)':<35} {stats_no_noise['within_agent_std_max']:>12.6f} {stats_noise['within_agent_std_max']:>12.6f}")
        print(f"{'PC1 variance explained':<35} {stats_no_noise['pc1_var']:>12.1%} {stats_noise['pc1_var']:>12.1%}")
        print(f"{'PC2 variance explained':<35} {stats_no_noise['pc2_var']:>12.1%} {stats_noise['pc2_var']:>12.1%}")
        print(f"{'Mean |corr| with ability':<35} {stats_no_noise['ability_corr_mean']:>12.3f} {stats_noise['ability_corr_mean']:>12.3f}")

        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        for i, (stats, df) in enumerate([(stats_no_noise, df_no_noise), (stats_noise, df_noise)]):
            # PCA plot
            ax = axes[0, i]
            scatter = ax.scatter(
                stats["X_2d"][:, 0], stats["X_2d"][:, 1],
                c=df["ability"], cmap="RdYlGn", s=50, alpha=0.7,
                edgecolors="black", linewidths=0.5,
            )
            plt.colorbar(scatter, ax=ax, label="IRT Ability")
            ax.set_xlabel(f"PC1 ({stats['pc1_var']:.1%} var)")
            ax.set_ylabel(f"PC2 ({stats['pc2_var']:.1%} var)")
            ax.set_title(f"{stats['label']}\nWithin-agent std: {stats['within_agent_std_mean']:.4f}")

            # PC1 vs ability
            ax2 = axes[1, i]
            corr = np.corrcoef(df["ability"], stats["X_2d"][:, 0])[0, 1]
            ax2.scatter(df["ability"], stats["X_2d"][:, 0], alpha=0.6, s=30)
            ax2.set_xlabel("IRT Ability (θ)")
            ax2.set_ylabel("PC1")
            ax2.set_title(f"PC1 vs Ability (r = {corr:.3f})")

        plt.tight_layout()
        output_path = ROOT / "chris_output/experiment_a/mlp_embedding/agent_embeddings_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved comparison plot to: {output_path}")
        plt.show()
        return

    # Single file mode
    if args.noise is not None and args.noise > 0:
        emb_path = ROOT / f"chris_output/experiment_a/mlp_embedding/agent_embeddings_noise{args.noise}.csv"
    else:
        emb_path = ROOT / "chris_output/experiment_a/mlp_embedding/agent_embeddings.csv"

    if not emb_path.exists():
        print(f"Embeddings not found at: {emb_path}")
        print("Run extract_agent_embeddings.py on cluster first, then transfer the file.")
        return

    df = pd.read_csv(emb_path)
    print(f"Loaded {len(df)} agent embeddings")

    # Extract embedding columns
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values
    print(f"Embedding shape: {X.shape}")

    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained: {sum(pca.explained_variance_ratio_):.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Colored by ability
    ax = axes[0]
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=df["ability"],
        cmap="RdYlGn",
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="IRT Ability (θ)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Agent Embeddings (64D → 2D via PCA)\nColored by IRT Ability")

    # Add labels for top/bottom agents
    df_sorted = df.sort_values("ability", ascending=False)
    for _, row in df_sorted.head(5).iterrows():
        idx = df[df["agent_id"] == row["agent_id"]].index[0]
        name = row["agent_id"].split("/")[-1][:15]  # Truncate long names
        ax.annotate(name, (X_2d[idx, 0], X_2d[idx, 1]), fontsize=7, alpha=0.8)

    for _, row in df_sorted.tail(5).iterrows():
        idx = df[df["agent_id"] == row["agent_id"]].index[0]
        name = row["agent_id"].split("/")[-1][:15]
        ax.annotate(name, (X_2d[idx, 0], X_2d[idx, 1]), fontsize=7, alpha=0.8)

    # Plot 2: Histogram of PC1 values vs ability
    ax2 = axes[1]
    ax2.scatter(df["ability"], X_2d[:, 0], alpha=0.6, s=30)
    ax2.set_xlabel("IRT Ability (θ)")
    ax2.set_ylabel("PC1")
    ax2.set_title("PC1 vs IRT Ability\n(should be correlated if initialized from IRT)")

    # Add correlation
    corr = np.corrcoef(df["ability"], X_2d[:, 0])[0, 1]
    ax2.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax2.transAxes,
             fontsize=12, verticalalignment='top')

    plt.tight_layout()

    # Save
    output_path = ROOT / "chris_output/experiment_a/mlp_embedding/agent_embeddings_pca.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {output_path}")

    plt.show()

    # Also print some stats
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Check if embeddings have diverged from IRT initialization
    # If init was ability broadcast to all dims, then std across dims should be low initially
    std_per_agent = X.std(axis=1)
    print(f"\nStd of embedding dims per agent:")
    print(f"  Mean: {std_per_agent.mean():.4f}")
    print(f"  Min:  {std_per_agent.min():.4f}")
    print(f"  Max:  {std_per_agent.max():.4f}")

    # Correlation of each embedding dim with ability
    print(f"\nCorrelation of each embedding dim with ability:")
    correlations = []
    for i in range(X.shape[1]):
        r = np.corrcoef(df["ability"], X[:, i])[0, 1]
        correlations.append(r)
    correlations = np.array(correlations)
    print(f"  Mean |r|: {np.abs(correlations).mean():.3f}")
    print(f"  Max r:    {correlations.max():.3f} (dim {correlations.argmax()})")
    print(f"  Min r:    {correlations.min():.3f} (dim {correlations.argmin()})")

    # Variance explained by PCA
    print(f"\nPCA variance explained:")
    pca_full = PCA().fit(X)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    for n_comp in [2, 5, 10, 20]:
        print(f"  {n_comp} components: {cumsum[n_comp-1]:.1%}")


if __name__ == "__main__":
    main()
