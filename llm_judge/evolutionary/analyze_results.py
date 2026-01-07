"""Analyze and visualize evolution results."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .feature_store import Feature, FeatureEvaluation, FeatureStore, GenerationSummary


def load_all_generations(store: FeatureStore) -> List[GenerationSummary]:
    """Load summaries for all generations.

    Args:
        store: Feature store instance.

    Returns:
        List of GenerationSummary objects.
    """
    summaries = []
    gen = 0
    while True:
        summary = store.load_summary(gen)
        if summary is None:
            break
        summaries.append(summary)
        gen += 1
    return summaries


def plot_correlation_progression(
    summaries: List[GenerationSummary],
    output_path: Optional[Path] = None,
):
    """Plot correlation progression across generations.

    Args:
        summaries: List of generation summaries.
        output_path: Optional path to save figure.
    """
    generations = [s.generation for s in summaries]
    best_corrs = [s.best_correlation for s in summaries]
    mean_corrs = [s.mean_correlation for s in summaries]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(generations, best_corrs, 'b-o', label='Best correlation', linewidth=2)
    ax.plot(generations, mean_corrs, 'g--s', label='Mean correlation', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Correlation with IRT Difficulty', fontsize=12)
    ax.set_title('Evolution Progress: Feature Correlation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotations for best points
    best_idx = np.argmax([abs(c) for c in best_corrs])
    ax.annotate(
        f'r = {best_corrs[best_idx]:+.3f}',
        xy=(generations[best_idx], best_corrs[best_idx]),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='gray'),
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved correlation plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_genealogy(
    store: FeatureStore,
    output_path: Optional[Path] = None,
):
    """Plot feature evolution genealogy.

    Args:
        store: Feature store instance.
        output_path: Optional path to save figure.
    """
    # Collect all features across generations
    all_features = []
    gen = 0
    while True:
        features = store.load_features(gen)
        if not features:
            break
        all_features.extend(features)
        gen += 1

    if not all_features:
        print("No features found to plot genealogy")
        return

    # Build parent-child relationships
    feature_map = {f.id: f for f in all_features}
    evaluations_map = {}

    for g in range(gen):
        evals = store.load_evaluations(g)
        for e in evals:
            evaluations_map[e.feature_id] = e

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Position features by generation
    gen_features: Dict[int, List[Feature]] = {}
    for f in all_features:
        if f.generation not in gen_features:
            gen_features[f.generation] = []
        gen_features[f.generation].append(f)

    positions = {}
    for g, features in gen_features.items():
        for i, f in enumerate(features):
            x = g
            y = i - len(features) / 2
            positions[f.id] = (x, y)

    # Draw edges (parent-child relationships)
    for f in all_features:
        if f.parent_id and f.parent_id in positions:
            if "+" in f.parent_id:  # EDA crossover
                parents = f.parent_id.split("+")
                for parent_id in parents:
                    if parent_id in positions:
                        px, py = positions[parent_id]
                        cx, cy = positions[f.id]
                        ax.plot([px, cx], [py, cy], 'gray', alpha=0.3, linewidth=0.5)
            else:
                px, py = positions[f.parent_id]
                cx, cy = positions[f.id]
                ax.plot([px, cx], [py, cy], 'gray', alpha=0.5, linewidth=1)

    # Draw nodes
    for f in all_features:
        x, y = positions[f.id]
        eval_result = evaluations_map.get(f.id)
        corr = eval_result.correlation if eval_result else 0

        # Color by correlation
        color = plt.cm.RdYlGn((corr + 1) / 2)  # Map [-1, 1] to [0, 1]

        ax.scatter(x, y, c=[color], s=100, edgecolors='black', linewidth=0.5)

        # Label with name and correlation
        label = f"{f.name[:8]}\n{corr:+.2f}" if eval_result else f.name[:8]
        ax.annotate(label, (x, y), fontsize=6, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Feature Index', fontsize=12)
    ax.set_title('Feature Evolution Genealogy', fontsize=14)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Correlation', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved genealogy plot to {output_path}")
    else:
        plt.show()

    plt.close()


def generate_report(
    store: FeatureStore,
    output_path: Optional[Path] = None,
) -> str:
    """Generate a text report of evolution results.

    Args:
        store: Feature store instance.
        output_path: Optional path to save report.

    Returns:
        Report text.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EVOLUTIONARY FEATURE DISCOVERY REPORT")
    lines.append("=" * 70)

    # Load evolution log
    log = store.load_evolution_log()
    if log:
        lines.append("\n## Configuration")
        for k, v in log.get("config", {}).items():
            lines.append(f"  {k}: {v}")

        lines.append(f"\n## API Usage")
        usage = log.get("final_usage", {})
        lines.append(f"  Total calls: {usage.get('total_calls', 'N/A')}")
        lines.append(f"  Input tokens: {usage.get('total_input_tokens', 'N/A'):,}")
        lines.append(f"  Output tokens: {usage.get('total_output_tokens', 'N/A'):,}")
        lines.append(f"  Estimated cost: ${usage.get('estimated_cost_usd', 0):.2f}")

    # Load summaries
    summaries = load_all_generations(store)
    if summaries:
        lines.append(f"\n## Evolution Progress ({len(summaries)} generations)")
        lines.append("-" * 50)
        lines.append(f"{'Gen':>4} | {'Best r':>8} | {'Mean r':>8} | {'Features':>8}")
        lines.append("-" * 50)
        for s in summaries:
            lines.append(
                f"{s.generation:>4} | {s.best_correlation:>+8.3f} | "
                f"{s.mean_correlation:>+8.3f} | {s.n_features:>8}"
            )

    # Load best features
    best_features, best_evals = store.load_best_features()
    if best_features:
        lines.append(f"\n## Best Features (Top {len(best_features)})")
        lines.append("-" * 70)

        eval_map = {e.feature_id: e for e in best_evals}
        for i, f in enumerate(best_features, 1):
            e = eval_map.get(f.id)
            corr = e.correlation if e else 0

            lines.append(f"\n### {i}. {f.name} (r = {corr:+.3f})")
            lines.append(f"   Generation: {f.generation}, Mutation: {f.mutation_type}")
            lines.append(f"   Description: {f.description}")
            lines.append(f"   Scale: 1 = {f.scale_low}")
            lines.append(f"          5 = {f.scale_high}")
            lines.append(f"   Hypothesis: {f.hypothesis}")

    # Mutation operator analysis
    if best_features:
        lines.append("\n## Mutation Operator Success")
        mutation_counts = {}
        mutation_corrs = {}
        for f in best_features:
            mt = f.mutation_type or "unknown"
            e = eval_map.get(f.id)
            if e:
                if mt not in mutation_counts:
                    mutation_counts[mt] = 0
                    mutation_corrs[mt] = []
                mutation_counts[mt] += 1
                mutation_corrs[mt].append(abs(e.correlation))

        for mt in sorted(mutation_counts.keys()):
            avg_corr = np.mean(mutation_corrs[mt]) if mutation_corrs[mt] else 0
            lines.append(f"  {mt}: {mutation_counts[mt]} features, avg |r| = {avg_corr:.3f}")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Saved report to {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description='Analyze evolution results')

    parser.add_argument('--results_dir', type=str,
                        default='llm_judge/evolutionary_results',
                        help='Directory containing evolution results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for output files (default: results_dir)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    store = FeatureStore(results_dir)

    # Generate report
    print("\nGenerating report...")
    report = generate_report(store, output_dir / "report.txt")
    print(report)

    if not args.no_plots:
        # Load summaries for plots
        summaries = load_all_generations(store)

        if summaries:
            print("\nGenerating correlation plot...")
            plot_correlation_progression(
                summaries,
                output_dir / "correlation_progression.png"
            )

            print("Generating genealogy plot...")
            plot_feature_genealogy(
                store,
                output_dir / "feature_genealogy.png"
            )

    print(f"\nAnalysis complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
