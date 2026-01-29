"""Aggregate results from interaction architecture sweep parts."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT / "chris_output/experiment_a/mlp_embedding"


def load_and_merge() -> dict:
    """Load part1 and part2 JSON files and merge them."""
    results = {}

    for part in [1, 2]:
        path = OUTPUT_DIR / f"interaction_sweep_part{part}.json"
        if path.exists():
            with open(path) as f:
                part_results = json.load(f)
            results.update(part_results)
            print(f"Loaded {len(part_results)} results from {path.name}")
        else:
            print(f"Warning: {path.name} not found")

    return results


def main():
    results = load_and_merge()

    if not results:
        print("No results found!")
        return

    print(f"\nTotal configs: {len(results)}")

    # Sort by test AUC
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["mean_auc"],
        reverse=True
    )

    print("\n" + "=" * 85)
    print("INTERACTION ARCHITECTURE SWEEP - AGGREGATED RESULTS")
    print("=" * 85)
    print(f"\n{'Method':<45} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 85)

    for name, r in sorted_results:
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc")
        display_name = r["display_name"]

        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{display_name:<45} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{display_name:<45} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Key comparisons
    print(f"\n{'-' * 85}")
    print("KEY COMPARISONS:")

    if "ridge" in results:
        ridge_auc = results["ridge"]["mean_auc"]
        print(f"  Ridge baseline: {ridge_auc:.4f}")

        # Find best interaction model (exclude baselines)
        interaction_results = [(n, r) for n, r in results.items()
                               if n not in ["oracle", "ridge", "fullmlp_64"]]
        if interaction_results:
            best_name, best_r = max(interaction_results, key=lambda x: x[1]["mean_auc"])
            delta = best_r["mean_auc"] - ridge_auc
            print(f"  Best interaction: {best_r['display_name']}: {best_r['mean_auc']:.4f} ({delta:+.4f} vs Ridge)")

    if "fullmlp_64" in results:
        print(f"  FullMLP baseline: {results['fullmlp_64']['mean_auc']:.4f}")

    if "oracle" in results:
        print(f"  Oracle upper bound: {results['oracle']['mean_auc']:.4f}")

    # Group by architecture type
    print(f"\n{'-' * 85}")
    print("BY ARCHITECTURE TYPE:")

    for arch_type in ["two_tower", "bilinear", "ncf", "multiplicative", "agent_emb"]:
        arch_results = [(n, r) for n, r in results.items() if n.startswith(arch_type)]
        if arch_results:
            best_name, best_r = max(arch_results, key=lambda x: x[1]["mean_auc"])
            print(f"  Best {arch_type}: {best_r['display_name']}: {best_r['mean_auc']:.4f}")


if __name__ == "__main__":
    main()
