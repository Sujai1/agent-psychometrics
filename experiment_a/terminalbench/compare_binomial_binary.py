"""Fair comparison of binomial vs binary IRT training methods.

This script runs all 4 configurations of (training method × evaluation method):
1. Binomial Training + Multi-attempt AUC (expand to 5 observations per pair)
2. Binomial Training + Binary AUC (collapse to any_success = k > 0)
3. Binary Training + Multi-attempt AUC (expand using original binomial ground truth)
4. Binary Training + Binary AUC (single observation per pair)

This allows fair comparison of training methods by holding evaluation method constant.

Usage:
    # Basic comparison (5-fold CV)
    python -m experiment_a.terminalbench.compare_binomial_binary

    # Dry run to see what would be run
    python -m experiment_a.terminalbench.compare_binomial_binary --dry_run

    # Save results to JSON
    python -m experiment_a.terminalbench.compare_binomial_binary --output_json results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a.terminalbench.config import TerminalBenchConfig
from experiment_a.terminalbench.data_loader import load_task_data_from_repo
from experiment_a.shared.pipeline import ExperimentSpec, run_cross_validation
from experiment_ab_shared.dataset import _load_binomial_responses


# TerminalBench-specific LLM judge features
TERMINALBENCH_LLM_JUDGE_FEATURES = [
    "solution_in_instruction",
    "task_clarity",
    "solution_size",
    "domain_knowledge_required",
    "task_complexity",
    "logical_reasoning_required",
    "atypicality",
    "tooling_complexity",
]


def create_metadata_loader(repo_path: Path):
    """Create a metadata loader for Terminal Bench tasks."""
    def loader(task_ids: List[str]) -> Dict[str, Any]:
        return {"task_data": load_task_data_from_repo(task_ids, repo_path)}
    return loader


def print_config_table(
    title: str,
    description: str,
    cv_results: Dict[str, Any],
    methods: List[str],
    method_names: Dict[str, str],
) -> None:
    """Print a single configuration table showing all methods.

    Args:
        title: Table title (e.g., "Table 1: Binomial Training + Multi-attempt AUC")
        description: Description of the configuration
        cv_results: Dict from run_cross_validation containing method results
        methods: List of method keys to display
        method_names: Mapping from method key to display name
    """
    print(f"\n{title}")
    print(f"({description})")
    print("-" * 55)
    print(f"| {'Method':<22} | {'Mean AUC':>10} | {'Std':>8} |")
    print(f"|{'-'*24}|{'-'*12}|{'-'*10}|")

    for method in methods:
        result = cv_results.get(method, {})
        mean_auc = result.get("mean_auc")
        std_auc = result.get("std_auc")
        name = method_names.get(method, method)
        if mean_auc is not None:
            print(f"| {name:<22} | {mean_auc:>10.4f} | {std_auc:>8.4f} |")
        else:
            print(f"| {name:<22} | {'N/A':>10} | {'N/A':>8} |")


def run_comparison(
    k_folds: int = 5,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run fair comparison: 4 configurations (2 training × 2 evaluation methods).

    All configurations use k-fold cross-validation.

    Configurations:
    1. Binomial Training + Multi-attempt AUC (expand to 5 observations)
    2. Binomial Training + Binary AUC (collapse to any_success = k > 0)
    3. Binary Training + Multi-attempt AUC (expand using original binomial ground truth)
    4. Binary Training + Binary AUC (single observation per pair)

    Binary training uses pre-computed collapsed binary data where any_success = 1 if k > 0.
    Binomial training uses the full k/n successes data.

    Args:
        k_folds: Number of CV folds
        dry_run: Just print config without running

    Returns:
        Dict with comparison results for all 4 configurations
    """
    # Create configs for both training modes (using their respective defaults)
    binomial_config = TerminalBenchConfig(use_binary=False)  # k/n successes
    binary_config = TerminalBenchConfig(use_binary=True)     # any_success = 1

    # Paths
    binomial_responses_path = ROOT / binomial_config.responses_path
    repo_path = ROOT / binomial_config.repo_path

    # Create metadata loader
    metadata_loader = create_metadata_loader(repo_path)

    # Load binomial responses (needed for multi-attempt evaluation on binary-trained models)
    print("Loading binomial responses (for multi-attempt evaluation)...")
    binomial_responses = _load_binomial_responses(binomial_responses_path)
    n_agents = len(binomial_responses)
    n_pairs = sum(len(tasks) for tasks in binomial_responses.values())
    all_tasks = set()
    total_trials = 0
    for tasks in binomial_responses.values():
        all_tasks.update(tasks.keys())
        for resp in tasks.values():
            total_trials += resp["trials"]
    n_tasks = len(all_tasks)
    mean_trials = total_trials / n_pairs if n_pairs > 0 else 0

    print(f"   Agents: {n_agents}")
    print(f"   Tasks: {n_tasks}")
    print(f"   Agent-task pairs: {n_pairs}")
    print(f"   Mean trials per pair: {mean_trials:.2f}")

    if dry_run:
        print("\n[DRY RUN] Would run 4 configurations (each with {}-fold CV):".format(k_folds))
        print(f"  1. Binomial Training + Multi-attempt AUC")
        print(f"  2. Binomial Training + Binary AUC")
        print(f"  3. Binary Training + Multi-attempt AUC")
        print(f"  4. Binary Training + Binary AUC")
        return {}

    results = {
        "config": {
            "k_folds": k_folds,
        },
        "data_summary": {
            "n_agents": n_agents,
            "n_tasks": n_tasks,
            "n_pairs": n_pairs,
            "mean_trials": mean_trials,
        },
    }

    # Method display configuration
    methods = ["oracle", "embedding_predictor", "llm_judge_predictor",
               "constant_baseline", "agent_only_baseline"]
    method_names = {
        "oracle": "Oracle (true β)",
        "embedding_predictor": "Embedding",
        "llm_judge_predictor": "LLM Judge",
        "constant_baseline": "Constant (mean β)",
        "agent_only_baseline": "Agent-only",
    }

    # =========================================================================
    # Configuration 1: Binomial Training + Multi-attempt AUC
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 1: Binomial Training + Multi-attempt AUC")
    print("=" * 70)

    binomial_spec = ExperimentSpec(
        name="Binomial Train + Multi-attempt Eval",
        is_binomial=True,
        irt_cache_dir=ROOT / "chris_output/experiment_a_terminalbench/irt_splits",
        llm_judge_features=TERMINALBENCH_LLM_JUDGE_FEATURES,
    )

    config1_results = run_cross_validation(
        binomial_config, binomial_spec, ROOT, k=k_folds,
        metadata_loader=metadata_loader,
        expansion_mode=None,  # Default: expand for binomial data
    )
    results["config1_binomial_train_expand_eval"] = config1_results

    # =========================================================================
    # Configuration 2: Binomial Training + Binary AUC
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 2: Binomial Training + Binary AUC")
    print("=" * 70)

    binomial_spec_binary_eval = ExperimentSpec(
        name="Binomial Train + Binary Eval",
        is_binomial=True,
        irt_cache_dir=ROOT / "chris_output/experiment_a_terminalbench/irt_splits",
        llm_judge_features=TERMINALBENCH_LLM_JUDGE_FEATURES,
    )

    config2_results = run_cross_validation(
        binomial_config, binomial_spec_binary_eval, ROOT, k=k_folds,
        metadata_loader=metadata_loader,
        expansion_mode="binary",  # Override: collapse to any_success
    )
    results["config2_binomial_train_binary_eval"] = config2_results

    # =========================================================================
    # Configuration 3: Binary Training + Multi-attempt AUC
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 3: Binary Training + Multi-attempt AUC")
    print("=" * 70)

    binary_spec_expand_eval = ExperimentSpec(
        name="Binary Train + Multi-attempt Eval",
        is_binomial=False,
        irt_cache_dir=ROOT / "chris_output/experiment_a_terminalbench_binary/irt_splits",
        llm_judge_features=TERMINALBENCH_LLM_JUDGE_FEATURES,
    )

    config3_results = run_cross_validation(
        binary_config, binary_spec_expand_eval, ROOT, k=k_folds,
        metadata_loader=metadata_loader,
        expansion_mode="expand",  # Override: expand using binomial ground truth
        binomial_responses=binomial_responses,  # Required for expand mode
    )
    results["config3_binary_train_expand_eval"] = config3_results

    # =========================================================================
    # Configuration 4: Binary Training + Binary AUC
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 4: Binary Training + Binary AUC")
    print("=" * 70)

    binary_spec_binary_eval = ExperimentSpec(
        name="Binary Train + Binary Eval",
        is_binomial=False,
        irt_cache_dir=ROOT / "chris_output/experiment_a_terminalbench_binary/irt_splits",
        llm_judge_features=TERMINALBENCH_LLM_JUDGE_FEATURES,
    )

    config4_results = run_cross_validation(
        binary_config, binary_spec_binary_eval, ROOT, k=k_folds,
        metadata_loader=metadata_loader,
        expansion_mode=None,  # Default: binary for binary data
    )
    results["config4_binary_train_binary_eval"] = config4_results

    # =========================================================================
    # Print Summary: 4 Tables (one per configuration)
    # =========================================================================
    print("\n" + "=" * 70)
    print("FAIR COMPARISON: Training Method vs Evaluation Method")
    print("=" * 70)

    print_config_table(
        "Table 1: Binomial Training + Multi-attempt AUC",
        "Train with binomial likelihood (k/n), evaluate by expanding to 5 observations per pair",
        config1_results.get("cv_results", {}),
        methods, method_names,
    )

    print_config_table(
        "Table 2: Binomial Training + Binary AUC",
        "Train with binomial likelihood (k/n), evaluate with any_success = k > 0",
        config2_results.get("cv_results", {}),
        methods, method_names,
    )

    print_config_table(
        "Table 3: Binary Training + Multi-attempt AUC",
        "Train with binary likelihood (any_success), evaluate by expanding to 5 observations",
        config3_results.get("cv_results", {}),
        methods, method_names,
    )

    print_config_table(
        "Table 4: Binary Training + Binary AUC",
        "Train with binary likelihood (any_success), evaluate with 1 observation per pair",
        config4_results.get("cv_results", {}),
        methods, method_names,
    )

    # =========================================================================
    # Print Key Comparisons
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)

    print("\n** Holding evaluation method constant (compare training methods): **")
    print("Tables 1 vs 3 (Multi-attempt AUC): Does binomial training help?")
    print("Tables 2 vs 4 (Binary AUC): Does binomial training help?")

    print("\n** Holding training method constant (compare evaluation methods): **")
    print("Tables 1 vs 2 (Binomial training): How much does evaluation affect AUC?")
    print("Tables 3 vs 4 (Binary training): How much does evaluation affect AUC?")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fair comparison of binomial vs binary IRT training methods"
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show configuration without running",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save comparison results JSON",
    )
    args = parser.parse_args()

    # Run comparison
    results = run_comparison(
        k_folds=args.k_folds,
        dry_run=args.dry_run,
    )

    # Save results
    if args.output_json and results:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        with open(output_path, "w") as f:
            json.dump(convert_numpy(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
