"""Compare cross-validation AUC between binomial (5 runs) and sampled binary (1 run) IRT.

This script compares how Terminal Bench cross-validation AUC changes when:
1. Using all 5 runs per agent-task pair with binomial IRT (current approach)
2. Sampling 1 binary outcome per pair and using standard binary IRT

Usage:
    # Basic comparison (single sample)
    python -m experiment_a_terminalbench.compare_binomial_binary

    # Multiple samples for confidence intervals
    python -m experiment_a_terminalbench.compare_binomial_binary --n_samples 10

    # Custom seed
    python -m experiment_a_terminalbench.compare_binomial_binary --sample_seed 42

    # Dry run
    python -m experiment_a_terminalbench.compare_binomial_binary --dry_run
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a.terminalbench.config import TerminalBenchConfig
from experiment_a.terminalbench.data_loader import load_task_data_from_repo
from experiment_a.shared.pipeline import ExperimentSpec, run_cross_validation
from experiment_ab_shared.sampling import (
    sample_binary_from_binomial,
    load_binomial_responses,
    save_binary_responses,
)


@dataclass
class SampledBinaryConfig:
    """Configuration for sampled binary experiment.

    This config points to sampled binary data instead of binomial data.
    """

    # Data paths (standard binary IRT model outputs)
    abilities_path: Path
    items_path: Path
    responses_path: Path  # Sampled binary responses
    repo_path: Path = Path("terminal-bench")
    output_dir: Path = Path("chris_output/experiment_a_terminalbench/sampled_binary")

    # Train/test splitting
    test_fraction: float = 0.2
    split_seed: int = 0

    # Embedding predictor config
    embeddings_path: Optional[Path] = None
    ridge_alphas: tuple = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)

    # LLM Judge predictor config
    llm_judge_features_path: Optional[Path] = None
    llm_judge_ridge_alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    llm_judge_max_features: Optional[int] = None

    # Task filtering
    exclude_unsolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d


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


def train_binary_irt(
    responses_path: Path,
    output_dir: Path,
    epochs: int = 2000,
    force_retrain: bool = False,
) -> Path:
    """Train standard binary IRT on sampled responses.

    Args:
        responses_path: Path to sampled binary JSONL
        output_dir: Directory to save IRT outputs
        epochs: Training epochs
        force_retrain: Force retrain even if cached

    Returns:
        Path to IRT output directory (contains abilities.csv, items.csv)
    """
    irt_dir = output_dir / "1d"
    abilities_path = irt_dir / "abilities.csv"
    items_path = irt_dir / "items.csv"

    if not force_retrain and abilities_path.exists() and items_path.exists():
        print(f"Found cached binary IRT model at {irt_dir}")
        return irt_dir

    print(f"Training binary IRT on {responses_path}...")
    irt_dir.mkdir(parents=True, exist_ok=True)

    from py_irt.config import IrtConfig
    from py_irt.training import IrtModelTrainer

    config = IrtConfig(
        model_type="1pl",
        epochs=epochs,
        priors="hierarchical",
        dims=1,
    )

    trainer = IrtModelTrainer(
        data_path=responses_path,
        config=config,
    )
    n_subjects = len(trainer._dataset.subject_ids)
    n_items = len(trainer._dataset.item_ids)
    print(f"   Dataset: {n_subjects} subjects, {n_items} items")

    trainer.train(device="cpu")

    # Save results
    abilities = trainer.best_params["ability"]
    difficulties = trainer.best_params["diff"]
    item_ids = trainer.best_params["item_ids"]
    subject_ids = trainer.best_params["subject_ids"]

    abilities_df = pd.DataFrame({
        "theta": [abilities[i] for i in range(len(subject_ids))],
    }, index=[subject_ids[i] for i in range(len(subject_ids))])
    abilities_df.index.name = "subject_id"
    abilities_df.to_csv(abilities_path)

    items_df = pd.DataFrame({
        "b": [difficulties[i] for i in range(len(item_ids))],
    }, index=[item_ids[i] for i in range(len(item_ids))])
    items_df.index.name = "item_id"
    items_df.to_csv(items_path)

    print(f"   Saved abilities: {abilities_path}")
    print(f"   Saved items: {items_path}")

    return irt_dir


def run_comparison(
    binomial_config: TerminalBenchConfig,
    n_samples: int = 1,
    sample_seed: int = 0,
    k_folds: int = 5,
    force_retrain: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run comparison between binomial and sampled binary IRT.

    Args:
        binomial_config: Configuration for binomial experiment
        n_samples: Number of binary samplings for confidence intervals
        sample_seed: Base seed for sampling (sample i uses seed = base + i)
        k_folds: Number of CV folds
        force_retrain: Force retrain IRT models
        dry_run: Just print config without running

    Returns:
        Dict with comparison results
    """
    # Paths
    binomial_responses_path = ROOT / binomial_config.responses_path
    repo_path = ROOT / binomial_config.repo_path
    output_base = ROOT / "chris_output/experiment_a_terminalbench/binomial_vs_binary"
    output_base.mkdir(parents=True, exist_ok=True)

    # Create metadata loader
    metadata_loader = create_metadata_loader(repo_path)

    # Load binomial responses
    print("Loading binomial responses...")
    binomial_responses = load_binomial_responses(binomial_responses_path)
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
        print("\n[DRY RUN] Would run:")
        print(f"  - Binomial IRT cross-validation ({k_folds} folds)")
        print(f"  - {n_samples} sampled binary IRT cross-validations")
        print(f"  - Sample seeds: {sample_seed} to {sample_seed + n_samples - 1}")
        return {}

    results = {
        "config": {
            "n_samples": n_samples,
            "sample_seed": sample_seed,
            "k_folds": k_folds,
        },
        "data_summary": {
            "n_agents": n_agents,
            "n_tasks": n_tasks,
            "n_pairs": n_pairs,
            "mean_trials": mean_trials,
        },
    }

    # =========================================================================
    # Run binomial IRT cross-validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("BINOMIAL IRT (all trials per pair)")
    print("=" * 70)

    binomial_spec = ExperimentSpec(
        name="TerminalBench (Binomial)",
        is_binomial=True,
        irt_cache_dir=ROOT / "chris_output/experiment_a_terminalbench/irt_splits",
        llm_judge_features=TERMINALBENCH_LLM_JUDGE_FEATURES,
    )

    binomial_results = run_cross_validation(
        binomial_config, binomial_spec, ROOT, k=k_folds, metadata_loader=metadata_loader
    )
    results["binomial"] = binomial_results

    # =========================================================================
    # Run sampled binary IRT cross-validation(s)
    # =========================================================================
    results["sampled_binary"] = []

    for sample_i in range(n_samples):
        current_seed = sample_seed + sample_i
        print("\n" + "=" * 70)
        print(f"SAMPLED BINARY IRT (1 trial per pair, seed={current_seed})")
        print("=" * 70)

        # Sample binary responses
        print(f"\nSampling binary responses (seed={current_seed})...")
        binary_responses = sample_binary_from_binomial(binomial_responses, seed=current_seed)

        # Save sampled responses
        sample_dir = output_base / f"sampled_seed{current_seed}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sampled_responses_path = sample_dir / "sampled_responses.jsonl"
        save_binary_responses(binary_responses, sampled_responses_path)
        print(f"   Saved to: {sampled_responses_path}")

        # Count sampled successes for sanity check
        n_successes = sum(sum(tasks.values()) for tasks in binary_responses.values())
        print(f"   Sampled successes: {n_successes}/{n_pairs} ({100*n_successes/n_pairs:.1f}%)")

        # Train binary IRT on sampled data
        irt_dir = train_binary_irt(
            sampled_responses_path,
            sample_dir,
            force_retrain=force_retrain,
        )

        # Create config for sampled binary
        sampled_config = SampledBinaryConfig(
            abilities_path=irt_dir / "abilities.csv",
            items_path=irt_dir / "items.csv",
            responses_path=sampled_responses_path,
            repo_path=binomial_config.repo_path,
            output_dir=sample_dir,
            test_fraction=binomial_config.test_fraction,
            split_seed=binomial_config.split_seed,
            embeddings_path=binomial_config.embeddings_path,
            llm_judge_features_path=binomial_config.llm_judge_features_path,
            llm_judge_max_features=binomial_config.llm_judge_max_features,
        )

        # Create spec for sampled binary (is_binomial=False)
        sampled_spec = ExperimentSpec(
            name=f"TerminalBench (Sampled Binary, seed={current_seed})",
            is_binomial=False,  # Key difference: use binary IRT
            irt_cache_dir=sample_dir / "irt_splits",
            llm_judge_features=TERMINALBENCH_LLM_JUDGE_FEATURES,
        )

        # Run cross-validation
        sampled_results = run_cross_validation(
            sampled_config, sampled_spec, ROOT, k=k_folds, metadata_loader=metadata_loader
        )
        sampled_results["seed"] = current_seed
        results["sampled_binary"].append(sampled_results)

    # =========================================================================
    # Print comparison summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    # Extract AUC values
    methods = ["oracle", "embedding_predictor", "llm_judge_predictor", "constant_baseline", "agent_only_baseline"]
    method_names = {
        "oracle": "Oracle (true b)",
        "embedding_predictor": "Embedding",
        "llm_judge_predictor": "LLM Judge",
        "constant_baseline": "Constant (mean b)",
        "agent_only_baseline": "Agent-only",
    }

    # Get binomial AUCs
    binomial_aucs = {}
    for method in methods:
        cv_result = binomial_results.get("cv_results", {}).get(method, {})
        binomial_aucs[method] = cv_result.get("mean_auc")

    # Get sampled binary AUCs (average over samples)
    sampled_aucs = {method: [] for method in methods}
    for sample_result in results["sampled_binary"]:
        for method in methods:
            cv_result = sample_result.get("cv_results", {}).get(method, {})
            auc = cv_result.get("mean_auc")
            if auc is not None:
                sampled_aucs[method].append(auc)

    # Compute mean and std for sampled
    sampled_mean = {}
    sampled_std = {}
    for method in methods:
        if sampled_aucs[method]:
            sampled_mean[method] = np.mean(sampled_aucs[method])
            sampled_std[method] = np.std(sampled_aucs[method]) if len(sampled_aucs[method]) > 1 else 0.0
        else:
            sampled_mean[method] = None
            sampled_std[method] = None

    # Print table
    if n_samples > 1:
        print(f"\n{'Method':<25} {'Binomial':>12} {'Binary (n={})':>18} {'Delta':>10}".format(n_samples))
    else:
        print(f"\n{'Method':<25} {'Binomial':>12} {'Binary':>12} {'Delta':>10}")
    print("-" * 60)

    for method in methods:
        name = method_names.get(method, method)
        bin_auc = binomial_aucs.get(method)
        samp_mean = sampled_mean.get(method)
        samp_std = sampled_std.get(method)

        if bin_auc is None or samp_mean is None:
            continue

        delta = samp_mean - bin_auc

        if n_samples > 1 and samp_std is not None:
            samp_str = f"{samp_mean:.4f} ± {samp_std:.4f}"
        else:
            samp_str = f"{samp_mean:.4f}"

        print(f"{name:<25} {bin_auc:>12.4f} {samp_str:>18} {delta:>+10.4f}")

    # Store comparison in results
    results["comparison"] = {
        method: {
            "binomial_auc": binomial_aucs.get(method),
            "binary_mean_auc": sampled_mean.get(method),
            "binary_std_auc": sampled_std.get(method),
            "delta": sampled_mean.get(method) - binomial_aucs.get(method)
                if sampled_mean.get(method) is not None and binomial_aucs.get(method) is not None
                else None,
        }
        for method in methods
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare binomial (5 runs) vs sampled binary (1 run) IRT cross-validation"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of binary samplings for confidence intervals (default: 1)",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="Base seed for sampling (sample i uses seed = base + i)",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Random seed for train/test split (default: 0)",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=None,
        help="Path to pre-computed embeddings .npz file",
    )
    parser.add_argument(
        "--llm_judge_features_path",
        type=str,
        default=None,
        help="Path to LLM judge features CSV file",
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retrain IRT models even if cached",
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

    # Build binomial config
    config_kwargs = {
        "split_seed": args.split_seed,
    }
    if args.embeddings_path:
        config_kwargs["embeddings_path"] = Path(args.embeddings_path)
    if args.llm_judge_features_path:
        config_kwargs["llm_judge_features_path"] = Path(args.llm_judge_features_path)

    binomial_config = TerminalBenchConfig(**config_kwargs)

    # Run comparison
    results = run_comparison(
        binomial_config,
        n_samples=args.n_samples,
        sample_seed=args.sample_seed,
        k_folds=args.k_folds,
        force_retrain=args.force_retrain,
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
