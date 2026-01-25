"""Analyze stacked predictor coefficients across all 4 datasets.

This script runs 5-fold CV for selected methods only and extracts LLM judge
coefficients from the stacked predictor to analyze:
1. Which features have highest/lowest coefficients
2. Whether feature importance is consistent across datasets
3. What percentage of predictions come from embeddings vs LLM judge

Usage:
    python -m experiment_a.analyze_stacked_coefficients
    python -m experiment_a.analyze_stacked_coefficients --dataset swebench
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiment_ab_shared.feature_source import (
    EmbeddingFeatureSource,
    CSVFeatureSource,
    build_feature_sources,
)
from experiment_ab_shared.feature_predictor import (
    FeatureBasedPredictor,
    StackedResidualPredictor,
)
from experiment_ab_shared import load_dataset_for_fold
from experiment_ab_shared.dataset import _load_binary_responses, _load_binomial_responses

from experiment_a.shared.cross_validation import (
    k_fold_split_tasks,
    run_cv,
    CrossValidationResult,
)
from experiment_a.shared.baselines import (
    AgentOnlyPredictor,
    ConstantPredictor,
    OraclePredictor,
    DifficultyPredictorAdapter,
)
from experiment_a.shared.pipeline import ExperimentSpec

# Import dataset configs
from experiment_a.swebench.config import ExperimentAConfig
from experiment_a.swebench_pro.config import SWEBenchProConfig
from experiment_a.gso.config import GSOConfig
from experiment_a.terminalbench.config import TerminalBenchConfig

ROOT = Path(__file__).resolve().parents[1]


# Dataset configurations with unified feature paths
DATASETS = {
    "swebench": {
        "config_class": ExperimentAConfig,
        "display_name": "SWE-bench Verified",
        "is_binomial": False,
        "irt_cache_dir": ROOT / "chris_output" / "experiment_a" / "irt_splits",
        "unified_llm_path": ROOT / "chris_output" / "llm_judge_features" / "swebench_unified" / "llm_judge_features.csv",
    },
    "swebench_pro": {
        "config_class": SWEBenchProConfig,
        "display_name": "SWE-bench Pro",
        "is_binomial": False,
        "irt_cache_dir": ROOT / "chris_output" / "experiment_a_swebench_pro" / "irt_splits",
        "unified_llm_path": ROOT / "chris_output" / "llm_judge_features" / "swebench_pro_unified" / "llm_judge_features.csv",
    },
    "gso": {
        "config_class": GSOConfig,
        "display_name": "GSO",
        "is_binomial": False,
        "irt_cache_dir": ROOT / "chris_output" / "experiment_a_gso" / "irt_splits",
        "unified_llm_path": ROOT / "chris_output" / "llm_judge_features" / "gso_unified" / "llm_judge_features.csv",
    },
    "terminalbench": {
        "config_class": TerminalBenchConfig,
        "display_name": "TerminalBench",
        "is_binomial": True,  # Default binomial mode
        "irt_cache_dir": ROOT / "chris_output" / "experiment_a_terminalbench" / "irt_splits",
        "unified_llm_path": ROOT / "chris_output" / "llm_judge_features" / "terminalbench_unified" / "llm_judge_features.csv",
    },
}


@dataclass
class CoefficientAnalysis:
    """Analysis of LLM judge coefficients from stacked predictor."""

    feature_names: List[str]
    mean_coef: np.ndarray  # Standardized coefficients (mean across folds)
    std_coef: np.ndarray   # Std across folds
    mean_unscaled_coef: np.ndarray  # Unscaled coefficients (per 1-unit change)
    std_unscaled_coef: np.ndarray


@dataclass
class ContributionAnalysis:
    """Variance contribution analysis for embeddings vs LLM judge."""

    # Train set
    train_embedding_pct: float
    train_llm_pct: float
    train_embedding_var: float
    train_llm_var: float

    # Test set
    test_embedding_pct: float
    test_llm_pct: float
    test_embedding_var: float
    test_llm_var: float


@dataclass
class DatasetResults:
    """Results for one dataset."""

    dataset_name: str
    display_name: str
    n_tasks: int
    n_agents: int

    # AUC results per method
    auc_results: Dict[str, CrossValidationResult]

    # Coefficient analysis (from stacked predictor)
    coefficient_analysis: CoefficientAnalysis

    # Contribution analysis (per fold, then averaged)
    contribution_analysis: ContributionAnalysis


def extract_coefficients(predictor: StackedResidualPredictor) -> Dict[str, Any]:
    """Extract and interpret coefficients from fitted stacked predictor."""
    if not predictor._is_fitted:
        raise RuntimeError("Predictor must be fitted")

    # Raw coefficients (for standardized features)
    coef = predictor._residual_model.coef_
    feature_names = predictor.residual_source.feature_names

    # Convert to unscaled (effect per 1-unit change in raw feature)
    scale = predictor._residual_scaler.scale_
    unscaled_coef = coef / scale

    return {
        "feature_names": list(feature_names),
        "standardized_coef": coef.copy(),
        "unscaled_coef": unscaled_coef.copy(),
    }


def compute_contributions(
    predictor: StackedResidualPredictor,
    task_ids: List[str],
) -> Dict[str, float]:
    """Compute variance contribution from embedding vs LLM judge."""
    if not predictor._is_fitted:
        raise RuntimeError("Predictor must be fitted")

    # Get features
    X_base = predictor.base_source.get_features(task_ids)
    X_base_scaled = predictor._base_scaler.transform(X_base)
    X_residual = predictor.residual_source.get_features(task_ids)
    X_residual_scaled = predictor._residual_scaler.transform(X_residual)

    # Get component predictions
    base_preds = predictor._base_model.predict(X_base_scaled)
    residual_preds = predictor._residual_model.predict(X_residual_scaled)

    # Compute variances
    base_var = float(np.var(base_preds))
    residual_var = float(np.var(residual_preds))
    total_var = float(np.var(base_preds + residual_preds))

    # Avoid division by zero
    if total_var < 1e-10:
        return {
            "embedding_var": base_var,
            "llm_var": residual_var,
            "embedding_pct": 50.0,
            "llm_pct": 50.0,
        }

    return {
        "embedding_var": base_var,
        "llm_var": residual_var,
        "embedding_pct": base_var / total_var * 100,
        "llm_pct": residual_var / total_var * 100,
    }


def run_analysis_for_dataset(
    dataset_name: str,
    k_folds: int = 5,
    verbose: bool = True,
) -> DatasetResults:
    """Run CV and extract coefficients for one dataset."""

    dataset_info = DATASETS[dataset_name]
    config_class = dataset_info["config_class"]
    display_name = dataset_info["display_name"]
    is_binomial = dataset_info["is_binomial"]
    irt_cache_dir = dataset_info["irt_cache_dir"]
    unified_llm_path = dataset_info["unified_llm_path"]

    # Create config with unified LLM features
    config = config_class(llm_judge_features_path=unified_llm_path)

    # Resolve paths
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path
    embeddings_path = ROOT / config.embeddings_path if config.embeddings_path else None
    llm_judge_path = ROOT / unified_llm_path

    if verbose:
        print(f"\n{'='*70}")
        print(f"DATASET: {display_name}")
        print(f"{'='*70}")
        print(f"LLM Judge Features: {llm_judge_path}")

    # Load full items to get all task IDs
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)
    n_tasks = len(all_task_ids)

    # Load abilities to get agent count
    abilities = pd.read_csv(abilities_path, index_col=0)
    n_agents = len(abilities)

    if verbose:
        print(f"Tasks: {n_tasks}, Agents: {n_agents}")

    # Generate k folds
    folds = k_fold_split_tasks(all_task_ids, k=k_folds, seed=config.split_seed)

    # Build feature sources
    feature_sources = build_feature_sources(
        embeddings_path=embeddings_path,
        llm_judge_path=llm_judge_path,
        llm_judge_feature_cols=None,  # Auto-detect from CSV
        verbose=False,
    )
    source_by_name = {name: source for name, source in feature_sources}

    emb_source = source_by_name.get("Embedding")
    llm_source = source_by_name.get("LLM Judge")

    if emb_source is None or llm_source is None:
        raise ValueError(f"Missing feature sources for {dataset_name}")

    if verbose:
        print(f"LLM Judge features: {llm_source.feature_names}")

    # Create fold data loader
    def load_fold_data(train_tasks: List[str], test_tasks: List[str], fold_idx: int):
        return load_dataset_for_fold(
            abilities_path=abilities_path,
            items_path=items_path,
            responses_path=responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=k_folds,
            split_seed=config.split_seed,
            is_binomial=is_binomial,
            irt_cache_dir=irt_cache_dir,
            metadata_loader=None,
            exclude_unsolved=False,
        )

    # Build predictors (only the 6 we want)
    predictors = {}

    # Oracle
    predictors["oracle"] = OraclePredictor()

    # Embedding only
    emb_predictor = FeatureBasedPredictor(
        emb_source,
        alphas=list(config.ridge_alphas) if hasattr(config, 'ridge_alphas') else [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
    )
    predictors["embedding_predictor"] = DifficultyPredictorAdapter(emb_predictor)

    # LLM Judge only
    llm_predictor = FeatureBasedPredictor(
        llm_source,
        alphas=list(config.ridge_alphas) if hasattr(config, 'ridge_alphas') else [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
    )
    predictors["llm_judge_predictor"] = DifficultyPredictorAdapter(llm_predictor)

    # Stacked (Emb → LLM)
    stacked_predictor = StackedResidualPredictor(
        base_source=emb_source,
        residual_source=llm_source,
    )
    predictors["stacked_emb_llm"] = DifficultyPredictorAdapter(stacked_predictor)

    # Baselines
    predictors["constant_baseline"] = ConstantPredictor()
    predictors["agent_only_baseline"] = AgentOnlyPredictor()

    # Run CV and collect results
    auc_results: Dict[str, CrossValidationResult] = {}

    # For stacked predictor, also collect coefficient and contribution data per fold
    fold_coefs: List[Dict[str, Any]] = []
    fold_train_contribs: List[Dict[str, float]] = []
    fold_test_contribs: List[Dict[str, float]] = []

    method_display_names = {
        "oracle": "Oracle (true b)",
        "embedding_predictor": "Embedding",
        "llm_judge_predictor": "LLM Judge",
        "stacked_emb_llm": "Stacked (Emb → LLM)",
        "constant_baseline": "Constant (mean b)",
        "agent_only_baseline": "Agent-only",
    }

    # Run manually to extract coefficients
    for name, predictor in predictors.items():
        if verbose:
            print(f"\n{method_display_names[name]}:")

        fold_aucs: List[Optional[float]] = []

        for fold_idx, (train_tasks, test_tasks) in enumerate(folds):
            # Load fold data
            data = load_fold_data(train_tasks, test_tasks, fold_idx)

            # Fit predictor
            predictor.fit(data, train_tasks)

            # For stacked predictor, extract coefficients
            if name == "stacked_emb_llm":
                inner_predictor = predictor._predictor  # Access inner StackedResidualPredictor

                # Extract coefficients
                coef_info = extract_coefficients(inner_predictor)
                fold_coefs.append(coef_info)

                # Compute contributions on train and test
                train_contrib = compute_contributions(inner_predictor, train_tasks)
                test_contrib = compute_contributions(inner_predictor, test_tasks)
                fold_train_contribs.append(train_contrib)
                fold_test_contribs.append(test_contrib)

            # Evaluate AUC
            from sklearn.metrics import roc_auc_score
            y_true: List[int] = []
            y_scores: List[float] = []

            for task_id in test_tasks:
                for agent_id in data.train_abilities.index:
                    if agent_id not in data.responses:
                        continue
                    if task_id not in data.responses[agent_id]:
                        continue

                    prob = predictor.predict_probability(data, agent_id, task_id)
                    outcomes, _ = data.expand_for_auc(agent_id, task_id, prob)
                    y_true.extend(outcomes)
                    y_scores.extend([prob] * len(outcomes))

            if len(y_true) >= 2 and len(set(y_true)) >= 2:
                auc = float(roc_auc_score(y_true, y_scores))
            else:
                auc = None
            fold_aucs.append(auc)

            if verbose:
                print(f"   Fold {fold_idx + 1}: AUC = {auc:.4f}" if auc else f"   Fold {fold_idx + 1}: AUC = N/A")

        # Aggregate AUC results
        valid_aucs = [a for a in fold_aucs if a is not None]
        auc_results[name] = CrossValidationResult(
            mean_auc=float(np.mean(valid_aucs)) if valid_aucs else None,
            std_auc=float(np.std(valid_aucs)) if valid_aucs else None,
            fold_aucs=fold_aucs,
            k=k_folds,
        )

        if verbose and valid_aucs:
            print(f"   Mean AUC: {np.mean(valid_aucs):.4f} ± {np.std(valid_aucs):.4f}")

    # Aggregate coefficient analysis across folds
    feature_names = fold_coefs[0]["feature_names"]
    all_std_coefs = np.array([c["standardized_coef"] for c in fold_coefs])
    all_unscaled_coefs = np.array([c["unscaled_coef"] for c in fold_coefs])

    coefficient_analysis = CoefficientAnalysis(
        feature_names=feature_names,
        mean_coef=np.mean(all_std_coefs, axis=0),
        std_coef=np.std(all_std_coefs, axis=0),
        mean_unscaled_coef=np.mean(all_unscaled_coefs, axis=0),
        std_unscaled_coef=np.std(all_unscaled_coefs, axis=0),
    )

    # Aggregate contribution analysis
    contribution_analysis = ContributionAnalysis(
        train_embedding_pct=float(np.mean([c["embedding_pct"] for c in fold_train_contribs])),
        train_llm_pct=float(np.mean([c["llm_pct"] for c in fold_train_contribs])),
        train_embedding_var=float(np.mean([c["embedding_var"] for c in fold_train_contribs])),
        train_llm_var=float(np.mean([c["llm_var"] for c in fold_train_contribs])),
        test_embedding_pct=float(np.mean([c["embedding_pct"] for c in fold_test_contribs])),
        test_llm_pct=float(np.mean([c["llm_pct"] for c in fold_test_contribs])),
        test_embedding_var=float(np.mean([c["embedding_var"] for c in fold_test_contribs])),
        test_llm_var=float(np.mean([c["llm_var"] for c in fold_test_contribs])),
    )

    return DatasetResults(
        dataset_name=dataset_name,
        display_name=display_name,
        n_tasks=n_tasks,
        n_agents=n_agents,
        auc_results=auc_results,
        coefficient_analysis=coefficient_analysis,
        contribution_analysis=contribution_analysis,
    )


def print_results_table(results: Dict[str, DatasetResults]) -> None:
    """Print formatted results for all datasets."""

    method_display_names = {
        "oracle": "Oracle (true b)",
        "embedding_predictor": "Embedding",
        "llm_judge_predictor": "LLM Judge",
        "stacked_emb_llm": "Stacked (Emb → LLM)",
        "constant_baseline": "Constant (mean b)",
        "agent_only_baseline": "Agent-only",
    }

    # Order for display (by expected AUC)
    method_order = [
        "oracle",
        "stacked_emb_llm",
        "embedding_predictor",
        "llm_judge_predictor",
        "constant_baseline",
        "agent_only_baseline",
    ]

    for dataset_name, result in results.items():
        print(f"\n{'='*75}")
        print(f"{result.display_name} (Unified Features)")
        print(f"{'='*75}")
        print(f"Tasks: {result.n_tasks}, Agents: {result.n_agents}")

        # AUC Table
        print(f"\nAUC Results (5-Fold CV):")
        print(f"{'Method':<35} {'Mean AUC':>12} {'Std':>10}")
        print("-" * 60)

        for method in method_order:
            if method in result.auc_results:
                auc_result = result.auc_results[method]
                if auc_result.mean_auc is not None:
                    print(f"{method_display_names[method]:<35} {auc_result.mean_auc:>12.4f} {auc_result.std_auc:>10.4f}")
                else:
                    print(f"{method_display_names[method]:<35} {'N/A':>12} {'N/A':>10}")

        # Coefficient Table
        print(f"\nLLM Judge Coefficients (Residual Stage):")
        print(f"{'Feature':<35} {'Coef (std)':<15} {'|Coef|':>8} {'Rank':>6}")
        print("-" * 70)

        coef_analysis = result.coefficient_analysis
        # Sort by absolute coefficient
        indices = np.argsort(np.abs(coef_analysis.mean_coef))[::-1]

        for rank, idx in enumerate(indices, 1):
            name = coef_analysis.feature_names[idx]
            coef = coef_analysis.mean_coef[idx]
            std = coef_analysis.std_coef[idx]
            abs_coef = abs(coef)
            coef_str = f"{coef:+.3f}±{std:.3f}"
            print(f"{name:<35} {coef_str:<15} {abs_coef:>8.3f} {rank:>6}")

        # Contribution Analysis
        print(f"\nContribution Analysis:")
        contrib = result.contribution_analysis
        print(f"{'':20} {'Train Set':>12} {'Test Set':>12}")
        print("-" * 50)
        print(f"{'Embedding base:':<20} {contrib.train_embedding_pct:>10.1f}% {contrib.test_embedding_pct:>10.1f}%")
        print(f"{'LLM residual:':<20} {contrib.train_llm_pct:>10.1f}% {contrib.test_llm_pct:>10.1f}%")


def print_cross_dataset_comparison(results: Dict[str, DatasetResults]) -> None:
    """Print cross-dataset comparison of feature importance."""

    print(f"\n{'='*85}")
    print("CROSS-DATASET COEFFICIENT COMPARISON")
    print(f"{'='*85}")

    # Get all feature names (should be the same across datasets with unified features)
    all_features = set()
    for result in results.values():
        all_features.update(result.coefficient_analysis.feature_names)
    all_features = sorted(all_features)

    # Build ranking table
    print(f"\nFeature Importance Ranking (by |coefficient|):")

    # Header
    dataset_abbrevs = {
        "swebench": "SWE",
        "swebench_pro": "Pro",
        "gso": "GSO",
        "terminalbench": "Term",
    }

    header = f"{'Feature':<32}"
    for ds in results.keys():
        header += f" {dataset_abbrevs.get(ds, ds[:4]):>6}"
    header += f" {'Avg':>6}"
    print(header)
    print("-" * (32 + 7 * (len(results) + 1)))

    # Build rankings per dataset
    rankings: Dict[str, Dict[str, int]] = {}
    for dataset_name, result in results.items():
        coef_analysis = result.coefficient_analysis
        indices = np.argsort(np.abs(coef_analysis.mean_coef))[::-1]
        rankings[dataset_name] = {
            coef_analysis.feature_names[idx]: rank + 1
            for rank, idx in enumerate(indices)
        }

    # Compute average rank and sort features by it
    avg_ranks = {}
    for feature in all_features:
        ranks = [rankings[ds].get(feature, len(all_features)) for ds in results.keys()]
        avg_ranks[feature] = np.mean(ranks)

    sorted_features = sorted(all_features, key=lambda f: avg_ranks[f])

    for feature in sorted_features:
        row = f"{feature:<32}"
        for ds in results.keys():
            rank = rankings[ds].get(feature, "-")
            if isinstance(rank, int):
                row += f" {rank:>6}"
            else:
                row += f" {rank:>6}"
        row += f" {avg_ranks[feature]:>6.1f}"
        print(row)

    # Print mean absolute coefficients across datasets
    print(f"\nMean Absolute Coefficient by Feature:")
    header = f"{'Feature':<32}"
    for ds in results.keys():
        header += f" {dataset_abbrevs.get(ds, ds[:4]):>8}"
    header += f" {'Mean':>8}"
    print(header)
    print("-" * (32 + 9 * (len(results) + 1)))

    for feature in sorted_features:
        row = f"{feature:<32}"
        coefs = []
        for ds, result in results.items():
            coef_analysis = result.coefficient_analysis
            if feature in coef_analysis.feature_names:
                idx = coef_analysis.feature_names.index(feature)
                coef = abs(coef_analysis.mean_coef[idx])
                coefs.append(coef)
                row += f" {coef:>8.3f}"
            else:
                row += f" {'N/A':>8}"
        row += f" {np.mean(coefs):>8.3f}" if coefs else f" {'N/A':>8}"
        print(row)

    # Contribution summary
    print(f"\nContribution Summary (Test Set):")
    header = f"{'':20}"
    for ds in results.keys():
        header += f" {dataset_abbrevs.get(ds, ds[:4]):>8}"
    print(header)
    print("-" * (20 + 9 * len(results)))

    row = f"{'Embedding %':<20}"
    for ds, result in results.items():
        row += f" {result.contribution_analysis.test_embedding_pct:>7.1f}%"
    print(row)

    row = f"{'LLM Judge %':<20}"
    for ds, result in results.items():
        row += f" {result.contribution_analysis.test_llm_pct:>7.1f}%"
    print(row)


def save_results(results: Dict[str, DatasetResults], output_path: Path) -> None:
    """Save results to JSON file."""

    def convert_for_json(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif hasattr(obj, '__dataclass_fields__'):
            return {k: convert_for_json(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    output_data = convert_for_json(results)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze stacked predictor coefficients across datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        default=None,
        help="Run analysis for single dataset (default: all datasets)",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(ROOT / "chris_output" / "stacked_coefficient_analysis.json"),
        help="Output path for JSON results",
    )
    args = parser.parse_args()

    # Run analysis
    datasets_to_run = [args.dataset] if args.dataset else list(DATASETS.keys())

    results: Dict[str, DatasetResults] = {}
    for dataset_name in datasets_to_run:
        results[dataset_name] = run_analysis_for_dataset(
            dataset_name,
            k_folds=args.k_folds,
            verbose=True,
        )

    # Print results
    print_results_table(results)

    if len(results) > 1:
        print_cross_dataset_comparison(results)

    # Save results
    save_results(results, Path(args.output_path))


if __name__ == "__main__":
    main()
