"""Diagnostic script for debugging combined feature sources in Experiment A.

This script analyzes why combining embeddings and LLM judge features
performs worse than individual sources. It provides:
1. Prediction correlation analysis
2. Coefficient and alpha analysis
3. Alpha grid sweep

Usage:
    python -m experiment_a.swebench.debug_combined_features
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error

from experiment_a.swebench.config import ExperimentAConfig
from experiment_a.shared.cross_validation import k_fold_split_tasks
from experiment_ab_shared import load_dataset_for_fold
from experiment_ab_shared.feature_source import (
    EmbeddingFeatureSource,
    CSVFeatureSource,
    RegularizedFeatureSource,
    GroupedFeatureSource,
    build_feature_sources,
)
from experiment_ab_shared.feature_predictor import (
    FeatureBasedPredictor,
    GroupedRidgePredictor,
)
from experiment_ab_shared.evaluator import compute_auc

# Root directory
ROOT = Path(__file__).resolve().parents[2]


def run_prediction_analysis(
    config: ExperimentAConfig,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Analyze prediction correlations between individual and combined predictors.

    Returns:
        Dictionary with correlation metrics and per-fold predictions
    """
    print("=" * 60)
    print("PREDICTION CORRELATION ANALYSIS")
    print("=" * 60)

    # Resolve paths
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path
    embeddings_path = ROOT / config.embeddings_path if config.embeddings_path else None
    llm_judge_path = ROOT / config.llm_judge_features_path if config.llm_judge_features_path else None

    # Build feature sources
    feature_sources = build_feature_sources(
        embeddings_path=embeddings_path,
        llm_judge_path=llm_judge_path,
        llm_judge_feature_cols=None,  # Auto-detect
        verbose=False,
    )
    source_by_name = {name: source for name, source in feature_sources}

    if "Embedding" not in source_by_name or "LLM Judge" not in source_by_name:
        raise ValueError("Both Embedding and LLM Judge sources required")

    emb_source = source_by_name["Embedding"]
    llm_source = source_by_name["LLM Judge"]

    print(f"  Embedding features: {emb_source.feature_dim}")
    print(f"  LLM Judge features: {llm_source.feature_dim}")
    if llm_source.feature_names:
        print(f"    Features: {llm_source.feature_names}")

    # Get all task IDs from items file
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    print(f"\nTotal tasks: {len(all_task_ids)}")

    # Generate folds
    folds = k_fold_split_tasks(all_task_ids, k=n_folds, seed=config.split_seed)

    # Results storage
    all_results = []

    for fold_idx, (train_tasks, test_tasks) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        print(f"  Train: {len(train_tasks)}, Test: {len(test_tasks)}")

        # Load data for this fold
        data = load_dataset_for_fold(
            abilities_path=abilities_path,
            items_path=items_path,
            responses_path=responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=n_folds,
            split_seed=config.split_seed,
            is_binomial=False,
            irt_cache_dir=ROOT / "chris_output" / "experiment_a" / "irt_splits",
        )

        # Get ground truth difficulties for training
        train_difficulties = data.get_train_difficulties()

        # Create individual predictors
        emb_predictor = FeatureBasedPredictor(emb_source, ridge_alphas=list(config.ridge_alphas))
        llm_predictor = FeatureBasedPredictor(llm_source, ridge_alphas=list(config.ridge_alphas))

        # Create grouped predictor
        grouped_source = GroupedFeatureSource([
            RegularizedFeatureSource(emb_source),
            RegularizedFeatureSource(llm_source),
        ])
        grouped_predictor = GroupedRidgePredictor(grouped_source)

        # Fit predictors
        emb_predictor.fit(train_tasks, train_difficulties)
        llm_predictor.fit(train_tasks, train_difficulties)
        grouped_predictor.fit(train_tasks, train_difficulties)

        # Get predictions on TEST tasks
        # Note: We need ground truth from full IRT for test tasks
        test_true = data.full_items.loc[test_tasks, "b"].values

        emb_preds = emb_predictor.predict(test_tasks)
        llm_preds = llm_predictor.predict(test_tasks)
        grouped_preds = grouped_predictor.predict(test_tasks)

        # Convert to arrays in same order
        emb_arr = np.array([emb_preds[t] for t in test_tasks])
        llm_arr = np.array([llm_preds[t] for t in test_tasks])
        grouped_arr = np.array([grouped_preds[t] for t in test_tasks])

        # Compute correlations
        emb_llm_pearson = stats.pearsonr(emb_arr, llm_arr)
        emb_llm_spearman = stats.spearmanr(emb_arr, llm_arr)

        emb_grouped_pearson = stats.pearsonr(emb_arr, grouped_arr)
        llm_grouped_pearson = stats.pearsonr(llm_arr, grouped_arr)

        # Compute errors
        emb_error = emb_arr - test_true
        llm_error = llm_arr - test_true
        grouped_error = grouped_arr - test_true

        error_corr = stats.pearsonr(emb_error, llm_error)

        # Compute MSEs
        emb_mse = mean_squared_error(test_true, emb_arr)
        llm_mse = mean_squared_error(test_true, llm_arr)
        grouped_mse = mean_squared_error(test_true, grouped_arr)

        print(f"\n  Prediction correlations:")
        print(f"    Embedding vs LLM Judge:   r={emb_llm_pearson[0]:.3f} (p={emb_llm_pearson[1]:.4f}), "
              f"rho={emb_llm_spearman.correlation:.3f}")
        print(f"    Embedding vs Combined:    r={emb_grouped_pearson[0]:.3f}")
        print(f"    LLM Judge vs Combined:    r={llm_grouped_pearson[0]:.3f}")

        print(f"\n  Error correlation:")
        print(f"    Embedding error vs LLM error: r={error_corr[0]:.3f}")

        print(f"\n  MSE (lower is better):")
        print(f"    Embedding:  {emb_mse:.4f}")
        print(f"    LLM Judge:  {llm_mse:.4f}")
        print(f"    Combined:   {grouped_mse:.4f}")
        improvement = (min(emb_mse, llm_mse) - grouped_mse) / min(emb_mse, llm_mse) * 100
        print(f"    Combined improvement: {improvement:+.1f}%")

        # Get model info
        emb_info = emb_predictor.get_model_info()
        llm_info = llm_predictor.get_model_info()
        grouped_info = grouped_predictor.get_model_info()
        grouped_diag = grouped_predictor.get_detailed_diagnostics()

        print(f"\n  Selected alphas:")
        print(f"    Embedding:  {emb_info['best_alpha']:.1f}")
        print(f"    LLM Judge:  {llm_info['best_alpha']:.1f}")
        print(f"    Combined:   Emb={grouped_diag['selected_alphas']['Embedding']:.1f}, "
              f"LLM={grouped_diag['selected_alphas']['LLM Judge']:.1f}")

        fold_result = {
            "fold_idx": fold_idx,
            "n_train": len(train_tasks),
            "n_test": len(test_tasks),
            "emb_llm_pearson": emb_llm_pearson[0],
            "emb_llm_spearman": emb_llm_spearman.correlation,
            "error_corr": error_corr[0],
            "emb_mse": emb_mse,
            "llm_mse": llm_mse,
            "grouped_mse": grouped_mse,
            "emb_alpha": emb_info["best_alpha"],
            "llm_alpha": llm_info["best_alpha"],
            "grouped_alphas": grouped_diag["selected_alphas"],
        }
        all_results.append(fold_result)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY ACROSS FOLDS")
    print("=" * 60)

    pred_corrs = [r["emb_llm_pearson"] for r in all_results]
    err_corrs = [r["error_corr"] for r in all_results]
    emb_mses = [r["emb_mse"] for r in all_results]
    llm_mses = [r["llm_mse"] for r in all_results]
    grouped_mses = [r["grouped_mse"] for r in all_results]

    print(f"\nPrediction correlation (Embedding vs LLM Judge):")
    print(f"  Mean: {np.mean(pred_corrs):.3f}, Std: {np.std(pred_corrs):.3f}")

    print(f"\nError correlation:")
    print(f"  Mean: {np.mean(err_corrs):.3f}, Std: {np.std(err_corrs):.3f}")

    print(f"\nMSE comparison:")
    print(f"  Embedding:  {np.mean(emb_mses):.4f} +/- {np.std(emb_mses):.4f}")
    print(f"  LLM Judge:  {np.mean(llm_mses):.4f} +/- {np.std(llm_mses):.4f}")
    print(f"  Combined:   {np.mean(grouped_mses):.4f} +/- {np.std(grouped_mses):.4f}")

    return {"fold_results": all_results}


def run_coefficient_analysis(
    config: ExperimentAConfig,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Analyze coefficient distribution in grouped ridge predictor."""
    print("\n" + "=" * 60)
    print("COEFFICIENT ANALYSIS")
    print("=" * 60)

    # Resolve paths
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path
    embeddings_path = ROOT / config.embeddings_path if config.embeddings_path else None
    llm_judge_path = ROOT / config.llm_judge_features_path if config.llm_judge_features_path else None

    # Build feature sources
    feature_sources = build_feature_sources(
        embeddings_path=embeddings_path,
        llm_judge_path=llm_judge_path,
        llm_judge_feature_cols=None,
        verbose=False,
    )
    source_by_name = {name: source for name, source in feature_sources}
    emb_source = source_by_name["Embedding"]
    llm_source = source_by_name["LLM Judge"]

    # Get all task IDs
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)
    folds = k_fold_split_tasks(all_task_ids, k=n_folds, seed=config.split_seed)

    all_coef_results = []

    for fold_idx, (train_tasks, test_tasks) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        # Load data
        data = load_dataset_for_fold(
            abilities_path=abilities_path,
            items_path=items_path,
            responses_path=responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=n_folds,
            split_seed=config.split_seed,
            is_binomial=False,
            irt_cache_dir=ROOT / "chris_output" / "experiment_a" / "irt_splits",
        )

        train_difficulties = data.get_train_difficulties()

        # Create and fit grouped predictor
        grouped_source = GroupedFeatureSource([
            RegularizedFeatureSource(emb_source),
            RegularizedFeatureSource(llm_source),
        ])
        grouped_predictor = GroupedRidgePredictor(grouped_source)
        grouped_predictor.fit(train_tasks, train_difficulties)

        # Get diagnostics
        diag = grouped_predictor.get_detailed_diagnostics()

        print(f"\n  Selected alphas:")
        for name, alpha in diag["selected_alphas"].items():
            print(f"    {name}: {alpha:.1f}")

        print(f"\n  Coefficient distribution by source:")
        print(f"    {'Source':<15} | {'Features':>8} | {'L2 Norm':>8} | {'Mean|c|':>8} | {'Contrib':>8}")
        print(f"    {'-' * 15}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}")

        for name, info in diag["coef_by_source"].items():
            print(f"    {name:<15} | {info['n_features']:>8} | {info['l2_norm']:>8.4f} | "
                  f"{info['mean_abs']:>8.5f} | {info['contribution_pct']:>7.1f}%")

        # Print LLM Judge feature coefficients if available
        if "LLM Judge" in diag["coef_by_source"]:
            llm_info = diag["coef_by_source"]["LLM Judge"]
            if "named_coefficients" in llm_info:
                print(f"\n  LLM Judge feature coefficients:")
                sorted_coefs = sorted(
                    llm_info["named_coefficients"].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                for name, coef in sorted_coefs:
                    print(f"    {name:<30}: {coef:+.4f}")

        all_coef_results.append(diag)

    # Summary
    print("\n" + "=" * 60)
    print("COEFFICIENT SUMMARY ACROSS FOLDS")
    print("=" * 60)

    emb_contribs = [r["coef_by_source"]["Embedding"]["contribution_pct"] for r in all_coef_results]
    llm_contribs = [r["coef_by_source"]["LLM Judge"]["contribution_pct"] for r in all_coef_results]

    print(f"\nContribution percentages:")
    print(f"  Embedding:  {np.mean(emb_contribs):.1f}% +/- {np.std(emb_contribs):.1f}%")
    print(f"  LLM Judge:  {np.mean(llm_contribs):.1f}% +/- {np.std(llm_contribs):.1f}%")

    return {"coef_results": all_coef_results}


def run_alpha_grid_analysis(
    config: ExperimentAConfig,
    n_folds: int = 5,
    alpha_grid: List[float] = None,
) -> Dict[str, Any]:
    """Sweep alpha grid and compute AUC for each combination."""
    if alpha_grid is None:
        alpha_grid = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    print("\n" + "=" * 60)
    print("ALPHA GRID ANALYSIS")
    print("=" * 60)
    print(f"Testing alpha grid: {alpha_grid}")
    print(f"Total combinations: {len(alpha_grid) ** 2}")

    # Resolve paths
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path
    embeddings_path = ROOT / config.embeddings_path if config.embeddings_path else None
    llm_judge_path = ROOT / config.llm_judge_features_path if config.llm_judge_features_path else None

    # Build feature sources
    feature_sources = build_feature_sources(
        embeddings_path=embeddings_path,
        llm_judge_path=llm_judge_path,
        llm_judge_feature_cols=None,
        verbose=False,
    )
    source_by_name = {name: source for name, source in feature_sources}
    emb_source = source_by_name["Embedding"]
    llm_source = source_by_name["LLM Judge"]

    # Get all task IDs
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)
    folds = k_fold_split_tasks(all_task_ids, k=n_folds, seed=config.split_seed)

    # Results: (alpha_emb, alpha_llm) -> list of AUCs per fold
    grid_results = {}

    # Also track individual predictor AUCs for baseline
    emb_aucs = []
    llm_aucs = []

    for fold_idx, (train_tasks, test_tasks) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        # Load data
        data = load_dataset_for_fold(
            abilities_path=abilities_path,
            items_path=items_path,
            responses_path=responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=n_folds,
            split_seed=config.split_seed,
            is_binomial=False,
            irt_cache_dir=ROOT / "chris_output" / "experiment_a" / "irt_splits",
        )

        train_difficulties = data.get_train_difficulties()

        # Individual predictor baselines
        emb_predictor = FeatureBasedPredictor(emb_source, ridge_alphas=list(config.ridge_alphas))
        llm_predictor = FeatureBasedPredictor(llm_source, ridge_alphas=list(config.ridge_alphas))

        emb_predictor.fit(train_tasks, train_difficulties)
        llm_predictor.fit(train_tasks, train_difficulties)

        emb_preds = emb_predictor.predict(test_tasks)
        llm_preds = llm_predictor.predict(test_tasks)

        emb_result = compute_auc(data, emb_preds)
        llm_result = compute_auc(data, llm_preds)

        emb_aucs.append(emb_result["auc"])
        llm_aucs.append(llm_result["auc"])

        # Grid sweep
        grouped_source = GroupedFeatureSource([
            RegularizedFeatureSource(emb_source),
            RegularizedFeatureSource(llm_source),
        ])

        for alpha_emb in alpha_grid:
            for alpha_llm in alpha_grid:
                fixed_alphas = {"Embedding": alpha_emb, "LLM Judge": alpha_llm}
                predictor = GroupedRidgePredictor(grouped_source, fixed_alphas=fixed_alphas)
                predictor.fit(train_tasks, train_difficulties)

                preds = predictor.predict(test_tasks)
                result = compute_auc(data, preds)

                key = (alpha_emb, alpha_llm)
                if key not in grid_results:
                    grid_results[key] = []
                grid_results[key].append(result["auc"])

        print(f"  Embedding AUC: {emb_aucs[-1]:.4f}")
        print(f"  LLM Judge AUC: {llm_aucs[-1]:.4f}")

    # Compute mean AUCs
    mean_grid = {k: np.mean(v) for k, v in grid_results.items()}

    # Print results as heatmap
    print("\n" + "=" * 60)
    print("MEAN AUC BY ALPHA COMBINATION")
    print("=" * 60)

    # Header
    header = f"{'α_emb \\ α_llm':>12}"
    for alpha_llm in alpha_grid:
        header += f" | {alpha_llm:>7}"
    print(header)
    print("-" * len(header))

    # Rows
    for alpha_emb in alpha_grid:
        row = f"{alpha_emb:>12}"
        for alpha_llm in alpha_grid:
            auc = mean_grid[(alpha_emb, alpha_llm)]
            row += f" | {auc:>7.4f}"
        print(row)

    # Find best
    best_key = max(mean_grid, key=mean_grid.get)
    best_auc = mean_grid[best_key]

    print(f"\n  Best combination: α_emb={best_key[0]}, α_llm={best_key[1]} -> AUC={best_auc:.4f}")
    print(f"\n  Individual baselines:")
    print(f"    Embedding only:  {np.mean(emb_aucs):.4f} +/- {np.std(emb_aucs):.4f}")
    print(f"    LLM Judge only:  {np.mean(llm_aucs):.4f} +/- {np.std(llm_aucs):.4f}")

    best_individual = max(np.mean(emb_aucs), np.mean(llm_aucs))
    improvement = (best_auc - best_individual) / best_individual * 100
    print(f"\n  Combined improvement over best individual: {improvement:+.2f}%")

    return {
        "grid_results": {str(k): v for k, v in grid_results.items()},
        "mean_grid": {str(k): v for k, v in mean_grid.items()},
        "best_alphas": best_key,
        "best_auc": best_auc,
        "emb_aucs": emb_aucs,
        "llm_aucs": llm_aucs,
    }


def main():
    """Run all diagnostic analyses."""
    config = ExperimentAConfig()
    n_folds = 5

    print("=" * 60)
    print("DEBUGGING COMBINED FEATURE SOURCES")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Embeddings: {config.embeddings_path}")
    print(f"  LLM Judge:  {config.llm_judge_features_path}")
    print(f"  Ridge alphas: {config.ridge_alphas}")
    print(f"  K-folds: {n_folds}")

    # Run analyses
    pred_results = run_prediction_analysis(config, n_folds)
    coef_results = run_coefficient_analysis(config, n_folds)
    alpha_results = run_alpha_grid_analysis(config, n_folds)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    # Key findings summary
    print("\nKEY FINDINGS:")

    # 1. Prediction correlation
    pred_corrs = [r["emb_llm_pearson"] for r in pred_results["fold_results"]]
    mean_corr = np.mean(pred_corrs)
    if mean_corr > 0.7:
        print(f"  1. HIGH prediction correlation (r={mean_corr:.3f}): "
              "Embeddings and LLM features make similar predictions")
    elif mean_corr > 0.4:
        print(f"  1. MODERATE prediction correlation (r={mean_corr:.3f}): "
              "Some redundancy between feature sources")
    else:
        print(f"  1. LOW prediction correlation (r={mean_corr:.3f}): "
              "Features capture different aspects (good for combination)")

    # 2. Coefficient contribution
    emb_contribs = [r["coef_by_source"]["Embedding"]["contribution_pct"]
                    for r in coef_results["coef_results"]]
    mean_emb_contrib = np.mean(emb_contribs)
    if mean_emb_contrib > 90:
        print(f"  2. EXTREME coefficient imbalance: Embedding contributes {mean_emb_contrib:.1f}% "
              "of total coefficient magnitude")
    elif mean_emb_contrib > 70:
        print(f"  2. HIGH coefficient imbalance: Embedding contributes {mean_emb_contrib:.1f}% "
              "of total coefficient magnitude")
    else:
        print(f"  2. BALANCED coefficients: Embedding contributes {mean_emb_contrib:.1f}% "
              "of total coefficient magnitude")

    # 3. Alpha grid
    best_individual = max(np.mean(alpha_results["emb_aucs"]), np.mean(alpha_results["llm_aucs"]))
    best_combined = alpha_results["best_auc"]
    if best_combined > best_individual:
        improvement = (best_combined - best_individual) / best_individual * 100
        print(f"  3. COMBINATION HELPS: Best combined AUC ({best_combined:.4f}) beats "
              f"best individual ({best_individual:.4f}) by {improvement:.2f}%")
    else:
        deficit = (best_individual - best_combined) / best_individual * 100
        print(f"  3. COMBINATION HURTS: Best combined AUC ({best_combined:.4f}) is worse than "
              f"best individual ({best_individual:.4f}) by {deficit:.2f}%")


if __name__ == "__main__":
    main()
