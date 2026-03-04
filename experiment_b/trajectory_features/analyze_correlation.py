"""Analyze correlation between trajectory features and task difficulty.

Usage:
    python -m experiment_b.trajectory_features.analyze_correlation \
        --features_path chris_output/trajectory_features/raw_features.csv

    # With aggregated features
    python -m experiment_b.trajectory_features.analyze_correlation \
        --features_path chris_output/trajectory_features/aggregated_features.csv \
        --aggregated
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_NAMES, TRAJECTORY_FEATURES


def load_oracle_difficulty(
    items_path: str = "data/swebench/irt/1d_1pl/items.csv",
) -> pd.Series:
    """Load oracle IRT difficulty (beta) for tasks."""
    df = pd.read_csv(items_path, index_col=0)
    return df["b"]


def compute_correlations(
    features_df: pd.DataFrame,
    difficulty: pd.Series,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute Spearman and Pearson correlations between features and difficulty.

    Args:
        features_df: DataFrame with task_id as index and feature columns
        difficulty: Series of oracle difficulties indexed by task_id
        feature_cols: Columns to analyze (default: all numeric)

    Returns:
        DataFrame with correlation statistics per feature
    """
    # Align indices
    common_tasks = features_df.index.intersection(difficulty.index)
    if len(common_tasks) == 0:
        raise ValueError("No common tasks between features and difficulty")

    features_aligned = features_df.loc[common_tasks]
    difficulty_aligned = difficulty.loc[common_tasks]

    if feature_cols is None:
        feature_cols = features_aligned.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for col in feature_cols:
        values = features_aligned[col].dropna()
        diff_values = difficulty_aligned.loc[values.index]

        if len(values) < 5:
            continue

        # Spearman (rank correlation)
        spearman_r, spearman_p = stats.spearmanr(values, diff_values)

        # Pearson (linear correlation)
        pearson_r, pearson_p = stats.pearsonr(values, diff_values)

        results.append({
            "feature": col,
            "n_samples": len(values),
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "significant_05": spearman_p < 0.05,
            "significant_01": spearman_p < 0.01,
        })

    return pd.DataFrame(results).sort_values("spearman_p")


def fit_regression_models(
    features_df: pd.DataFrame,
    difficulty: pd.Series,
    feature_cols: Optional[List[str]] = None,
    alpha_ridge: float = 1.0,
    alpha_lasso: float = 0.1,
) -> Dict:
    """Fit Ridge and Lasso regression models and report R² and selected features.

    Args:
        features_df: DataFrame with task_id as index
        difficulty: Series of oracle difficulties
        feature_cols: Columns to use as features
        alpha_ridge: Regularization strength for Ridge
        alpha_lasso: Regularization strength for Lasso

    Returns:
        Dict with model results
    """
    # Align indices
    common_tasks = features_df.index.intersection(difficulty.index)
    features_aligned = features_df.loc[common_tasks]
    y = difficulty.loc[common_tasks].values

    if feature_cols is None:
        feature_cols = features_aligned.select_dtypes(include=[np.number]).columns.tolist()

    X = features_aligned[feature_cols].values

    # Handle missing values (fill with mean)
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_means[j]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {
        "n_samples": len(y),
        "n_features": X.shape[1],
    }

    # Ridge regression
    ridge = Ridge(alpha=alpha_ridge)
    ridge_scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring="r2")
    ridge.fit(X_scaled, y)
    results["ridge_cv_r2_mean"] = ridge_scores.mean()
    results["ridge_cv_r2_std"] = ridge_scores.std()
    results["ridge_train_r2"] = ridge.score(X_scaled, y)

    # Lasso regression (for feature selection)
    lasso = Lasso(alpha=alpha_lasso, max_iter=10000)
    lasso_scores = cross_val_score(lasso, X_scaled, y, cv=5, scoring="r2")
    lasso.fit(X_scaled, y)
    results["lasso_cv_r2_mean"] = lasso_scores.mean()
    results["lasso_cv_r2_std"] = lasso_scores.std()
    results["lasso_train_r2"] = lasso.score(X_scaled, y)

    # Selected features by Lasso (non-zero coefficients)
    selected_mask = np.abs(lasso.coef_) > 1e-6
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
    results["lasso_n_selected"] = len(selected_features)
    results["lasso_selected_features"] = selected_features

    # Feature importance (Ridge coefficients)
    ridge_importance = list(zip(feature_cols, ridge.coef_))
    ridge_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    results["ridge_top_features"] = ridge_importance[:10]

    return results


def analyze_raw_features(
    raw_df: pd.DataFrame,
    difficulty: pd.Series,
) -> Tuple[pd.DataFrame, Dict]:
    """Analyze raw (per-agent) features.

    Computes correlations at the trajectory level and tests if features
    predict outcome (resolved) controlling for agent ability.
    """
    # Join difficulty
    raw_with_diff = raw_df.copy()
    raw_with_diff["difficulty"] = raw_with_diff["task_id"].map(difficulty)
    raw_with_diff = raw_with_diff.dropna(subset=["difficulty"])

    # Correlation of raw features with difficulty
    feature_cols = [c for c in FEATURE_NAMES if c in raw_with_diff.columns]
    correlations = []

    for col in feature_cols:
        values = raw_with_diff[col].dropna()
        diff_values = raw_with_diff.loc[values.index, "difficulty"]

        if len(values) < 10:
            continue

        r, p = stats.spearmanr(values, diff_values)
        correlations.append({
            "feature": col,
            "spearman_r": r,
            "spearman_p": p,
            "n_samples": len(values),
        })

    corr_df = pd.DataFrame(correlations)

    # Also test correlation with resolved (at trajectory level)
    resolved_corrs = []
    for col in feature_cols:
        values = raw_with_diff[col].dropna()
        resolved = raw_with_diff.loc[values.index, "resolved"].astype(int)

        if len(values) < 10:
            continue

        r, p = stats.pointbiserialr(resolved, values)
        resolved_corrs.append({
            "feature": col,
            "pointbiserial_r": r,
            "pointbiserial_p": p,
        })

    resolved_df = pd.DataFrame(resolved_corrs)

    return corr_df.merge(resolved_df, on="feature"), {
        "n_trajectories": len(raw_with_diff),
        "n_tasks": raw_with_diff["task_id"].nunique(),
        "n_agents": raw_with_diff["agent"].nunique(),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory feature correlations")
    parser.add_argument(
        "--features_path",
        type=str,
        default="chris_output/trajectory_features/raw_features.csv",
        help="Path to features CSV",
    )
    parser.add_argument(
        "--items_path",
        type=str,
        default="data/swebench/irt/1d_1pl/items.csv",
        help="Path to IRT items CSV",
    )
    parser.add_argument(
        "--aggregated",
        action="store_true",
        help="Features are already aggregated (task_id as index)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for correlation results",
    )
    args = parser.parse_args()

    print(f"Loading features from {args.features_path}...")
    features_df = pd.read_csv(args.features_path)
    print(f"  Loaded {len(features_df)} rows")

    print(f"Loading oracle difficulty from {args.items_path}...")
    difficulty = load_oracle_difficulty(args.items_path)
    print(f"  Loaded difficulty for {len(difficulty)} tasks")

    if args.aggregated:
        # Aggregated features: task_id is index or first column
        if "task_id" in features_df.columns:
            features_df = features_df.set_index("task_id")

        print("\n" + "=" * 60)
        print("AGGREGATED FEATURE CORRELATIONS WITH DIFFICULTY")
        print("=" * 60)

        corr_df = compute_correlations(features_df, difficulty)
        print(f"\n{'Feature':<40} {'Spearman r':>12} {'p-value':>12} {'Sig?':>6}")
        print("-" * 70)
        for _, row in corr_df.iterrows():
            sig = "**" if row["significant_01"] else ("*" if row["significant_05"] else "")
            print(f"{row['feature']:<40} {row['spearman_r']:>12.3f} {row['spearman_p']:>12.4f} {sig:>6}")

        print("\n" + "=" * 60)
        print("REGRESSION ANALYSIS")
        print("=" * 60)

        reg_results = fit_regression_models(features_df, difficulty)
        print(f"\nSamples: {reg_results['n_samples']}, Features: {reg_results['n_features']}")
        print(f"\nRidge (CV R²): {reg_results['ridge_cv_r2_mean']:.3f} ± {reg_results['ridge_cv_r2_std']:.3f}")
        print(f"Lasso (CV R²): {reg_results['lasso_cv_r2_mean']:.3f} ± {reg_results['lasso_cv_r2_std']:.3f}")
        print(f"\nLasso selected {reg_results['lasso_n_selected']} features:")
        for feat in reg_results["lasso_selected_features"]:
            print(f"  - {feat}")

        print("\nTop Ridge coefficients:")
        for feat, coef in reg_results["ridge_top_features"]:
            print(f"  {feat:<40}: {coef:+.4f}")

    else:
        # Raw features: one row per trajectory
        print("\n" + "=" * 60)
        print("RAW FEATURE CORRELATIONS")
        print("=" * 60)

        corr_df, stats_dict = analyze_raw_features(features_df, difficulty)
        print(f"\nAnalyzed {stats_dict['n_trajectories']} trajectories")
        print(f"  Tasks: {stats_dict['n_tasks']}, Agents: {stats_dict['n_agents']}")

        print(f"\n{'Feature':<25} {'vs Difficulty':>15} {'vs Resolved':>15}")
        print(f"{'':25} {'(Spearman r)':>15} {'(Pointbiserial)':>15}")
        print("-" * 60)
        for _, row in corr_df.iterrows():
            print(f"{row['feature']:<25} {row['spearman_r']:>+15.3f} {row['pointbiserial_r']:>+15.3f}")

        # Check expected directions
        print("\n=== Expected Direction Check ===")
        for feat_spec in TRAJECTORY_FEATURES:
            if feat_spec.name not in corr_df["feature"].values:
                continue

            row = corr_df[corr_df["feature"] == feat_spec.name].iloc[0]
            r = row["spearman_r"]
            expected = feat_spec.expected_direction

            if expected == "positive" and r > 0:
                status = "✓ Correct"
            elif expected == "negative" and r < 0:
                status = "✓ Correct"
            elif expected == "unknown":
                status = "- Unknown"
            else:
                status = "✗ Unexpected"

            print(f"  {feat_spec.name}: r={r:+.3f}, expected={expected}, {status}")

    if args.output_path:
        corr_df.to_csv(args.output_path, index=False)
        print(f"\nSaved results to {args.output_path}")


if __name__ == "__main__":
    main()
