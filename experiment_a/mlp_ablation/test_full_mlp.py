"""Ablation study: Full MLP on embedding features.

Tests the FullMLPPredictor which takes [agent_one_hot | task_features] as
concatenated input through a hidden layer, unlike the IRT-style MLP.

Ablation dimensions:
1. Hidden size: [32, 64, 128, 256]
2. Weight decay: [0.1, 1.0, 10.0] (high values needed to prevent overfitting)
3. Initialization: random vs IRT abilities
4. Early stopping: with vs without

Usage:
    python -m experiment_a.mlp_ablation.test_full_mlp

    # Quick test with fewer configs
    python -m experiment_a.mlp_ablation.test_full_mlp --quick
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiment_ab_shared.feature_source import EmbeddingFeatureSource
from experiment_ab_shared.feature_predictor import FeatureBasedPredictor
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import FullMLPPredictor
from experiment_a.shared.baselines import (
    OraclePredictor,
    ConstantPredictor,
    DifficultyPredictorAdapter,
)
from experiment_a.swebench.config import ExperimentAConfig
from experiment_ab_shared import load_dataset_for_fold

ROOT = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Full MLP on embeddings ablation")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer configs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--part", type=int, choices=[1, 2], default=None,
                        help="Run only part 1 or 2 of configs (for parallel execution)")
    args = parser.parse_args()

    config = ExperimentAConfig()

    # Resolve paths
    embeddings_path = ROOT / config.embeddings_path
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    # Build embedding source
    embedding_source = EmbeddingFeatureSource(embeddings_path)
    print(f"Loaded embeddings: {embedding_source.feature_dim} dimensions")

    # Load task IDs
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    # Generate folds
    k_folds = args.k_folds
    folds = k_fold_split_tasks(all_task_ids, k=k_folds, seed=config.split_seed)

    # Create fold data loader
    def load_fold_data(train_tasks, test_tasks, fold_idx):
        return load_dataset_for_fold(
            abilities_path=abilities_path,
            items_path=items_path,
            responses_path=responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=k_folds,
            split_seed=config.split_seed,
            is_binomial=False,
            irt_cache_dir=Path("chris_output/experiment_a/irt_splits"),
        )

    # Training parameters
    n_epochs = 500
    learning_rate = 0.01

    configs: List[CVPredictorConfig] = []

    # === Baselines ===
    configs.append(
        CVPredictorConfig(
            predictor=OraclePredictor(),
            name="oracle",
            display_name="Oracle (true β)",
        )
    )
    configs.append(
        CVPredictorConfig(
            predictor=ConstantPredictor(),
            name="constant",
            display_name="Constant (mean β)",
        )
    )

    # Ridge baseline
    configs.append(
        CVPredictorConfig(
            predictor=DifficultyPredictorAdapter(
                FeatureBasedPredictor(embedding_source, alphas=config.ridge_alphas)
            ),
            name="ridge",
            display_name="Ridge (Embedding)",
        )
    )

    # === Full MLP Experiments ===

    # Ablation parameters (higher weight decays based on IRT-MLP findings)
    if args.quick:
        hidden_sizes = [64, 128]
        weight_decays = [1.0, 10.0]
    else:
        hidden_sizes = [32, 64, 128, 256]
        weight_decays = [0.1, 1.0, 10.0]

    # 1. Hidden size ablation (with weight_decay=1.0)
    for hidden_size in hidden_sizes:
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=hidden_size,
                    weight_decay=1.0,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    init_from_irt=False,
                    verbose=True,
                ),
                name=f"full_mlp_h{hidden_size}",
                display_name=f"FullMLP (h={hidden_size})",
            )
        )

    # 2. Weight decay ablation (with hidden_size=128)
    for wd in weight_decays:
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=128,
                    weight_decay=wd,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    init_from_irt=False,
                    verbose=True,
                ),
                name=f"full_mlp_wd{wd}",
                display_name=f"FullMLP (wd={wd})",
            )
        )

    # 3. IRT initialization ablation (with weight_decay=1.0)
    for hidden_size in [64, 128]:
        # With IRT init
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=hidden_size,
                    weight_decay=1.0,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    init_from_irt=True,
                    verbose=True,
                ),
                name=f"full_mlp_h{hidden_size}_irt",
                display_name=f"FullMLP (h={hidden_size}, IRT init)",
            )
        )

    # 4. Early stopping ablation
    for init_from_irt in [False, True]:
        init_str = "irt" if init_from_irt else "random"
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=128,
                    weight_decay=1.0,
                    learning_rate=learning_rate,
                    n_epochs=1000,  # More epochs since early stopping will cut it
                    init_from_irt=init_from_irt,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name=f"full_mlp_earlystop_{init_str}",
                display_name=f"FullMLP early stop ({init_str})",
            )
        )

    # 5. Best config candidates (combining good settings)
    if not args.quick:
        # Larger hidden + IRT init + early stopping
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=256,
                    weight_decay=1.0,
                    learning_rate=learning_rate,
                    n_epochs=1000,
                    init_from_irt=True,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name="full_mlp_best_v1",
                display_name="FullMLP (h=256, IRT, early stop)",
            )
        )

        # Higher regularization + early stopping
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=128,
                    weight_decay=10.0,
                    learning_rate=learning_rate,
                    n_epochs=1000,
                    init_from_irt=True,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name="full_mlp_best_v2",
                display_name="FullMLP (h=128, wd=10, IRT, early stop)",
            )
        )

        # Dropout experiment
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=128,
                    dropout=0.3,
                    weight_decay=1.0,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    init_from_irt=True,
                    verbose=True,
                ),
                name="full_mlp_dropout",
                display_name="FullMLP (h=128, dropout=0.3, IRT)",
            )
        )

    # Filter configs by part if specified (for parallel execution)
    if args.part is not None:
        # Split configs roughly in half
        # Part 1: Baselines + hidden size + weight decay ablations
        # Part 2: IRT init + early stopping + best configs
        part1_keywords = ["oracle", "constant", "ridge", "full_mlp_h32", "full_mlp_h64", "full_mlp_h128", "full_mlp_h256", "full_mlp_wd"]
        part2_keywords = ["full_mlp_h64_irt", "full_mlp_h128_irt", "full_mlp_earlystop", "full_mlp_best", "full_mlp_dropout"]

        if args.part == 1:
            configs = [c for c in configs if any(kw in c.name for kw in part1_keywords)]
        else:
            configs = [c for c in configs if any(kw in c.name for kw in part2_keywords)]

        print(f"\n*** Running PART {args.part} only ({len(configs)} configs) ***")

    # Run CV
    results = {}

    print("\n" + "=" * 80)
    print("FULL MLP ON EMBEDDINGS ABLATION")
    print(f"Feature dim: {embedding_source.feature_dim}, Epochs: {n_epochs}, LR: {learning_rate}")
    print(f"Configs to run: {len(configs)}")
    print("=" * 80)

    for i, pc in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {pc.display_name}")
        print("-" * 60)

        cv_result = run_cv(
            pc.predictor,
            folds,
            load_fold_data,
            verbose=True,
        )

        results[pc.name] = {
            "display_name": pc.display_name,
            "mean_auc": cv_result.mean_auc,
            "std_auc": cv_result.std_auc,
            "fold_aucs": cv_result.fold_aucs,
        }

        # Get train AUC if available
        if hasattr(pc.predictor, 'get_train_auc') and pc.predictor.get_train_auc() is not None:
            results[pc.name]["train_auc"] = pc.predictor.get_train_auc()

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} ± {cv_result.std_auc:.4f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<45} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 80)

    # Sort by test AUC descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_auc"], reverse=True)

    for name, r in sorted_results:
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc", None)
        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{r['display_name']:<45} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{r['display_name']:<45} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Highlight key comparisons
    print("\n" + "-" * 80)
    print("KEY COMPARISONS:")
    if "ridge" in results:
        ridge_auc = results["ridge"]["mean_auc"]
        print(f"  Ridge baseline: {ridge_auc:.4f}")

        # Find best Full MLP
        best_full = max(
            [(name, r) for name, r in results.items() if name.startswith("full_mlp")],
            key=lambda x: x[1]["mean_auc"],
            default=None
        )
        if best_full:
            best_name, best_r = best_full
            delta = best_r["mean_auc"] - ridge_auc
            print(f"  Best FullMLP: {best_r['display_name']}: {best_r['mean_auc']:.4f} ({delta:+.4f} vs Ridge)")

    # Save results
    output_dir = ROOT / "chris_output/experiment_a/mlp_embedding"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.part is not None:
        output_path = output_dir / f"full_mlp_results_part{args.part}.json"
    else:
        output_path = output_dir / "full_mlp_results.json"

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
