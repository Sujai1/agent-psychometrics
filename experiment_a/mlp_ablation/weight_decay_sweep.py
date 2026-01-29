"""Weight decay sweep for FullMLP with early stopping.

Holds constant:
- hidden_size = 1024
- early_stopping = True
- init_from_irt = True
- learning_rate = 0.01

Varies weight_decay across a wide range to find optimal regularization.

Usage:
    python -m experiment_a.mlp_ablation.weight_decay_sweep
    python -m experiment_a.mlp_ablation.weight_decay_sweep --part 1  # GPU 0
    python -m experiment_a.mlp_ablation.weight_decay_sweep --part 2  # GPU 1
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiment_ab_shared.feature_source import EmbeddingFeatureSource
from experiment_ab_shared.feature_predictor import FeatureBasedPredictor
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv, CVPredictor
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import FullMLPPredictor
from experiment_a.shared.baselines import (
    OraclePredictor,
    ConstantPredictor,
    DifficultyPredictorAdapter,
)
from experiment_a.swebench.config import ExperimentAConfig
from experiment_ab_shared import load_dataset_for_fold


def extract_train_auc(predictor: CVPredictor, fold_idx: int) -> float | None:
    """Extract train AUC from fitted predictor for diagnostics."""
    if hasattr(predictor, 'get_train_auc'):
        return predictor.get_train_auc()
    return None


ROOT = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Weight decay sweep for FullMLP")
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

    # Fixed parameters
    hidden_size = 1024
    learning_rate = 0.01
    n_epochs = 1000  # Early stopping will cut this

    configs: List[CVPredictorConfig] = []

    # === Baselines (only in part 1) ===
    if args.part is None or args.part == 1:
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
        configs.append(
            CVPredictorConfig(
                predictor=DifficultyPredictorAdapter(
                    FeatureBasedPredictor(embedding_source, alphas=config.ridge_alphas)
                ),
                name="ridge",
                display_name="Ridge (Embedding)",
            )
        )

    # === Weight decay sweep ===
    # Wide range from 0.1 to 10
    weight_decays = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0]

    # Split for parallel execution
    if args.part == 1:
        # Part 1: lower weight decays
        weight_decays_to_run = [0.1, 0.2, 0.3, 0.5]
    elif args.part == 2:
        # Part 2: higher weight decays
        weight_decays_to_run = [0.7, 1.0, 2.0, 5.0, 10.0]
    else:
        weight_decays_to_run = weight_decays

    for wd in weight_decays_to_run:
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=hidden_size,
                    weight_decay=wd,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    init_from_irt=True,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name=f"full_mlp_wd{wd}",
                display_name=f"FullMLP (h={hidden_size}, wd={wd}, early stop)",
            )
        )

    if args.part is not None:
        print(f"\n*** Running PART {args.part} only ({len(configs)} configs) ***")

    # Run CV
    results = {}

    print("\n" + "=" * 80)
    print("WEIGHT DECAY SWEEP FOR FULLMLP")
    print(f"Fixed: hidden_size={hidden_size}, early_stopping=True, init_from_irt=True")
    print(f"Varying: weight_decay in {weight_decays_to_run}")
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
            diagnostics_extractor=extract_train_auc,
        )

        results[pc.name] = {
            "display_name": pc.display_name,
            "mean_auc": cv_result.mean_auc,
            "std_auc": cv_result.std_auc,
            "fold_aucs": cv_result.fold_aucs,
        }

        # Get mean train AUC from fold diagnostics
        if cv_result.fold_diagnostics:
            valid_train_aucs = [t for t in cv_result.fold_diagnostics if t is not None]
            if valid_train_aucs:
                results[pc.name]["train_auc"] = float(np.mean(valid_train_aucs))
                results[pc.name]["fold_train_aucs"] = valid_train_aucs

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} ± {cv_result.std_auc:.4f}")

    # Print summary table
    print("\n" + "=" * 85)
    print("WEIGHT DECAY SWEEP RESULTS")
    print("=" * 85)
    print(f"\n{'Method':<50} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 85)

    # Sort by test AUC descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_auc"], reverse=True)

    for name, r in sorted_results:
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc", None)
        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{r['display_name']:<50} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{r['display_name']:<50} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Highlight key comparisons
    print("\n" + "-" * 85)
    print("KEY COMPARISONS:")
    if "ridge" in results:
        ridge_auc = results["ridge"]["mean_auc"]
        print(f"  Ridge baseline: {ridge_auc:.4f}")

        # Find best MLP from this sweep
        best_mlp = max(
            [(name, r) for name, r in results.items() if name.startswith("full_mlp")],
            key=lambda x: x[1]["mean_auc"],
            default=None
        )
        if best_mlp:
            best_name, best_r = best_mlp
            delta = best_r["mean_auc"] - ridge_auc
            print(f"  Best MLP: {best_r['display_name']}: {best_r['mean_auc']:.4f} ({delta:+.4f} vs Ridge)")

    if "oracle" in results:
        print(f"  Oracle upper bound: {results['oracle']['mean_auc']:.4f}")

    # Save results
    output_dir = ROOT / "chris_output/experiment_a/mlp_embedding"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.part is not None:
        output_path = output_dir / f"weight_decay_sweep_part{args.part}.json"
    else:
        output_path = output_dir / "weight_decay_sweep.json"

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
