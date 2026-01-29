"""Quick ablation: test two-stage training approach.

Two-stage training:
  Stage 1: Initialize agents from IRT, freeze them, train features only
  Stage 2: Unfreeze agents, fine-tune both with low agent LR

This combines good initialization from IRT with joint adaptation.

Usage:
    python -m experiment_a.mlp_ablation.test_two_stage
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiment_ab_shared.feature_source import build_feature_sources
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import MLPPredictor
from experiment_a.shared.baselines import OraclePredictor, ConstantPredictor
from experiment_a.swebench.config import ExperimentAConfig
from experiment_ab_shared import load_dataset_for_fold

ROOT = Path(__file__).parent.parent.parent


def main():
    config = ExperimentAConfig()

    # Resolve paths
    llm_judge_path = ROOT / config.llm_judge_features_path
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path

    # Build feature sources (just LLM Judge)
    feature_source_list = build_feature_sources(
        llm_judge_path=llm_judge_path,
        verbose=True,
    )
    source = feature_source_list[0][1]  # LLM Judge source

    # Load task IDs
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    # Generate folds
    k_folds = 5
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

    # Common training params
    n_epochs = 500
    learning_rate = 0.01

    configs: List[CVPredictorConfig] = []

    # Baseline: learned from scratch (current broken behavior)
    configs.append(
        CVPredictorConfig(
            predictor=MLPPredictor(
                source,
                freeze_abilities=False,
                two_stage=False,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                verbose=True,
            ),
            name="baseline",
            display_name="Baseline (learned from scratch)",
        )
    )

    # Frozen IRT (known good result)
    configs.append(
        CVPredictorConfig(
            predictor=MLPPredictor(
                source,
                freeze_abilities=True,
                two_stage=False,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                verbose=True,
            ),
            name="frozen_irt",
            display_name="Frozen IRT",
        )
    )

    # Two-stage training with different stage splits and stage2 LR scales
    stage_splits = [0.5, 0.75]  # fraction of epochs for stage 1
    stage2_lr_scales = [0.1, 0.01]

    for split in stage_splits:
        for lr_scale in stage2_lr_scales:
            stage1_epochs = int(n_epochs * split)
            configs.append(
                CVPredictorConfig(
                    predictor=MLPPredictor(
                        source,
                        freeze_abilities=False,
                        two_stage=True,
                        stage1_epochs=stage1_epochs,
                        stage2_agent_lr_scale=lr_scale,
                        learning_rate=learning_rate,
                        n_epochs=n_epochs,
                        verbose=True,
                    ),
                    name=f"two_stage_s1={split}_s2lr={lr_scale}",
                    display_name=f"Two-stage (s1={int(split*100)}%, s2_lr={lr_scale})",
                )
            )

    # Add baselines
    configs.append(
        CVPredictorConfig(
            predictor=OraclePredictor(),
            name="oracle",
            display_name="Oracle (true b)",
        )
    )
    configs.append(
        CVPredictorConfig(
            predictor=ConstantPredictor(),
            name="constant",
            display_name="Constant (mean b)",
        )
    )

    # Run CV
    results = {}

    print("\n" + "=" * 80)
    print("TWO-STAGE TRAINING ABLATION (LLM Judge)")
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

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} +/- {cv_result.std_auc:.4f}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<45} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 80)

    for name, r in results.items():
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc", None)
        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{r['display_name']:<45} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{r['display_name']:<45} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Save results
    output_dir = ROOT / "chris_output/experiment_a/mlp_two_stage"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "two_stage_results.json"

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
