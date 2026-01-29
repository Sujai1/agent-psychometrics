"""Architecture sweep Part 6: Balanced representation architectures.

Tests architectures that create balanced agent/task representations:
1. TaskBottleneck: Compress task features (5120 -> 64) before combining with agent
2. CrossAttention: Agent embedding attends to task feature chunks
3. FeatureGated: Agent embedding gates which task features matter

These build on AgentEmb's success (0.8194 AUC) by also compressing/balancing
the task representation.

Usage:
    python -m experiment_a.mlp_ablation.interaction_sweep_v2
    python -m experiment_a.mlp_ablation.interaction_sweep_v2 --quick
    sbatch experiment_a/mlp_ablation/slurm_interaction_sweep_v2.sh
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiment_ab_shared.feature_source import EmbeddingFeatureSource
from experiment_ab_shared.feature_predictor import FeatureBasedPredictor
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv, CVPredictor
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import (
    AgentEmbeddingPredictor,
    TaskBottleneckPredictor,
    CrossAttentionPredictor,
    FeatureGatedPredictor,
)
from experiment_a.shared.baselines import (
    OraclePredictor,
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
    main_start = time.time()
    print(f"Script starting at: {time.strftime('%H:%M:%S')}")

    parser = argparse.ArgumentParser(description="Balanced representation architecture sweep")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer configs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    config = ExperimentAConfig()

    # Resolve paths
    embeddings_path = ROOT / config.embeddings_path
    items_path = ROOT / config.items_path

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    print(f"Loading embeddings at: {time.strftime('%H:%M:%S')}")
    embedding_source = EmbeddingFeatureSource(embeddings_path)
    print(f"Loaded embeddings: {embedding_source.feature_dim} dimensions (took {time.time() - main_start:.1f}s)")

    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    folds = k_fold_split_tasks(all_task_ids, k=args.k_folds, seed=config.split_seed)

    def load_fold_data(train_tasks, test_tasks, fold_idx):
        return load_dataset_for_fold(
            abilities_path=ROOT / config.abilities_path,
            items_path=ROOT / config.items_path,
            responses_path=ROOT / config.responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=args.k_folds,
            split_seed=config.split_seed,
            is_binomial=False,
            irt_cache_dir=Path("chris_output/experiment_a/irt_splits"),
        )

    # Training parameters (use best from previous sweeps)
    learning_rate = 0.01
    weight_decay = 0.2
    n_epochs = 1000

    configs: List[CVPredictorConfig] = []

    # === Baselines ===
    configs.append(CVPredictorConfig(
        predictor=OraclePredictor(),
        name="oracle",
        display_name="Oracle (true beta)",
    ))
    configs.append(CVPredictorConfig(
        predictor=DifficultyPredictorAdapter(
            FeatureBasedPredictor(embedding_source, alphas=config.ridge_alphas)
        ),
        name="ridge",
        display_name="Ridge (Embedding)",
    ))

    # Reference: Best from Part 5 (AgentEmb)
    configs.append(CVPredictorConfig(
        predictor=AgentEmbeddingPredictor(
            embedding_source,
            agent_emb_dim=64,
            hidden_sizes=[64, 64],
            dropout=0.0,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            init_from_irt=True,
            early_stopping=True,
            val_fraction=0.1,
            patience=30,
            verbose=True,
        ),
        name="agent_emb_64_h64-64",
        display_name="AgentEmb (d=64, h=64-64) [best from Part 5]",
    ))

    # === New Architectures ===

    if args.quick:
        agent_emb_dims = [32]
        bottleneck_dims = [64]
        n_chunks_list = [64]
        compressed_dims = [64]
    else:
        agent_emb_dims = [32, 64]
        bottleneck_dims = [32, 64, 128]
        n_chunks_list = [32, 64, 128]
        compressed_dims = [32, 64, 128]

    # 1. TaskBottleneck: Compress task features before combining
    for agent_emb in agent_emb_dims:
        for bottleneck in bottleneck_dims:
            configs.append(CVPredictorConfig(
                predictor=TaskBottleneckPredictor(
                    embedding_source,
                    agent_emb_dim=agent_emb,
                    task_bottleneck_dim=bottleneck,
                    hidden_sizes=[64, 32],
                    dropout=0.0,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    n_epochs=n_epochs,
                    init_from_irt=True,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name=f"bottleneck_a{agent_emb}_t{bottleneck}",
                display_name=f"TaskBottleneck (agent={agent_emb}, task={bottleneck})",
            ))

    # 2. CrossAttention: Agent attends to task feature chunks
    for agent_emb in agent_emb_dims:
        for n_chunks in n_chunks_list:
            configs.append(CVPredictorConfig(
                predictor=CrossAttentionPredictor(
                    embedding_source,
                    agent_emb_dim=agent_emb,
                    n_chunks=n_chunks,
                    hidden_sizes=[64, 32],
                    dropout=0.0,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    n_epochs=n_epochs,
                    init_from_irt=True,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name=f"crossattn_a{agent_emb}_c{n_chunks}",
                display_name=f"CrossAttention (agent={agent_emb}, chunks={n_chunks})",
            ))

    # 3. FeatureGated: Agent gates task features
    for agent_emb in agent_emb_dims:
        for compressed in compressed_dims:
            configs.append(CVPredictorConfig(
                predictor=FeatureGatedPredictor(
                    embedding_source,
                    agent_emb_dim=agent_emb,
                    compressed_dim=compressed,
                    hidden_sizes=[64, 32],
                    dropout=0.0,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    n_epochs=n_epochs,
                    init_from_irt=True,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name=f"gated_a{agent_emb}_c{compressed}",
                display_name=f"FeatureGated (agent={agent_emb}, compress={compressed})",
            ))

    if args.quick:
        # Filter to just a few representative configs
        quick_names = ["oracle", "ridge", "agent_emb_64_h64-64",
                       "bottleneck_a32_t64", "crossattn_a32_c64", "gated_a32_c64"]
        configs = [c for c in configs if c.name in quick_names]

    print(f"\n*** Running {len(configs)} configs ***")

    # Run CV
    results = {}
    cv_start = time.time()

    print("\n" + "=" * 85)
    print(f"BALANCED REPRESENTATION SWEEP (starting CV at {time.strftime('%H:%M:%S')})")
    print(f"Fixed: weight_decay={weight_decay}, init_from_irt=True, early_stopping=True")
    print(f"Configs to run: {len(configs)}")
    print("=" * 85)

    for i, pc in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {pc.display_name}")
        print("-" * 60)

        config_start = time.time()
        cv_result = run_cv(
            pc.predictor,
            folds,
            load_fold_data,
            verbose=True,
            diagnostics_extractor=extract_train_auc,
        )
        config_elapsed = time.time() - config_start

        results[pc.name] = {
            "display_name": pc.display_name,
            "mean_auc": cv_result.mean_auc,
            "std_auc": cv_result.std_auc,
            "fold_aucs": cv_result.fold_aucs,
        }

        if cv_result.fold_diagnostics:
            valid_train_aucs = [t for t in cv_result.fold_diagnostics if t is not None]
            if valid_train_aucs:
                results[pc.name]["train_auc"] = float(np.mean(valid_train_aucs))

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} +/- {cv_result.std_auc:.4f}")
        print(f"   Config time: {config_elapsed:.1f}s")

    # Print summary
    print("\n" + "=" * 85)
    print("BALANCED REPRESENTATION SWEEP RESULTS")
    print("=" * 85)
    print(f"\n{'Method':<50} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 85)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["mean_auc"],
        reverse=True
    )

    for name, r in sorted_results:
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc")
        display_name = r["display_name"]

        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{display_name:<50} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{display_name:<50} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Key comparisons
    print(f"\n{'-' * 85}")
    print("KEY COMPARISONS:")

    if "ridge" in results:
        ridge_auc = results["ridge"]["mean_auc"]
        print(f"  Ridge baseline: {ridge_auc:.4f}")

        # Find best new architecture (exclude baselines and AgentEmb reference)
        new_arch_results = [(n, r) for n, r in results.items()
                           if n not in ["oracle", "ridge", "agent_emb_64_h64-64"]]
        if new_arch_results:
            best_name, best_r = max(new_arch_results, key=lambda x: x[1]["mean_auc"])
            delta = best_r["mean_auc"] - ridge_auc
            print(f"  Best new arch: {best_r['display_name']}: {best_r['mean_auc']:.4f} ({delta:+.4f} vs Ridge)")

    if "agent_emb_64_h64-64" in results:
        print(f"  AgentEmb reference: {results['agent_emb_64_h64-64']['mean_auc']:.4f}")

    if "oracle" in results:
        print(f"  Oracle upper bound: {results['oracle']['mean_auc']:.4f}")

    # Group by architecture type
    print(f"\n{'-' * 85}")
    print("BY ARCHITECTURE TYPE:")

    for arch_type in ["bottleneck", "crossattn", "gated"]:
        arch_results = [(n, r) for n, r in results.items() if n.startswith(arch_type)]
        if arch_results:
            best_name, best_r = max(arch_results, key=lambda x: x[1]["mean_auc"])
            print(f"  Best {arch_type}: {best_r['display_name']}: {best_r['mean_auc']:.4f}")

    # Save results
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    output_dir = ROOT / "chris_output/experiment_a/mlp_embedding"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "interaction_sweep_v2.json"

    with open(output_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {time.time() - main_start:.1f}s")


if __name__ == "__main__":
    main()
