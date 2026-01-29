"""Fast hyperparameter search split across 2 GPUs.

Usage:
    python -m experiment_a.mlp_ablation.test_optimal_hyperparams_fast --gpu 0
    python -m experiment_a.mlp_ablation.test_optimal_hyperparams_fast --gpu 1
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
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import AgentEmbeddingPredictor
from experiment_a.shared.baselines import OraclePredictor, DifficultyPredictorAdapter
from experiment_a.swebench.config import ExperimentAConfig
from experiment_ab_shared import load_dataset_for_fold


def extract_train_auc(predictor, fold_idx):
    if hasattr(predictor, 'get_train_auc'):
        return predictor.get_train_auc()
    return None


ROOT = Path(__file__).parent.parent.parent


def main():
    main_start = time.time()

    parser = argparse.ArgumentParser(description="Fast hyperparameter search")
    parser.add_argument("--gpu", type=int, required=True, choices=[0, 1], help="GPU index (0 or 1)")
    args = parser.parse_args()

    print(f"Script starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running on GPU {args.gpu}")

    config = ExperimentAConfig()
    k_folds = 5  # Use 5 folds for proper evaluation

    embeddings_path = ROOT / config.embeddings_path
    items_path = ROOT / config.items_path

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    print(f"Loading embeddings...")
    embedding_source = EmbeddingFeatureSource(embeddings_path)
    print(f"Loaded embeddings: {embedding_source.feature_dim} dimensions")

    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    folds = k_fold_split_tasks(all_task_ids, k=k_folds, seed=config.split_seed)

    def load_fold_data(train_tasks, test_tasks, fold_idx):
        return load_dataset_for_fold(
            abilities_path=ROOT / config.abilities_path,
            items_path=ROOT / config.items_path,
            responses_path=ROOT / config.responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=k_folds,
            split_seed=config.split_seed,
            is_binomial=False,
            irt_cache_dir=Path("chris_output/experiment_a/irt_splits"),
        )

    # All configurations - will be split between GPUs
    all_configs: List[CVPredictorConfig] = []

    # Baselines (GPU 0 only)
    if args.gpu == 0:
        all_configs.append(CVPredictorConfig(
            predictor=OraclePredictor(),
            name="oracle",
            display_name="Oracle (true beta)",
        ))
        all_configs.append(CVPredictorConfig(
            predictor=DifficultyPredictorAdapter(
                FeatureBasedPredictor(embedding_source, alphas=config.ridge_alphas)
            ),
            name="ridge",
            display_name="Ridge (Embedding)",
        ))

    # Define all MLP configs
    mlp_configs = []

    # Daria-style baselines (scaled vs raw)
    for scale in [True, False]:
        scale_str = "scaled" if scale else "raw"
        mlp_configs.append({
            "name": f"daria_style_{scale_str}_random",
            "display": f"Daria-style ({scale_str}, random)",
            "emb": 64, "hid": [256, 128], "dp": 0.1, "lr": 0.001,
            "wd": 0.01, "ep": 10, "bs": 256, "es": False, "scale": scale
        })

    # LR search (raw features)
    for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
        mlp_configs.append({
            "name": f"raw_lr{lr}_random",
            "display": f"Raw, LR={lr}",
            "emb": 64, "hid": [256, 128], "dp": 0.1, "lr": lr,
            "wd": 0.01, "ep": 10, "bs": 256, "es": False, "scale": False
        })

    # Batch size search
    for bs in [64, 128, 256, 512]:
        mlp_configs.append({
            "name": f"raw_bs{bs}_random",
            "display": f"Raw, BS={bs}",
            "emb": 64, "hid": [256, 128], "dp": 0.1, "lr": 0.001,
            "wd": 0.01, "ep": 10, "bs": bs, "es": False, "scale": False
        })

    # Weight decay search
    for wd in [0.001, 0.01, 0.1]:
        mlp_configs.append({
            "name": f"raw_wd{wd}_random",
            "display": f"Raw, WD={wd}",
            "emb": 64, "hid": [256, 128], "dp": 0.1, "lr": 0.001,
            "wd": wd, "ep": 10, "bs": 256, "es": False, "scale": False
        })

    # Dropout search
    for dp in [0.0, 0.1, 0.2, 0.3]:
        mlp_configs.append({
            "name": f"raw_dp{dp}_random",
            "display": f"Raw, DP={dp}",
            "emb": 64, "hid": [256, 128], "dp": dp, "lr": 0.001,
            "wd": 0.01, "ep": 10, "bs": 256, "es": False, "scale": False
        })

    # Epochs search
    for ep in [10, 20, 50]:
        mlp_configs.append({
            "name": f"raw_ep{ep}_random",
            "display": f"Raw, Epochs={ep}",
            "emb": 64, "hid": [256, 128], "dp": 0.1, "lr": 0.001,
            "wd": 0.01, "ep": ep, "bs": 256, "es": False, "scale": False
        })

    # Embedding dim search
    for ed in [32, 64, 128]:
        mlp_configs.append({
            "name": f"raw_emb{ed}_random",
            "display": f"Raw, Emb={ed}",
            "emb": ed, "hid": [256, 128], "dp": 0.1, "lr": 0.001,
            "wd": 0.01, "ep": 10, "bs": 256, "es": False, "scale": False
        })

    # Promising combinations
    combos = [
        {"lr": 0.005, "ep": 20, "dp": 0.1, "bs": 256, "emb": 64},
        {"lr": 0.01, "ep": 20, "dp": 0.1, "bs": 256, "emb": 64},
        {"lr": 0.005, "ep": 20, "dp": 0.2, "bs": 256, "emb": 64},
        {"lr": 0.01, "ep": 20, "dp": 0.2, "bs": 256, "emb": 64},
        {"lr": 0.005, "ep": 20, "dp": 0.1, "bs": 128, "emb": 64},
        {"lr": 0.01, "ep": 20, "dp": 0.1, "bs": 128, "emb": 64},
        {"lr": 0.005, "ep": 20, "dp": 0.1, "bs": 256, "emb": 128},
        {"lr": 0.01, "ep": 20, "dp": 0.1, "bs": 256, "emb": 128},
    ]
    for i, c in enumerate(combos):
        mlp_configs.append({
            "name": f"raw_combo{i+1}_random",
            "display": f"Raw, LR={c['lr']}, DP={c['dp']}, BS={c['bs']}, Emb={c['emb']}",
            "emb": c["emb"], "hid": [256, 128], "dp": c["dp"], "lr": c["lr"],
            "wd": 0.01, "ep": c["ep"], "bs": c["bs"], "es": False, "scale": False
        })

    # Split configs between GPUs
    gpu0_configs = mlp_configs[::2]  # Even indices
    gpu1_configs = mlp_configs[1::2]  # Odd indices

    my_configs = gpu0_configs if args.gpu == 0 else gpu1_configs

    # Convert to CVPredictorConfig objects
    for cfg in my_configs:
        all_configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=cfg["emb"],
                hidden_sizes=cfg["hid"],
                dropout=cfg["dp"],
                learning_rate=cfg["lr"],
                weight_decay=cfg["wd"],
                n_epochs=cfg["ep"],
                batch_size=cfg["bs"],
                init_from_irt=False,
                early_stopping=cfg["es"],
                val_fraction=0.1 if cfg["es"] else 0.0,
                patience=20 if cfg["es"] else 0,
                verbose=False,
                scale_features=cfg["scale"],
            ),
            name=cfg["name"],
            display_name=cfg["display"],
        ))

    print(f"\n*** GPU {args.gpu}: Running {len(all_configs)} configs ***")

    results = {}

    for i, pc in enumerate(all_configs, 1):
        print(f"\n[{i}/{len(all_configs)}] {pc.display_name}")
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
            "elapsed_seconds": config_elapsed,
        }

        if cv_result.fold_diagnostics:
            valid_train_aucs = [t for t in cv_result.fold_diagnostics if t is not None]
            if valid_train_aucs:
                results[pc.name]["train_auc"] = float(np.mean(valid_train_aucs))

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} +/- {cv_result.std_auc:.4f}")
        print(f"   Time: {config_elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 85)
    print(f"GPU {args.gpu} RESULTS SUMMARY")
    print("=" * 85)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_auc"], reverse=True)
    for name, r in sorted_results[:10]:
        print(f"{r['display_name'][:50]:<52} {r['mean_auc']:.4f}")

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

    output_dir = ROOT / "chris_output/experiment_a/mlp_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"optimal_hyperparams_gpu{args.gpu}.json"

    with open(output_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {time.time() - main_start:.1f}s")


if __name__ == "__main__":
    main()
