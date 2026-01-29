"""Find optimal hyperparameters for mini-batch + AdamW + BCEWithLogitsLoss.

Tests:
- With and without StandardScaler on task features
- Various learning rates, weight decay, dropout, epochs, batch sizes
- Different hidden layer configurations and embedding dimensions

Usage:
    python -m experiment_a.mlp_ablation.test_optimal_hyperparams
    python -m experiment_a.mlp_ablation.test_optimal_hyperparams --quick
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
    print(f"Script starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    parser = argparse.ArgumentParser(description="Find optimal hyperparameters")
    parser.add_argument("--quick", action="store_true", help="Quick test with 2 folds")
    args = parser.parse_args()

    config = ExperimentAConfig()
    k_folds = 2 if args.quick else 5

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

    configs: List[CVPredictorConfig] = []

    # Baselines
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

    # Hyperparameter search space
    # Focus on mini-batch configurations

    # Core hyperparameters to search
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    weight_decays = [0.001, 0.01, 0.1]
    dropouts = [0.0, 0.1, 0.2, 0.3]
    batch_sizes = [64, 128, 256, 512]
    hidden_configs = [[256, 128], [128, 64], [256, 128, 64]]
    epochs_list = [10, 20, 50]
    emb_dims = [32, 64, 128]
    scale_features_options = [True, False]

    # Create configurations
    # Start with Daria's exact config as baseline (both with and without scaling)
    for scale in scale_features_options:
        scale_str = "scaled" if scale else "raw"
        configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=64,
                hidden_sizes=[256, 128],
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=0.01,
                n_epochs=10,
                batch_size=256,
                init_from_irt=False,
                early_stopping=False,
                verbose=False,
                scale_features=scale,
            ),
            name=f"daria_style_{scale_str}_random",
            display_name=f"Daria-style ({scale_str}, random)",
        ))

    # Search over key hyperparameters with scale_features=False (match Daria)
    # LR search
    for lr in learning_rates:
        configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=64,
                hidden_sizes=[256, 128],
                dropout=0.1,
                learning_rate=lr,
                weight_decay=0.01,
                n_epochs=10,
                batch_size=256,
                init_from_irt=False,
                early_stopping=False,
                verbose=False,
                scale_features=False,
            ),
            name=f"raw_lr{lr}_random",
            display_name=f"Raw, LR={lr}, random",
        ))

    # Batch size search
    for bs in batch_sizes:
        configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=64,
                hidden_sizes=[256, 128],
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=0.01,
                n_epochs=10,
                batch_size=bs,
                init_from_irt=False,
                early_stopping=False,
                verbose=False,
                scale_features=False,
            ),
            name=f"raw_bs{bs}_random",
            display_name=f"Raw, BS={bs}, random",
        ))

    # Weight decay search
    for wd in weight_decays:
        configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=64,
                hidden_sizes=[256, 128],
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=wd,
                n_epochs=10,
                batch_size=256,
                init_from_irt=False,
                early_stopping=False,
                verbose=False,
                scale_features=False,
            ),
            name=f"raw_wd{wd}_random",
            display_name=f"Raw, WD={wd}, random",
        ))

    # Dropout search
    for dp in dropouts:
        configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=64,
                hidden_sizes=[256, 128],
                dropout=dp,
                learning_rate=0.001,
                weight_decay=0.01,
                n_epochs=10,
                batch_size=256,
                init_from_irt=False,
                early_stopping=False,
                verbose=False,
                scale_features=False,
            ),
            name=f"raw_dp{dp}_random",
            display_name=f"Raw, Dropout={dp}, random",
        ))

    # Epochs search
    for ep in epochs_list:
        configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=64,
                hidden_sizes=[256, 128],
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=0.01,
                n_epochs=ep,
                batch_size=256,
                init_from_irt=False,
                early_stopping=False,
                verbose=False,
                scale_features=False,
            ),
            name=f"raw_ep{ep}_random",
            display_name=f"Raw, Epochs={ep}, random",
        ))

    # Hidden configs search
    for hc in hidden_configs:
        hc_str = "-".join(map(str, hc))
        configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=64,
                hidden_sizes=hc,
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=0.01,
                n_epochs=10,
                batch_size=256,
                init_from_irt=False,
                early_stopping=False,
                verbose=False,
                scale_features=False,
            ),
            name=f"raw_hid{hc_str}_random",
            display_name=f"Raw, Hidden={hc_str}, random",
        ))

    # Embedding dim search
    for ed in emb_dims:
        configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=ed,
                hidden_sizes=[256, 128],
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=0.01,
                n_epochs=10,
                batch_size=256,
                init_from_irt=False,
                early_stopping=False,
                verbose=False,
                scale_features=False,
            ),
            name=f"raw_emb{ed}_random",
            display_name=f"Raw, EmbDim={ed}, random",
        ))

    # More epochs with early stopping
    for ep in [50, 100, 200]:
        configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=64,
                hidden_sizes=[256, 128],
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=0.01,
                n_epochs=ep,
                batch_size=256,
                init_from_irt=False,
                early_stopping=True,
                val_fraction=0.1,
                patience=20,
                verbose=False,
                scale_features=False,
            ),
            name=f"raw_ep{ep}_es_random",
            display_name=f"Raw, Epochs={ep}+ES, random",
        ))

    # Promising combinations based on single-variable search
    promising_configs = [
        # Higher LR with more epochs and early stopping
        {"lr": 0.005, "ep": 50, "es": True, "dp": 0.1, "wd": 0.01, "bs": 256, "hid": [256, 128], "emb": 64},
        {"lr": 0.01, "ep": 50, "es": True, "dp": 0.1, "wd": 0.01, "bs": 256, "hid": [256, 128], "emb": 64},
        # More dropout with higher LR
        {"lr": 0.005, "ep": 20, "es": False, "dp": 0.2, "wd": 0.01, "bs": 256, "hid": [256, 128], "emb": 64},
        {"lr": 0.01, "ep": 20, "es": False, "dp": 0.3, "wd": 0.01, "bs": 256, "hid": [256, 128], "emb": 64},
        # Smaller batch with higher LR
        {"lr": 0.005, "ep": 20, "es": False, "dp": 0.1, "wd": 0.01, "bs": 128, "hid": [256, 128], "emb": 64},
        {"lr": 0.01, "ep": 20, "es": False, "dp": 0.1, "wd": 0.01, "bs": 64, "hid": [256, 128], "emb": 64},
        # Larger embedding dim
        {"lr": 0.001, "ep": 10, "es": False, "dp": 0.1, "wd": 0.01, "bs": 256, "hid": [256, 128], "emb": 128},
        {"lr": 0.005, "ep": 20, "es": True, "dp": 0.1, "wd": 0.01, "bs": 256, "hid": [256, 128], "emb": 128},
        # Deeper network
        {"lr": 0.001, "ep": 20, "es": False, "dp": 0.1, "wd": 0.01, "bs": 256, "hid": [256, 128, 64], "emb": 64},
        {"lr": 0.005, "ep": 50, "es": True, "dp": 0.2, "wd": 0.01, "bs": 256, "hid": [256, 128, 64], "emb": 64},
        # Lower weight decay
        {"lr": 0.001, "ep": 10, "es": False, "dp": 0.1, "wd": 0.001, "bs": 256, "hid": [256, 128], "emb": 64},
        {"lr": 0.005, "ep": 20, "es": False, "dp": 0.1, "wd": 0.001, "bs": 256, "hid": [256, 128], "emb": 64},
    ]

    for i, pc in enumerate(promising_configs):
        hid_str = "-".join(map(str, pc["hid"]))
        es_str = "_es" if pc["es"] else ""
        name = f"raw_combo{i+1}_lr{pc['lr']}_dp{pc['dp']}_emb{pc['emb']}{es_str}_random"
        display = f"Raw, LR={pc['lr']}, DP={pc['dp']}, Emb={pc['emb']}, H={hid_str}{', ES' if pc['es'] else ''}"

        configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=pc["emb"],
                hidden_sizes=pc["hid"],
                dropout=pc["dp"],
                learning_rate=pc["lr"],
                weight_decay=pc["wd"],
                n_epochs=pc["ep"],
                batch_size=pc["bs"],
                init_from_irt=False,
                early_stopping=pc["es"],
                val_fraction=0.1 if pc["es"] else 0.0,
                patience=20 if pc["es"] else 0,
                verbose=False,
                scale_features=False,
            ),
            name=name,
            display_name=display,
        ))

    print(f"\n*** Running {len(configs)} configs ***")

    results = {}

    print("\n" + "=" * 85)
    print(f"OPTIMAL HYPERPARAMETER SEARCH (starting at {time.strftime('%H:%M:%S')})")
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
    print("RESULTS SUMMARY")
    print("=" * 85)
    print(f"\n{'Method':<55} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 95)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_auc"], reverse=True)

    for name, r in sorted_results:
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc")
        display_name = r["display_name"][:53]

        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{display_name:<55} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{display_name:<55} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Key comparisons
    print("\n" + "=" * 85)
    print("KEY COMPARISONS")
    print("=" * 85)

    if "ridge" in results:
        print(f"\nRidge baseline: {results['ridge']['mean_auc']:.4f}")

    if "daria_style_scaled_random" in results and "daria_style_raw_random" in results:
        scaled = results["daria_style_scaled_random"]["mean_auc"]
        raw = results["daria_style_raw_random"]["mean_auc"]
        print(f"Daria-style (scaled): {scaled:.4f}")
        print(f"Daria-style (raw): {raw:.4f}")
        print(f"Scaling impact: {scaled - raw:+.4f}")

    # Best raw feature config
    raw_results = [(n, r) for n, r in sorted_results if "raw" in n.lower() and n != "ridge"]
    if raw_results:
        best_raw = raw_results[0]
        print(f"\nBest raw features config: {best_raw[1]['mean_auc']:.4f} ({best_raw[1]['display_name']})")
        if "ridge" in results:
            gap_to_ridge = results["ridge"]["mean_auc"] - best_raw[1]["mean_auc"]
            print(f"Gap to Ridge: {gap_to_ridge:+.4f}")

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
    output_path = output_dir / "optimal_hyperparams_test.json"

    with open(output_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {time.time() - main_start:.1f}s")


if __name__ == "__main__":
    main()
