"""Extract learned agent embeddings from trained AgentEmb model.

Trains an AgentEmbeddingPredictor on the full dataset and extracts the
learned 64-dim agent embeddings. Saves them to CSV for visualization.

Usage:
    # On cluster:
    python -m experiment_a.mlp_ablation.extract_agent_embeddings
    python -m experiment_a.mlp_ablation.extract_agent_embeddings --noise 1.0

    # Then transfer: chris_output/experiment_a/mlp_embedding/agent_embeddings*.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_ab_shared.feature_source import EmbeddingFeatureSource
from experiment_ab_shared import load_dataset_for_fold
from experiment_a.shared.cross_validation import k_fold_split_tasks
from experiment_a.shared.mlp_predictor import AgentEmbeddingPredictor
from experiment_a.swebench.config import ExperimentAConfig


ROOT = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Extract agent embeddings")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Init noise scale (0.0 = no noise)")
    args = parser.parse_args()

    start = time.time()
    print(f"Starting at: {time.strftime('%H:%M:%S')}")
    print(f"Init noise scale: {args.noise}")

    config = ExperimentAConfig()

    # Load embeddings
    embeddings_path = ROOT / config.embeddings_path
    embedding_source = EmbeddingFeatureSource(embeddings_path)
    print(f"Loaded embeddings: {embedding_source.feature_dim} dimensions")

    # Load items to get all task IDs
    items_path = ROOT / config.items_path
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    # Use fold 0 for training (80% of tasks)
    folds = k_fold_split_tasks(all_task_ids, k=5, seed=config.split_seed)
    train_tasks, test_tasks = folds[0]

    print(f"Training on {len(train_tasks)} tasks, holding out {len(test_tasks)}")

    # Load data for fold 0
    data = load_dataset_for_fold(
        abilities_path=ROOT / config.abilities_path,
        items_path=ROOT / config.items_path,
        responses_path=ROOT / config.responses_path,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        fold_idx=0,
        k_folds=5,
        split_seed=config.split_seed,
        is_binomial=False,
        irt_cache_dir=Path("chris_output/experiment_a/irt_splits"),
    )

    # Train AgentEmb model with best config from sweep
    print("\nTraining AgentEmbeddingPredictor...")
    predictor = AgentEmbeddingPredictor(
        embedding_source,
        agent_emb_dim=64,
        hidden_sizes=[64, 64],
        dropout=0.0,
        learning_rate=0.01,
        weight_decay=0.2,
        n_epochs=1000,
        init_from_irt=True,
        init_noise_scale=args.noise,
        early_stopping=True,
        val_fraction=0.1,
        patience=30,
        verbose=True,
    )

    predictor.fit(data, train_tasks)
    print(f"\nTraining complete. Train AUC: {predictor.get_train_auc():.4f}")

    # Extract agent embeddings from the model
    model = predictor._model
    agent_to_idx = predictor._agent_to_idx

    # Get embedding weights: shape (n_agents, agent_emb_dim)
    with torch.no_grad():
        embeddings = model.agent_embedding.weight.cpu().numpy()

    print(f"\nExtracted embeddings: {embeddings.shape}")

    # Build dataframe with agent info
    rows = []
    for agent_id, idx in agent_to_idx.items():
        emb = embeddings[idx]

        # Get IRT ability
        if agent_id in data.train_abilities.index:
            ability = float(data.train_abilities.loc[agent_id, "ability"])
        else:
            ability = np.nan

        row = {
            "agent_id": agent_id,
            "ability": ability,
            "idx": idx,
        }
        # Add embedding dimensions
        for i, val in enumerate(emb):
            row[f"emb_{i}"] = val

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by ability for easier inspection
    df = df.sort_values("ability", ascending=False)

    # Save
    output_dir = ROOT / "chris_output/experiment_a/mlp_embedding"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.noise > 0:
        output_path = output_dir / f"agent_embeddings_noise{args.noise}.csv"
    else:
        output_path = output_dir / "agent_embeddings.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved embeddings to: {output_path}")
    print(f"Columns: {list(df.columns[:5])} + {embeddings.shape[1]} embedding dims")
    print(f"\nTop 5 agents by ability:")
    print(df[["agent_id", "ability"]].head())
    print(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
