#!/usr/bin/env python3
"""Generate embeddings for TerminalBench tasks.

Embeds task instruction + solution from the local terminal-bench repo
using a VLM backbone for difficulty prediction.

This script reuses the embedding infrastructure from predict_question_difficulty.py
(developed by Daria with careful ablations) to ensure consistent methodology.

The format is:
    Task statement:
    {instruction from task.yaml}

    Solution:
    {content of solution.sh}

    How difficult is the above task for a coding agent? ...

Example usage:
    python -m experiment_a_terminalbench.generate_embeddings \
        --out_dir chris_output/experiment_a_terminalbench/embeddings \
        --batch_size 1 \
        --device_map auto

For SLURM cluster:
    sbatch slurm_scripts/terminalbench_embeddings.sh
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a_terminalbench.data_loader import load_task_data_from_repo

# Reuse the embedding infrastructure from predict_question_difficulty.py
# This uses the exact same methodology developed for experiment_a (SWE-bench)
from predict_question_difficulty import (
    DIFFICULTY_INSTRUCTION,
    ItemRecord,
    embed_items,
)


def load_terminalbench_items(
    items_path: Path,
    repo_path: Path,
) -> List[ItemRecord]:
    """Load TerminalBench tasks as ItemRecords for embedding.

    The ItemRecord format maps:
    - item_id -> task_id (e.g., "3d-model-format-legacy")
    - question_statement -> instruction from task.yaml
    - solution -> content of solution.sh

    Args:
        items_path: Path to 1PL items.csv (to get list of task IDs)
        repo_path: Path to cloned terminal-bench repo

    Returns:
        List of ItemRecord objects compatible with embed_items()
    """
    # Load task IDs from items.csv
    items_df = pd.read_csv(items_path, index_col=0)
    task_ids = list(items_df.index)
    print(f"Found {len(task_ids)} task IDs in items.csv")

    # Load task data from repo (reusing existing function)
    task_data = load_task_data_from_repo(task_ids, repo_path)
    print(f"Loaded task data for {len(task_data)} tasks")

    # Convert to ItemRecord format (compatible with embed_items from predict_question_difficulty.py)
    # question_statement = instruction from task.yaml
    # solution = content of solution.sh
    items = []
    for task_id in task_ids:
        if task_id not in task_data:
            print(f"Warning: no task data for {task_id}")
            continue

        data = task_data[task_id]
        items.append(ItemRecord(
            item_id=task_id,
            question_statement=data["instruction"],
            solution=data["solution"],
        ))

    return items


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for TerminalBench tasks"
    )
    parser.add_argument(
        "--items_path",
        type=str,
        default="chris_output/terminal_bench_2.0_binomial_1pl/1d/items.csv",
        help="Path to 1PL items.csv",
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        default="terminal-bench",
        help="Path to cloned terminal-bench repo",
    )
    # Use same defaults as predict_question_difficulty.py for experiment_a
    parser.add_argument(
        "--backbone",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model for embeddings (same as experiment_a)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length (same as experiment_a)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for embedding",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map (e.g., auto, none)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        help="Attention implementation (e.g., auto, flash_attention_2)",
    )
    parser.add_argument(
        "--embedding_layer",
        type=int,
        default=-1,
        help="Which hidden layer to pool embeddings from (-1 = last)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for model loading",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="chris_output/experiment_a_terminalbench/embeddings",
        help="Output directory",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DIFFICULTY_INSTRUCTION,
        help="Difficulty instruction to append",
    )
    args = parser.parse_args()

    # Resolve paths
    items_path = ROOT / args.items_path
    repo_path = ROOT / args.repo_path
    out_dir = ROOT / args.out_dir

    print(f"Loading tasks from: {items_path}")
    print(f"Terminal-bench repo: {repo_path}")

    # Load tasks as ItemRecords
    items = load_terminalbench_items(items_path, repo_path)
    print(f"Prepared {len(items)} items for embedding")

    # Embed tasks using the exact same infrastructure as experiment_a
    # This function is from predict_question_difficulty.py (Daria's carefully tuned setup)
    print(f"Embedding with backbone: {args.backbone}")
    print(f"Max length: {args.max_length}")
    print(f"Embedding layer: {args.embedding_layer}")

    ids_sorted, embeddings_by_id, counts_by_id, embedding_dim = embed_items(
        items=items,
        backbone=args.backbone,
        trust_remote_code=args.trust_remote_code,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        instruction=args.instruction,
        embedding_layer=args.embedding_layer,
    )

    print(f"Embedded {len(ids_sorted)} tasks with dim={embedding_dim}")

    # Build embedding matrix
    X = np.stack([embeddings_by_id[tid] for tid in ids_sorted], axis=0).astype(np.float32)

    # Save embeddings
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with model info (similar format to predict_question_difficulty.py)
    safe_backbone = args.backbone.replace("/", "__")
    layer_flag = "" if args.embedding_layer == -1 else f"__layer{args.embedding_layer}"
    out_path = out_dir / f"embeddings__{safe_backbone}__pool-lasttoken{layer_flag}__maxlen{args.max_length}.npz"

    np.savez_compressed(
        out_path,
        task_ids=np.array(ids_sorted, dtype=object),
        X=X,
        backbone=np.array([args.backbone], dtype=object),
        max_length=np.array([args.max_length], dtype=np.int64),
        embedding_dim=np.array([embedding_dim], dtype=np.int64),
        embedding_layer=np.array([args.embedding_layer], dtype=np.int64),
        instruction=np.array([args.instruction], dtype=object),
    )

    print(f"Saved embeddings to: {out_path}")


if __name__ == "__main__":
    main()
