#!/usr/bin/env python3
"""Analyze psi distribution from SAD-IRT checkpoint.

Computes:
1. Per-agent mean(psi_ij) across all tasks
2. Distribution statistics on per-agent means
3. Overall mean and std across all psi_ij values

Usage:
    python -m experiment_sad_irt.analyze_psi_distribution \
        --checkpoint chris_output/sad_irt_long/full_20260118_024625_psi_batchnorm_lora_r64/checkpoint_epoch_9_step4248_20260118_044922.pt

Requires GPU for efficient forward passes through the encoder.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint_utils import load_checkpoint_for_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_psi_distribution(
    checkpoint_path: str,
    batch_size: int = 16,
    output_path: str | None = None,
) -> Dict:
    """Analyze psi distribution from a SAD-IRT checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        batch_size: Batch size for inference
        output_path: Optional path to save JSON results

    Returns:
        Dict with analysis results
    """
    checkpoint_path = Path(checkpoint_path)

    # Load checkpoint for inference
    ckpt = load_checkpoint_for_inference(str(checkpoint_path))
    dataloader = ckpt.create_dataloader(batch_size=batch_size)

    # Collect psi values per agent
    agent_psi_values: Dict[int, List[float]] = defaultdict(list)
    all_psi_values: List[float] = []

    logger.info(f"Computing psi for {len(ckpt.dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch = {k: v.to(ckpt.device) for k, v in batch.items()}

            _, _, _, psi = ckpt.model.forward_with_components(
                agent_idx=batch["agent_idx"],
                task_idx=batch["task_idx"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            psi_np = psi.cpu().numpy()
            agent_idx_np = batch["agent_idx"].cpu().numpy()

            for a_idx, p in zip(agent_idx_np, psi_np):
                agent_psi_values[int(a_idx)].append(float(p))
                all_psi_values.append(float(p))

    # Compute statistics
    logger.info("\n" + "=" * 70)
    logger.info("PSI DISTRIBUTION ANALYSIS")
    logger.info("=" * 70)

    # 1. Per-agent mean(psi_ij) statistics
    per_agent_means = []
    agent_stats = []

    for agent_idx in sorted(agent_psi_values.keys()):
        psi_vals = agent_psi_values[agent_idx]
        mean_psi = np.mean(psi_vals)
        std_psi = np.std(psi_vals)
        per_agent_means.append(mean_psi)

        agent_name = ckpt.agent_ids[agent_idx] if agent_idx < len(ckpt.agent_ids) else f"agent_{agent_idx}"
        agent_stats.append({
            "agent_idx": agent_idx,
            "agent_name": agent_name,
            "mean_psi": mean_psi,
            "std_psi": std_psi,
            "n_samples": len(psi_vals),
        })

    logger.info("\n1. Per-agent mean(psi_ij) across all tasks:")
    for stat in agent_stats:
        logger.info(
            f"   Agent {stat['agent_idx']:2d} ({stat['agent_name'][:40]:40s}): "
            f"mean={stat['mean_psi']:+.4f}, std={stat['std_psi']:.4f}, n={stat['n_samples']}"
        )

    # Distribution of per-agent means
    per_agent_means_arr = np.array(per_agent_means)
    logger.info("\n   Distribution of per-agent means:")
    logger.info(f"      Mean of means: {np.mean(per_agent_means_arr):.4f}")
    logger.info(f"      Std of means:  {np.std(per_agent_means_arr):.4f}")
    logger.info(f"      Min mean:      {np.min(per_agent_means_arr):.4f}")
    logger.info(f"      Max mean:      {np.max(per_agent_means_arr):.4f}")
    logger.info(f"      Median mean:   {np.median(per_agent_means_arr):.4f}")

    # 2. Overall psi statistics
    all_psi_arr = np.array(all_psi_values)
    logger.info("\n2. Overall psi_ij statistics (all agent-task pairs):")
    logger.info(f"   N samples:  {len(all_psi_arr)}")
    logger.info(f"   Mean:       {np.mean(all_psi_arr):.4f}")
    logger.info(f"   Std:        {np.std(all_psi_arr):.4f}")
    logger.info(f"   Min:        {np.min(all_psi_arr):.4f}")
    logger.info(f"   Max:        {np.max(all_psi_arr):.4f}")
    logger.info(f"   P5:         {np.percentile(all_psi_arr, 5):.4f}")
    logger.info(f"   P25:        {np.percentile(all_psi_arr, 25):.4f}")
    logger.info(f"   P50:        {np.percentile(all_psi_arr, 50):.4f}")
    logger.info(f"   P75:        {np.percentile(all_psi_arr, 75):.4f}")
    logger.info(f"   P95:        {np.percentile(all_psi_arr, 95):.4f}")

    # 3. BatchNorm running stats (for comparison)
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bn_mean = raw_ckpt["model_state_dict"].get("psi_bn.running_mean")
    bn_var = raw_ckpt["model_state_dict"].get("psi_bn.running_var")
    if bn_mean is not None and bn_var is not None:
        logger.info("\n3. BatchNorm running statistics (from training):")
        logger.info(f"   Running mean:  {bn_mean.item():.4f}")
        logger.info(f"   Running var:   {bn_var.item():.4f}")
        logger.info(f"   Running std:   {np.sqrt(bn_var.item()):.4f}")

    # Build results dict
    results = {
        "checkpoint": str(checkpoint_path),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in ckpt.config.items()},
        "n_agents": len(agent_stats),
        "n_samples": len(all_psi_arr),
        "per_agent_stats": agent_stats,
        "per_agent_mean_distribution": {
            "mean_of_means": float(np.mean(per_agent_means_arr)),
            "std_of_means": float(np.std(per_agent_means_arr)),
            "min_mean": float(np.min(per_agent_means_arr)),
            "max_mean": float(np.max(per_agent_means_arr)),
            "median_mean": float(np.median(per_agent_means_arr)),
        },
        "overall_psi_stats": {
            "mean": float(np.mean(all_psi_arr)),
            "std": float(np.std(all_psi_arr)),
            "min": float(np.min(all_psi_arr)),
            "max": float(np.max(all_psi_arr)),
            "p5": float(np.percentile(all_psi_arr, 5)),
            "p25": float(np.percentile(all_psi_arr, 25)),
            "p50": float(np.percentile(all_psi_arr, 50)),
            "p75": float(np.percentile(all_psi_arr, 75)),
            "p95": float(np.percentile(all_psi_arr, 95)),
        },
    }

    if bn_mean is not None:
        results["batchnorm_running_stats"] = {
            "running_mean": float(bn_mean.item()),
            "running_var": float(bn_var.item()),
            "running_std": float(np.sqrt(bn_var.item())),
        }

    # Save results
    if output_path is None:
        output_path = checkpoint_path.parent / "psi_analysis.json"
    else:
        output_path = Path(output_path)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze psi distribution from SAD-IRT checkpoint")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to SAD-IRT checkpoint file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSON results (default: checkpoint_dir/psi_analysis.json)",
    )
    args = parser.parse_args()

    analyze_psi_distribution(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
