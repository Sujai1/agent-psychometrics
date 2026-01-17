#!/usr/bin/env python3
"""Analyze SAD-IRT checkpoints for parameter changes and gradient flow.

This script checks:
1. Whether LoRA parameters changed from initialization
2. Whether LoRA parameters are nonzero
3. Parameter statistics across training

Usage:
    python -m experiment_sad_irt.analyze_checkpoint chris_output/sad_irt_long/full
    python -m experiment_sad_irt.analyze_checkpoint chris_output/sad_irt_long/freeze_irt
"""

import argparse
import re
from pathlib import Path

import torch
import numpy as np


def analyze_lora_params(state_dict: dict) -> dict:
    """Analyze LoRA parameters in a state dict."""
    lora_stats = {
        "lora_A": {"count": 0, "total_params": 0, "nonzero": 0, "norm": 0.0, "max_abs": 0.0},
        "lora_B": {"count": 0, "total_params": 0, "nonzero": 0, "norm": 0.0, "max_abs": 0.0},
    }

    for key, param in state_dict.items():
        if "lora_A" in key:
            cat = "lora_A"
        elif "lora_B" in key:
            cat = "lora_B"
        else:
            continue

        param_np = param.cpu().numpy().flatten()
        lora_stats[cat]["count"] += 1
        lora_stats[cat]["total_params"] += len(param_np)
        lora_stats[cat]["nonzero"] += np.count_nonzero(param_np)
        lora_stats[cat]["norm"] += np.sum(param_np ** 2)
        lora_stats[cat]["max_abs"] = max(lora_stats[cat]["max_abs"], np.max(np.abs(param_np)))

    # Finalize norm
    for cat in lora_stats:
        lora_stats[cat]["norm"] = np.sqrt(lora_stats[cat]["norm"])

    return lora_stats


def analyze_embeddings(state_dict: dict) -> dict:
    """Analyze θ and β embeddings."""
    stats = {}

    for key in ["theta_embedding.weight", "beta_embedding.weight"]:
        if key in state_dict:
            param = state_dict[key].cpu().numpy()
            stats[key] = {
                "shape": param.shape,
                "mean": float(np.mean(param)),
                "std": float(np.std(param)),
                "min": float(np.min(param)),
                "max": float(np.max(param)),
                "norm": float(np.linalg.norm(param)),
            }

    return stats


def analyze_psi_head(state_dict: dict) -> dict:
    """Analyze ψ head parameters."""
    stats = {}

    for suffix in ["weight", "bias"]:
        key = f"psi_head.{suffix}"
        if key in state_dict:
            param = state_dict[key].cpu().numpy()
            stats[key] = {
                "shape": param.shape,
                "mean": float(np.mean(param)),
                "std": float(np.std(param)),
                "min": float(np.min(param)),
                "max": float(np.max(param)),
                "norm": float(np.linalg.norm(param)),
                "nonzero_frac": float(np.count_nonzero(param) / param.size),
            }

    return stats


def compare_checkpoints(checkpoint1: dict, checkpoint2: dict, label1: str, label2: str) -> dict:
    """Compare two checkpoints to see what changed."""
    changes = {}

    sd1 = checkpoint1.get("model_state_dict", checkpoint1)
    sd2 = checkpoint2.get("model_state_dict", checkpoint2)

    # Find common keys
    common_keys = set(sd1.keys()) & set(sd2.keys())

    for key in sorted(common_keys):
        p1 = sd1[key].cpu().numpy()
        p2 = sd2[key].cpu().numpy()

        diff = p2 - p1
        diff_norm = np.linalg.norm(diff)
        p1_norm = np.linalg.norm(p1)

        if diff_norm > 1e-10:  # Only report if actually changed
            changes[key] = {
                "diff_norm": float(diff_norm),
                "relative_change": float(diff_norm / (p1_norm + 1e-10)),
                "max_abs_diff": float(np.max(np.abs(diff))),
            }

    return changes


def parse_gradient_logs(log_path: Path) -> list:
    """Parse gradient information from training log."""
    gradients = []

    if not log_path.exists():
        return gradients

    with open(log_path, "r") as f:
        content = f.read()

    # Pattern for gradient debug output
    # Looking for patterns like: "lora grad norm: 0.0000"
    pattern = r"Gradient norms.*?lora.*?:\s*([\d.e+-]+)"

    for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
        try:
            gradients.append(float(match.group(1)))
        except ValueError:
            pass

    # Also look for detailed gradient logs
    detail_pattern = r"lora_[AB].*?grad.*?norm.*?:\s*([\d.e+-]+)"
    for match in re.finditer(detail_pattern, content, re.IGNORECASE):
        try:
            val = float(match.group(1))
            if val not in gradients:
                gradients.append(val)
        except ValueError:
            pass

    return gradients


def analyze_checkpoint_dir(output_dir: Path) -> None:
    """Analyze all checkpoints in a directory."""
    output_dir = Path(output_dir)

    print(f"\n{'='*70}")
    print(f"Analyzing: {output_dir}")
    print(f"{'='*70}")

    # Find all checkpoints
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))

    if not checkpoints:
        print("ERROR: No checkpoints found")
        return

    print(f"\nFound {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp.name}")

    # Load and analyze each checkpoint
    checkpoint_data = {}
    for cp_path in checkpoints:
        print(f"\n--- Analyzing {cp_path.name} ---")

        checkpoint = torch.load(cp_path, map_location="cpu")
        checkpoint_data[cp_path.name] = checkpoint

        # Get state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Analyze LoRA parameters
        lora_stats = analyze_lora_params(state_dict)
        print(f"\nLoRA Parameters:")
        for cat, stats in lora_stats.items():
            if stats["count"] > 0:
                print(f"  {cat}:")
                print(f"    Layers: {stats['count']}")
                print(f"    Total params: {stats['total_params']}")
                print(f"    Nonzero params: {stats['nonzero']} ({100*stats['nonzero']/max(1,stats['total_params']):.2f}%)")
                print(f"    L2 norm: {stats['norm']:.6f}")
                print(f"    Max |value|: {stats['max_abs']:.6f}")

        # Analyze embeddings
        emb_stats = analyze_embeddings(state_dict)
        if emb_stats:
            print(f"\nEmbeddings:")
            for key, stats in emb_stats.items():
                print(f"  {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, norm={stats['norm']:.4f}")

        # Analyze ψ head
        psi_stats = analyze_psi_head(state_dict)
        if psi_stats:
            print(f"\nψ Head:")
            for key, stats in psi_stats.items():
                print(f"  {key}: norm={stats['norm']:.6f}, nonzero={stats['nonzero_frac']*100:.1f}%")

        # Print checkpoint metadata
        if "epoch" in checkpoint:
            print(f"\nMetadata:")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Global step: {checkpoint.get('global_step', 'N/A')}")
            if "best_spearman_rho" in checkpoint:
                print(f"  Best Spearman ρ: {checkpoint['best_spearman_rho']:.4f}")

    # Compare first and last checkpoints
    if len(checkpoints) >= 2:
        first_cp = checkpoint_data[checkpoints[0].name]
        last_cp = checkpoint_data[checkpoints[-1].name]

        print(f"\n{'='*70}")
        print(f"COMPARISON: {checkpoints[0].name} → {checkpoints[-1].name}")
        print(f"{'='*70}")

        changes = compare_checkpoints(first_cp, last_cp, checkpoints[0].name, checkpoints[-1].name)

        if not changes:
            print("\nWARNING: No parameters changed between checkpoints!")
        else:
            # Categorize changes
            lora_changes = {k: v for k, v in changes.items() if "lora" in k.lower()}
            emb_changes = {k: v for k, v in changes.items() if "embedding" in k.lower()}
            psi_changes = {k: v for k, v in changes.items() if "psi" in k.lower()}
            other_changes = {k: v for k, v in changes.items()
                           if k not in lora_changes and k not in emb_changes and k not in psi_changes}

            print(f"\nLoRA parameter changes: {len(lora_changes)}")
            if lora_changes:
                total_lora_diff = sum(v["diff_norm"] for v in lora_changes.values())
                max_lora_diff = max(v["diff_norm"] for v in lora_changes.values())
                print(f"  Total L2 change: {total_lora_diff:.6f}")
                print(f"  Max single param change: {max_lora_diff:.6f}")
                # Show top 5 changes
                sorted_lora = sorted(lora_changes.items(), key=lambda x: x[1]["diff_norm"], reverse=True)[:5]
                for key, stats in sorted_lora:
                    short_key = key.split(".")[-3] + "." + key.split(".")[-2] + "." + key.split(".")[-1]
                    print(f"    {short_key}: Δ={stats['diff_norm']:.6f}")
            else:
                print("  *** NO LORA PARAMETERS CHANGED ***")

            print(f"\nEmbedding changes: {len(emb_changes)}")
            for key, stats in emb_changes.items():
                print(f"  {key}: Δ={stats['diff_norm']:.4f} ({stats['relative_change']*100:.2f}%)")

            print(f"\nψ head changes: {len(psi_changes)}")
            for key, stats in psi_changes.items():
                print(f"  {key}: Δ={stats['diff_norm']:.6f}")

            if other_changes:
                print(f"\nOther changes: {len(other_changes)}")

    # Try to parse gradient logs
    log_patterns = [
        output_dir.parent / f"logs/sad_irt_long_*.out",
        Path("logs") / f"sad_irt_long_*.out",
    ]

    for pattern in log_patterns:
        log_files = list(pattern.parent.glob(pattern.name)) if pattern.parent.exists() else []
        if log_files:
            print(f"\n{'='*70}")
            print("GRADIENT LOG ANALYSIS")
            print(f"{'='*70}")

            for log_file in log_files:
                print(f"\nParsing: {log_file}")
                gradients = parse_gradient_logs(log_file)
                if gradients:
                    print(f"  Found {len(gradients)} LoRA gradient values")
                    print(f"  Min: {min(gradients):.6e}")
                    print(f"  Max: {max(gradients):.6e}")
                    print(f"  Nonzero: {sum(1 for g in gradients if g > 1e-10)}/{len(gradients)}")
                else:
                    print("  No gradient values found in log")
            break


def main():
    parser = argparse.ArgumentParser(description="Analyze SAD-IRT checkpoints")
    parser.add_argument("output_dirs", nargs="+", help="Output directories to analyze")
    args = parser.parse_args()

    for d in args.output_dirs:
        analyze_checkpoint_dir(Path(d))


if __name__ == "__main__":
    main()
