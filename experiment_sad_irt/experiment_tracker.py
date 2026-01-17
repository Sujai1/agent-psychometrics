#!/usr/bin/env python3
"""Experiment tracking for SAD-IRT ablations.

Logs hyperparameters, architectural choices, and results to a central CSV file
for easy comparison across runs.

Usage:
    # At end of training (called automatically by Trainer)
    tracker = ExperimentTracker("chris_output/sad_irt_experiments.csv")
    tracker.log_experiment(config, history, final_metrics)

    # View all experiments
    python -m experiment_sad_irt.experiment_tracker --summary
    python -m experiment_sad_irt.experiment_tracker --compare chris_output/sad_irt_experiments.csv
"""

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class ExperimentTracker:
    """Track experiments and their results in a central CSV file."""

    def __init__(self, csv_path: str = "chris_output/sad_irt_experiments.csv"):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def log_experiment(
        self,
        config,
        history: Dict[str, List],
        final_metrics: Dict[str, float],
        output_dir: str,
        notes: str = "",
    ) -> None:
        """Log a completed experiment.

        Args:
            config: SADIRTConfig object or dict
            history: Training history dict with loss, gradients, etc.
            final_metrics: Final evaluation metrics
            output_dir: Path to experiment output directory
            notes: Optional notes about this run
        """
        # Convert config to dict if needed
        if hasattr(config, "__dataclass_fields__"):
            config_dict = asdict(config)
        else:
            config_dict = dict(config)

        # Analyze training dynamics
        training_analysis = self._analyze_training(history)

        # Build experiment record
        record = {
            # Metadata
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(output_dir),
            "notes": notes,

            # Key hyperparameters
            "model_name": config_dict.get("model_name", ""),
            "lora_r": config_dict.get("lora_r", 16),
            "lora_alpha": config_dict.get("lora_alpha", 32),
            "lora_dropout": config_dict.get("lora_dropout", 0.1),
            "psi_normalization": config_dict.get("psi_normalization", "center"),
            "freeze_irt": config_dict.get("freeze_irt", False),

            # Training hyperparameters
            "epochs": config_dict.get("epochs", 0),
            "batch_size": config_dict.get("batch_size", 0),
            "gradient_accumulation_steps": config_dict.get("gradient_accumulation_steps", 1),
            "effective_batch_size": config_dict.get("batch_size", 0) * config_dict.get("gradient_accumulation_steps", 1),
            "learning_rate_encoder": config_dict.get("learning_rate_encoder", 0),
            "learning_rate_embeddings": config_dict.get("learning_rate_embeddings", 0),
            "max_length": config_dict.get("max_length", 0),
            "warmup_ratio": config_dict.get("warmup_ratio", 0.1),
            "max_grad_norm": config_dict.get("max_grad_norm", 1.0),

            # Data settings
            "frontier_cutoff_date": config_dict.get("frontier_cutoff_date", ""),
            "pre_frontier_threshold": config_dict.get("pre_frontier_threshold", 0.1),
            "post_frontier_threshold": config_dict.get("post_frontier_threshold", 0.1),

            # Results - primary metric
            "final_spearman_rho": final_metrics.get("frontier_spearman_rho", None),
            "final_spearman_p": final_metrics.get("frontier_spearman_p", None),
            "best_spearman_rho": max(
                [r for r in history.get("frontier_spearman_rho", []) if r is not None],
                default=None
            ),
            "best_epoch": self._find_best_epoch(history),

            # Training dynamics analysis
            "training_healthy": training_analysis["healthy"],
            "loss_decreased": training_analysis["loss_decreased"],
            "final_loss": training_analysis["final_loss"],
            "loss_reduction_pct": training_analysis["loss_reduction_pct"],
            "gradients_stable": training_analysis["gradients_stable"],
            "encoder_grads_nonzero": training_analysis["encoder_grads_nonzero"],
            "grad_norm_mean": training_analysis["grad_norm_mean"],
            "grad_norm_std": training_analysis["grad_norm_std"],

            # Number of frontier tasks (for context)
            "num_frontier_tasks": final_metrics.get("num_frontier_tasks", None),
        }

        # Append to CSV
        df_new = pd.DataFrame([record])

        if self.csv_path.exists():
            df_existing = pd.read_csv(self.csv_path)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new

        df.to_csv(self.csv_path, index=False)

    def _analyze_training(self, history: Dict[str, List]) -> Dict:
        """Analyze training dynamics to check for issues."""
        analysis = {
            "healthy": True,
            "loss_decreased": False,
            "final_loss": None,
            "loss_reduction_pct": None,
            "gradients_stable": True,
            "encoder_grads_nonzero": True,
            "grad_norm_mean": None,
            "grad_norm_std": None,
        }

        losses = history.get("loss", [])
        if len(losses) >= 2:
            initial_loss = losses[0]
            final_loss = losses[-1]
            analysis["final_loss"] = final_loss
            analysis["loss_decreased"] = final_loss < initial_loss
            if initial_loss > 0:
                analysis["loss_reduction_pct"] = (initial_loss - final_loss) / initial_loss * 100

        # Check encoder gradients
        encoder_grads = history.get("grad_norm_encoder", [])
        if encoder_grads:
            import numpy as np
            grads_array = np.array(encoder_grads)
            analysis["grad_norm_mean"] = float(np.mean(grads_array))
            analysis["grad_norm_std"] = float(np.std(grads_array))

            # Check for vanishing gradients
            if np.mean(grads_array) < 1e-6:
                analysis["encoder_grads_nonzero"] = False
                analysis["healthy"] = False

            # Check for exploding gradients (std > 10x mean suggests instability)
            if np.std(grads_array) > 10 * np.mean(grads_array) and np.mean(grads_array) > 0:
                analysis["gradients_stable"] = False
                analysis["healthy"] = False

        # Overall health check
        if not analysis["loss_decreased"]:
            analysis["healthy"] = False

        return analysis

    def _find_best_epoch(self, history: Dict[str, List]) -> Optional[int]:
        """Find epoch with best Spearman rho."""
        rhos = history.get("frontier_spearman_rho", [])
        epochs = history.get("epoch", [])

        if not rhos or not epochs:
            return None

        valid_pairs = [(e, r) for e, r in zip(epochs, rhos) if r is not None]
        if not valid_pairs:
            return None

        best_epoch, best_rho = max(valid_pairs, key=lambda x: x[1])
        return best_epoch

    def get_all_experiments(self) -> pd.DataFrame:
        """Load all logged experiments."""
        if not self.csv_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.csv_path)

    def get_summary(self) -> pd.DataFrame:
        """Get summary of all experiments sorted by performance."""
        df = self.get_all_experiments()
        if df.empty:
            return df

        # Select key columns for summary
        summary_cols = [
            "timestamp",
            "output_dir",
            "psi_normalization",
            "freeze_irt",
            "lora_r",
            "epochs",
            "effective_batch_size",
            "learning_rate_encoder",
            "best_spearman_rho",
            "final_spearman_rho",
            "training_healthy",
            "loss_reduction_pct",
            "notes",
        ]

        # Only include columns that exist
        summary_cols = [c for c in summary_cols if c in df.columns]

        summary = df[summary_cols].copy()
        summary = summary.sort_values("best_spearman_rho", ascending=False)

        return summary


def print_summary(csv_path: str = "chris_output/sad_irt_experiments.csv"):
    """Print summary of all experiments."""
    tracker = ExperimentTracker(csv_path)
    df = tracker.get_summary()

    if df.empty:
        print(f"No experiments found in {csv_path}")
        return

    print(f"\n{'='*80}")
    print(f"SAD-IRT Experiment Summary ({len(df)} runs)")
    print(f"{'='*80}\n")

    # Best experiment
    best = df.iloc[0]
    print(f"Best experiment:")
    print(f"  Output: {best.get('output_dir', 'N/A')}")
    print(f"  Best Spearman ρ: {best.get('best_spearman_rho', 'N/A'):.4f}")
    print(f"  Config: freeze_irt={best.get('freeze_irt', 'N/A')}, psi_norm={best.get('psi_normalization', 'N/A')}, lora_r={best.get('lora_r', 'N/A')}")
    print()

    # Table of all experiments
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.to_string(index=False))


def compare_experiments(csv_path: str = "chris_output/sad_irt_experiments.csv"):
    """Compare experiments with detailed analysis."""
    tracker = ExperimentTracker(csv_path)
    df = tracker.get_all_experiments()

    if df.empty:
        print(f"No experiments found in {csv_path}")
        return

    print(f"\n{'='*80}")
    print("Ablation Analysis")
    print(f"{'='*80}\n")

    # Group by key ablation dimensions
    if "freeze_irt" in df.columns and df["freeze_irt"].nunique() > 1:
        print("Effect of freezing IRT parameters:")
        for freeze, group in df.groupby("freeze_irt"):
            rhos = group["best_spearman_rho"].dropna()
            if len(rhos) > 0:
                print(f"  freeze_irt={freeze}: mean ρ = {rhos.mean():.4f} (n={len(rhos)})")
        print()

    if "psi_normalization" in df.columns and df["psi_normalization"].nunique() > 1:
        print("Effect of ψ normalization:")
        for norm, group in df.groupby("psi_normalization"):
            rhos = group["best_spearman_rho"].dropna()
            if len(rhos) > 0:
                print(f"  {norm}: mean ρ = {rhos.mean():.4f} (n={len(rhos)})")
        print()

    if "lora_r" in df.columns and df["lora_r"].nunique() > 1:
        print("Effect of LoRA rank:")
        for r, group in df.groupby("lora_r"):
            rhos = group["best_spearman_rho"].dropna()
            if len(rhos) > 0:
                print(f"  r={r}: mean ρ = {rhos.mean():.4f} (n={len(rhos)})")
        print()

    # Training health summary
    print("Training health:")
    healthy = df["training_healthy"].sum() if "training_healthy" in df.columns else 0
    total = len(df)
    print(f"  Healthy runs: {healthy}/{total} ({100*healthy/total:.0f}%)")

    unhealthy = df[df["training_healthy"] == False] if "training_healthy" in df.columns else pd.DataFrame()
    if not unhealthy.empty:
        print(f"  Unhealthy runs:")
        for _, row in unhealthy.iterrows():
            issues = []
            if not row.get("loss_decreased", True):
                issues.append("loss didn't decrease")
            if not row.get("encoder_grads_nonzero", True):
                issues.append("vanishing gradients")
            if not row.get("gradients_stable", True):
                issues.append("unstable gradients")
            print(f"    {row.get('output_dir', 'unknown')}: {', '.join(issues)}")


def main():
    parser = argparse.ArgumentParser(description="SAD-IRT Experiment Tracker")
    parser.add_argument("--csv", default="chris_output/sad_irt_experiments.csv",
                        help="Path to experiments CSV")
    parser.add_argument("--summary", action="store_true", help="Print summary of all experiments")
    parser.add_argument("--compare", action="store_true", help="Compare experiments by ablation dimension")
    args = parser.parse_args()

    if args.summary:
        print_summary(args.csv)
    elif args.compare:
        compare_experiments(args.csv)
    else:
        print_summary(args.csv)


if __name__ == "__main__":
    main()
