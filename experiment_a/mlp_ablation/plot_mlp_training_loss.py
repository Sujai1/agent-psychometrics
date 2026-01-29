#!/usr/bin/env python3
"""Plot MLP training loss over iterations to verify convergence.

This script can be used to visualize the training loss curves for MLP predictors
across cross-validation folds.

Usage:
    python -m experiment_a.mlp_ablation.plot_mlp_training_loss --losses_json path/to/losses.json
    python -m experiment_a.mlp_ablation.plot_mlp_training_loss --run_cv  # Run CV and plot
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_training_losses(
    fold_losses: List[List[float]],
    output_path: Path,
    title: str = "MLP Training Loss per Iteration",
) -> None:
    """Plot training loss curves for each fold.

    Args:
        fold_losses: List of loss lists, one per fold.
        output_path: Path to save the plot.
        title: Plot title.
    """
    plt.figure(figsize=(12, 6))

    for fold_idx, losses in enumerate(fold_losses):
        plt.plot(losses, label=f"Fold {fold_idx + 1}", alpha=0.7)

    plt.xlabel("Iteration")
    plt.ylabel("BCE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add final loss annotation
    final_losses = [losses[-1] for losses in fold_losses if losses]
    if final_losses:
        mean_final = np.mean(final_losses)
        plt.axhline(y=mean_final, color="r", linestyle="--", alpha=0.5,
                   label=f"Mean final loss: {mean_final:.4f}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved training loss plot to: {output_path}")


def mlp_diagnostics_extractor(predictor: Any, fold_idx: int) -> Optional[Dict[str, Any]]:
    """Extract training diagnostics from an MLP predictor.

    This function can be passed to run_cv() as diagnostics_extractor.

    Args:
        predictor: The predictor (should be MLPPredictor).
        fold_idx: Current fold index.

    Returns:
        Dict with 'losses' (list) and 'train_auc' (float), or None if not an MLP.
    """
    if hasattr(predictor, "get_training_losses"):
        return {
            "losses": predictor.get_training_losses(),
            "train_auc": predictor.get_train_auc() if hasattr(predictor, "get_train_auc") else None,
        }
    return None


def run_cv_with_mlp_diagnostics(
    dataset: str = "swebench",
    output_dir: Path = Path("chris_output/experiment_a/mlp_diagnostics"),
) -> Dict[str, List[List[float]]]:
    """Run cross-validation and extract MLP training losses.

    Args:
        dataset: Dataset to run on ("swebench", "gso", "terminalbench", "swebench_pro").
        output_dir: Directory to save diagnostics.

    Returns:
        Dict mapping predictor name -> list of fold losses.
    """
    import importlib

    from experiment_a.shared.pipeline import run_cross_validation, build_cv_predictors
    from experiment_a.shared.cross_validation import k_fold_split_tasks
    from experiment_ab_shared.dataset import _load_binary_responses

    # Import dataset-specific config and spec
    if dataset == "swebench":
        from experiment_a.swebench.config import ExperimentAConfig as Config
        from experiment_a.swebench.train_evaluate import SPEC, ROOT
    elif dataset == "gso":
        from experiment_a.gso.config import GSOConfig as Config
        from experiment_a.gso.train_evaluate import SPEC, ROOT
    elif dataset == "terminalbench":
        from experiment_a.terminalbench.config import TerminalBenchConfig as Config
        from experiment_a.terminalbench.train_evaluate import SPEC, ROOT
    elif dataset == "swebench_pro":
        from experiment_a.swebench_pro.config import SWEBenchProConfig as Config
        from experiment_a.swebench_pro.train_evaluate import SPEC, ROOT
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    config = Config()

    # Run with diagnostics extraction
    results = run_cross_validation(
        config, SPEC, ROOT, k=5,
        diagnostics_extractors={
            "mlp_embedding": mlp_diagnostics_extractor,
            "mlp_llm_judge": mlp_diagnostics_extractor,
            "mlp_grouped": mlp_diagnostics_extractor,
        },
    )

    # Extract MLP diagnostics from results
    mlp_diagnostics: Dict[str, Dict[str, Any]] = {}
    cv_results = results.get("cv_results", {})

    for name in ["mlp_embedding", "mlp_llm_judge", "mlp_grouped"]:
        if name in cv_results:
            fold_diagnostics = cv_results[name].get("fold_diagnostics", [])
            test_auc = cv_results[name].get("mean_auc")
            fold_aucs = cv_results[name].get("fold_aucs", [])

            if fold_diagnostics and fold_diagnostics[0] is not None:
                fold_losses = [d["losses"] for d in fold_diagnostics]
                train_aucs = [d.get("train_auc") for d in fold_diagnostics]

                mlp_diagnostics[name] = {
                    "fold_losses": fold_losses,
                    "train_aucs": train_aucs,
                    "test_auc": test_auc,
                    "fold_test_aucs": fold_aucs,
                }

    # Save and plot
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print train vs test AUC comparison
    print("\n" + "=" * 70)
    print("MLP TRAIN vs TEST AUC COMPARISON (Overfitting Diagnostic)")
    print("=" * 70)
    print(f"{'Method':<30} {'Train AUC':>12} {'Test AUC':>12} {'Gap':>10}")
    print("-" * 70)

    for name, diag in mlp_diagnostics.items():
        train_aucs = [a for a in diag["train_aucs"] if a is not None]
        mean_train_auc = np.mean(train_aucs) if train_aucs else None
        test_auc = diag["test_auc"]

        if mean_train_auc is not None and test_auc is not None:
            gap = mean_train_auc - test_auc
            print(f"{name:<30} {mean_train_auc:>12.4f} {test_auc:>12.4f} {gap:>+10.4f}")
        else:
            train_str = f"{mean_train_auc:.4f}" if mean_train_auc else "N/A"
            test_str = f"{test_auc:.4f}" if test_auc else "N/A"
            print(f"{name:<30} {train_str:>12} {test_str:>12} {'N/A':>10}")

    print("=" * 70)
    print("(Large positive gap = overfitting)")
    print()

    for name, diag in mlp_diagnostics.items():
        fold_losses = diag["fold_losses"]

        # Save JSON with all diagnostics
        json_path = output_dir / f"{name}_diagnostics.json"
        with open(json_path, "w") as f:
            json.dump(diag, f, indent=2)
        print(f"Saved {name} diagnostics to: {json_path}")

        # Plot losses
        plot_path = output_dir / f"{name}_training_loss.png"
        plot_training_losses(fold_losses, plot_path, title=f"{name} Training Loss")

    return mlp_diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Plot MLP training loss curves"
    )
    parser.add_argument(
        "--losses_json",
        type=Path,
        help="Path to JSON file with fold losses (list of lists)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mlp_training_loss.png"),
        help="Output plot path",
    )
    parser.add_argument(
        "--run_cv",
        action="store_true",
        help="Run cross-validation to collect MLP training losses",
    )
    parser.add_argument(
        "--dataset",
        choices=["swebench", "gso", "terminalbench", "swebench_pro"],
        default="swebench",
        help="Dataset for --run_cv mode",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/experiment_a/mlp_diagnostics"),
        help="Output directory for --run_cv mode",
    )

    args = parser.parse_args()

    if args.run_cv:
        run_cv_with_mlp_diagnostics(args.dataset, args.output_dir)
    elif args.losses_json:
        with open(args.losses_json) as f:
            fold_losses = json.load(f)
        plot_training_losses(fold_losses, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
