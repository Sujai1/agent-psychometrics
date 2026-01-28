#!/usr/bin/env python3
"""Plot MLP training loss over iterations to verify convergence.

This script can be used to visualize the training loss curves for MLP predictors
across cross-validation folds.

Usage:
    python -m experiment_a.plot_mlp_training_loss --losses_json path/to/losses.json
    python -m experiment_a.plot_mlp_training_loss --run_cv  # Run CV and plot
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


def mlp_diagnostics_extractor(predictor: Any, fold_idx: int) -> Optional[List[float]]:
    """Extract training losses from an MLP predictor.

    This function can be passed to run_cv() as diagnostics_extractor.

    Args:
        predictor: The predictor (should be MLPPredictor).
        fold_idx: Current fold index.

    Returns:
        List of training losses per iteration, or None if not an MLP.
    """
    if hasattr(predictor, "get_training_losses"):
        return predictor.get_training_losses()
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

    # Extract MLP losses from results
    mlp_losses: Dict[str, List[List[float]]] = {}
    cv_results = results.get("cv_results", {})

    for name in ["mlp_embedding", "mlp_llm_judge", "mlp_grouped"]:
        if name in cv_results:
            fold_diagnostics = cv_results[name].get("fold_diagnostics", [])
            if fold_diagnostics and fold_diagnostics[0] is not None:
                mlp_losses[name] = fold_diagnostics

    # Save and plot
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fold_losses in mlp_losses.items():
        # Save JSON
        json_path = output_dir / f"{name}_losses.json"
        with open(json_path, "w") as f:
            json.dump(fold_losses, f)
        print(f"Saved {name} losses to: {json_path}")

        # Plot
        plot_path = output_dir / f"{name}_training_loss.png"
        plot_training_losses(fold_losses, plot_path, title=f"{name} Training Loss")

    return mlp_losses


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
