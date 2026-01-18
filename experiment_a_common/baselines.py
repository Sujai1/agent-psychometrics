"""Generic baseline methods for Experiment A evaluation.

These baselines work with any ExperimentData subclass (binary or binomial)
by using the dataset's expand_for_auc() method.
"""

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from experiment_a_common.dataset import ExperimentData


def agent_only_baseline(data: ExperimentData) -> Dict[str, Any]:
    """Baseline: P(success) = agent's success rate on TRAINING tasks only.

    This baseline ignores task difficulty entirely and uses each agent's
    average performance on training tasks as the prediction for test tasks.

    Args:
        data: ExperimentData with responses and train/test splits

    Returns:
        Dict with 'auc', 'n_pairs', 'n_observations', 'method'
    """
    y_true: List[int] = []
    y_scores: List[float] = []

    # Pre-compute agent success rates using ONLY training tasks
    agent_success_rates: Dict[str, float] = {}

    for agent_id in data.train_abilities.index:
        if agent_id not in data.responses:
            continue

        # Use expand_for_auc to get binary observations
        all_true: List[int] = []
        for task_id in data.train_tasks:
            if task_id in data.responses[agent_id]:
                yt, _ = data.expand_for_auc(agent_id, task_id, 0.0)  # prob unused
                all_true.extend(yt)

        if all_true:
            agent_success_rates[agent_id] = float(np.mean(all_true))
        else:
            agent_success_rates[agent_id] = 0.5  # Default

    # Evaluate predictions on test tasks
    n_pairs = 0
    for task_id in data.test_tasks:
        for agent_id in data.train_abilities.index:
            if agent_id not in data.responses:
                continue
            if task_id not in data.responses[agent_id]:
                continue

            pred_prob = agent_success_rates.get(agent_id, 0.5)

            # Use expand_for_auc to handle binary vs binomial
            yt, _ = data.expand_for_auc(agent_id, task_id, pred_prob)
            y_true.extend(yt)
            y_scores.extend([pred_prob] * len(yt))
            n_pairs += 1

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {
            "error": "Insufficient data",
            "n_pairs": n_pairs,
            "n_observations": len(y_true),
            "method": "agent_only",
        }

    auc = roc_auc_score(y_true, y_scores)
    return {
        "auc": float(auc),
        "n_pairs": n_pairs,
        "n_observations": len(y_true),
        "method": "agent_only",
    }


def random_baseline(data: ExperimentData, seed: int = 42) -> Dict[str, Any]:
    """Baseline: Random predictions (expected AUC ~ 0.5).

    Args:
        data: ExperimentData with responses and test tasks
        seed: Random seed for reproducibility

    Returns:
        Dict with 'auc', 'n_pairs', 'n_observations', 'method'
    """
    rng = np.random.RandomState(seed)

    y_true: List[int] = []
    n_pairs = 0

    # Collect all binary observations from test tasks
    for task_id in data.test_tasks:
        for agent_id in data.responses:
            if task_id not in data.responses[agent_id]:
                continue
            yt, _ = data.expand_for_auc(agent_id, task_id, 0.0)
            y_true.extend(yt)
            n_pairs += 1

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {
            "error": "Insufficient data",
            "n_pairs": n_pairs,
            "n_observations": len(y_true),
            "method": "random",
        }

    # Generate random predictions
    y_scores = rng.random(len(y_true)).tolist()

    auc = roc_auc_score(y_true, y_scores)
    return {
        "auc": float(auc),
        "n_pairs": n_pairs,
        "n_observations": len(y_true),
        "method": "random",
    }


def verify_random_baseline_sanity(
    data: ExperimentData,
    n_trials: int = 100,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Verify that random baseline gives AUC ~ 0.5 as a sanity check.

    Args:
        data: ExperimentData with responses and test tasks
        n_trials: Number of random trials to average
        tolerance: Acceptable deviation from 0.5

    Returns:
        Dict with 'mean_auc', 'std_auc', 'passed', 'n_trials'
    """
    aucs = []
    for seed in range(n_trials):
        result = random_baseline(data, seed=seed)
        if "error" not in result:
            aucs.append(result["auc"])

    if not aucs:
        return {"error": "No successful random baseline runs", "passed": False}

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    passed = abs(mean_auc - 0.5) < tolerance

    return {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "min_auc": float(np.min(aucs)),
        "max_auc": float(np.max(aucs)),
        "passed": passed,
        "n_trials": len(aucs),
        "expected": 0.5,
        "tolerance": tolerance,
    }
