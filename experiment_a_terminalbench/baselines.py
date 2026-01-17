"""Baseline methods for Experiment A on TerminalBench with binomial data.

All baselines only use training data when computing statistics (success rates,
mean difficulties, etc.) and only use test data for evaluation.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def agent_only_baseline_binomial(
    abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, Dict[str, int]]],
    train_tasks: List[str],
    test_tasks: List[str],
) -> Dict[str, Any]:
    """Baseline: P(success) = agent's success rate on TRAINING tasks only.

    This baseline ignores task difficulty entirely and uses each agent's
    average performance on training tasks as the prediction for test tasks.

    For binomial data, computes success rate as sum(successes)/sum(trials).

    Args:
        abilities: DataFrame with index=agent_id (used for agent list)
        responses: Dict mapping agent_id -> {task_id -> {successes, trials}}
        train_tasks: List of training task identifiers (used for computing rates)
        test_tasks: List of test task identifiers (used for evaluation)

    Returns:
        Dict with 'auc', 'n_pairs', 'n_observations', 'method'
    """
    y_true: List[int] = []
    y_scores: List[float] = []

    # Pre-compute agent success rates using ONLY training tasks
    agent_success_rates: Dict[str, float] = {}
    train_tasks_set = set(train_tasks)
    for agent_id in abilities.index:
        if agent_id not in responses:
            continue
        total_successes = 0
        total_trials = 0
        for task_id, resp in responses[agent_id].items():
            if task_id in train_tasks_set:  # Only use training tasks
                total_successes += resp["successes"]
                total_trials += resp["trials"]
        if total_trials > 0:
            agent_success_rates[agent_id] = total_successes / total_trials
        else:
            agent_success_rates[agent_id] = 0.5  # Default

    # Evaluate predictions on test tasks with per-trial expansion
    n_pairs = 0
    for task_id in test_tasks:
        for agent_id in abilities.index:
            if agent_id not in responses:
                continue
            if task_id not in responses[agent_id]:
                continue

            resp = responses[agent_id][task_id]
            k = resp["successes"]
            n = resp["trials"]
            pred_prob = agent_success_rates.get(agent_id, 0.5)

            # Expand binomial to binary observations
            y_true.extend([1] * k + [0] * (n - k))
            y_scores.extend([pred_prob] * n)
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


def task_only_baseline_binomial(
    responses: Dict[str, Dict[str, Dict[str, int]]],
    train_tasks: List[str],
    test_tasks: List[str],
) -> Dict[str, Any]:
    """Baseline: P(success) = mean success rate from TRAINING tasks only.

    This baseline ignores agent ability entirely and uses the mean pass rate
    from training tasks as the prediction for all test tasks.

    For binomial data, computes mean rate as sum(successes)/sum(trials).

    Args:
        responses: Dict mapping agent_id -> {task_id -> {successes, trials}}
        train_tasks: List of training task identifiers (used for computing rate)
        test_tasks: List of test task identifiers (used for evaluation)

    Returns:
        Dict with 'auc', 'n_pairs', 'n_observations', 'method', 'mean_train_rate'
    """
    # Compute mean pass rate from TRAINING tasks only (weighted by trials)
    total_train_successes = 0
    total_train_trials = 0
    for task_id in train_tasks:
        for agent_id in responses:
            if task_id in responses[agent_id]:
                resp = responses[agent_id][task_id]
                total_train_successes += resp["successes"]
                total_train_trials += resp["trials"]

    if total_train_trials > 0:
        mean_train_rate = total_train_successes / total_train_trials
    else:
        mean_train_rate = 0.5  # Default

    # Evaluate predictions on test tasks with per-trial expansion
    y_true: List[int] = []
    y_scores: List[float] = []
    n_pairs = 0

    for task_id in test_tasks:
        for agent_id in responses:
            if task_id not in responses[agent_id]:
                continue

            resp = responses[agent_id][task_id]
            k = resp["successes"]
            n = resp["trials"]

            # Expand binomial to binary observations
            y_true.extend([1] * k + [0] * (n - k))
            y_scores.extend([mean_train_rate] * n)
            n_pairs += 1

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {
            "error": "Insufficient data",
            "n_pairs": n_pairs,
            "n_observations": len(y_true),
            "method": "task_only",
            "mean_train_rate": float(mean_train_rate),
        }

    auc = roc_auc_score(y_true, y_scores)
    return {
        "auc": float(auc),
        "n_pairs": n_pairs,
        "n_observations": len(y_true),
        "method": "task_only",
        "mean_train_rate": float(mean_train_rate),
    }


def constant_baseline_binomial(
    items: pd.DataFrame,
    abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, Dict[str, int]]],
    train_tasks: List[str],
    test_tasks: List[str],
) -> Dict[str, Any]:
    """Baseline: Predict mean difficulty from TRAINING set for all test tasks.

    Uses the mean ground-truth difficulty (b) from training tasks as the
    predicted difficulty for all test tasks, then computes IRT probability.

    Args:
        items: DataFrame with index=task_id, column 'b' for difficulty
        abilities: DataFrame with index=agent_id, column 'theta' for ability
        responses: Dict mapping agent_id -> {task_id -> {successes, trials}}
        train_tasks: List of training task identifiers (used for computing mean)
        test_tasks: List of test task identifiers (used for evaluation)

    Returns:
        Dict with 'auc', 'n_pairs', 'n_observations', 'method', 'mean_train_difficulty'
    """
    from scipy.special import expit

    # Compute mean difficulty from TRAINING tasks only
    train_difficulties = [
        float(items.loc[task_id, "b"])
        for task_id in train_tasks
        if task_id in items.index
    ]
    if train_difficulties:
        mean_difficulty = float(np.mean(train_difficulties))
    else:
        mean_difficulty = 0.0  # Default

    # Evaluate predictions on test tasks using IRT formula with mean difficulty
    y_true: List[int] = []
    y_scores: List[float] = []
    n_pairs = 0

    for task_id in test_tasks:
        for agent_id in abilities.index:
            if agent_id not in responses:
                continue
            if task_id not in responses[agent_id]:
                continue

            theta = float(abilities.loc[agent_id, "theta"])
            resp = responses[agent_id][task_id]
            k = resp["successes"]
            n = resp["trials"]

            # IRT probability with mean difficulty
            prob = float(expit(theta - mean_difficulty))

            # Expand binomial to binary observations
            y_true.extend([1] * k + [0] * (n - k))
            y_scores.extend([prob] * n)
            n_pairs += 1

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {
            "error": "Insufficient data",
            "n_pairs": n_pairs,
            "n_observations": len(y_true),
            "method": "constant",
            "mean_train_difficulty": mean_difficulty,
        }

    auc = roc_auc_score(y_true, y_scores)
    return {
        "auc": float(auc),
        "n_pairs": n_pairs,
        "n_observations": len(y_true),
        "method": "constant",
        "mean_train_difficulty": mean_difficulty,
    }


def random_baseline_binomial(
    responses: Dict[str, Dict[str, Dict[str, int]]],
    test_tasks: List[str],
    seed: int = 42,
) -> Dict[str, Any]:
    """Baseline: Random predictions (expected AUC ~ 0.5).

    For binomial data, expands to binary observations first.

    Args:
        responses: Dict mapping agent_id -> {task_id -> {successes, trials}}
        test_tasks: List of test task identifiers to evaluate
        seed: Random seed for reproducibility

    Returns:
        Dict with 'auc', 'n_pairs', 'n_observations', 'method'
    """
    rng = np.random.RandomState(seed)

    y_true: List[int] = []
    n_pairs = 0

    # Collect all binary observations from test tasks only
    for task_id in test_tasks:
        for agent_id in responses:
            if task_id not in responses[agent_id]:
                continue
            resp = responses[agent_id][task_id]
            k = resp["successes"]
            n = resp["trials"]
            y_true.extend([1] * k + [0] * (n - k))
            n_pairs += 1

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {
            "error": "Insufficient data",
            "n_pairs": n_pairs,
            "n_observations": len(y_true),
            "method": "random",
        }

    # Generate random predictions
    y_scores = rng.random(len(y_true))

    auc = roc_auc_score(y_true, y_scores)
    return {
        "auc": float(auc),
        "n_pairs": n_pairs,
        "n_observations": len(y_true),
        "method": "random",
    }


def verify_random_baseline_sanity(
    responses: Dict[str, Dict[str, Dict[str, int]]],
    test_tasks: List[str],
    n_trials: int = 100,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Verify that random baseline gives AUC ≈ 0.5 as a sanity check.

    Runs multiple trials with different seeds and checks the mean AUC.

    Args:
        responses: Dict mapping agent_id -> {task_id -> {successes, trials}}
        test_tasks: List of test task identifiers to evaluate
        n_trials: Number of random trials to average
        tolerance: Acceptable deviation from 0.5

    Returns:
        Dict with 'mean_auc', 'std_auc', 'min_auc', 'max_auc', 'passed', 'n_trials'
    """
    aucs = []
    for seed in range(n_trials):
        result = random_baseline_binomial(responses, test_tasks, seed=seed)
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
