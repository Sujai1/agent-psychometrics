"""Binomial metrics for pass rate prediction evaluation.

Computes metrics comparing predicted pass rates to observed outcomes
for binomial response data (k successes out of n trials).

Two subsets of metrics are computed:
- MAE/RMSE: Over ALL (agent, task) pairs using actual trial counts
- Accuracy/Confusion Matrix: Only on 5-trial responses for clean 6-class classification
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from experiment_a_common.dataset import BinomialExperimentData
from experiment_a_common.evaluator import compute_irt_probability


@dataclass
class BinomialMetricsResult:
    """Results from binomial metrics computation."""

    # MAE/RMSE computed over all trials
    mae: float
    rmse: float
    mean_predicted: float
    mean_actual: float
    n_pairs: int

    # MSE computed only on 5-trial responses (predicted prob vs empirical rate)
    pass5_mse: float
    n_pass5_pairs: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mean_predicted": self.mean_predicted,
            "mean_actual": self.mean_actual,
            "n_pairs": self.n_pairs,
            "pass5_mse": self.pass5_mse,
            "n_pass5_pairs": self.n_pass5_pairs,
        }


def compute_binomial_metrics(
    data: BinomialExperimentData,
    predicted_difficulties: Dict[str, float],
    use_full_abilities: bool = False,
) -> BinomialMetricsResult:
    """Compute binomial metrics for predicted difficulties.

    For each (agent, task) pair in test set:
        - prob = sigmoid(theta - beta_predicted)
        - expected = prob * trials
        - actual = successes

    Computes:
    - MAE, RMSE over all (agent, task) pairs
    - Accuracy, confusion matrix only for 5-trial responses

    Args:
        data: BinomialExperimentData with responses and abilities
        predicted_difficulties: Mapping of task_id -> predicted difficulty
        use_full_abilities: If True, use full IRT abilities (oracle only)

    Returns:
        BinomialMetricsResult with all computed metrics
    """
    abilities = data.full_abilities if use_full_abilities else data.train_abilities

    # Lists for all-trials metrics (MAE/RMSE)
    all_predicted: List[float] = []
    all_actual: List[int] = []

    # Lists for 5-trial MSE (predicted prob vs empirical rate)
    pass5_pred_prob: List[float] = []
    pass5_empirical_rate: List[float] = []

    for task_id in data.test_tasks:
        beta_pred = predicted_difficulties.get(task_id)
        if beta_pred is None:
            continue

        for agent_id in abilities.index:
            if agent_id not in data.responses:
                continue
            if task_id not in data.responses[agent_id]:
                continue

            resp = data.responses[agent_id][task_id]
            k = resp["successes"]
            n = resp["trials"]

            theta = abilities.loc[agent_id, "ability"]
            prob = compute_irt_probability(theta, beta_pred)

            expected = prob * n
            all_predicted.append(expected)
            all_actual.append(k)

            # For 5-trial responses, compute MSE of predicted prob vs empirical rate
            if n == 5:
                empirical_rate = k / 5.0  # e.g., 4/5 = 0.8
                pass5_pred_prob.append(prob)
                pass5_empirical_rate.append(empirical_rate)

    # Handle empty case
    if len(all_predicted) == 0:
        return BinomialMetricsResult(
            mae=float("nan"),
            rmse=float("nan"),
            mean_predicted=float("nan"),
            mean_actual=float("nan"),
            n_pairs=0,
            pass5_mse=float("nan"),
            n_pass5_pairs=0,
        )

    # Compute MAE/RMSE over all pairs
    pred_arr = np.array(all_predicted)
    actual_arr = np.array(all_actual)
    errors = pred_arr - actual_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # Compute MSE for 5-trial responses (predicted prob vs empirical rate)
    if len(pass5_pred_prob) > 0:
        pass5_pred_arr = np.array(pass5_pred_prob)
        pass5_emp_arr = np.array(pass5_empirical_rate)
        pass5_mse = float(np.mean((pass5_pred_arr - pass5_emp_arr) ** 2))
    else:
        pass5_mse = float("nan")

    return BinomialMetricsResult(
        mae=mae,
        rmse=rmse,
        mean_predicted=float(np.mean(pred_arr)),
        mean_actual=float(np.mean(actual_arr)),
        n_pairs=len(all_predicted),
        pass5_mse=pass5_mse,
        n_pass5_pairs=len(pass5_pred_prob),
    )
