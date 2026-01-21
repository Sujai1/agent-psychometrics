"""Date forecasting utilities for experiment B.

Predicts when tasks will become solvable (50% probability) based on
the linear relationship between difficulty and time.

Key insight: From IRT, P(success) = sigmoid(theta - beta) = 0.5 when theta = beta.
So a task is solvable with 50% probability when an agent's ability >= task difficulty.
Combined with Experiment D's finding that frontier ability is linear over time,
we can predict when a task will become solvable.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def parse_date(date_str: str) -> datetime:
    """Parse YYYYMMDD date string to datetime."""
    return datetime.strptime(date_str, "%Y%m%d")


@dataclass
class FirstCapableDatesResult:
    """Result from compute_first_capable_dates().

    Attributes:
        first_capable_dates: Dict mapping task_id -> datetime of earliest capable agent.
            Tasks with NO capable agent are NOT included in this dict.
        tasks_without_capable_agent: List of task_ids where no agent has theta >= beta.
            These tasks are "too hard" for any current agent to solve with 50% prob.
        earliest_agent_date: Earliest agent submission date in the dataset.
        latest_agent_date: Latest agent submission date in the dataset.
    """

    first_capable_dates: Dict[str, datetime]
    tasks_without_capable_agent: List[str]
    earliest_agent_date: datetime
    latest_agent_date: datetime


def compute_first_capable_dates(
    oracle_items: pd.DataFrame,
    oracle_abilities: pd.DataFrame,
    agent_dates: Dict[str, str],
) -> FirstCapableDatesResult:
    """Compute the first date when each task became solvable with 50% prob.

    A task is solvable with 50% probability when theta_agent >= beta_task
    (since P(success) = sigmoid(theta - beta) = 0.5 when theta = beta).

    Args:
        oracle_items: DataFrame with 'b' column (oracle task difficulties)
        oracle_abilities: DataFrame with 'theta' column (oracle agent abilities)
        agent_dates: Dict mapping agent_id -> date string (YYYYMMDD)

    Returns:
        FirstCapableDatesResult with:
            - first_capable_dates: task_id -> datetime (only tasks WITH capable agents)
            - tasks_without_capable_agent: list of task_ids with NO capable agent
            - earliest_agent_date, latest_agent_date: date range of agents

    Raises:
        ValueError: If required data columns are missing or no agents have dates
    """
    if "b" not in oracle_items.columns:
        raise ValueError("oracle_items must have 'b' column")
    if "theta" not in oracle_abilities.columns:
        raise ValueError("oracle_abilities must have 'theta' column")

    # Build agent -> (theta, date) mapping
    agent_info = {}
    for agent_id in oracle_abilities.index:
        if agent_id not in agent_dates:
            continue
        theta = oracle_abilities.loc[agent_id, "theta"]
        date = parse_date(agent_dates[agent_id])
        agent_info[agent_id] = (theta, date)

    if not agent_info:
        raise ValueError("No agents found with both abilities and dates")

    # Get date range
    all_dates = [info[1] for info in agent_info.values()]
    earliest_agent_date = min(all_dates)
    latest_agent_date = max(all_dates)

    # For each task, find earliest agent where theta >= beta
    first_capable_dates = {}
    tasks_without_capable_agent = []

    for task_id in oracle_items.index:
        beta = oracle_items.loc[task_id, "b"]

        # Find all agents capable of solving with 50% prob (theta >= beta)
        capable_agents = [
            (agent_id, info[1])  # (agent_id, date)
            for agent_id, info in agent_info.items()
            if info[0] >= beta
        ]

        if capable_agents:
            # Find earliest by date
            earliest = min(capable_agents, key=lambda x: x[1])
            first_capable_dates[task_id] = earliest[1]
        else:
            # No agent can currently solve this task with 50% probability
            tasks_without_capable_agent.append(task_id)

    return FirstCapableDatesResult(
        first_capable_dates=first_capable_dates,
        tasks_without_capable_agent=tasks_without_capable_agent,
        earliest_agent_date=earliest_agent_date,
        latest_agent_date=latest_agent_date,
    )


def split_tasks_by_first_capable_date(
    first_capable_dates: Dict[str, datetime],
    cutoff_date: datetime,
) -> Tuple[List[str], List[str]]:
    """Split tasks by whether first capable agent is before/after cutoff.

    Args:
        first_capable_dates: Dict from FirstCapableDatesResult.first_capable_dates
        cutoff_date: Date to split on (e.g., frontier cutoff)

    Returns:
        Tuple of (pre_cutoff_tasks, post_cutoff_tasks):
            - pre_cutoff_tasks: Tasks where first capable agent is before cutoff (for training)
            - post_cutoff_tasks: Tasks where first capable agent is on/after cutoff (for eval)
    """
    pre_cutoff_tasks = []
    post_cutoff_tasks = []

    for task_id, first_date in first_capable_dates.items():
        if first_date < cutoff_date:
            pre_cutoff_tasks.append(task_id)
        else:
            post_cutoff_tasks.append(task_id)

    return pre_cutoff_tasks, post_cutoff_tasks


def compute_ground_truth_days(
    task_ids: List[str],
    first_capable_dates: Dict[str, datetime],
    reference_date: datetime,
) -> Dict[str, float]:
    """Convert dates to days since reference date.

    Args:
        task_ids: List of task IDs to process
        first_capable_dates: Dict from FirstCapableDatesResult.first_capable_dates
        reference_date: Reference date (typically earliest date in dataset)

    Returns:
        Dict mapping task_id -> days since reference.
        Tasks NOT in first_capable_dates are excluded from result.
    """
    result = {}
    for task_id in task_ids:
        if task_id in first_capable_dates:
            delta = first_capable_dates[task_id] - reference_date
            result[task_id] = delta.days
    return result


class DateForecastModel:
    """Linear model for predicting solvability dates from difficulties."""

    def __init__(self):
        self._model: Optional[LinearRegression] = None
        self._is_fitted: bool = False
        self._reference_date: Optional[datetime] = None
        self._slope: Optional[float] = None
        self._intercept: Optional[float] = None
        self._r_squared: Optional[float] = None
        self._n_train: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def slope(self) -> Optional[float]:
        return self._slope

    @property
    def intercept(self) -> Optional[float]:
        return self._intercept

    @property
    def r_squared(self) -> Optional[float]:
        return self._r_squared

    def fit(
        self,
        predicted_beta: Dict[str, float],
        ground_truth_days: Dict[str, float],
        task_ids: List[str],
        reference_date: datetime,
    ) -> Dict[str, float]:
        """Fit linear model: days = slope * predicted_beta + intercept.

        Args:
            predicted_beta: Raw predicted difficulties (NOT oracle-aligned)
            ground_truth_days: Days since reference for each task
            task_ids: Training task IDs
            reference_date: Reference date for converting days back to dates

        Returns:
            Dict with fit statistics (slope, intercept, r_squared, n_train)
        """
        self._reference_date = reference_date

        # Collect training data
        X = []
        y = []
        for task_id in task_ids:
            if task_id in predicted_beta and task_id in ground_truth_days:
                X.append(predicted_beta[task_id])
                y.append(ground_truth_days[task_id])

        if len(X) < 3:
            raise ValueError(f"Insufficient training data: only {len(X)} tasks")

        X_arr = np.array(X).reshape(-1, 1)
        y_arr = np.array(y)

        # Fit linear regression
        self._model = LinearRegression()
        self._model.fit(X_arr, y_arr)

        self._slope = float(self._model.coef_[0])
        self._intercept = float(self._model.intercept_)
        self._n_train = len(X)
        self._is_fitted = True

        # Compute R²
        y_pred = self._model.predict(X_arr)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        self._r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "slope": self._slope,
            "intercept": self._intercept,
            "r_squared": self._r_squared,
            "n_train": self._n_train,
        }

    def predict(
        self,
        predicted_beta: Dict[str, float],
        task_ids: List[str],
    ) -> Dict[str, Tuple[float, datetime]]:
        """Predict solvability dates for tasks.

        Args:
            predicted_beta: Raw predicted difficulties
            task_ids: Task IDs to predict for

        Returns:
            Dict mapping task_id -> (predicted_days, predicted_date)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        result = {}
        for task_id in task_ids:
            if task_id not in predicted_beta:
                continue

            beta = predicted_beta[task_id]
            days = self._slope * beta + self._intercept
            date = self._reference_date + pd.Timedelta(days=int(round(days)))
            result[task_id] = (float(days), date)

        return result


def compute_date_forecast_metrics(
    predicted: Dict[str, Tuple[float, datetime]],
    ground_truth_days: Dict[str, float],
    task_ids: List[str],
) -> Dict[str, float]:
    """Compute evaluation metrics for date forecasting.

    Args:
        predicted: Dict from DateForecastModel.predict() (task_id -> (days, date))
        ground_truth_days: Dict from compute_ground_truth_days()
        task_ids: Task IDs to evaluate on

    Returns:
        Dict with metrics:
            - mae_days: Mean absolute error in days
            - rmse_days: Root mean square error in days
            - pearson_r: Pearson correlation
            - pearson_p: Pearson p-value
            - spearman_rho: Spearman correlation
            - spearman_p: Spearman p-value
            - n_tasks: Number of evaluated tasks
            - early_pct: % of predictions earlier than actual
    """
    pred_days = []
    actual_days = []

    for task_id in task_ids:
        if task_id in predicted and task_id in ground_truth_days:
            pred_days.append(predicted[task_id][0])
            actual_days.append(ground_truth_days[task_id])

    if len(pred_days) < 3:
        return {
            "mae_days": float("nan"),
            "rmse_days": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n_tasks": len(pred_days),
            "early_pct": float("nan"),
        }

    pred_arr = np.array(pred_days)
    actual_arr = np.array(actual_days)
    errors = pred_arr - actual_arr

    # MAE and RMSE
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # Correlations
    pearson_r, pearson_p = stats.pearsonr(pred_arr, actual_arr)
    spearman_rho, spearman_p = stats.spearmanr(pred_arr, actual_arr)

    # Early prediction percentage (pred < actual)
    early_pct = float(np.mean(pred_arr < actual_arr) * 100)

    return {
        "mae_days": mae,
        "rmse_days": rmse,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "n_tasks": len(pred_days),
        "early_pct": early_pct,
    }
