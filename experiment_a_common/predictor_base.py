"""Base classes for difficulty predictors.

This module defines the abstract base class and simple concrete predictors
that are shared across experiments.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd


class DifficultyPredictorBase(ABC):
    """Abstract base class for all difficulty predictors.

    All predictors must implement:
    - fit(): Train on tasks with known difficulties
    - predict(): Predict difficulties for new tasks
    - name: Human-readable predictor name
    """

    @abstractmethod
    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Train on tasks with known IRT difficulties.

        Args:
            task_ids: List of task identifiers
            ground_truth_b: Array of ground truth difficulty values (b parameters)
        """
        ...

    @abstractmethod
    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable predictor name."""
        ...


class ConstantPredictor(DifficultyPredictorBase):
    """Baseline predictor that always predicts the mean training difficulty.

    This serves as a baseline - any useful predictor should beat this.
    """

    def __init__(self):
        self._mean_b: float = 0.0

    @property
    def name(self) -> str:
        return "Constant (mean b)"

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Compute mean difficulty from training data."""
        self._mean_b = float(np.mean(ground_truth_b))

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict mean difficulty for all tasks."""
        return {task_id: self._mean_b for task_id in task_ids}


class GroundTruthPredictor(DifficultyPredictorBase):
    """Oracle predictor that uses the true IRT difficulties.

    This serves as an upper bound - no predictor can beat this.
    Used to establish the maximum achievable AUC.
    """

    def __init__(self, items_df: pd.DataFrame):
        """Initialize with ground truth items.

        Args:
            items_df: DataFrame with task difficulties indexed by task_id,
                containing a 'b' column with difficulty values.
        """
        self._items_df = items_df

    @property
    def name(self) -> str:
        return "Oracle (true b)"

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """No-op - oracle doesn't need training."""
        pass

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Return ground truth difficulties."""
        predictions = {}
        for task_id in task_ids:
            if task_id in self._items_df.index:
                predictions[task_id] = float(self._items_df.loc[task_id, "b"])
            else:
                raise ValueError(f"Task {task_id} not found in ground truth items")
        return predictions
