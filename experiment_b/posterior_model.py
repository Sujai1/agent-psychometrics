"""Posterior model: Prior + linear correction from trajectory features."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import Ridge

from .prior_model import PriorModel
from .trajectory_features import (
    TRAJECTORY_FEATURE_NAMES,
    load_trajectories_for_task,
    aggregate_trajectory_features,
)


class PosteriorModel:
    """
    Posterior difficulty = Prior(x_i) + psi^T * avg_features(trajectories)

    From the proposal:
    posterior_difficulty_i = prior(x_i) + psi^T * (1/|M|) * sum_j f(tau_ij)
    """

    def __init__(
        self,
        prior_model: PriorModel,
        alpha: float = 1.0,
    ):
        """Initialize posterior model.

        Args:
            prior_model: Trained prior model
            alpha: Ridge regularization parameter for psi
        """
        self.prior_model = prior_model
        self.alpha = alpha
        self.psi_model: Optional[Ridge] = None
        self.training_stats: Dict = {}

    def fit(
        self,
        task_ids: List[str],
        ground_truth_difficulties: np.ndarray,
        weak_agents: List[str],
        trajectories_dir: Path,
    ) -> "PosteriorModel":
        """Fit the correction term psi.

        Args:
            task_ids: Training task IDs (D_train)
            ground_truth_difficulties: IRT b values for tasks (aligned with task_ids)
            weak_agents: M1 agents whose trajectories to use
            trajectories_dir: Base directory for trajectories
        """
        # Get prior predictions
        prior_preds = self.prior_model.get_prior_predictions(task_ids)

        # Compute residuals (what prior doesn't explain)
        X_features = []
        y_residuals = []
        valid_task_ids = []
        tasks_with_trajs = 0

        for i, task_id in enumerate(task_ids):
            if task_id not in prior_preds:
                continue

            # Load trajectory features for this task
            traj_features = load_trajectories_for_task(task_id, weak_agents, trajectories_dir)

            if not traj_features:
                continue  # No trajectories available

            tasks_with_trajs += 1

            # Aggregate features
            feat_vec = aggregate_trajectory_features(traj_features)
            X_features.append(feat_vec)

            # Residual = ground_truth - prior
            residual = ground_truth_difficulties[i] - prior_preds[task_id]
            y_residuals.append(residual)
            valid_task_ids.append(task_id)

        self.training_stats = {
            "total_tasks": len(task_ids),
            "tasks_with_trajectories": tasks_with_trajs,
            "tasks_used_for_training": len(valid_task_ids),
            "agents_used": len(weak_agents),
        }

        if not X_features:
            print("Warning: No valid training data for posterior model")
            self.psi_model = None
            return self

        X = np.array(X_features)
        y = np.array(y_residuals)

        # Fit Ridge regression for psi
        self.psi_model = Ridge(alpha=self.alpha)
        self.psi_model.fit(X, y)

        print(f"Posterior model trained on {len(valid_task_ids)} tasks")
        print(f"  Tasks with trajectories: {tasks_with_trajs}")
        print(f"  Psi coefficients: {dict(zip(TRAJECTORY_FEATURE_NAMES, self.psi_model.coef_))}")

        return self

    def predict(
        self,
        task_ids: List[str],
        weak_agents: List[str],
        trajectories_dir: Path,
    ) -> Dict[str, float]:
        """Predict posterior difficulty.

        posterior = prior + psi^T * trajectory_features
        """
        # Get prior predictions
        prior_preds = self.prior_model.get_prior_predictions(task_ids)

        predictions = {}
        for task_id in task_ids:
            if task_id not in prior_preds:
                continue

            prior = prior_preds[task_id]

            # If no psi model or no trajectories, just use prior
            if self.psi_model is None:
                predictions[task_id] = prior
                continue

            # Load and aggregate trajectory features
            traj_features = load_trajectories_for_task(task_id, weak_agents, trajectories_dir)

            if not traj_features:
                predictions[task_id] = prior
                continue

            feat_vec = aggregate_trajectory_features(traj_features)
            correction = self.psi_model.predict([feat_vec])[0]

            predictions[task_id] = prior + correction

        return predictions

    def get_feature_importance(self) -> Dict[str, float]:
        """Get psi coefficients as feature importance."""
        if self.psi_model is None:
            return {}
        return dict(zip(TRAJECTORY_FEATURE_NAMES, self.psi_model.coef_))

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return self.training_stats
