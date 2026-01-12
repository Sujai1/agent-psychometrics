"""Simple linear model for prior difficulty prediction."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def extract_task_features(task_ids: List[str]) -> pd.DataFrame:
    """Extract simple features from task data.

    Features (kept simple per requirements):
    - problem_len: Length of problem statement
    - problem_lines: Number of lines
    - patch_len: Length of gold patch
    - patch_files: Number of files in patch
    - repo: Repository name (categorical)
    """
    # Load SWE-bench data
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        task_data = {ex["instance_id"]: ex for ex in ds}
    except Exception as e:
        print(f"Warning: Could not load SWE-bench dataset: {e}")
        # Return empty dataframe if dataset unavailable
        return pd.DataFrame(columns=["task_id", "problem_len", "problem_lines", "patch_len", "patch_files", "repo"]).set_index("task_id")

    features = []
    for task_id in task_ids:
        if task_id not in task_data:
            continue
        task = task_data[task_id]

        problem = task["problem_statement"]
        patch = task["patch"]

        features.append(
            {
                "task_id": task_id,
                "problem_len": len(problem),
                "problem_lines": problem.count("\n"),
                "patch_len": len(patch),
                "patch_files": patch.count("diff --git"),
                "repo": task["repo"],
            }
        )

    return pd.DataFrame(features).set_index("task_id")


class PriorModel:
    """Simple linear model predicting difficulty from task features."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model: Optional[Pipeline] = None
        self.feature_names = ["problem_len", "problem_lines", "patch_len", "patch_files"]
        self._features_cache: Optional[pd.DataFrame] = None

    def fit(self, task_ids: List[str], difficulties: np.ndarray) -> "PriorModel":
        """Fit prior model on task features.

        Args:
            task_ids: List of task IDs
            difficulties: Array of IRT b values (aligned with task_ids)
        """
        # Extract features
        features_df = extract_task_features(task_ids)
        self._features_cache = features_df

        if features_df.empty:
            print("Warning: No features extracted, prior model will return zeros")
            return self

        # Build pipeline
        preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), self.feature_names),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["repo"]),
            ]
        )

        self.model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", Ridge(alpha=self.alpha)),
            ]
        )

        # Align features with difficulties
        aligned_tasks = [t for t in task_ids if t in features_df.index]
        if not aligned_tasks:
            print("Warning: No aligned tasks found")
            return self

        X = features_df.loc[aligned_tasks]
        y = pd.Series(difficulties, index=task_ids).loc[aligned_tasks]

        self.model.fit(X, y)
        print(f"Prior model trained on {len(aligned_tasks)} tasks")

        return self

    def predict(self, task_ids: List[str]) -> np.ndarray:
        """Predict difficulty for tasks."""
        if self.model is None:
            return np.zeros(len(task_ids))

        # Use cached features or extract new ones
        if self._features_cache is not None:
            features_df = self._features_cache
            # Extract features for any new tasks not in cache
            new_tasks = [t for t in task_ids if t not in features_df.index]
            if new_tasks:
                new_features = extract_task_features(new_tasks)
                features_df = pd.concat([features_df, new_features])
                self._features_cache = features_df
        else:
            features_df = extract_task_features(task_ids)

        valid_tasks = [t for t in task_ids if t in features_df.index]
        if not valid_tasks:
            return np.zeros(len(task_ids))

        X = features_df.loc[valid_tasks]
        predictions = self.model.predict(X)

        # Return predictions aligned with input task_ids
        result = np.zeros(len(task_ids))
        for i, t in enumerate(task_ids):
            if t in valid_tasks:
                idx = valid_tasks.index(t)
                result[i] = predictions[idx]

        return result

    def get_prior_predictions(self, task_ids: List[str]) -> Dict[str, float]:
        """Get prior predictions as a dict."""
        predictions = self.predict(task_ids)
        return dict(zip(task_ids, predictions))

    def get_feature_coefficients(self) -> Dict[str, float]:
        """Get coefficients of the linear model for interpretability."""
        if self.model is None:
            return {}

        regressor = self.model.named_steps["regressor"]
        preprocessor = self.model.named_steps["preprocessor"]

        # Get feature names after preprocessing
        num_names = self.feature_names
        cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(["repo"]))

        all_names = num_names + cat_names
        coeffs = regressor.coef_

        return dict(zip(all_names, coeffs))
