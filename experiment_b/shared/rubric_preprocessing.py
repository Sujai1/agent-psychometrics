"""Rubric data preprocessing for Ordered Logit IRT model.

This module handles loading and preprocessing trajectory rubric data,
transforming all items to a uniform 0-5 ordinal scale where higher
values indicate easier tasks (better agent performance).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RubricPreprocessor:
    """Preprocess raw rubric data for ordered logit modeling.

    Transforms all rubric items to a uniform 0-5 ordinal scale where
    higher values indicate easier tasks (better agent performance).

    Transform rules:
    - REVERSE: loop_detection, focus_drift (higher raw = worse, so invert)
    - BIN: debugging_cycles, exploration_breadth (unbounded counts -> 0-5)
    - STANDARD: Others are already 0-5, just cast to int
    """

    # Items where higher = worse (need reversal)
    REVERSE_ITEMS: Tuple[str, ...] = ("loop_detection", "focus_drift")

    # Items that are unbounded counts (need binning)
    # Format: item_name -> bin edges (results in 0 to len(edges)-2 categories)
    BIN_ITEMS: Dict[str, List[float]] = field(default_factory=dict)

    # Items already on 0-5 scale (just need int cast)
    STANDARD_ITEMS: Tuple[str, ...] = (
        "localization_quality",
        "error_recovery",
        "solution_completeness",
        "edge_case_handling",
        "test_verification",
    )

    def __post_init__(self):
        if not self.BIN_ITEMS:
            self.BIN_ITEMS = {
                # debugging_cycles: 0->0, 1->1, 2->2, 3-4->3, 5-9->4, 10+->5
                "debugging_cycles": [-np.inf, 1, 2, 3, 5, 10, np.inf],
                # exploration_breadth: 0-2->0, 3-4->1, 5-6->2, 7-8->3, 9-11->4, 12+->5
                "exploration_breadth": [-np.inf, 3, 5, 7, 9, 12, np.inf],
            }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw rubric data to uniform 0-5 ordinal scale.

        After preprocessing:
        - All items are integers 0-5
        - Higher values = better agent performance = easier task

        Args:
            df: Raw rubric DataFrame

        Returns:
            Preprocessed DataFrame with transformed columns
        """
        result = df.copy()

        # 1. Reverse items (so higher = better for all)
        for item in self.REVERSE_ITEMS:
            if item in result.columns:
                max_val = result[item].max()
                result[item] = max_val - result[item]
                # Round and clip to 0-5 range
                result[item] = result[item].round().clip(0, 5).astype(int)

        # 2. Bin unbounded counts to 0-5 scale
        for item, bin_edges in self.BIN_ITEMS.items():
            if item in result.columns:
                # pd.cut with labels gives categories 0, 1, ..., n_bins-1
                n_bins = len(bin_edges) - 1
                result[item] = pd.cut(
                    result[item],
                    bins=bin_edges,
                    labels=list(range(n_bins)),
                    include_lowest=True,
                )
                result[item] = result[item].astype(int)

        # 3. Ensure standard items are integers in 0-5 range
        for item in self.STANDARD_ITEMS:
            if item in result.columns:
                result[item] = result[item].round().clip(0, 5).astype(int)

        return result


class RubricDataSource:
    """Load and serve rubric data for ordered logit model.

    IMPORTANT: This class enforces strict data completeness requirements.
    Every task must have exactly 6 agent observations, and every observation
    must have values for all rubric items. Missing data will raise an error.
    """

    ALL_RUBRIC_ITEMS = [
        "loop_detection",
        "localization_quality",
        "debugging_cycles",
        "error_recovery",
        "exploration_breadth",
        "focus_drift",
        "solution_completeness",
        "edge_case_handling",
        "test_verification",
    ]

    # Items with positive correlation to eta (validated via rubric_correlation_eda.py)
    # Excludes: debugging_cycles (r=-0.048), exploration_breadth (r=-0.087)
    SELECTED_RUBRIC_ITEMS = [
        "loop_detection",       # r=0.221
        "localization_quality", # r=0.338
        "error_recovery",       # r=0.177
        "focus_drift",          # r=0.218
        "solution_completeness",# r=0.465 (strongest)
        "edge_case_handling",   # r=0.307
        "test_verification",    # r=0.249
    ]

    EXPECTED_AGENTS_PER_TASK = 6

    def __init__(
        self,
        rubric_path: Path,
        preprocessor: Optional[RubricPreprocessor] = None,
        rubric_items: Optional[List[str]] = None,
    ):
        """Load rubric data with preprocessing.

        Args:
            rubric_path: Path to raw_features CSV
            preprocessor: Optional preprocessor (default creates standard one)
            rubric_items: Optional list of rubric items to use (default: all)

        Raises:
            FileNotFoundError: If rubric file doesn't exist
            ValueError: If required columns are missing, data is incomplete,
                or any task doesn't have exactly 6 agents
        """
        rubric_path = Path(rubric_path)
        if not rubric_path.exists():
            raise FileNotFoundError(f"Rubric file not found: {rubric_path}")

        self._raw_df = pd.read_csv(rubric_path)

        # Set rubric items to use (default: selected items with positive correlation)
        self._rubric_items = rubric_items or self.SELECTED_RUBRIC_ITEMS

        # Validate required columns exist
        required_cols = ["task_id", "agent"] + self._rubric_items
        missing_cols = set(required_cols) - set(self._raw_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for any NaN values in rubric items
        for item in self._rubric_items:
            nan_count = self._raw_df[item].isna().sum()
            if nan_count > 0:
                nan_rows = self._raw_df[self._raw_df[item].isna()][
                    ["task_id", "agent"]
                ].head(5)
                raise ValueError(
                    f"Missing values found in rubric item '{item}' ({nan_count} total). "
                    f"First few: {nan_rows.to_dict('records')}"
                )

        # Validate every task has exactly 6 agents
        task_counts = self._raw_df.groupby("task_id").size()
        incomplete_tasks = task_counts[task_counts != self.EXPECTED_AGENTS_PER_TASK]
        if len(incomplete_tasks) > 0:
            raise ValueError(
                f"Expected {self.EXPECTED_AGENTS_PER_TASK} agents per task, but found "
                f"{len(incomplete_tasks)} tasks with different counts: "
                f"{incomplete_tasks.head(10).to_dict()}"
            )

        # Now preprocess
        self._preprocessor = preprocessor or RubricPreprocessor()
        self._df = self._preprocessor.preprocess(self._raw_df)

        # Build indices
        self._task_ids = self._df["task_id"].unique().tolist()
        self._agent_ids = self._df["agent"].unique().tolist()
        self._build_index()

        print(
            f"  Loaded rubric data: {len(self._task_ids)} tasks x {len(self._agent_ids)} agents "
            f"= {len(self._df)} observations"
        )

    def _build_index(self):
        """Build (task_id, agent_id) -> row index mapping."""
        self._pair_to_idx = {}
        for idx, row in self._df.iterrows():
            key = (row["task_id"], row["agent"])
            self._pair_to_idx[key] = idx

    @property
    def task_ids(self) -> List[str]:
        return self._task_ids.copy()

    @property
    def agent_ids(self) -> List[str]:
        return self._agent_ids.copy()

    @property
    def rubric_items(self) -> List[str]:
        return self._rubric_items.copy()

    @property
    def raw_df(self) -> pd.DataFrame:
        """Get raw (unprocessed) DataFrame."""
        return self._raw_df.copy()

    @property
    def preprocessed_df(self) -> pd.DataFrame:
        """Get preprocessed DataFrame."""
        return self._df.copy()

    def get_scores_for_task(
        self, task_id: str
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Get all rubric scores for a single task.

        Args:
            task_id: Task ID to get scores for

        Returns:
            Tuple of (agent_ids, scores_dict) where:
            - agent_ids: list of agents with data for this task (exactly 6)
            - scores_dict: item -> (6,) array of scores

        Raises:
            ValueError: If task_id not found or doesn't have exactly 6 agents
        """
        task_df = self._df[self._df["task_id"] == task_id]

        if len(task_df) == 0:
            raise ValueError(f"Task '{task_id}' not found in rubric data")

        if len(task_df) != self.EXPECTED_AGENTS_PER_TASK:
            raise ValueError(
                f"Task '{task_id}' has {len(task_df)} agents, "
                f"expected {self.EXPECTED_AGENTS_PER_TASK}"
            )

        agent_ids = task_df["agent"].tolist()

        scores = {}
        for item in self._rubric_items:
            scores[item] = task_df[item].values.astype(np.int64)

        return agent_ids, scores

    def get_all_observations(
        self,
    ) -> Tuple[List[str], List[str], Dict[str, np.ndarray]]:
        """Get all observations for training ordered logit parameters.

        Returns flat arrays suitable for batch training.

        Returns:
            Tuple of (task_ids, agent_ids, scores_dict) where:
            - task_ids: list of task IDs for each observation
            - agent_ids: list of agent IDs for each observation
            - scores_dict: item -> (n_obs,) array of scores
        """
        task_ids = self._df["task_id"].tolist()
        agent_ids = self._df["agent"].tolist()

        scores = {}
        for item in self._rubric_items:
            scores[item] = self._df[item].values.astype(np.int64)

        return task_ids, agent_ids, scores

    def get_agents_for_task(self, task_id: str) -> List[str]:
        """Get agents that have rubric data for a task.

        Args:
            task_id: Task ID

        Returns:
            List of 6 agent IDs

        Raises:
            ValueError: If task_id not found
        """
        agents = self._df[self._df["task_id"] == task_id]["agent"].tolist()
        if len(agents) == 0:
            raise ValueError(f"Task '{task_id}' not found in rubric data")
        return agents
