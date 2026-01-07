"""Data loading utilities for evolutionary feature discovery."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Task:
    """A single SWE-bench task with IRT difficulty."""

    task_id: str
    repo: str
    problem_statement: str
    patch: str
    difficulty: float  # IRT b parameter
    discrimination: float  # IRT a parameter


class DataLoader:
    """Load and sample SWE-bench tasks with IRT parameters."""

    def __init__(self, items_path: Path):
        """Initialize data loader.

        Args:
            items_path: Path to IRT items.csv file.
        """
        self.items_path = items_path
        self._items_df: Optional[pd.DataFrame] = None
        self._swebench_df: Optional[pd.DataFrame] = None
        self._merged_df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load and merge IRT items with SWE-bench data.

        Returns:
            DataFrame with task_id as index, columns: repo, problem_statement,
            patch, b (difficulty), a (discrimination).
        """
        if self._merged_df is not None:
            return self._merged_df

        # Load IRT items
        self._items_df = pd.read_csv(self.items_path, index_col=0)

        # Load SWE-bench from HuggingFace
        from datasets import load_dataset

        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        self._swebench_df = pd.DataFrame({
            'instance_id': ds['instance_id'],
            'repo': ds['repo'],
            'problem_statement': ds['problem_statement'],
            'patch': ds['patch'],
        }).set_index('instance_id')

        # Merge
        self._merged_df = self._swebench_df.join(
            self._items_df[['b', 'a']], how='inner'
        )

        return self._merged_df

    def get_task(self, task_id: str) -> Task:
        """Get a single task by ID.

        Args:
            task_id: The task instance ID (e.g., "django__django-12345").

        Returns:
            Task object with all fields populated.
        """
        df = self.load()
        row = df.loc[task_id]
        return Task(
            task_id=task_id,
            repo=row['repo'],
            problem_statement=row['problem_statement'],
            patch=row['patch'],
            difficulty=row['b'],
            discrimination=row['a'],
        )

    def get_tasks(self, task_ids: List[str]) -> List[Task]:
        """Get multiple tasks by ID.

        Args:
            task_ids: List of task instance IDs.

        Returns:
            List of Task objects.
        """
        return [self.get_task(tid) for tid in task_ids]

    def get_difficulty_extremes(
        self,
        percentile: int = 20,
        n_per_group: Optional[int] = None,
    ) -> Tuple[List[Task], List[Task]]:
        """Get tasks from high and low difficulty extremes.

        Args:
            percentile: Percentile threshold for extremes (e.g., 20 means
                        top/bottom 20%).
            n_per_group: Optional limit on number of tasks per group.

        Returns:
            Tuple of (easy_tasks, hard_tasks).
        """
        df = self.load()

        low_threshold = np.percentile(df['b'], percentile)
        high_threshold = np.percentile(df['b'], 100 - percentile)

        easy_ids = df[df['b'] <= low_threshold].index.tolist()
        hard_ids = df[df['b'] >= high_threshold].index.tolist()

        if n_per_group is not None:
            easy_ids = easy_ids[:n_per_group]
            hard_ids = hard_ids[:n_per_group]

        return self.get_tasks(easy_ids), self.get_tasks(hard_ids)

    def stratified_sample(
        self,
        n: int,
        seed: Optional[int] = None,
        n_strata: int = 5,
    ) -> List[Task]:
        """Sample tasks stratified by difficulty.

        Args:
            n: Total number of tasks to sample.
            seed: Random seed for reproducibility.
            n_strata: Number of difficulty strata.

        Returns:
            List of sampled Task objects.
        """
        df = self.load()
        rng = np.random.default_rng(seed)

        # Create difficulty strata
        df = df.copy()
        df['stratum'] = pd.qcut(df['b'], n_strata, labels=False)

        # Sample from each stratum
        samples_per_stratum = n // n_strata
        remainder = n % n_strata

        sampled_ids = []
        for stratum in range(n_strata):
            stratum_ids = df[df['stratum'] == stratum].index.tolist()
            n_sample = samples_per_stratum + (1 if stratum < remainder else 0)
            n_sample = min(n_sample, len(stratum_ids))

            if n_sample > 0:
                sampled = rng.choice(stratum_ids, size=n_sample, replace=False)
                sampled_ids.extend(sampled)

        return self.get_tasks(sampled_ids)

    def random_sample(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> List[Task]:
        """Randomly sample tasks.

        Args:
            n: Number of tasks to sample.
            seed: Random seed for reproducibility.

        Returns:
            List of sampled Task objects.
        """
        df = self.load()
        rng = np.random.default_rng(seed)

        n = min(n, len(df))
        sampled_ids = rng.choice(df.index.tolist(), size=n, replace=False)

        return self.get_tasks(list(sampled_ids))

    @property
    def num_tasks(self) -> int:
        """Total number of tasks available."""
        return len(self.load())

    @property
    def difficulty_range(self) -> Tuple[float, float]:
        """Min and max difficulty values."""
        df = self.load()
        return float(df['b'].min()), float(df['b'].max())
