"""Data loader for uploaded Lunette trajectories.

This loader reads from the _lunette_uploads.json files created by the batch
upload script and combines them with IRT difficulty parameters.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrajectoryInfo:
    """Information about an uploaded trajectory."""

    task_id: str
    agent: str
    run_id: str
    trajectory_id: str
    resolved: bool
    difficulty: float  # IRT b parameter
    discrimination: float  # IRT a parameter


class LunetteDataLoader:
    """Load uploaded Lunette trajectories with IRT parameters."""

    def __init__(
        self,
        items_path: Path,
        trajectories_dir: Path = Path("trajectory_data/unified_trajs"),
    ):
        """Initialize Lunette data loader.

        Args:
            items_path: Path to IRT items.csv file.
            trajectories_dir: Base directory containing agent folders with
                _lunette_uploads.json files.
        """
        self.items_path = items_path
        self.trajectories_dir = trajectories_dir
        self._items_df: Optional[pd.DataFrame] = None
        self._trajectories: Optional[List[TrajectoryInfo]] = None

    def _load_items(self) -> pd.DataFrame:
        """Load IRT items."""
        if self._items_df is None:
            self._items_df = pd.read_csv(self.items_path, index_col=0)
        return self._items_df

    def _load_agent_trajectories(self, agent_dir: Path) -> List[TrajectoryInfo]:
        """Load trajectories for a single agent from upload tracking file.

        Args:
            agent_dir: Path to agent directory.

        Returns:
            List of TrajectoryInfo objects.
        """
        upload_file = agent_dir / "_lunette_uploads.json"
        if not upload_file.exists():
            return []

        with open(upload_file) as f:
            upload_data = json.load(f)

        agent = upload_data["agent"]
        trajectories_list = upload_data.get("trajectories", [])

        # Map run_ids to trajectories (handle multi-run uploads)
        run_ids = upload_data.get("run_ids", [upload_data.get("run_id")])
        if run_ids and run_ids[0] is None:
            return []

        # Get IRT parameters
        items_df = self._load_items()

        results = []
        for traj_data in trajectories_list:
            task_id = traj_data["task_id"]
            trajectory_id = traj_data.get("trajectory_id")

            if task_id not in items_df.index:
                continue  # Task not in IRT data

            # Find which run this trajectory belongs to
            # (For now, use first run_id; could be more sophisticated)
            run_id = run_ids[0]

            results.append(TrajectoryInfo(
                task_id=task_id,
                agent=agent,
                run_id=run_id,
                trajectory_id=trajectory_id,
                resolved=traj_data.get("resolved", False),
                difficulty=items_df.loc[task_id, "b"],
                discrimination=items_df.loc[task_id, "a"],
            ))

        return results

    def load_all(self) -> List[TrajectoryInfo]:
        """Load all uploaded trajectories from all agents.

        Returns:
            List of TrajectoryInfo objects.
        """
        if self._trajectories is not None:
            return self._trajectories

        all_trajectories = []

        for agent_dir in sorted(self.trajectories_dir.iterdir()):
            if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
                continue

            agent_trajs = self._load_agent_trajectories(agent_dir)
            all_trajectories.extend(agent_trajs)

        self._trajectories = all_trajectories
        return all_trajectories

    def get_agents(self) -> List[str]:
        """Get list of agents with uploaded trajectories.

        Returns:
            List of agent names.
        """
        trajs = self.load_all()
        return sorted(set(t.agent for t in trajs))

    def filter_by_agent(self, agent: str) -> List[TrajectoryInfo]:
        """Get trajectories for a specific agent.

        Args:
            agent: Agent name.

        Returns:
            List of TrajectoryInfo for that agent.
        """
        trajs = self.load_all()
        return [t for t in trajs if t.agent == agent]

    def filter_by_tasks(self, task_ids: List[str]) -> List[TrajectoryInfo]:
        """Get trajectories for specific tasks.

        Args:
            task_ids: List of task instance IDs.

        Returns:
            List of TrajectoryInfo for those tasks.
        """
        trajs = self.load_all()
        task_set = set(task_ids)
        return [t for t in trajs if t.task_id in task_set]

    def stratified_sample(
        self,
        n_tasks: int,
        n_agents: Optional[int] = None,
        seed: Optional[int] = None,
        n_strata: int = 5,
    ) -> List[TrajectoryInfo]:
        """Sample trajectories stratified by difficulty.

        Samples tasks stratified by difficulty, then for each task selects
        one or more agent trajectories.

        Args:
            n_tasks: Number of tasks to sample.
            n_agents: Number of agents per task (None = all available).
            seed: Random seed for reproducibility.
            n_strata: Number of difficulty strata.

        Returns:
            List of TrajectoryInfo objects.
        """
        trajs = self.load_all()
        rng = np.random.default_rng(seed)

        # Group by task
        task_groups: Dict[str, List[TrajectoryInfo]] = {}
        for traj in trajs:
            if traj.task_id not in task_groups:
                task_groups[traj.task_id] = []
            task_groups[traj.task_id].append(traj)

        # Get difficulty for each task (from first trajectory)
        task_difficulties = {
            task_id: trajs_list[0].difficulty
            for task_id, trajs_list in task_groups.items()
        }

        # Create difficulty strata
        difficulties_df = pd.DataFrame({
            "task_id": list(task_difficulties.keys()),
            "difficulty": list(task_difficulties.values()),
        })
        difficulties_df["stratum"] = pd.qcut(
            difficulties_df["difficulty"],
            n_strata,
            labels=False,
            duplicates="drop",
        )

        # Sample tasks from each stratum
        samples_per_stratum = n_tasks // n_strata
        remainder = n_tasks % n_strata

        sampled_task_ids = []
        for stratum in range(n_strata):
            stratum_tasks = difficulties_df[
                difficulties_df["stratum"] == stratum
            ]["task_id"].tolist()
            n_sample = samples_per_stratum + (1 if stratum < remainder else 0)
            n_sample = min(n_sample, len(stratum_tasks))

            if n_sample > 0:
                sampled = rng.choice(stratum_tasks, size=n_sample, replace=False)
                sampled_task_ids.extend(sampled)

        # For each sampled task, select agent trajectories
        sampled_trajs = []
        for task_id in sampled_task_ids:
            available_trajs = task_groups[task_id]

            if n_agents is None or n_agents >= len(available_trajs):
                # Use all available
                sampled_trajs.extend(available_trajs)
            else:
                # Sample n_agents
                selected = rng.choice(available_trajs, size=n_agents, replace=False)
                sampled_trajs.extend(selected)

        return sampled_trajs

    def get_task_coverage(self) -> Dict[str, int]:
        """Get number of agents per task.

        Returns:
            Dict mapping task_id to number of agents with trajectories.
        """
        trajs = self.load_all()
        task_agents: Dict[str, set] = {}

        for traj in trajs:
            if traj.task_id not in task_agents:
                task_agents[traj.task_id] = set()
            task_agents[traj.task_id].add(traj.agent)

        return {task_id: len(agents) for task_id, agents in task_agents.items()}

    def to_eval_format(
        self,
        trajectories: List[TrajectoryInfo],
    ) -> List[Dict]:
        """Convert to format expected by LunetteFeatureEvaluator.

        Args:
            trajectories: List of TrajectoryInfo objects.

        Returns:
            List of dicts with keys: task_id, run_id, trajectory_id, difficulty.
        """
        return [
            {
                "task_id": t.task_id,
                "run_id": t.run_id,
                "trajectory_id": t.trajectory_id,
                "difficulty": t.difficulty,
            }
            for t in trajectories
        ]

    @property
    def num_trajectories(self) -> int:
        """Total number of uploaded trajectories."""
        return len(self.load_all())

    @property
    def num_tasks(self) -> int:
        """Number of unique tasks with trajectories."""
        trajs = self.load_all()
        return len(set(t.task_id for t in trajs))

    @property
    def num_agents(self) -> int:
        """Number of unique agents with trajectories."""
        return len(self.get_agents())
