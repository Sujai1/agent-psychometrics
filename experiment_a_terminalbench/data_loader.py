"""Data loading and splitting for Experiment A on TerminalBench."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

# Reuse stable_split_tasks from experiment_a
from experiment_a.data_loader import stable_split_tasks


def load_abilities(abilities_path: Path) -> pd.DataFrame:
    """Load agent abilities from 1PL IRT model.

    Args:
        abilities_path: Path to abilities.csv

    Returns:
        DataFrame with index=agent_id, columns=['theta', 'theta_std']
    """
    df = pd.read_csv(abilities_path, index_col=0)
    return df


def load_items(items_path: Path) -> pd.DataFrame:
    """Load IRT item parameters (ground truth difficulties).

    Args:
        items_path: Path to items.csv

    Returns:
        DataFrame with index=task_id, columns=['b', 'b_std']
    """
    df = pd.read_csv(items_path, index_col=0)
    return df


def load_binomial_responses(responses_path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Load binomial response matrix from JSONL.

    Args:
        responses_path: Path to response matrix JSONL file with binomial data

    Returns:
        Dict mapping agent_id -> {task_id -> {successes: int, trials: int}}
    """
    responses = {}
    with open(responses_path, "r") as f:
        for line in f:
            record = json.loads(line)
            agent_id = record["subject_id"]
            responses[agent_id] = record["responses"]
    return responses


def load_task_data_from_repo(
    task_ids: List[str],
    repo_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load task instruction and solution from local terminal-bench repo.

    Args:
        task_ids: List of task IDs to load
        repo_path: Path to cloned terminal-bench repo

    Returns:
        Dict mapping task_id -> {instruction, solution, difficulty, category, tags}
    """
    tasks = {}
    for task_id in task_ids:
        task_dir = repo_path / "tasks" / task_id
        if not task_dir.exists():
            print(f"Warning: Task directory not found: {task_dir}")
            continue

        # Load task.yaml
        task_yaml_path = task_dir / "task.yaml"
        if task_yaml_path.exists():
            with open(task_yaml_path) as f:
                task_yaml = yaml.safe_load(f)
        else:
            task_yaml = {}

        # Load solution.sh
        solution_path = task_dir / "solution.sh"
        if solution_path.exists():
            solution = solution_path.read_text()
        else:
            solution = ""

        tasks[task_id] = {
            "instruction": task_yaml.get("instruction", ""),
            "solution": solution,
            "difficulty": task_yaml.get("difficulty"),
            "category": task_yaml.get("category"),
            "tags": task_yaml.get("tags", []),
        }

    return tasks


@dataclass
class TerminalBenchData:
    """Container for all loaded TerminalBench data."""

    abilities: pd.DataFrame  # Agent abilities (theta), index=agent_id
    items: pd.DataFrame  # Ground truth difficulties (b), index=task_id
    responses: Dict[str, Dict[str, Dict[str, int]]]  # agent_id -> {task_id -> {successes, trials}}
    task_data: Dict[str, Dict[str, Any]]  # task_id -> {instruction, solution, ...}
    train_tasks: List[str]
    test_tasks: List[str]
    all_agents: List[str]

    @property
    def n_agents(self) -> int:
        return len(self.all_agents)

    @property
    def n_tasks(self) -> int:
        return len(self.items)

    @property
    def n_train_tasks(self) -> int:
        return len(self.train_tasks)

    @property
    def n_test_tasks(self) -> int:
        return len(self.test_tasks)


def load_terminalbench_data(
    abilities_path: Path,
    items_path: Path,
    responses_path: Path,
    repo_path: Path,
    test_fraction: float,
    split_seed: int,
) -> TerminalBenchData:
    """Load all TerminalBench data and create train/test splits.

    Args:
        abilities_path: Path to 1PL abilities.csv
        items_path: Path to 1PL items.csv
        responses_path: Path to binomial response matrix JSONL
        repo_path: Path to cloned terminal-bench repo
        test_fraction: Fraction of tasks for test set
        split_seed: Random seed for splits

    Returns:
        TerminalBenchData with all loaded data and splits
    """
    abilities = load_abilities(abilities_path)
    items = load_items(items_path)
    responses = load_binomial_responses(responses_path)

    # Get all task IDs from items (ground truth)
    all_task_ids = list(items.index)

    # Load task data from repo
    task_data = load_task_data_from_repo(all_task_ids, repo_path)

    # Create train/test split on tasks (reusing from experiment_a)
    train_tasks, test_tasks = stable_split_tasks(
        all_task_ids, test_fraction, split_seed
    )

    # Get agents that are in both abilities and responses
    all_agents = [a for a in abilities.index if a in responses]

    return TerminalBenchData(
        abilities=abilities,
        items=items,
        responses=responses,
        task_data=task_data,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        all_agents=all_agents,
    )
