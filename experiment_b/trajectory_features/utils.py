"""Shared utilities for frontier trajectory feature extraction.

This module provides common functions used by both the EDA script
and the feature extraction script.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from experiment_b.shared.data_preparation import (
    identify_frontier_tasks_human_hard,
    identify_frontier_tasks_zero_pre,
    split_agents_by_dates,
)
from experiment_b.swebench.config import SWEBenchConfig
from experiment_b.trajectory_features.prompts import format_trajectory_for_prompt


def load_frontier_tasks_with_difficulties(
    config: SWEBenchConfig,
    frontier_def: str = "zero_pre",
) -> Tuple[List[str], pd.DataFrame, List[str], List[str]]:
    """Load frontier tasks, oracle difficulties, and agent splits.

    Args:
        config: SWE-bench dataset configuration
        frontier_def: Frontier definition to use. Options:
            - 'zero_pre': Tasks with 0% pre-frontier, >0% post-frontier solve rate
            - 'human_hard': Tasks labeled "1-4 hours" or ">4 hours" by human estimate

    Returns:
        Tuple of:
        - frontier_task_ids: List of frontier task IDs
        - oracle_items: DataFrame with oracle IRT difficulties (column 'b')
        - pre_frontier_agents: List of pre-frontier agent names
        - post_frontier_agents: List of post-frontier agent names

    Raises:
        ValueError: If frontier_def is not recognized
    """
    # Load oracle IRT difficulties
    oracle_items = pd.read_csv(config.oracle_irt_path, index_col=0)

    # Get all agents and their dates
    all_agents = config.all_agents
    agent_dates = config.get_agent_dates(all_agents)

    # Split by cutoff date
    pre_frontier, post_frontier = split_agents_by_dates(
        all_agents, agent_dates, config.cutoff_date
    )

    # Identify frontier tasks based on definition
    if frontier_def == "zero_pre":
        frontier_tasks = identify_frontier_tasks_zero_pre(
            config.responses_path,
            pre_frontier,
            post_frontier,
        )
    elif frontier_def == "human_hard":
        frontier_tasks = identify_frontier_tasks_human_hard(
            config.all_task_ids,
        )
    else:
        raise ValueError(
            f"Unknown frontier_def: {frontier_def}. "
            f"Must be one of: 'zero_pre', 'human_hard'"
        )

    return frontier_tasks, oracle_items, pre_frontier, post_frontier


def load_trajectory(agent: str, task_id: str, trajs_dir: Path) -> dict:
    """Load a single trajectory file.

    Args:
        agent: Agent name (directory name)
        task_id: Task ID (filename without .json)
        trajs_dir: Base directory for trajectories

    Returns:
        Trajectory dictionary with 'task_id', 'agent', 'resolved', 'messages'

    Raises:
        FileNotFoundError: If trajectory file doesn't exist
    """
    traj_path = trajs_dir / agent / f"{task_id}.json"
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    with open(traj_path) as f:
        return json.load(f)


def build_task_dicts(
    frontier_tasks: List[str],
    agent: str,
    trajs_dir: Path,
    max_messages: int = 100,
    max_chars_per_message: int = 2000,
) -> Tuple[List[Dict], List[str]]:
    """Build task dictionaries for the extractor.

    Each task dict contains the fields needed by the prompt template:
    - task_id
    - agent
    - trajectory_content

    Args:
        frontier_tasks: List of task IDs to process
        agent: Agent name to load trajectories from
        trajs_dir: Base directory for trajectories
        max_messages: Max messages to include in trajectory
        max_chars_per_message: Max chars per message

    Returns:
        Tuple of:
        - task_dicts: List of task dictionaries
        - missing: List of task IDs with missing trajectories
    """
    task_dicts = []
    missing = []

    for task_id in frontier_tasks:
        try:
            trajectory = load_trajectory(agent, task_id, trajs_dir)
            trajectory_content = format_trajectory_for_prompt(
                trajectory,
                max_messages=max_messages,
                max_chars_per_message=max_chars_per_message,
            )

            task_dicts.append({
                "task_id": task_id,
                "agent": agent,
                "trajectory_content": trajectory_content,
            })
        except FileNotFoundError:
            missing.append(task_id)

    return task_dicts, missing
