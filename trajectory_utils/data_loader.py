"""Load and format trajectories for summarization.

This module provides shared utilities for loading trajectory data
used by both local (vLLM) and API-based summarization pipelines.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryData:
    """Loaded trajectory data."""

    task_id: str
    agent: str
    resolved: bool
    messages: List[Dict[str, str]]
    filepath: Path


def discover_trajectories(
    trajectory_dir: Path,
    agents: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> List[Tuple[str, str, Path]]:
    """Discover all trajectories with optional filtering and sharding.

    Args:
        trajectory_dir: Root directory containing agent subdirectories
        agents: Optional list of agent IDs to include (None = all)
        task_ids: Optional list of task IDs to include (None = all)
        shard_id: Which shard to process (0-indexed), for distributed processing
        num_shards: Total number of shards

    Returns:
        List of (agent_id, task_id, filepath) tuples
    """
    results = []

    for agent_dir in sorted(trajectory_dir.iterdir()):
        if not agent_dir.is_dir():
            continue

        # Skip hidden dirs and special dirs starting with _
        if agent_dir.name.startswith(".") or agent_dir.name.startswith("_"):
            continue

        agent_id = agent_dir.name

        # Filter by agent if specified
        if agents is not None and agent_id not in agents:
            continue

        for json_file in sorted(agent_dir.glob("*.json")):
            # Skip special files starting with _
            if json_file.name.startswith("_"):
                continue
            task_id = json_file.stem

            # Filter by task if specified
            if task_ids is not None and task_id not in task_ids:
                continue

            results.append((agent_id, task_id, json_file))

    # Sort for deterministic ordering
    results = sorted(results, key=lambda x: (x[0], x[1]))

    # Return shard subset if sharding is enabled
    if num_shards > 1:
        results = [r for i, r in enumerate(results) if i % num_shards == shard_id]

    return results


def load_trajectory(filepath: Path) -> Optional[TrajectoryData]:
    """Load a single trajectory file.

    Args:
        filepath: Path to trajectory JSON file

    Returns:
        TrajectoryData object or None if loading failed
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
        return TrajectoryData(
            task_id=data.get("task_id", ""),
            agent=data.get("agent", ""),
            resolved=data.get("resolved", False),
            messages=data.get("messages", []),
            filepath=filepath,
        )
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load trajectory {filepath}: {e}")
        return None


def format_trajectory(
    messages: List[Dict[str, str]],
    max_chars: Optional[int] = None,
) -> str:
    """Convert trajectory messages to text format.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        max_chars: Optional maximum character limit (truncates from middle if exceeded)

    Returns:
        Formatted trajectory text with role markers
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Skip empty content
        if not content.strip():
            continue

        # Handle content that might be a list (normalize to string)
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)

        parts.append(f"[{role.upper()}]\n{content}")

    full_text = "\n\n".join(parts)

    # Truncate from middle if too long
    if max_chars and len(full_text) > max_chars:
        half = max_chars // 2
        truncation_marker = "\n\n[... TRAJECTORY TRUNCATED ...]\n\n"
        full_text = full_text[:half] + truncation_marker + full_text[-half:]

    return full_text
