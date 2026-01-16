"""Load and format trajectories for summarization.

This module re-exports from the shared trajectory_utils for backward compatibility.
"""

from trajectory_utils import (
    TrajectoryData,
    discover_trajectories,
    load_trajectory,
    format_trajectory,
)

__all__ = [
    "TrajectoryData",
    "discover_trajectories",
    "load_trajectory",
    "format_trajectory",
]


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (chars / 4).

    This is a fast approximation. For accurate counts, use a tokenizer.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // 4
