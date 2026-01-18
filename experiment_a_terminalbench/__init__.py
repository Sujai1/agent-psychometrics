"""Experiment A for TerminalBench: Prior Validation (IRT AUC) with Binomial data."""

from .config import TerminalBenchConfig
from .data_loader import (
    TerminalBenchData,
    load_terminalbench_data,
    load_task_data_from_repo,
)
from experiment_a_common import (
    compute_auc,
    agent_only_baseline,
    random_baseline,
)

__all__ = [
    "TerminalBenchConfig",
    "load_terminalbench_data",
    "TerminalBenchData",
    "load_task_data_from_repo",
    "compute_auc",
    "agent_only_baseline",
    "random_baseline",
]
