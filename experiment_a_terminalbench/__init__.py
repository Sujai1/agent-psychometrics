"""Experiment A for TerminalBench: Prior Validation (IRT AUC) with Binomial data."""

from .baselines import (
    agent_only_baseline_binomial,
    constant_baseline_binomial,
    random_baseline_binomial,
    task_only_baseline_binomial,
    verify_random_baseline_sanity,
)
from .config import TerminalBenchConfig
from .data_loader import TerminalBenchData, load_terminalbench_data
from .irt_evaluation import compute_binomial_auc, compute_irt_probability

__all__ = [
    "TerminalBenchConfig",
    "load_terminalbench_data",
    "TerminalBenchData",
    "compute_binomial_auc",
    "compute_irt_probability",
    "agent_only_baseline_binomial",
    "constant_baseline_binomial",
    "random_baseline_binomial",
    "task_only_baseline_binomial",
    "verify_random_baseline_sanity",
]
