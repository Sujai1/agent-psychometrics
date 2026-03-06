"""Experiment B: Frontier Task Difficulty Prediction.

Predicts difficulty of frontier tasks (tasks only solvable by newer models)
using various methods WITHOUT access to held-out post-frontier agents.

Setting:
- Date-based split: pre-frontier (< 20250807) vs post-frontier (>= 20250807)
- Frontier tasks: ≤10% pre-frontier pass rate, >10% post-frontier pass rate
- Evaluation: ROC-AUC after projecting predicted difficulties onto oracle IRT scale

Methods compared:
- Oracle (upper bound): Use true IRT difficulties
- Baseline IRT: Train IRT on pre-frontier agents only
- Embedding + Ridge: Task embeddings from any backbone model
- LLM Judge + Ridge: LLM-extracted semantic features
Usage:
    python -m experiment_appendix_h_hard_tasks.swebench.compare_methods
    python -m experiment_appendix_h_hard_tasks.terminalbench.compare_methods

See README.md for full documentation.
"""

from experiment_appendix_h_hard_tasks.swebench.config import SWEBenchConfig
from experiment_appendix_h_hard_tasks.swebench_pro.config import SWEBenchProConfig
from experiment_appendix_h_hard_tasks.terminalbench.config import TerminalBenchConfig
from experiment_appendix_h_hard_tasks.gso.config import GSOConfig
from experiment_appendix_h_hard_tasks.shared.config_base import DatasetConfig


# Registry of available dataset configurations
DATASET_CONFIGS = {
    "swebench": SWEBenchConfig,
    "swebench_pro": SWEBenchProConfig,
    "terminalbench": TerminalBenchConfig,
    "gso": GSOConfig,
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Get a dataset configuration by name.

    Args:
        name: Dataset name (e.g., "swebench", "terminalbench")

    Returns:
        DatasetConfig instance for the specified dataset

    Raises:
        ValueError: If the dataset name is not recognized
    """
    if name not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    return DATASET_CONFIGS[name]()


def list_datasets() -> list:
    """List available dataset names."""
    return list(DATASET_CONFIGS.keys())


__all__ = [
    "DatasetConfig",
    "SWEBenchConfig",
    "SWEBenchProConfig",
    "TerminalBenchConfig",
    "GSOConfig",
    "get_dataset_config",
    "list_datasets",
    "DATASET_CONFIGS",
]
