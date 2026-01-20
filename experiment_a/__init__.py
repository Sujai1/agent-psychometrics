"""Experiment A: Prior Validation (IRT AUC).

Evaluates how well a difficulty predictor can predict agent success on held-out
tasks using the 1PL IRT model.

Entry points:
- python -m experiment_a.swebench.train_evaluate  # SWE-bench
- python -m experiment_a.terminalbench.train_evaluate  # TerminalBench
"""

# Re-export core classes from shared modules for convenience
from shared import (
    DifficultyPredictorBase,
    ConstantPredictor,
    GroundTruthPredictor,
    FeatureBasedPredictor,
    EmbeddingFeatureSource,
    CSVFeatureSource,
    compute_auc,
    agent_only_baseline,
    random_baseline,
    stable_split_tasks,
)

__all__ = [
    "DifficultyPredictorBase",
    "ConstantPredictor",
    "GroundTruthPredictor",
    "FeatureBasedPredictor",
    "EmbeddingFeatureSource",
    "CSVFeatureSource",
    "compute_auc",
    "agent_only_baseline",
    "random_baseline",
    "stable_split_tasks",
]
