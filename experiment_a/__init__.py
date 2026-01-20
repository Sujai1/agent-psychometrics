"""Experiment A: Prior Validation (IRT AUC) Pipeline.

This module evaluates how well a difficulty predictor can predict agent success
on held-out tasks using the 1PL IRT model.

Core formula: P(success) = sigmoid(theta_j - beta_i)

Where:
- theta_j: Agent j's ability (from 1PL IRT model)
- beta_i: Task i's predicted difficulty

Evaluation metric: AUC comparing predicted probabilities to actual outcomes.
"""

from experiment_a.config import ExperimentAConfig
from experiment_a.data_loader import (
    ExperimentAData,
    load_abilities,
    load_items,
    load_responses,
    stable_split_tasks,
)
from experiment_a.difficulty_predictor import (
    DifficultyPredictorBase,
    ConstantPredictor,
    GroundTruthPredictor,
    FeatureBasedPredictor,
    EmbeddingFeatureSource,
    CSVFeatureSource,
)
from experiment_a_common import (
    compute_auc,
    agent_only_baseline,
    random_baseline,
)

__all__ = [
    "ExperimentAConfig",
    "ExperimentAData",
    "load_abilities",
    "load_items",
    "load_responses",
    "stable_split_tasks",
    "DifficultyPredictorBase",
    "ConstantPredictor",
    "GroundTruthPredictor",
    "FeatureBasedPredictor",
    "EmbeddingFeatureSource",
    "CSVFeatureSource",
    "compute_auc",
    "agent_only_baseline",
    "random_baseline",
]
