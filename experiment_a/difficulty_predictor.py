"""Difficulty predictor base class and implementations.

This module re-exports from experiment_a_common for backward compatibility.
New code should import directly from experiment_a_common.
"""

# Re-export base classes and simple predictors from shared module
from experiment_a_common.predictor_base import (
    DifficultyPredictorBase,
    ConstantPredictor,
    GroundTruthPredictor,
)

# Re-export feature infrastructure
from experiment_a_common.feature_source import (
    TaskFeatureSource,
    EmbeddingFeatureSource,
    CSVFeatureSource,
    ConcatenatedFeatureSource,
)

from experiment_a_common.feature_predictor import FeatureBasedPredictor

__all__ = [
    # Base classes
    "DifficultyPredictorBase",
    "ConstantPredictor",
    "GroundTruthPredictor",
    # Feature sources
    "TaskFeatureSource",
    "EmbeddingFeatureSource",
    "CSVFeatureSource",
    "ConcatenatedFeatureSource",
    # Predictor
    "FeatureBasedPredictor",
]
