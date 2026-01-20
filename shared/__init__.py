"""Root-level shared utilities for experiments A and B.

This module provides the core abstractions for difficulty prediction:
- TaskFeatureSource: Plug-and-play interface for any feature type
- FeatureBasedPredictor: Source-agnostic Ridge regression predictor
- DifficultyPredictorBase: Abstract base for all predictors
"""

from shared.dataset import (
    ExperimentData,
    BinaryExperimentData,
    BinomialExperimentData,
    load_dataset,
    load_dataset_for_fold,
    stable_split_tasks,
    filter_unsolved_tasks,
)
from shared.evaluator import (
    compute_auc,
    compute_irt_probability,
    convert_numpy,
    evaluate_single_predictor,
    run_evaluation_pipeline,
    PredictorConfig,
    PredictorResult,
)
from shared.baselines import (
    agent_only_baseline,
    random_baseline,
    verify_random_baseline_sanity,
)
from shared.cross_validation import (
    CrossValidationResult,
    k_fold_split_tasks,
    run_cv_for_predictor,
    run_cv_for_baseline,
)
from shared.binomial_metrics import (
    BinomialMetricsResult,
    compute_binomial_metrics,
)
from shared.feature_source import (
    TaskFeatureSource,
    EmbeddingFeatureSource,
    CSVFeatureSource,
    ConcatenatedFeatureSource,
)
from shared.feature_predictor import (
    FeatureBasedPredictor,
)
from shared.predictor_base import (
    DifficultyPredictorBase,
    ConstantPredictor,
    GroundTruthPredictor,
)
from shared.train_irt_split import (
    get_or_train_split_irt,
    get_split_cache_dir,
    check_cached_irt,
)

__all__ = [
    # Dataset
    "ExperimentData",
    "BinaryExperimentData",
    "BinomialExperimentData",
    "load_dataset",
    "load_dataset_for_fold",
    "stable_split_tasks",
    "filter_unsolved_tasks",
    # Evaluator
    "compute_auc",
    "compute_irt_probability",
    "convert_numpy",
    "evaluate_single_predictor",
    "run_evaluation_pipeline",
    "PredictorConfig",
    "PredictorResult",
    # Baselines
    "agent_only_baseline",
    "random_baseline",
    "verify_random_baseline_sanity",
    # Cross-validation
    "CrossValidationResult",
    "k_fold_split_tasks",
    "run_cv_for_predictor",
    "run_cv_for_baseline",
    # Binomial metrics
    "BinomialMetricsResult",
    "compute_binomial_metrics",
    # Feature sources
    "TaskFeatureSource",
    "EmbeddingFeatureSource",
    "CSVFeatureSource",
    "ConcatenatedFeatureSource",
    # Feature-based predictor
    "FeatureBasedPredictor",
    # Predictor base classes
    "DifficultyPredictorBase",
    "ConstantPredictor",
    "GroundTruthPredictor",
    # IRT training
    "get_or_train_split_irt",
    "get_split_cache_dir",
    "check_cached_irt",
]
