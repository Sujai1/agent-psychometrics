"""Feature and evaluation persistence."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Feature:
    """A single feature definition."""

    id: str  # Unique identifier (e.g., "gen0_feat3")
    name: str  # Human-readable name
    description: str  # What the feature measures
    extraction_prompt: str  # Prompt to extract score from task
    scale_low: str  # Description of score 1
    scale_high: str  # Description of score 5
    hypothesis: str  # Expected correlation direction and why
    parent_id: Optional[str] = None  # ID of parent feature (if evolved)
    mutation_type: Optional[str] = None  # How this feature was created
    mutation_prompt: Optional[str] = None  # Prompt used for self-referential mutation
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Feature":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class FeatureEvaluation:
    """Evaluation results for a single feature."""

    feature_id: str
    correlation: float  # Pearson correlation with IRT difficulty
    abs_correlation: float  # Absolute correlation (for ranking)
    mean_score: float
    std_score: float
    n_tasks: int
    task_scores: Dict[str, float] = field(default_factory=dict)  # task_id -> score
    task_errors: Dict[str, float] = field(default_factory=dict)  # task_id -> prediction error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureEvaluation":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GenerationSummary:
    """Summary of a single generation."""

    generation: int
    n_features: int
    n_surviving: int
    best_correlation: float
    best_feature_id: str
    mean_correlation: float
    timestamp: str
    token_usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationSummary":
        """Create from dictionary."""
        return cls(**data)


class FeatureStore:
    """Persistent storage for features and evaluations."""

    def __init__(self, output_dir: Path):
        """Initialize feature store.

        Args:
            output_dir: Root directory for storing results.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _gen_dir(self, generation: int) -> Path:
        """Get directory for a specific generation."""
        gen_dir = self.output_dir / "generations" / f"gen_{generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        return gen_dir

    def save_features(self, generation: int, features: List[Feature]):
        """Save features for a generation.

        Args:
            generation: Generation number.
            features: List of Feature objects.
        """
        gen_dir = self._gen_dir(generation)
        path = gen_dir / "features.json"

        data = [f.to_dict() for f in features]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_features(self, generation: int) -> List[Feature]:
        """Load features from a generation.

        Args:
            generation: Generation number.

        Returns:
            List of Feature objects.
        """
        gen_dir = self._gen_dir(generation)
        path = gen_dir / "features.json"

        if not path.exists():
            return []

        with open(path) as f:
            data = json.load(f)

        return [Feature.from_dict(d) for d in data]

    def save_evaluations(self, generation: int, evaluations: List[FeatureEvaluation]):
        """Save evaluations for a generation.

        Args:
            generation: Generation number.
            evaluations: List of FeatureEvaluation objects.
        """
        gen_dir = self._gen_dir(generation)
        path = gen_dir / "evaluations.json"

        data = [e.to_dict() for e in evaluations]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_evaluations(self, generation: int) -> List[FeatureEvaluation]:
        """Load evaluations from a generation.

        Args:
            generation: Generation number.

        Returns:
            List of FeatureEvaluation objects.
        """
        gen_dir = self._gen_dir(generation)
        path = gen_dir / "evaluations.json"

        if not path.exists():
            return []

        with open(path) as f:
            data = json.load(f)

        return [FeatureEvaluation.from_dict(d) for d in data]

    def save_summary(self, summary: GenerationSummary):
        """Save generation summary.

        Args:
            summary: GenerationSummary object.
        """
        gen_dir = self._gen_dir(summary.generation)
        path = gen_dir / "summary.json"

        with open(path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)

    def load_summary(self, generation: int) -> Optional[GenerationSummary]:
        """Load generation summary.

        Args:
            generation: Generation number.

        Returns:
            GenerationSummary object or None if not found.
        """
        gen_dir = self._gen_dir(generation)
        path = gen_dir / "summary.json"

        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return GenerationSummary.from_dict(data)

    def save_best_features(self, features: List[Feature], evaluations: List[FeatureEvaluation]):
        """Save the best features across all generations.

        Args:
            features: List of best Feature objects.
            evaluations: Corresponding evaluations.
        """
        path = self.output_dir / "best_features.json"

        data = {
            "features": [f.to_dict() for f in features],
            "evaluations": [e.to_dict() for e in evaluations],
            "timestamp": datetime.now().isoformat(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_best_features(self) -> tuple:
        """Load best features.

        Returns:
            Tuple of (features, evaluations) or ([], []) if not found.
        """
        path = self.output_dir / "best_features.json"

        if not path.exists():
            return [], []

        with open(path) as f:
            data = json.load(f)

        features = [Feature.from_dict(d) for d in data["features"]]
        evaluations = [FeatureEvaluation.from_dict(d) for d in data["evaluations"]]

        return features, evaluations

    def save_checkpoint(self, generation: int, state: Dict[str, Any]):
        """Save checkpoint for resuming.

        Args:
            generation: Current generation number.
            state: State dictionary to save.
        """
        path = self.output_dir / "checkpoint.json"

        data = {
            "generation": generation,
            "state": state,
            "timestamp": datetime.now().isoformat(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint.

        Returns:
            Checkpoint data or None if not found.
        """
        path = self.output_dir / "checkpoint.json"

        if not path.exists():
            return None

        with open(path) as f:
            return json.load(f)

    def save_evolution_log(self, log: Dict[str, Any]):
        """Save evolution log.

        Args:
            log: Log data dictionary.
        """
        path = self.output_dir / "evolution_log.json"

        with open(path, "w") as f:
            json.dump(log, f, indent=2)

    def load_evolution_log(self) -> Optional[Dict[str, Any]]:
        """Load evolution log.

        Returns:
            Log data or None if not found.
        """
        path = self.output_dir / "evolution_log.json"

        if not path.exists():
            return None

        with open(path) as f:
            return json.load(f)

    def get_latest_generation(self) -> int:
        """Get the latest completed generation number.

        Returns:
            Latest generation number, or -1 if none found.
        """
        gen_dir = self.output_dir / "generations"
        if not gen_dir.exists():
            return -1

        generations = []
        for d in gen_dir.iterdir():
            if d.is_dir() and d.name.startswith("gen_"):
                try:
                    gen_num = int(d.name.split("_")[1])
                    # Only count if summary exists (completed)
                    if (d / "summary.json").exists():
                        generations.append(gen_num)
                except ValueError:
                    pass

        return max(generations) if generations else -1
