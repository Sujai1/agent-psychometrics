"""Configuration for Experiment B."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    """Configuration for Experiment B."""

    # Data paths
    items_path: Path = Path("clean_data/swebench_verified_20251115_full/1d/items.csv")
    responses_path: Path = Path("chris_output/clean_data/swebench_verified/swebench_verified_20251115_full.jsonl")
    trajectories_dir: Path = Path("trajectory_data/unified_trajs")
    output_dir: Path = Path("chris_output/experiment_b")

    # Agent splitting
    m1_fraction: float = 0.4  # Oldest 40%
    m2_fraction: float = 0.4  # Middle 40%
    # M3 = remaining 20%

    # Task selection
    weak_threshold: float = 0.2  # Max pass rate for "hard" tasks
    strong_min_improvement: float = 0.1  # Min improvement for strong group

    # Model parameters
    prior_alpha: float = 1.0  # Ridge alpha for prior
    posterior_alpha: float = 1.0  # Ridge alpha for psi

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dict, converting strings to Paths."""
        path_fields = {"items_path", "responses_path", "trajectories_dir", "output_dir"}
        converted = {}
        for k, v in d.items():
            if k in path_fields and isinstance(v, str):
                converted[k] = Path(v)
            else:
                converted[k] = v
        return cls(**converted)
