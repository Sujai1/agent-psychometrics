"""Main training and evaluation pipeline for Experiment A (GSO).

This is a thin wrapper around the shared pipeline in experiment_a.shared.pipeline.
"""

from pathlib import Path

from experiment_a.gso.config import GSOConfig
from experiment_a.shared.pipeline import ExperimentSpec, run_experiment_main


# Root directory for resolving relative paths
ROOT = Path(__file__).resolve().parents[2]

# Experiment specification for GSO
# LLM judge features are auto-detected from the CSV (no need to specify)
SPEC = ExperimentSpec(
    name="GSO",
    is_binomial=False,  # GSO uses binary 0/1 responses
    irt_cache_dir=ROOT / "chris_output" / "experiment_a_gso" / "irt_splits",
    llm_judge_features=None,  # Auto-detect from CSV
)


def main():
    """Run Experiment A on GSO."""
    run_experiment_main(GSOConfig, SPEC, ROOT)


if __name__ == "__main__":
    main()
