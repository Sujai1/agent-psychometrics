"""Main training and evaluation pipeline for Experiment A (SWE-bench Pro).

This is a thin wrapper around the shared pipeline in experiment_a.shared.pipeline.
"""

from pathlib import Path

from experiment_a.swebench_pro.config import SWEBenchProConfig
from experiment_a.shared.pipeline import ExperimentSpec, run_experiment_main


# Root directory for resolving relative paths
ROOT = Path(__file__).resolve().parents[2]

# Experiment specification for SWE-bench Pro
# LLM judge features are auto-detected from the CSV (no need to specify)
SPEC = ExperimentSpec(
    name="SWE-bench Pro",
    is_binomial=False,  # SWE-bench Pro uses binary 0/1 responses
    irt_cache_dir=ROOT / "chris_output" / "experiment_a_swebench_pro" / "irt_splits",
    llm_judge_features=None,  # Auto-detect from CSV
)


def main():
    """Run Experiment A on SWE-bench Pro."""
    run_experiment_main(SWEBenchProConfig, SPEC, ROOT)


if __name__ == "__main__":
    main()
