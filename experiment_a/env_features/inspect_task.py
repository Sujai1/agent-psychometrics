"""Inspect task definition for environment feature extraction.

This task uses dynamic sandboxing to run each SWE-bench instance in its own
Docker container, then extracts deterministic features from the environment.

Usage:
    # Test on 2 tasks
    inspect eval experiment_a/env_features/inspect_task.py --limit 2

    # With parallelism
    inspect eval experiment_a/env_features/inspect_task.py --limit 10 --max-connections 10
"""

import sys
from pathlib import Path

# Add project root to path so we can import experiment_a modules
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_evals.utils.huggingface import hf_dataset

from experiment_a.env_features.extractor_solver import env_feature_extractor
from experiment_a.sandbox_utils import get_sandbox_config


@task
def env_feature_extraction(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
) -> Task:
    """Extract environment features from SWE-bench tasks.

    Args:
        dataset: HuggingFace dataset name
        split: Dataset split (default: test)

    Returns:
        Inspect Task configured with dynamic sandboxing per instance
    """
    # Load dataset from HuggingFace
    samples = hf_dataset(
        path=dataset,
        split=split,
        sample_fields=FieldSpec(
            input="problem_statement",
            id="instance_id",
            metadata=[
                "base_commit",
                "patch",
                "repo",
                "version",
            ],
        ),
    )

    # Add sandbox config to each sample (dynamic per-instance Docker image)
    for sample in samples:
        sample.sandbox = SandboxEnvironmentSpec(
            type="docker",
            config=get_sandbox_config(str(sample.id)),
        )

    return Task(
        dataset=samples,
        solver=env_feature_extractor(),
        scorer=None,  # No scoring needed - we just extract features
        name="swe_bench_env_features",
    )


@task
def env_feature_extraction_mini(
    split: str = "test",
) -> Task:
    """Extract features from SWE-bench verified mini (smaller dataset for testing)."""
    return env_feature_extraction(
        dataset="MariusHobbhahn/swe-bench-verified-mini",
        split=split,
    )
