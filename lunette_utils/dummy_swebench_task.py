"""
SWE-bench task with a dummy solver for Experiment C.

This creates a minimal task that:
1. Uses Lunette sandbox to get environment access
2. Runs a dummy solver that immediately fails
3. Allows us to measure grading cost without agent cost

Usage:
    # With lunette sandbox
    inspect eval lunette_utils/dummy_swebench_task.py \
        --model mockllm/model \
        --sandbox lunette \
        --limit 1 \
        --no-score

    # Or use lunette CLI directly
    lunette eval lunette_utils/dummy_swebench_task.py --model mockllm/model --limit 1
"""

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver, TaskState, solver
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_evals.utils.huggingface import hf_dataset


@solver
def dummy_solver() -> Solver:
    """A minimal solver that does absolutely nothing.

    Lunette's investigator agent has its own sandbox access and can explore
    the environment independently - the solver doesn't need to do any work.
    This produces zero messages so we can measure pure Lunette grading cost.
    """

    async def solve(state: TaskState, generate) -> TaskState:
        """Do nothing - just return the state unchanged."""
        return state

    return solve


@solver
def instant_fail_solver() -> Solver:
    """Even more minimal - just immediately returns without any exploration."""

    async def solve(state: TaskState, generate) -> TaskState:
        state.messages.append(
            ChatMessageAssistant(content="Dummy agent: No attempt made.")
        )
        return state

    return solve


def get_swebench_image_name(instance_id: str) -> str:
    """Get Docker image name for a SWE-bench instance."""
    import platform
    updated_id = instance_id.replace("__", "_1776_")
    if platform.machine() in {"aarch64", "arm64"}:
        arch = "arm64"
    else:
        arch = "x86_64"
    return f"swebench/sweb.eval.{arch}.{updated_id}:latest"


def get_sandbox_config(instance_id: str) -> str:
    """Generate sandbox config for a SWE-bench instance."""
    import tempfile
    from pathlib import Path

    image_name = get_swebench_image_name(instance_id)

    content = f"""services:
    default:
        image: {image_name}
        command: "sleep infinity"
        working_dir: /testbed
        deploy:
            resources:
                limits:
                    cpus: '1'
"""

    # Write to temp file
    config_dir = Path(tempfile.gettempdir()) / "swebench_configs"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / f"{instance_id}-compose.yaml"
    config_file.write_text(content)

    return str(config_file)


@task
def dummy_swebench(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    explore: bool = True,
) -> Task:
    """SWE-bench with a dummy solver for measuring grading cost.

    Args:
        dataset: HuggingFace dataset name
        split: Dataset split
        explore: If True, use dummy_solver (explores env). If False, use instant_fail_solver.
    """
    samples = hf_dataset(
        path=dataset,
        split=split,
        sample_fields=FieldSpec(
            input="problem_statement",
            id="instance_id",
            metadata=[
                "base_commit",
                "patch",
                "PASS_TO_PASS",
                "FAIL_TO_PASS",
                "test_patch",
                "version",
                "repo",
            ],
        ),
    )

    # Add sandbox config to each sample
    for sample in samples:
        sample.sandbox = SandboxEnvironmentSpec(
            type="lunette",
            config=get_sandbox_config(str(sample.id)),
        )

    return Task(
        dataset=samples,
        solver=dummy_solver() if explore else instant_fail_solver(),
        scorer=None,  # Don't score - we know it will fail
        message_limit=10,
    )


@task
def dummy_swebench_mini(
    split: str = "test",
    explore: bool = True,
) -> Task:
    """SWE-bench verified mini with a dummy solver."""
    return dummy_swebench(
        dataset="MariusHobbhahn/swe-bench-verified-mini",
        split=split,
        explore=explore,
    )
