"""Inspect task definition for auditor agent on Terminal Bench.

Terminal Bench uses pre-built Docker images from Docker Hub at:
    xiangyangli/{task_id}:20260204

Each container provides an Ubuntu/Debian environment with the task
setup pre-configured. The agent interacts via /app as the working directory.

Task metadata (instruction, category, etc.) is loaded from the local
terminal-bench-2 repo at terminal-bench-2/{task_id}/instruction.md and task.toml.

Usage:
    inspect eval experiment_a/terminalbench/inspect_task.py@auditor_task_v4_terminalbench \
        --model anthropic/claude-opus-4-5-20251101 --limit 1
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import bash
from inspect_ai.util import SandboxEnvironmentSpec

from experiment_a.sandbox_utils import get_sandbox_config
from experiment_a.auditor_agent.prompts_v4 import build_auditor_system_prompt_v4


DOCKER_REPO = "xiangyangli"
DOCKER_TAG = "20260204"

# Default paths
DEFAULT_ITEMS_PATH = Path("chris_output/terminal_bench_2.0/1d_1pl/items.csv")
DEFAULT_REPO_PATH = Path("terminal-bench-2")


def _get_terminalbench_image(task_id: str) -> str:
    """Get Docker image name for a Terminal Bench task."""
    return f"{DOCKER_REPO}/{task_id}:{DOCKER_TAG}"


def load_terminalbench_samples(
    items_path: Path = DEFAULT_ITEMS_PATH,
    repo_path: Path = DEFAULT_REPO_PATH,
):
    """Load Terminal Bench tasks as Inspect samples with Docker sandbox configs.

    Task IDs come from the IRT items file. Task instructions come from
    the terminal-bench-2 repo's instruction.md files.

    Args:
        items_path: Path to IRT items.csv (provides the list of task IDs).
        repo_path: Path to the cloned terminal-bench-2 repo.

    Returns:
        List of Inspect samples with sandbox configs attached.

    Raises:
        FileNotFoundError: If items_path or repo_path doesn't exist.
        ValueError: If a task directory is missing required files.
    """
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not repo_path.exists():
        raise FileNotFoundError(
            f"terminal-bench-2 repo not found: {repo_path}. "
            f"Clone it with: git clone https://github.com/harbor-framework/terminal-bench-2"
        )

    items_df = pd.read_csv(items_path, index_col=0)
    task_ids = list(items_df.index)

    samples = []
    for task_id in task_ids:
        task_dir = repo_path / task_id
        instruction_md = task_dir / "instruction.md"
        task_toml_path = task_dir / "task.toml"

        if not instruction_md.exists():
            raise ValueError(
                f"instruction.md not found for '{task_id}' at {instruction_md}"
            )
        if not task_toml_path.exists():
            raise ValueError(
                f"task.toml not found for '{task_id}' at {task_toml_path}"
            )

        instruction = instruction_md.read_text(encoding="utf-8").strip()
        if not instruction:
            raise ValueError(f"Empty instruction.md for '{task_id}'")

        with open(task_toml_path, "rb") as f:
            task_toml = tomllib.load(f)
        metadata_section = task_toml.get("metadata", {})

        image = _get_terminalbench_image(task_id)

        sample = Sample(
            input=instruction,
            id=task_id,
            metadata={
                "category": metadata_section.get("category", ""),
                "tags": metadata_section.get("tags", []),
                "difficulty": metadata_section.get("difficulty", ""),
            },
        )
        sample.sandbox = SandboxEnvironmentSpec(
            type="docker",
            config=get_sandbox_config(
                task_id, image_name=image, working_dir="/app"
            ),
        )
        samples.append(sample)

    return samples


@task
def auditor_task_v4_terminalbench(
    max_attempts: int = 1,
    message_limit: int = 50,
) -> Task:
    """Run V4 auditor agent on Terminal Bench tasks."""
    samples = load_terminalbench_samples()

    auditor_agent = basic_agent(
        init=system_message(
            build_auditor_system_prompt_v4(task_type="terminalbench")
        ),
        tools=[bash(timeout=120)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 8 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,
        name="auditor_v4_terminalbench",
    )
