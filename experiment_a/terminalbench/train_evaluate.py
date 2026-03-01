"""Single-dataset CLI entry point for Experiment A (TerminalBench).

TerminalBench has unique behavior not shared by other datasets:
- Binary/binomial mode switching (--binary flag)
- Task metadata loading from the terminal-bench repo

For running TerminalBench alongside other datasets, use run_all_datasets.py instead.

Supports two modes:
- Binomial (default): Uses k/n successes per agent-task pair
- Binary (--binary flag): Uses collapsed binary (any success = 1)
"""

from pathlib import Path
from typing import Any, Dict, List

from experiment_a.shared.config import TerminalBenchConfig, build_spec
from experiment_a.terminalbench.data_loader import load_task_data_from_repo
from experiment_a.shared.pipeline import ExperimentSpec, run_experiment_main


# Root directory for resolving relative paths
ROOT = Path(__file__).resolve().parents[2]


def get_spec(use_binary: bool) -> ExperimentSpec:
    """Get experiment specification based on binary/binomial mode."""
    if use_binary:
        return ExperimentSpec(
            name="TerminalBench (Binary)",
            is_binomial=False,
            irt_cache_dir=ROOT / "chris_output" / "experiment_a_terminalbench_binary" / "irt_splits",
        )
    return build_spec("terminalbench", ROOT)


def create_metadata_loader(config: TerminalBenchConfig):
    """Create a metadata loader that loads task data from the terminal-bench repo."""
    repo_path = ROOT / config.repo_path

    def loader(task_ids: List[str]) -> Dict[str, Any]:
        return {"task_data": load_task_data_from_repo(task_ids, repo_path)}

    return loader


def main():
    """Run Experiment A on TerminalBench.

    Default is binomial mode (k/n successes per agent-task pair).
    Use --binary flag to use collapsed binary mode (any success = 1).
    """
    run_experiment_main(
        "terminalbench",
        build_spec("terminalbench", ROOT),
        ROOT,
        config_class=TerminalBenchConfig,
        metadata_loader_factory=create_metadata_loader,
        spec_factory=get_spec,
    )


if __name__ == "__main__":
    main()
