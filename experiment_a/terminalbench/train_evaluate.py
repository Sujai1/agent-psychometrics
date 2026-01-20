"""Main training and evaluation pipeline for Experiment A (TerminalBench).

This is a thin wrapper around the shared pipeline in experiment_a.shared.pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List

from experiment_a.terminalbench.config import TerminalBenchConfig
from experiment_a.terminalbench.data_loader import load_task_data_from_repo
from experiment_a.shared.pipeline import ExperimentSpec, run_experiment_main


# Root directory for resolving relative paths
ROOT = Path(__file__).resolve().parents[2]

# TerminalBench-specific LLM judge features (4 pre-selected features)
# Pre-selected subset that works well with Ridge regression (verified by comparing
# Ridge-only vs Lasso+Ridge performance). The full 8 features extracted by
# experiment_a/terminalbench/llm_judge_prompt.py are available, but using all 8
# with Ridge-only hurts performance compared to this pre-selected subset.
TERMINALBENCH_LLM_JUDGE_FEATURES = [
    "task_clarity",
    "domain_knowledge_required",
    "task_complexity",
    "atypicality",
]

# Experiment specification for TerminalBench
SPEC = ExperimentSpec(
    name="TerminalBench",
    is_binomial=True,  # TerminalBench uses binomial (successes/trials) responses
    irt_cache_dir=ROOT / "chris_output" / "experiment_a_terminalbench" / "irt_splits",
    llm_judge_features=TERMINALBENCH_LLM_JUDGE_FEATURES,
)


def create_metadata_loader(config: TerminalBenchConfig):
    """Create a metadata loader that loads task data from the terminal-bench repo.

    Args:
        config: TerminalBench configuration containing repo_path

    Returns:
        Callable that loads task metadata from the repo
    """
    repo_path = ROOT / config.repo_path

    def loader(task_ids: List[str]) -> Dict[str, Any]:
        return {"task_data": load_task_data_from_repo(task_ids, repo_path)}

    return loader


def main():
    """Run Experiment A on TerminalBench."""
    run_experiment_main(
        TerminalBenchConfig, SPEC, ROOT, metadata_loader_factory=create_metadata_loader
    )


if __name__ == "__main__":
    main()
