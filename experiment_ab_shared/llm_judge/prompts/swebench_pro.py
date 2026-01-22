"""SWE-bench Pro prompt configuration for LLM judge feature extraction.

SWE-bench Pro uses the same 9 semantic features as SWE-bench Verified since both
are code/patch datasets with similar structure. The prompt and feature definitions
are reused from swebench.py.

Data source: HuggingFace ScaleAI/SWE-bench_Pro
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import PromptConfig
from experiment_ab_shared.llm_judge.prompts.swebench import (
    SWEBENCH_FEATURES,
    SWEBENCH_PROMPT_TEMPLATE,
)


def format_swebench_pro_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench Pro prompt with task-specific information.

    SWE-bench Pro has similar structure to SWE-bench Verified, with fields:
    - instance_id: Task identifier
    - repo: Repository name (e.g., "flipt-io/flipt")
    - problem_statement: The issue description
    - patch: The gold solution patch
    - fail_to_pass: Tests that should pass after fix
    - pass_to_pass: Regression tests

    Note: SWE-bench Pro uses lowercase field names (fail_to_pass, pass_to_pass)
    while SWE-bench Verified uses uppercase (FAIL_TO_PASS, PASS_TO_PASS).

    Args:
        task: SWE-bench Pro task dict

    Returns:
        Formatted prompt string
    """
    # Handle potential empty or None hints
    hints_text = task.get("hints_text", "") or ""
    hints_section = ""
    if hints_text.strip():
        hints_section = f"**Hints:**\n{hints_text}"

    # Truncate very long fields to avoid context overflow
    problem_statement = task.get("problem_statement", "") or ""
    if len(problem_statement) > 12000:
        problem_statement = problem_statement[:12000]

    patch = task.get("patch", "") or ""
    if len(patch) > 8000:
        patch = patch[:8000]

    # SWE-bench Pro may use lowercase field names
    fail_to_pass = task.get("fail_to_pass") or task.get("FAIL_TO_PASS") or "[]"
    pass_to_pass = task.get("pass_to_pass") or task.get("PASS_TO_PASS") or "[]"

    return SWEBENCH_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        hints_section=hints_section,
    )


# The main configuration object for SWE-bench Pro
SWEBENCH_PRO_CONFIG = PromptConfig(
    name="swebench_pro",
    features=SWEBENCH_FEATURES,  # Reuse same 9 features as SWE-bench Verified
    prompt_template=SWEBENCH_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={
        "problem_statement": 12000,
        "patch": 8000,
    },
    format_prompt_fn=format_swebench_pro_prompt,
)
