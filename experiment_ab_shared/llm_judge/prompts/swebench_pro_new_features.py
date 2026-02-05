"""SWE-bench Pro prompt configuration for 7 new features.

This module extracts 7 additional features for SWE-bench Pro tasks,
to be merged with the existing 8 core unified features.

Features (7 total):
- integration_complexity, solution_locality, side_effect_risk (solution-focused)
- information_completeness, codebase_scope, error_specificity, debugging_complexity (problem-focused)
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import PromptConfig
from experiment_ab_shared.llm_judge.prompts.extended_features import (
    NEW_CODE_FEATURES,
    COMPLETENESS_INSTRUCTION,
    INTEGRATION_COMPLEXITY_SCALE,
    SOLUTION_LOCALITY_SCALE_CODE,
    SIDE_EFFECT_RISK_SCALE_CODE,
    INFORMATION_COMPLETENESS_SCALE,
    CODEBASE_SCOPE_SCALE_CODE,
    ERROR_SPECIFICITY_SCALE_CODE,
    DEBUGGING_COMPLEXITY_SCALE,
    OUTPUT_FORMAT_7_FEATURES_CODE,
)


SWEBENCH_PRO_NEW_FEATURES_PROMPT_TEMPLATE = """You are analyzing a SWE-bench Pro coding task to extract additional difficulty features.
This is a BUG FIX task in a Python repository. You will analyze the problem statement AND gold patch.

{completeness_instruction}

## TASK INFORMATION

**Instance ID:** {{instance_id}}
**Repository:** {{repo}}
**Version:** {{version}}

**Problem Statement:**
{{problem_statement}}

**Gold Patch (correct solution):**
```diff
{{patch}}
```

**Tests that should pass after fix (FAIL_TO_PASS):**
{{fail_to_pass}}

**Regression tests (PASS_TO_PASS):**
{{pass_to_pass}}

{{hints_section}}

## FEATURES TO EVALUATE (7 features)

Analyze the problem statement AND gold patch to evaluate these 7 features.
Focus on what makes the SOLUTION hard, not just what the PROBLEM looks like.

{integration_complexity_scale}

{solution_locality_scale}

{side_effect_risk_scale}

{information_completeness_scale}

{codebase_scope_scale}

{error_specificity_scale}

{debugging_complexity_scale}

{output_format}
""".format(
    completeness_instruction=COMPLETENESS_INSTRUCTION,
    integration_complexity_scale=INTEGRATION_COMPLEXITY_SCALE,
    solution_locality_scale=SOLUTION_LOCALITY_SCALE_CODE,
    side_effect_risk_scale=SIDE_EFFECT_RISK_SCALE_CODE,
    information_completeness_scale=INFORMATION_COMPLETENESS_SCALE,
    codebase_scope_scale=CODEBASE_SCOPE_SCALE_CODE,
    error_specificity_scale=ERROR_SPECIFICITY_SCALE_CODE,
    debugging_complexity_scale=DEBUGGING_COMPLEXITY_SCALE,
    output_format=OUTPUT_FORMAT_7_FEATURES_CODE,
)


def format_swebench_pro_new_features_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench Pro new features prompt with task-specific information.

    Args:
        task: SWE-bench Pro task dict with keys:
            - instance_id: SWE-bench instance ID
            - repo: Repository name (e.g., "django/django")
            - version: Version string
            - problem_statement: The issue description
            - patch: The gold solution patch
            - fail_to_pass or FAIL_TO_PASS: Tests that should pass after fix
            - pass_to_pass or PASS_TO_PASS: Regression tests
            - hints_text: Optional hints (may be empty)

    Returns:
        Formatted prompt string
    """
    hints_text = task.get("hints_text", "") or ""
    hints_section = ""
    if hints_text and hints_text.strip():
        hints_section = f"**Hints:**\n{hints_text}"

    # No truncation needed - Claude Opus 4.5 has 200K token context
    problem_statement = task.get("problem_statement", "") or ""
    patch = task.get("patch", "") or ""

    # Handle both lowercase and uppercase field names
    fail_to_pass = task.get("fail_to_pass") or task.get("FAIL_TO_PASS") or "[]"
    pass_to_pass = task.get("pass_to_pass") or task.get("PASS_TO_PASS") or "[]"

    return SWEBENCH_PRO_NEW_FEATURES_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        hints_section=hints_section,
    )


# The main configuration object
SWEBENCH_PRO_NEW_FEATURES_CONFIG = PromptConfig(
    name="swebench_pro_new_features",
    features=NEW_CODE_FEATURES,
    prompt_template=SWEBENCH_PRO_NEW_FEATURES_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={},  # No truncation needed with Claude Opus 4.5's 200K context
    format_prompt_fn=format_swebench_pro_new_features_prompt,
)
