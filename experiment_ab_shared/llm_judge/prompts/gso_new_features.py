"""GSO (Software Optimization Benchmark) prompt configuration for 7 new features.

This module extracts 7 additional features for GSO tasks,
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
    SOLUTION_LOCALITY_SCALE_OPTIMIZATION,
    SIDE_EFFECT_RISK_SCALE_OPTIMIZATION,
    INFORMATION_COMPLETENESS_SCALE,
    CODEBASE_SCOPE_SCALE_OPTIMIZATION,
    ERROR_SPECIFICITY_SCALE_OPTIMIZATION,
    DEBUGGING_COMPLEXITY_SCALE,
    OUTPUT_FORMAT_7_FEATURES_CODE,
)


GSO_NEW_FEATURES_PROMPT_TEMPLATE = """You are analyzing a GSO (Software Optimization Benchmark) task to extract additional difficulty features.
This is a PERFORMANCE OPTIMIZATION task, NOT a bug fix. The goal is to make code run faster
while maintaining correctness. You will analyze the test script AND gold optimization patch.

{completeness_instruction}

## TASK INFORMATION

**Instance ID:** {{instance_id}}
**Repository:** {{repo}}
**API/Function being optimized:** {{api}}

**Test Script (performance scenario to optimize):**
```python
{{prob_script}}
```

**Gold Patch (optimization solution):**
```diff
{{gt_diff}}
```

{{hints_section}}

## FEATURES TO EVALUATE (7 features)

Analyze the test script AND optimization patch to evaluate these 7 features.
Focus on what makes the OPTIMIZATION hard, not just what the code looks like.

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
    solution_locality_scale=SOLUTION_LOCALITY_SCALE_OPTIMIZATION,
    side_effect_risk_scale=SIDE_EFFECT_RISK_SCALE_OPTIMIZATION,
    information_completeness_scale=INFORMATION_COMPLETENESS_SCALE,
    codebase_scope_scale=CODEBASE_SCOPE_SCALE_OPTIMIZATION,
    error_specificity_scale=ERROR_SPECIFICITY_SCALE_OPTIMIZATION,
    debugging_complexity_scale=DEBUGGING_COMPLEXITY_SCALE,
    output_format=OUTPUT_FORMAT_7_FEATURES_CODE,
)


def format_gso_new_features_prompt(task: Dict[str, Any]) -> str:
    """Format the GSO new features prompt with task-specific information.

    Args:
        task: GSO task dict with keys:
            - instance_id: GSO task ID
            - repo: Repository name (e.g., "numpy/numpy")
            - api: API/function being optimized
            - prob_script: Test script showing performance scenario
            - gt_diff: Gold optimization patch
            - hints_text: Optional hints (may be empty)

    Returns:
        Formatted prompt string
    """
    hints_text = task.get("hints_text", "") or ""
    hints_section = ""
    if hints_text and hints_text.strip():
        hints_section = f"**Hints:**\n{hints_text}"

    # No truncation needed - Claude Opus 4.5 has 200K token context
    prob_script = task.get("prob_script", "") or ""
    gt_diff = task.get("gt_diff", "") or ""

    return GSO_NEW_FEATURES_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        api=task.get("api", "unknown"),
        prob_script=prob_script,
        gt_diff=gt_diff,
        hints_section=hints_section,
    )


# The main configuration object
GSO_NEW_FEATURES_CONFIG = PromptConfig(
    name="gso_new_features",
    features=NEW_CODE_FEATURES,
    prompt_template=GSO_NEW_FEATURES_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={},  # No truncation needed with Claude Opus 4.5's 200K context
    format_prompt_fn=format_gso_new_features_prompt,
)
