"""TerminalBench prompt configuration for 7 new features.

This module extracts 7 additional features for TerminalBench tasks,
to be merged with the existing 8 core unified features.

Features (7 total):
- tooling_complexity, solution_locality, side_effect_risk (solution-focused)
- information_completeness, codebase_scope, error_specificity, debugging_complexity (problem-focused)
"""

from typing import Any, Dict, List

from experiment_ab_shared.llm_judge.prompt_config import PromptConfig
from experiment_ab_shared.llm_judge.prompts.extended_features import (
    NEW_TERMINAL_FEATURES,
    COMPLETENESS_INSTRUCTION,
    TOOLING_COMPLEXITY_SCALE,
    SOLUTION_LOCALITY_SCALE_TERMINAL,
    SIDE_EFFECT_RISK_SCALE_TERMINAL,
    INFORMATION_COMPLETENESS_SCALE,
    CODEBASE_SCOPE_SCALE_TERMINAL,
    ERROR_SPECIFICITY_SCALE_TERMINAL,
    DEBUGGING_COMPLEXITY_SCALE,
    OUTPUT_FORMAT_7_FEATURES_TERMINAL,
)


TERMINALBENCH_NEW_FEATURES_PROMPT_TEMPLATE = """You are analyzing a TerminalBench terminal/shell task to extract additional difficulty features.
This task requires writing shell commands or scripts to accomplish a goal. You will analyze
the task instruction AND reference solution.

{completeness_instruction}

## TASK INFORMATION

**Task ID:** {{task_id}}
**Category:** {{category}}
**Tags:** {{tags}}
**Claimed Difficulty:** {{claimed_difficulty}}

**Task Instruction:**
{{instruction}}

**Reference Solution (solution.sh):**
```bash
{{solution}}
```

## FEATURES TO EVALUATE (7 features)

Analyze the instruction AND solution to evaluate these 7 features.
Focus on what makes the SOLUTION hard, not just what the TASK looks like.

{tooling_complexity_scale}

{solution_locality_scale}

{side_effect_risk_scale}

{information_completeness_scale}

{codebase_scope_scale}

{error_specificity_scale}

{debugging_complexity_scale}

{output_format}
""".format(
    completeness_instruction=COMPLETENESS_INSTRUCTION,
    tooling_complexity_scale=TOOLING_COMPLEXITY_SCALE,
    solution_locality_scale=SOLUTION_LOCALITY_SCALE_TERMINAL,
    side_effect_risk_scale=SIDE_EFFECT_RISK_SCALE_TERMINAL,
    information_completeness_scale=INFORMATION_COMPLETENESS_SCALE,
    codebase_scope_scale=CODEBASE_SCOPE_SCALE_TERMINAL,
    error_specificity_scale=ERROR_SPECIFICITY_SCALE_TERMINAL,
    debugging_complexity_scale=DEBUGGING_COMPLEXITY_SCALE,
    output_format=OUTPUT_FORMAT_7_FEATURES_TERMINAL,
)


def format_terminalbench_new_features_prompt(task: Dict[str, Any]) -> str:
    """Format the TerminalBench new features prompt with task-specific information.

    Args:
        task: TerminalBench task dict with keys:
            - task_id: TerminalBench task ID
            - instruction: Task description
            - solution: Reference solution script
            - tags: List of tags

    Returns:
        Formatted prompt string
    """
    # Extract tags as a comma-separated string
    tags: List[str] = task.get("tags", [])
    tags_str = ", ".join(tags) if tags else "None"

    # Get category from first tag if available
    category = tags[0] if tags else "general"

    # Claimed difficulty from tags
    claimed_difficulty = "unknown"
    for tag in tags:
        if tag in ["easy", "medium", "hard", "expert"]:
            claimed_difficulty = tag
            break

    return TERMINALBENCH_NEW_FEATURES_PROMPT_TEMPLATE.format(
        task_id=task.get("task_id", ""),
        category=category,
        tags=tags_str,
        claimed_difficulty=claimed_difficulty,
        instruction=task.get("instruction", ""),
        solution=task.get("solution", ""),
    )


# The main configuration object
TERMINALBENCH_NEW_FEATURES_CONFIG = PromptConfig(
    name="terminalbench_new_features",
    features=NEW_TERMINAL_FEATURES,
    prompt_template=TERMINALBENCH_NEW_FEATURES_PROMPT_TEMPLATE,
    task_id_field="task_id",
    truncation_limits={},  # No truncation needed with Claude Opus 4.5's 200K context
    format_prompt_fn=format_terminalbench_new_features_prompt,
)
