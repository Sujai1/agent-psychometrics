"""SWE-bench test quality features - trimmed to 3 significant features.

This module defines a prompt with only the 3 test patch features that showed
significant correlation with task difficulty in the 50-task pilot:
- test_assertion_complexity (r=0.355, p=0.0114)
- test_edge_case_coverage (r=0.346, p=0.0138)
- test_comprehensiveness (r=0.324, p=0.0216)

Use case: Experiment A difficulty prediction using test quality signals.
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig


# =============================================================================
# Feature Definitions (3 features - significant correlations only)
# =============================================================================

TEST_COMPREHENSIVENESS = FeatureDefinition(
    name="test_comprehensiveness",
    min_value=1,
    max_value=5,
    description="How thoroughly does the test cover the expected behavior? (1=minimal, 5=exhaustive)",
)

TEST_ASSERTION_COMPLEXITY = FeatureDefinition(
    name="test_assertion_complexity",
    min_value=1,
    max_value=5,
    description="How complex are the test assertions? (1=simple equality, 5=complex logic/mocking)",
)

TEST_EDGE_CASE_COVERAGE = FeatureDefinition(
    name="test_edge_case_coverage",
    min_value=1,
    max_value=5,
    description="Does the test cover edge cases and boundary conditions? (1=happy path only, 5=thorough)",
)

# All features (only the 3 significant ones)
SWEBENCH_TEST_QUALITY_FEATURES = [
    TEST_COMPREHENSIVENESS,
    TEST_ASSERTION_COMPLEXITY,
    TEST_EDGE_CASE_COVERAGE,
]


# =============================================================================
# Scale Descriptions
# =============================================================================

TEST_COMPREHENSIVENESS_SCALE = """### Test Comprehensiveness (test_comprehensiveness: 1-5)
How thoroughly does the test patch cover the expected behavior?
- 1: Minimal - tests only one basic case
- 2: Limited - tests a few cases but misses important scenarios
- 3: Moderate - covers main functionality with some gaps
- 4: Good - covers most expected behaviors and variations
- 5: Exhaustive - comprehensive coverage including corner cases"""

TEST_ASSERTION_COMPLEXITY_SCALE = """### Test Assertion Complexity (test_assertion_complexity: 1-5)
How complex are the assertions and test setup in the test patch?
- 1: Simple - basic equality checks (assertEqual, assertTrue)
- 2: Standard - uses common assertion patterns
- 3: Moderate - multiple assertions, some setup required
- 4: Complex - requires mocking, fixtures, or intricate setup
- 5: Very complex - extensive mocking, async testing, or multi-step verification"""

TEST_EDGE_CASE_COVERAGE_SCALE = """### Test Edge Case Coverage (test_edge_case_coverage: 1-5)
Does the test cover edge cases, boundary conditions, and error scenarios?
- 1: Happy path only - no edge cases tested
- 2: Minimal - one or two edge cases
- 3: Moderate - some boundary conditions checked
- 4: Good - most edge cases and error conditions tested
- 5: Thorough - comprehensive edge case and error handling coverage"""


# =============================================================================
# Prompt Template
# =============================================================================

COMPLETENESS_INSTRUCTION = """
CRITICAL: You MUST provide a value for EVERY feature listed below.
Do not skip any features. If uncertain, provide your best estimate.
Missing values will cause extraction to fail.
"""

OUTPUT_FORMAT = """## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "test_comprehensiveness": <1-5>,
    "test_assertion_complexity": <1-5>,
    "test_edge_case_coverage": <1-5>,
    "reasoning": "<1-2 sentence summary of the test quality>"
}}"""


SWEBENCH_TEST_QUALITY_PROMPT_TEMPLATE = """You are analyzing the TEST PATCH quality for a SWE-bench coding task.
This is a BUG FIX task in a Python repository. Focus on evaluating the test quality.

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

**Test Patch (tests that verify the fix):**
```diff
{{test_patch}}
```

**Tests that should pass after fix (FAIL_TO_PASS):**
{{fail_to_pass}}

## FEATURES TO EVALUATE

Analyze the test patch quality - how well does it verify the intended fix?

{test_comprehensiveness_scale}

{test_assertion_complexity_scale}

{test_edge_case_coverage_scale}

{output_format}
""".format(
    completeness_instruction=COMPLETENESS_INSTRUCTION,
    test_comprehensiveness_scale=TEST_COMPREHENSIVENESS_SCALE,
    test_assertion_complexity_scale=TEST_ASSERTION_COMPLEXITY_SCALE,
    test_edge_case_coverage_scale=TEST_EDGE_CASE_COVERAGE_SCALE,
    output_format=OUTPUT_FORMAT,
)


def format_swebench_test_quality_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench test quality prompt.

    Args:
        task: SWE-bench task dict with keys:
            - instance_id: SWE-bench instance ID
            - repo: Repository name (e.g., "django/django")
            - version: Version string
            - problem_statement: The issue description
            - patch: The gold solution patch
            - test_patch: The test patch diff
            - FAIL_TO_PASS: Tests that should pass after fix

    Returns:
        Formatted prompt string
    """
    # No truncation - Claude Opus 4.5 has 200K context
    problem_statement = task.get("problem_statement", "")
    patch = task.get("patch", "")
    test_patch = task.get("test_patch", "")

    return SWEBENCH_TEST_QUALITY_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=problem_statement,
        patch=patch,
        test_patch=test_patch if test_patch else "[No test patch available]",
        fail_to_pass=task.get("FAIL_TO_PASS", "[]"),
    )


# =============================================================================
# Configuration Object
# =============================================================================

SWEBENCH_TEST_QUALITY_CONFIG = PromptConfig(
    name="swebench_test_quality",
    features=SWEBENCH_TEST_QUALITY_FEATURES,
    prompt_template=SWEBENCH_TEST_QUALITY_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={},  # No truncation
    format_prompt_fn=format_swebench_test_quality_prompt,
)
