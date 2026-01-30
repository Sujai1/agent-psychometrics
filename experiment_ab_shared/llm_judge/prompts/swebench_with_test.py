"""SWE-bench Verified prompt configuration with test patch features.

This module defines a prompt that includes the test patch for extracting
semantic features about test quality and alignment with the problem/solution.

Features (7 total):
- 3 test-only features: comprehensiveness, assertion complexity, edge case coverage
- 2 test-problem alignment: test_problem_alignment, expected_behavior_clarity
- 2 test-solution alignment: test_solution_coverage, test_sufficient_for_solution

Use case: Experiment A difficulty prediction using test patch information.
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig


# =============================================================================
# Feature Definitions (7 features)
# =============================================================================

# Group 1: Test-Only Features (3)
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

# Group 2: Test-Problem Alignment (2)
TEST_PROBLEM_ALIGNMENT = FeatureDefinition(
    name="test_problem_alignment",
    min_value=1,
    max_value=5,
    description="Does the test verify what's described in the problem statement? (1=mismatch, 5=exact match)",
)

EXPECTED_BEHAVIOR_CLARITY = FeatureDefinition(
    name="expected_behavior_clarity",
    min_value=1,
    max_value=5,
    description="How clear is the expected behavior from the test? (1=ambiguous, 5=crystal clear)",
)

# Group 3: Test-Solution Alignment (2)
TEST_SOLUTION_COVERAGE = FeatureDefinition(
    name="test_solution_coverage",
    min_value=1,
    max_value=5,
    description="What fraction of the solution changes are verified by tests? (1=none, 5=all changes tested)",
)

TEST_SUFFICIENT_FOR_SOLUTION = FeatureDefinition(
    name="test_sufficient_for_solution",
    min_value=0,
    max_value=1,
    description="Would passing these tests guarantee a correct solution? (0=no, 1=yes)",
)

# All features
SWEBENCH_WITH_TEST_FEATURES = [
    TEST_COMPREHENSIVENESS,
    TEST_ASSERTION_COMPLEXITY,
    TEST_EDGE_CASE_COVERAGE,
    TEST_PROBLEM_ALIGNMENT,
    EXPECTED_BEHAVIOR_CLARITY,
    TEST_SOLUTION_COVERAGE,
    TEST_SUFFICIENT_FOR_SOLUTION,
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

TEST_PROBLEM_ALIGNMENT_SCALE = """### Test-Problem Alignment (test_problem_alignment: 1-5)
Does the test verify exactly what's described in the problem statement?
- 1: Mismatch - test seems unrelated to the problem description
- 2: Weak alignment - test partially addresses the problem
- 3: Moderate - test covers the main issue but not specifics mentioned
- 4: Good - test closely matches problem description
- 5: Exact match - test directly verifies all aspects of the problem"""

EXPECTED_BEHAVIOR_CLARITY_SCALE = """### Expected Behavior Clarity (expected_behavior_clarity: 1-5)
How clearly does the test communicate what the correct behavior should be?
- 1: Ambiguous - hard to understand what correct behavior is from the test
- 2: Unclear - requires significant inference
- 3: Moderate - some understanding needed but generally clear
- 4: Clear - expected behavior is evident from test structure
- 5: Crystal clear - test explicitly documents expected behavior"""

TEST_SOLUTION_COVERAGE_SCALE = """### Test-Solution Coverage (test_solution_coverage: 1-5)
What fraction of the gold patch's changes are verified by the test?
- 1: None - test doesn't exercise the changed code
- 2: Minimal - tests only a small part of the changes
- 3: Partial - tests about half of the significant changes
- 4: Most - tests most of the changes made
- 5: Complete - all meaningful changes are exercised by tests"""

TEST_SUFFICIENT_SCALE = """### Test Sufficient for Solution (test_sufficient_for_solution: 0/1)
Would passing these tests guarantee a correct, complete solution?
- 0: No - passing tests might still allow incorrect or incomplete solutions
- 1: Yes - tests are comprehensive enough that passing them implies correctness"""


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
    "test_problem_alignment": <1-5>,
    "expected_behavior_clarity": <1-5>,
    "test_solution_coverage": <1-5>,
    "test_sufficient_for_solution": <0 or 1>,
    "reasoning": "<2-3 sentence summary of the test quality and alignment>"
}}"""


SWEBENCH_WITH_TEST_PROMPT_TEMPLATE = """You are analyzing a SWE-bench coding task to evaluate the TEST PATCH quality and alignment.
This is a BUG FIX task in a Python repository. Focus on the relationship between the test, problem, and solution.

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

**Regression tests (PASS_TO_PASS):**
{{pass_to_pass}}

{{hints_section}}

## FEATURES TO EVALUATE

Analyze the test patch in relation to the problem statement and gold patch.
Focus on test quality, coverage, and alignment with the intended fix.

{test_comprehensiveness_scale}

{test_assertion_complexity_scale}

{test_edge_case_coverage_scale}

{test_problem_alignment_scale}

{expected_behavior_clarity_scale}

{test_solution_coverage_scale}

{test_sufficient_scale}

{output_format}
""".format(
    completeness_instruction=COMPLETENESS_INSTRUCTION,
    test_comprehensiveness_scale=TEST_COMPREHENSIVENESS_SCALE,
    test_assertion_complexity_scale=TEST_ASSERTION_COMPLEXITY_SCALE,
    test_edge_case_coverage_scale=TEST_EDGE_CASE_COVERAGE_SCALE,
    test_problem_alignment_scale=TEST_PROBLEM_ALIGNMENT_SCALE,
    expected_behavior_clarity_scale=EXPECTED_BEHAVIOR_CLARITY_SCALE,
    test_solution_coverage_scale=TEST_SOLUTION_COVERAGE_SCALE,
    test_sufficient_scale=TEST_SUFFICIENT_SCALE,
    output_format=OUTPUT_FORMAT,
)


def format_swebench_with_test_prompt(task: Dict[str, Any]) -> str:
    """Format the SWE-bench prompt with test patch included.

    Args:
        task: SWE-bench task dict with keys:
            - instance_id: SWE-bench instance ID
            - repo: Repository name (e.g., "django/django")
            - version: Version string
            - problem_statement: The issue description
            - patch: The gold solution patch
            - test_patch: The test patch diff
            - FAIL_TO_PASS: Tests that should pass after fix
            - PASS_TO_PASS: Regression tests
            - hints_text: Optional hints (may be empty)

    Returns:
        Formatted prompt string
    """
    hints_text = task.get("hints_text", "")
    hints_section = ""
    if hints_text and hints_text.strip():
        hints_section = f"**Hints:**\n{hints_text}"

    # No truncation - Claude Opus 4.5 has 200K context
    problem_statement = task.get("problem_statement", "")
    patch = task.get("patch", "")
    test_patch = task.get("test_patch", "")

    return SWEBENCH_WITH_TEST_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        version=task.get("version", "unknown"),
        problem_statement=problem_statement,
        patch=patch,
        test_patch=test_patch if test_patch else "[No test patch available]",
        fail_to_pass=task.get("FAIL_TO_PASS", "[]"),
        pass_to_pass=task.get("PASS_TO_PASS", "[]"),
        hints_section=hints_section,
    )


# =============================================================================
# Configuration Object
# =============================================================================

SWEBENCH_WITH_TEST_CONFIG = PromptConfig(
    name="swebench_with_test",
    features=SWEBENCH_WITH_TEST_FEATURES,
    prompt_template=SWEBENCH_WITH_TEST_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={},  # No truncation - Claude Opus 4.5 has 200K context
    format_prompt_fn=format_swebench_with_test_prompt,
)
