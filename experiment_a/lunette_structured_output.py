"""Lunette structured output support for Experiment A.

This module provides:
1. TaskDifficultyFeatures - Pydantic model defining the structured output schema
2. FeatureExtractionPlan - Custom AnalysisPlan that includes the schema in serialization

Usage:
    from experiment_a.lunette_structured_output import FeatureExtractionPlan, TaskDifficultyFeatures

    plan = FeatureExtractionPlan(
        name="task-difficulty-features",
        prompt=your_prompt,
    )

    results = await client.investigate(run_id=run_id, plan=plan)

    # Results will be structured as TaskDifficultyFeatures
    for result in results.results:
        features = result.data  # Dict matching TaskDifficultyFeatures schema
"""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from lunette.analysis import AnalysisPlanBase


class TaskDifficultyFeatures(BaseModel):
    """Pydantic model for structured output from Lunette task difficulty feature extraction.

    This model defines all the features we want to extract for predicting IRT difficulty.
    The Lunette judge will be forced to return output matching this schema.
    """

    # ===== Environment-based features (from shell commands) =====
    repo_file_count: int = Field(
        ge=0,
        description="Total Python files in the repository"
    )
    repo_line_count: int = Field(
        ge=0,
        description="Total lines of Python code in the repository"
    )
    patch_file_count: int = Field(
        ge=1,
        description="Number of files changed in the gold patch"
    )
    patch_line_count: int = Field(
        ge=1,
        description="Number of lines added/removed in the gold patch"
    )
    test_file_count: int = Field(
        ge=0,
        description="Number of test files in the repository"
    )
    related_file_count: int = Field(
        ge=0,
        description="Number of files in the same directory as patched files"
    )
    import_count: int = Field(
        ge=0,
        description="Number of import statements in the patched file"
    )
    class_count_in_file: int = Field(
        ge=0,
        description="Number of classes defined in the patched file"
    )
    function_count_in_file: int = Field(
        ge=0,
        description="Number of functions defined in the patched file"
    )
    test_count_fail_to_pass: int = Field(
        ge=0,
        description="Number of tests that must pass after the fix (FAIL_TO_PASS)"
    )
    test_count_pass_to_pass: int = Field(
        ge=0,
        description="Number of regression tests (PASS_TO_PASS)"
    )
    git_commit_count: int = Field(
        ge=0,
        description="Number of git commits touching the patched files"
    )
    directory_depth: int = Field(
        ge=1,
        description="Directory nesting level of the patched files"
    )
    has_conftest: Literal[0, 1] = Field(
        description="Whether pytest conftest.py is present in the directory (0 or 1)"
    )
    has_init: Literal[0, 1] = Field(
        description="Whether __init__.py is present in the directory (0 or 1)"
    )

    # ===== Semantic features (LLM judgment) =====
    fix_in_description: int = Field(
        ge=0, le=3,
        description="Does the problem statement hint at the fix? (0=no hint, 1=vague, 2=clear direction, 3=exact fix)"
    )
    problem_clarity: int = Field(
        ge=1, le=5,
        description="How clear is the issue description? (1=very vague, 5=crystal clear with repro steps)"
    )
    error_message_provided: Literal[0, 1] = Field(
        description="Is there an error message or traceback provided? (0 or 1)"
    )
    reproduction_steps: Literal[0, 1] = Field(
        description="Are concrete reproduction steps provided? (0 or 1)"
    )
    fix_locality: int = Field(
        ge=1, le=3,
        description="How localized is the fix? (1=single location, 2=same file, 3=multiple files)"
    )
    domain_knowledge_required: int = Field(
        ge=1, le=5,
        description="Specialized knowledge needed? (1=basic Python, 3=framework, 5=obscure APIs)"
    )
    fix_complexity: int = Field(
        ge=1, le=5,
        description="Complexity of the fix (1=trivial, 3=moderate, 5=very complex)"
    )
    logical_reasoning_required: int = Field(
        ge=1, le=5,
        description="Amount of reasoning needed (1=mechanical, 3=multi-step, 5=deep reasoning)"
    )
    atypicality: int = Field(
        ge=1, le=5,
        description="How unusual is this bug pattern? (1=common, 3=moderate, 5=rare/novel)"
    )

    # ===== Explanation =====
    reasoning: str = Field(
        description="2-3 sentence summary explaining the key difficulty factors for this task"
    )


class SemanticOnlyFeatures(BaseModel):
    """Simplified model with only semantic features (no environment features).

    Useful when you want to skip shell command exploration and focus on
    text-based analysis of the problem statement and patch.
    """

    fix_in_description: int = Field(
        ge=0, le=3,
        description="Does the problem statement hint at the fix? (0=no hint, 1=vague, 2=clear direction, 3=exact fix)"
    )
    problem_clarity: int = Field(
        ge=1, le=5,
        description="How clear is the issue description? (1=very vague, 5=crystal clear)"
    )
    error_message_provided: Literal[0, 1] = Field(
        description="Is there an error message or traceback provided? (0 or 1)"
    )
    reproduction_steps: Literal[0, 1] = Field(
        description="Are concrete reproduction steps provided? (0 or 1)"
    )
    fix_locality: int = Field(
        ge=1, le=3,
        description="How localized is the fix? (1=single location, 2=same file, 3=multiple files)"
    )
    domain_knowledge_required: int = Field(
        ge=1, le=5,
        description="Specialized knowledge needed? (1=basic Python, 5=obscure APIs)"
    )
    fix_complexity: int = Field(
        ge=1, le=5,
        description="Complexity of the fix (1=trivial, 5=very complex)"
    )
    logical_reasoning_required: int = Field(
        ge=1, le=5,
        description="Amount of reasoning needed (1=mechanical, 5=deep reasoning)"
    )
    atypicality: int = Field(
        ge=1, le=5,
        description="How unusual is this bug pattern? (1=common, 5=rare/novel)"
    )
    reasoning: str = Field(
        description="2-3 sentence summary of key difficulty factors"
    )


class FeatureExtractionPlan(AnalysisPlanBase):
    """AnalysisPlan for extracting TaskDifficultyFeatures with structured output.

    Uses lunette-sdk 0.2.3+ which auto-populates result_schema_json from the
    result_schema ClassVar via model_validator. No custom model_dump() needed.

    The Lunette backend uses the schema to force the judge model to output
    structured data matching the schema via tool use.
    """

    kind: Literal["grading"] = "grading"
    result_schema: ClassVar[type[TaskDifficultyFeatures]] = TaskDifficultyFeatures


class SemanticFeatureExtractionPlan(AnalysisPlanBase):
    """AnalysisPlan for extracting SemanticOnlyFeatures with structured output.

    Use this when you want to skip environment exploration and focus
    on semantic analysis of the problem statement and patch.
    """

    kind: Literal["grading"] = "grading"
    result_schema: ClassVar[type[SemanticOnlyFeatures]] = SemanticOnlyFeatures


# Grading prompt template for feature extraction
FEATURE_EXTRACTION_PROMPT = """You are analyzing a SWE-bench coding task to predict its difficulty.
You have FULL ACCESS to the codebase sandbox. Use shell commands to extract precise features.

## TASK INFORMATION

**Instance ID:** {instance_id}
**Repository:** {repo}
**Version:** {version}

**Problem Statement:**
{problem_statement}

**Gold Patch (correct solution):**
```diff
{patch}
```

**Tests that should pass after fix (FAIL_TO_PASS):**
{fail_to_pass}

**Regression tests (PASS_TO_PASS):**
{pass_to_pass}

{hints_section}

## INSTRUCTIONS

1. Use shell commands to explore the codebase and extract environment features:
   - Count Python files, lines, test files
   - Analyze the patched files (imports, classes, functions)
   - Check directory structure (conftest.py, __init__.py, depth)

2. Analyze the problem statement and gold patch to determine semantic features:
   - How clear is the problem description?
   - Does it hint at the solution?
   - How complex is the fix?
   - What domain knowledge is required?

3. Provide your analysis as structured output matching the required schema.
"""


def format_feature_extraction_prompt(
    instance_id: str,
    repo: str,
    version: str,
    problem_statement: str,
    patch: str,
    fail_to_pass: str = "[]",
    pass_to_pass: str = "[]",
    hints_text: str = "",
) -> str:
    """Format the feature extraction prompt with task-specific information."""
    hints_section = ""
    if hints_text and hints_text.strip():
        hints_section = f"**Hints:**\n{hints_text}"

    # Truncate very long fields to avoid context overflow
    problem_statement = problem_statement[:12000] if len(problem_statement) > 12000 else problem_statement
    patch = patch[:8000] if len(patch) > 8000 else patch

    return FEATURE_EXTRACTION_PROMPT.format(
        instance_id=instance_id,
        repo=repo,
        version=version,
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        hints_section=hints_section,
    )
