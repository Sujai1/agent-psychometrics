"""LLM judge prompt for semantic feature extraction (Experiment A).

This prompt extracts only the 9 semantic features from static task information
(problem statement, gold patch, tests, hints) WITHOUT sandbox/environment access.
This is the "ablation" version of Lunette grading that doesn't use shell commands.

The 9 semantic features match those in lunette_grading_prompt.py:
- fix_in_description (0-3)
- problem_clarity (1-5)
- error_message_provided (0/1)
- reproduction_steps (0/1)
- fix_locality (1-3)
- domain_knowledge_required (1-5)
- fix_complexity (1-5)
- logical_reasoning_required (1-5)
- atypicality (1-5)
"""

# Feature names for semantic-only extraction
LLM_JUDGE_SEMANTIC_FEATURES = [
    "fix_in_description",
    "problem_clarity",
    "error_message_provided",
    "reproduction_steps",
    "fix_locality",
    "domain_knowledge_required",
    "fix_complexity",
    "logical_reasoning_required",
    "atypicality",
]

# The semantic-only grading prompt (no shell commands)
LLM_JUDGE_PROMPT = """You are analyzing a SWE-bench coding task to predict its difficulty.
You will analyze ONLY the static task information (no code execution or environment access).

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

## FEATURES TO EVALUATE

Analyze the problem statement and gold patch to evaluate these 9 semantic features.
Be precise and consistent with your ratings.

### 1. Fix Information in Description (fix_in_description: 0-3)
Does the problem statement contain or hint at the solution?
- 0: No hint at the solution at all
- 1: Vague hint or general direction
- 2: Clear description of what needs to change
- 3: Exact code fix or detailed solution provided

### 2. Problem Clarity (problem_clarity: 1-5)
How clear and well-specified is the problem?
- 1: Very vague, unclear what's actually wrong
- 2: Somewhat clear but missing key details
- 3: Reasonably clear, some ambiguity
- 4: Clear with good context
- 5: Crystal clear with reproduction steps and expected behavior

### 3. Error Message/Traceback (error_message_provided: 0/1)
Does the problem include an error message or traceback?
- 0: No error message provided
- 1: Error message, traceback, or exception shown

### 4. Reproduction Steps (reproduction_steps: 0/1)
Are concrete reproduction steps provided?
- 0: No concrete reproduction steps
- 1: Has reproduction steps (code snippet, test case, or commands)

### 5. Fix Locality (fix_locality: 1-3)
How localized is the fix based on the patch?
- 1: Single location, few lines changed (1-5 lines)
- 2: Multiple locations in same file, or moderate changes (6-20 lines)
- 3: Multiple files or significant changes (>20 lines)

### 6. Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed to understand and fix this?
- 1: Basic Python, obvious fix anyone could make
- 2: Standard library knowledge needed
- 3: Framework-specific knowledge (Django, pytest, numpy, etc.)
- 4: Deep understanding of the library's internals
- 5: Obscure APIs, protocols, or highly specialized domain knowledge

### 7. Fix Complexity (fix_complexity: 1-5)
How complex is the actual fix?
- 1: Trivial (add parameter, change value, simple one-liner)
- 2: Simple (straightforward logic change)
- 3: Moderate (requires understanding context, multiple changes)
- 4: Complex (algorithmic changes, multiple interdependent fixes)
- 5: Very complex (architectural changes, subtle edge cases, tricky bugs)

### 8. Logical Reasoning Required (logical_reasoning_required: 1-5)
How much logical reasoning is needed to arrive at the fix?
- 1: Mechanical fix, no reasoning needed
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about edge cases, invariants, or system behavior

### 9. Atypicality (atypicality: 1-5)
How unusual is this bug pattern?
- 1: Very common bug pattern (typo, off-by-one, missing null check)
- 2: Common pattern (incorrect condition, wrong default)
- 3: Moderately unusual
- 4: Unusual bug pattern
- 5: Rare or novel bug pattern

## OUTPUT FORMAT

Respond with ONLY a JSON object containing all features. No markdown, no extra text.

{{
    "fix_in_description": <0-3>,
    "problem_clarity": <1-5>,
    "error_message_provided": <0 or 1>,
    "reproduction_steps": <0 or 1>,
    "fix_locality": <1-3>,
    "domain_knowledge_required": <1-5>,
    "fix_complexity": <1-5>,
    "logical_reasoning_required": <1-5>,
    "atypicality": <1-5>,
    "reasoning": "<2-3 sentence summary of the key difficulty factors>"
}}
"""


def format_llm_judge_prompt(
    instance_id: str,
    repo: str,
    version: str,
    problem_statement: str,
    patch: str,
    fail_to_pass: str,
    pass_to_pass: str,
    hints_text: str = "",
) -> str:
    """Format the LLM judge prompt with task-specific information.

    Args:
        instance_id: SWE-bench instance ID
        repo: Repository name (e.g., "django/django")
        version: Version string
        problem_statement: The issue description
        patch: The gold solution patch
        fail_to_pass: Tests that should pass after fix
        pass_to_pass: Regression tests
        hints_text: Optional hints

    Returns:
        Formatted prompt string
    """
    hints_section = ""
    if hints_text and hints_text.strip():
        hints_section = f"**Hints:**\n{hints_text}"

    # Truncate very long fields to avoid context overflow
    problem_statement = (
        problem_statement[:12000] if len(problem_statement) > 12000 else problem_statement
    )
    patch = patch[:8000] if len(patch) > 8000 else patch

    return LLM_JUDGE_PROMPT.format(
        instance_id=instance_id,
        repo=repo,
        version=version,
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        hints_section=hints_section,
    )
