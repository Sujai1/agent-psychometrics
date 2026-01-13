"""Lunette grading prompt for task difficulty feature extraction.

This prompt instructs the Lunette judge to:
1. Run shell commands to explore the codebase (environment access)
2. Extract semantic features from the problem statement and gold patch
3. Output structured JSON with all features

Features are designed for predicting IRT difficulty in Experiment A.
"""

# Feature names and their expected ranges
LUNETTE_FEATURE_NAMES = [
    # Environment-based features (from shell commands)
    "repo_file_count",           # int: Total Python files in repo
    "repo_line_count",           # int: Total lines of Python code
    "patch_file_count",          # int: Files changed in gold patch
    "patch_line_count",          # int: Lines added/removed in patch
    "test_file_count",           # int: Number of test files
    "related_file_count",        # int: Files in patched directories
    "import_count",              # int: Import statements in patched file
    "class_count_in_file",       # int: Classes in patched file
    "function_count_in_file",    # int: Functions in patched file
    "test_count_fail_to_pass",   # int: Number of tests that must pass
    "test_count_pass_to_pass",   # int: Number of regression tests
    "git_commit_count",          # int: Commits touching patched files
    "directory_depth",           # int: Nesting level of patched files
    "has_conftest",              # 0/1: pytest fixtures present
    "has_init",                  # 0/1: __init__.py present in directory
    # Semantic features (LLM judgment)
    "fix_in_description",        # 0-3: Does problem statement hint at fix?
    "problem_clarity",           # 1-5: How clear is the issue description?
    "error_message_provided",    # 0/1: Is there a traceback/error?
    "reproduction_steps",        # 0/1: Are repro steps provided?
    "fix_locality",              # 1-3: How localized is the change?
    "domain_knowledge_required", # 1-5: Specialized knowledge needed?
    "fix_complexity",            # 1-5: Complexity of the actual fix
    "logical_reasoning_required",# 1-5: Amount of reasoning needed
    "atypicality",               # 1-5: How unusual is this bug pattern?
]

# The main grading prompt
THOROUGH_GRADING_PROMPT = """You are analyzing a SWE-bench coding task to predict its difficulty.
You have FULL ACCESS to the codebase sandbox. Use this to extract precise features.

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

## STEP 1: Environment Exploration

Run these commands to extract EXACT numbers. Report the actual values.

### 1.1 Repository Structure
```bash
# Total Python files
find . -name "*.py" -type f 2>/dev/null | wc -l

# Total lines of Python code (approximate)
find . -name "*.py" -type f -exec cat {{}} \\; 2>/dev/null | wc -l

# Test files
find . -name "test_*.py" -o -name "*_test.py" 2>/dev/null | wc -l
```

### 1.2 Patch Analysis
For each file in the gold patch, run:
```bash
# File size
wc -l <patched_file>

# Classes and functions
grep -c "^class " <patched_file> 2>/dev/null || echo 0
grep -c "^def " <patched_file> 2>/dev/null || echo 0
grep -c "^    def " <patched_file> 2>/dev/null || echo 0

# Imports
grep -c "^import\\|^from" <patched_file> 2>/dev/null || echo 0

# Git history (if available)
git log --oneline <patched_file> 2>/dev/null | wc -l
```

### 1.3 Directory Analysis
```bash
# Files in the same directory as patched files
ls -la <patch_directory>/ | wc -l

# Check for pytest fixtures
test -f <patch_directory>/conftest.py && echo "1" || echo "0"

# Check for __init__.py
test -f <patch_directory>/__init__.py && echo "1" || echo "0"

# Directory depth (count slashes in path)
echo "<patched_file_path>" | tr -cd '/' | wc -c
```

## STEP 2: Semantic Analysis

Based on the problem statement and gold patch, evaluate these features:

### 2.1 Fix Information in Description (0-3)
- 0: No hint at the solution at all
- 1: Vague hint or general direction
- 2: Clear description of what needs to change
- 3: Exact code fix or detailed solution provided

### 2.2 Problem Clarity (1-5)
- 1: Very vague, unclear what's actually wrong
- 2: Somewhat clear but missing key details
- 3: Reasonably clear, some ambiguity
- 4: Clear with good context
- 5: Crystal clear with reproduction steps and expected behavior

### 2.3 Error Message/Traceback (0/1)
- 0: No error message provided
- 1: Error message, traceback, or exception shown

### 2.4 Reproduction Steps (0/1)
- 0: No concrete reproduction steps
- 1: Has reproduction steps (code snippet, test case, or commands)

### 2.5 Fix Locality (1-3)
- 1: Single location, few lines changed (1-5 lines)
- 2: Multiple locations in same file, or moderate changes (6-20 lines)
- 3: Multiple files or significant changes (>20 lines)

### 2.6 Domain Knowledge Required (1-5)
- 1: Basic Python, obvious fix anyone could make
- 2: Standard library knowledge needed
- 3: Framework-specific knowledge (Django, pytest, numpy, etc.)
- 4: Deep understanding of the library's internals
- 5: Obscure APIs, protocols, or highly specialized domain knowledge

### 2.7 Fix Complexity (1-5)
- 1: Trivial (add parameter, change value, simple one-liner)
- 2: Simple (straightforward logic change)
- 3: Moderate (requires understanding context, multiple changes)
- 4: Complex (algorithmic changes, multiple interdependent fixes)
- 5: Very complex (architectural changes, subtle edge cases, tricky bugs)

### 2.8 Logical Reasoning Required (1-5)
- 1: Mechanical fix, no reasoning needed
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about edge cases, invariants, or system behavior

### 2.9 Atypicality (1-5)
- 1: Very common bug pattern (typo, off-by-one, missing null check)
- 2: Common pattern (incorrect condition, wrong default)
- 3: Moderately unusual
- 4: Unusual bug pattern
- 5: Rare or novel bug pattern

## OUTPUT FORMAT

Respond with ONLY a JSON object containing ALL features. No markdown, no explanation outside JSON.

{{
    "repo_file_count": <int>,
    "repo_line_count": <int>,
    "patch_file_count": <int>,
    "patch_line_count": <int>,
    "test_file_count": <int>,
    "related_file_count": <int>,
    "import_count": <int>,
    "class_count_in_file": <int>,
    "function_count_in_file": <int>,
    "test_count_fail_to_pass": <int>,
    "test_count_pass_to_pass": <int>,
    "git_commit_count": <int>,
    "directory_depth": <int>,
    "has_conftest": <0 or 1>,
    "has_init": <0 or 1>,
    "fix_in_description": <0-3>,
    "problem_clarity": <1-5>,
    "error_message_provided": <0 or 1>,
    "reproduction_steps": <0 or 1>,
    "fix_locality": <1-3>,
    "domain_knowledge_required": <1-5>,
    "fix_complexity": <1-5>,
    "logical_reasoning_required": <1-5>,
    "atypicality": <1-5>,
    "reasoning": "<2-3 sentence summary of key difficulty factors>"
}}
"""


def format_grading_prompt(
    instance_id: str,
    repo: str,
    version: str,
    problem_statement: str,
    patch: str,
    fail_to_pass: str,
    pass_to_pass: str,
    hints_text: str = "",
) -> str:
    """Format the grading prompt with task-specific information.

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
    problem_statement = problem_statement[:12000] if len(problem_statement) > 12000 else problem_statement
    patch = patch[:8000] if len(patch) > 8000 else patch

    return THOROUGH_GRADING_PROMPT.format(
        instance_id=instance_id,
        repo=repo,
        version=version,
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        hints_section=hints_section,
    )
