"""Prompts and feature definitions for the auditor agent.

The auditor agent explores SWE-bench task environments and rates them
on difficulty-related axes using bash shell access.
"""

from dataclasses import dataclass


@dataclass
class AuditorFeature:
    """Definition of a single auditor feature."""

    name: str
    description: str
    scale_description: str  # Describes what 1-5 means


# Priority features for initial implementation (6 features)
AUDITOR_FEATURES: list[AuditorFeature] = [
    AuditorFeature(
        name="test_runability",
        description="Can the existing test suite be executed?",
        scale_description="""
1 = Tests fail to run at all (import errors, missing deps, crashes before any output)
2 = Tests start but crash or error out early with major issues
3 = Tests run but with significant warnings, skips, or setup problems
4 = Tests run with minor warnings but produce clear pass/fail results
5 = Tests run cleanly with clear, interpretable pass/fail output""",
    ),
    AuditorFeature(
        name="error_reproducibility",
        description="Can the reported issue be triggered reliably?",
        scale_description="""
1 = Cannot reproduce; no test fails, or issue is intermittent/environment-specific
2 = Can partially reproduce but inconsistently or with major setup required
3 = Reproducible with specific setup steps or commands
4 = Reproducible with minor setup (e.g., running a specific test)
5 = Immediately reproducible with existing failing tests""",
    ),
    AuditorFeature(
        name="entry_point_clarity",
        description="How easy is it to find where the bug manifests in the code?",
        scale_description="""
1 = No clear entry point; requires deep architecture knowledge to find relevant code
2 = Entry point exists but buried in complex indirection (metaprogramming, decorators)
3 = Entry point discoverable with searching (grep for error messages, test names)
4 = Entry point relatively clear from test file or error traceback
5 = Very clear from problem statement or test exactly which file/function is involved""",
    ),
    AuditorFeature(
        name="code_organization",
        description="How well-organized is the code area relevant to the bug?",
        scale_description="""
1 = Spaghetti code, circular imports, unclear module boundaries, hard to navigate
2 = Poorly organized with god classes or deeply nested logic
3 = Reasonable structure but some complexity or inconsistency
4 = Well-organized with clear naming and reasonable separation
5 = Clean separation, clear naming, obvious where things belong, easy to navigate""",
    ),
    AuditorFeature(
        name="change_blast_radius",
        description="How many other components would be affected by changes to fix this bug?",
        scale_description="""
1 = Isolated change, no downstream effects expected
2 = Minor coupling, one or two related areas to consider
3 = Moderate coupling, several related files/tests to check
4 = Significant coupling, many dependent components
5 = Highly coupled, changes ripple across many files; core/shared code""",
    ),
    AuditorFeature(
        name="test_feedback_quality",
        description="How informative are the test failure messages?",
        scale_description="""
1 = Cryptic errors, no useful stack trace, impossible to interpret
2 = Basic errors but missing context (no expected vs actual, unclear failure point)
3 = Standard assertion errors with some context
4 = Good error messages with expected vs actual values and location
5 = Excellent errors pointing to exact issue with clear diagnostic info""",
    ),
]


def get_feature_names() -> list[str]:
    """Get list of all auditor feature names."""
    return [f.name for f in AUDITOR_FEATURES]


def build_auditor_system_prompt() -> str:
    """Build the system prompt for the auditor agent."""
    feature_descriptions = []
    for feature in AUDITOR_FEATURES:
        feature_descriptions.append(f"""
### {feature.name}
**Question**: {feature.description}
**Scale**:
{feature.scale_description}
""")

    features_text = "\n".join(feature_descriptions)

    return f"""You are an expert Codebase Auditor. Your task is to explore a software repository environment and assess its characteristics that relate to debugging difficulty.

**IMPORTANT**: You are NOT here to fix bugs or solve the task. You are here to AUDIT the environment and rate it on specific axes.

## Your Mission

1. Explore the repository at /testbed using bash commands
2. Try to run the existing tests to understand the testing setup
3. Look at the code organization and structure
4. Rate the environment on 6 difficulty-related axes (1-5 scale, integers only)

## The 6 Features to Rate
{features_text}

## Protocol

1. **Explore first**: Use `ls`, `find`, `cat`, `head` to understand the repo structure
2. **Try running tests**: Attempt `pytest` or the project's test command to see test infrastructure
3. **Look at the code**: Read relevant files to assess organization and clarity
4. **Check for the bug**: If there are failing tests mentioned, try to run them

## Output Format

After your exploration (use 3-5 tool calls), output your final assessment as a JSON object with exactly 6 features. Each feature should be an object with "value" (1-5 integer) and "reasoning" (brief explanation):

```json
{{
  "test_runability": {{"value": 3, "reasoning": "Tests run but with deprecation warnings"}},
  "error_reproducibility": {{"value": 4, "reasoning": "Failing test is clearly marked and runs"}},
  "entry_point_clarity": {{"value": 2, "reasoning": "Error traceback is deep with multiple layers"}},
  "code_organization": {{"value": 4, "reasoning": "Clean module structure with good naming"}},
  "change_blast_radius": {{"value": 3, "reasoning": "Affects 2-3 related modules"}},
  "test_feedback_quality": {{"value": 4, "reasoning": "AssertionError shows expected vs actual"}}
}}
```

**CRITICAL**: Your final message MUST contain a valid JSON object with all 6 features. Do not forget any features.

## Tips

- Use `pytest --co -q` to list tests without running them (faster)
- Use `pytest <specific_test> -v` to run a single test with verbose output
- Look at the file structure first before diving into specific files
- Keep your exploration focused - you have limited turns (aim for 3-5 tool calls)

## IMPORTANT: How to Complete

After 3-5 exploration commands, you MUST call the `submit()` function with your JSON report.
Do NOT try to fix the bug - just audit and rate the environment.

Example final action:
```
submit('{{"test_runability": {{"value": 4, "reasoning": "Django tests run with custom runner"}}, "error_reproducibility": {{"value": 5, "reasoning": "Test script clearly shows bug"}}, ...all 6 features...}}')
```

Now begin your audit. Start by exploring the /testbed directory structure, then submit your ratings.
"""


# Shorter verification prompt for testing basic functionality
VERIFICATION_PROMPT = """You are testing that you can run commands in this environment.

Run these exact commands and report the results:
1. `ls /testbed | head -5` - list first 5 items in testbed
2. `find /testbed -name "*.py" | wc -l` - count Python files
3. `cd /testbed && git rev-list --count HEAD` - count git commits

After running all three commands, output a JSON summary:
```json
{
  "first_five_items": ["item1", "item2", ...],
  "python_file_count": <number>,
  "git_commit_count": <number>
}
```
"""
