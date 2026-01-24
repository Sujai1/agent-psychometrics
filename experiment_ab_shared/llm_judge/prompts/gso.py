"""GSO (Software Optimization Benchmark) prompt configuration for LLM judge feature extraction.

This module defines the prompt template and feature definitions for extracting
semantic features from GSO optimization tasks.

GSO is fundamentally different from SWE-bench:
- Tasks are performance OPTIMIZATIONS, not bug fixes
- Input: prob_script (test script showing performance scenario)
- Output: gt_diff (optimization patch)
- Success: achieving commit-level optimization

Features are adapted from SWE-bench Verified, Pro V5, and TerminalBench,
with descriptions adjusted for the optimization context.
"""

from typing import Any, Dict

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig

# Feature definitions for GSO - adapted from SWE-bench/TerminalBench
GSO_FEATURES = [
    FeatureDefinition(
        name="solution_in_problem",
        min_value=0,
        max_value=3,
        description="Does the test script hint at the optimization approach? (0=none, 3=clear optimization strategy)",
    ),
    FeatureDefinition(
        name="problem_clarity",
        min_value=1,
        max_value=5,
        description="How clear is what needs optimization? (1=vague bottleneck, 5=obvious target)",
    ),
    FeatureDefinition(
        name="fix_complexity",
        min_value=1,
        max_value=5,
        description="How complex is the optimization logic? (1=simple caching, 5=algorithmic redesign)",
    ),
    FeatureDefinition(
        name="verification_difficulty",
        min_value=1,
        max_value=5,
        description="How hard to verify the optimization is correct and faster? (1=trivial, 5=subtle correctness issues)",
    ),
    FeatureDefinition(
        name="standard_pattern_available",
        min_value=0,
        max_value=1,
        description="Is this a well-documented optimization pattern? (0=novel approach needed, 1=known pattern)",
    ),
    FeatureDefinition(
        name="integration_complexity",
        min_value=1,
        max_value=5,
        description="How integrated with existing code? (1=isolated function, 5=system-wide changes)",
    ),
    FeatureDefinition(
        name="domain_knowledge_required",
        min_value=1,
        max_value=5,
        description="How much library-specific knowledge needed? (1=basic Python, 5=deep library internals)",
    ),
    FeatureDefinition(
        name="atypicality",
        min_value=1,
        max_value=5,
        description="How unusual is this optimization scenario? (1=common pattern like vectorization, 5=rare/novel)",
    ),
]

# The prompt template for GSO tasks
GSO_PROMPT_TEMPLATE = """You are analyzing a GSO (Software Optimization Benchmark) task to predict its difficulty.
This is a PERFORMANCE OPTIMIZATION task, not a bug fix. You will analyze ONLY the static task information.

## TASK INFORMATION

**Instance ID:** {instance_id}
**Repository:** {repo}
**API/Function being optimized:** {api}

**Test Script (performance scenario to optimize):**
```python
{prob_script}
```

**Gold Patch (optimization solution):**
```diff
{gt_diff}
```

{hints_section}

## FEATURES TO EVALUATE

Analyze the test script and optimization patch to evaluate these 8 semantic features.
Focus on what makes the OPTIMIZATION hard, not just what the code looks like.

### 1. Solution Hint in Problem (solution_in_problem: 0-3)
Does the test script hint at the optimization approach?
- 0: No hint at optimization strategy
- 1: Vague hint (e.g., "this is slow")
- 2: Points to what needs optimization (e.g., specific function)
- 3: Clear optimization strategy suggested

### 2. Problem Clarity (problem_clarity: 1-5)
How clear is what needs optimization?
- 1: Very vague, unclear what's the bottleneck
- 2: Somewhat clear target but unclear approach
- 3: Reasonably clear optimization target
- 4: Clear bottleneck with good context
- 5: Obvious optimization target and clear metrics

### 3. Optimization Complexity (fix_complexity: 1-5)
How complex is the optimization logic itself?
- 1: Simple (add caching, use built-in function)
- 2: Standard (vectorization, batch processing)
- 3: Moderate (algorithm improvements, memory optimization)
- 4: Complex (significant algorithmic changes)
- 5: Very complex (architectural redesign, low-level optimization)

### 4. Verification Difficulty (verification_difficulty: 1-5)
How hard is it to verify the optimization is correct AND faster?
- 1: Trivial (obvious correctness, easy to benchmark)
- 2: Easy (straightforward test cases)
- 3: Moderate (some edge cases to consider)
- 4: Hard (subtle correctness issues, complex benchmarking)
- 5: Very hard (race conditions, platform-specific behavior)

### 5. Standard Pattern Available (standard_pattern_available: 0/1)
Is this a well-documented optimization pattern?
- 0: Novel optimization needed, no clear pattern to follow
- 1: Well-documented pattern (e.g., "use numpy instead of loops", "add LRU cache", "use SIMD")

### 6. Integration Complexity (integration_complexity: 1-5)
How tightly integrated with existing code?
- 1: Isolated function, self-contained optimization
- 2: Extends existing code with clear interface
- 3: Touches multiple components
- 4: Requires understanding multiple subsystems
- 5: System-wide changes, many touchpoints

### 7. Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed?
- 1: Basic Python performance (list comprehensions, generators)
- 2: Standard library optimization patterns
- 3: Library-specific knowledge (numpy, pandas internals)
- 4: Deep understanding of library implementation
- 5: Expert knowledge (SIMD, memory layout, CPU caches)

### 8. Atypicality (atypicality: 1-5)
How unusual is this optimization scenario?
- 1: Very common (vectorization, caching, batching)
- 2: Common pattern in this library
- 3: Moderately unusual
- 4: Unusual optimization scenario
- 5: Rare or novel optimization pattern

## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "solution_in_problem": <0-3>,
    "problem_clarity": <1-5>,
    "fix_complexity": <1-5>,
    "verification_difficulty": <1-5>,
    "standard_pattern_available": <0 or 1>,
    "integration_complexity": <1-5>,
    "domain_knowledge_required": <1-5>,
    "atypicality": <1-5>,
    "reasoning": "<2-3 sentences on what makes the OPTIMIZATION hard or easy>"
}}
"""


def format_gso_prompt(task: Dict[str, Any]) -> str:
    """Format the GSO prompt with task-specific information.

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

    # Truncate very long fields to avoid context overflow
    prob_script = task.get("prob_script", "") or ""
    if len(prob_script) > 12000:
        prob_script = prob_script[:12000] + "\n... [truncated]"

    gt_diff = task.get("gt_diff", "") or ""
    if len(gt_diff) > 8000:
        gt_diff = gt_diff[:8000] + "\n... [truncated]"

    return GSO_PROMPT_TEMPLATE.format(
        instance_id=task.get("instance_id", ""),
        repo=task.get("repo", ""),
        api=task.get("api", "unknown"),
        prob_script=prob_script,
        gt_diff=gt_diff,
        hints_section=hints_section,
    )


# The main configuration object
GSO_CONFIG = PromptConfig(
    name="gso",
    features=GSO_FEATURES,
    prompt_template=GSO_PROMPT_TEMPLATE,
    task_id_field="instance_id",
    truncation_limits={
        "prob_script": 12000,
        "gt_diff": 8000,
    },
    format_prompt_fn=format_gso_prompt,
)
