"""Extended feature definitions for 7 new LLM judge features.

This module defines 7 additional features to extract across all datasets,
complementing the existing 8 core unified features.

Features (7 total):
- 3 solution-focused: integration_complexity (or tooling_complexity), solution_locality, side_effect_risk
- 4 problem-focused: information_completeness, codebase_scope, error_specificity, debugging_complexity
"""

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition
from experiment_ab_shared.llm_judge.prompts.unified_features import (
    INTEGRATION_COMPLEXITY,
    TOOLING_COMPLEXITY,
    INTEGRATION_COMPLEXITY_SCALE,
    TOOLING_COMPLEXITY_SCALE,
    COMPLETENESS_INSTRUCTION,
)
from experiment_ab_shared.llm_judge.prompts.swebench_problem_extended import (
    INFORMATION_COMPLETENESS,
    CODEBASE_SCOPE,
    ERROR_SPECIFICITY,
    DEBUGGING_COMPLEXITY,
)


# =============================================================================
# 2 NEW Feature Definitions (solution-focused)
# =============================================================================

SOLUTION_LOCALITY = FeatureDefinition(
    name="solution_locality",
    min_value=1,
    max_value=3,
    description="How spread out are the changes? (1=single location, 2=few files, 3=many files)",
)

SIDE_EFFECT_RISK = FeatureDefinition(
    name="side_effect_risk",
    min_value=1,
    max_value=5,
    description="How likely could the solution introduce unintended effects? (1=isolated, 5=high risk)",
)


# =============================================================================
# Scale Descriptions for NEW Features
# =============================================================================

SOLUTION_LOCALITY_SCALE_CODE = """### Solution Locality (solution_locality: 1-3)
How spread out are the code changes across the codebase?
- 1: Single location - changes in one file or one function
- 2: Few files - changes in 2-3 related files or modules
- 3: Many files - changes spread across 4+ files or multiple subsystems"""

SOLUTION_LOCALITY_SCALE_TERMINAL = """### Solution Locality (solution_locality: 1-3)
How spread out is the solution across different tools/operations?
- 1: Single operation - one command or a simple sequence
- 2: Few operations - 2-3 different tools or command types
- 3: Many operations - 4+ different tools, scripts, or complex pipelines"""

SOLUTION_LOCALITY_SCALE_OPTIMIZATION = """### Solution Locality (solution_locality: 1-3)
How spread out are the optimization changes?
- 1: Single location - optimization in one function or file
- 2: Few locations - changes in 2-3 related functions or modules
- 3: Many locations - changes spread across multiple files or subsystems"""

SIDE_EFFECT_RISK_SCALE_CODE = """### Side Effect Risk (side_effect_risk: 1-5)
How likely could the solution introduce unintended effects?
- 1: Very low - isolated change with clear boundaries, no shared state
- 2: Low - limited scope, minimal interaction with other components
- 3: Moderate - touches shared code paths, some integration points
- 4: High - modifies core functionality, affects multiple call sites
- 5: Very high - system-wide implications, affects critical paths or shared state"""

SIDE_EFFECT_RISK_SCALE_TERMINAL = """### Side Effect Risk (side_effect_risk: 1-5)
How likely could the solution introduce unintended effects?
- 1: Very low - read-only operations, no system modifications
- 2: Low - writes to isolated location, easy to revert
- 3: Moderate - modifies files or configurations that others depend on
- 4: High - changes system settings, affects multiple processes
- 5: Very high - modifies critical system files, potentially destructive"""

SIDE_EFFECT_RISK_SCALE_OPTIMIZATION = """### Side Effect Risk (side_effect_risk: 1-5)
How likely could the optimization introduce unintended effects?
- 1: Very low - isolated optimization, no behavior change
- 2: Low - performance change only, correctness preserved
- 3: Moderate - may affect edge case behavior
- 4: High - changes algorithmic behavior, needs careful validation
- 5: Very high - fundamental change, may break assumptions elsewhere"""


# =============================================================================
# Scale Descriptions for Existing Features (from problem_extended)
# =============================================================================

INFORMATION_COMPLETENESS_SCALE = """### Information Completeness (information_completeness: 1-5)
How complete is the information provided in the problem statement?
- 1: Missing critical information, many unknowns
- 2: Key details missing
- 3: Adequate information but gaps exist
- 4: Good context provided
- 5: Comprehensive information including versions, configs, examples"""

CODEBASE_SCOPE_SCALE_CODE = """### Codebase Scope (codebase_scope: 1-5)
How much of the codebase might need to be understood or modified?
- 1: Likely isolated to single file/function
- 2: Few related files
- 3: Multiple components involved
- 4: Cross-cutting concern affecting many areas
- 5: System-wide implications"""

CODEBASE_SCOPE_SCALE_TERMINAL = """### Codebase Scope (codebase_scope: 1-5)
How much system state/configuration might be involved?
- 1: Single tool or simple file operation
- 2: Few related files or directories
- 3: Multiple system areas (files, processes, configs)
- 4: Cross-cutting system concerns
- 5: System-wide implications (kernel, network, etc.)"""

CODEBASE_SCOPE_SCALE_OPTIMIZATION = """### Codebase Scope (codebase_scope: 1-5)
How much of the codebase might need to be understood for optimization?
- 1: Isolated to single function
- 2: Few related functions or modules
- 3: Multiple interacting components
- 4: Understanding data flow across subsystems
- 5: System-wide performance considerations"""

ERROR_SPECIFICITY_SCALE_CODE = """### Error Specificity (error_specificity: 1-5)
How specific is the error or bug description?
- 1: Very vague symptoms, unclear what's actually broken
- 2: General description of misbehavior
- 3: Specific behavior described but no error details
- 4: Clear error description with some context
- 5: Exact error message, stack trace, or precise failure mode"""

ERROR_SPECIFICITY_SCALE_TERMINAL = """### Task Specificity (error_specificity: 1-5)
How specific is the task description?
- 1: Very vague, unclear what exactly needs to be done
- 2: General goal described but missing details
- 3: Task is described but implementation unclear
- 4: Clear task with some implementation hints
- 5: Precise specification with expected inputs/outputs"""

ERROR_SPECIFICITY_SCALE_OPTIMIZATION = """### Performance Issue Specificity (error_specificity: 1-5)
How specific is the performance issue description?
- 1: Vague "it's slow" without details
- 2: General area identified but no metrics
- 3: Specific bottleneck mentioned without profiling data
- 4: Clear performance issue with some context
- 5: Precise metrics, profiling data, or specific bottleneck"""

DEBUGGING_COMPLEXITY_SCALE = """### Debugging Complexity (debugging_complexity: 1-5)
Based on the problem description, how complex would root cause analysis be?
- 1: Obvious cause stated or implied
- 2: Straightforward to identify cause
- 3: Moderate investigation needed
- 4: Complex debugging likely required
- 5: Deep investigation into internals needed"""


# =============================================================================
# Combined Feature Lists (7 features each)
# =============================================================================

# For code-based datasets (SWE-bench Pro, GSO)
NEW_CODE_FEATURES = [
    INTEGRATION_COMPLEXITY,      # existing from unified_features.py
    SOLUTION_LOCALITY,           # NEW
    SIDE_EFFECT_RISK,            # NEW
    INFORMATION_COMPLETENESS,    # existing from swebench_problem_extended.py
    CODEBASE_SCOPE,              # existing from swebench_problem_extended.py
    ERROR_SPECIFICITY,           # existing from swebench_problem_extended.py
    DEBUGGING_COMPLEXITY,        # existing from swebench_problem_extended.py
]

# For terminal-based datasets (TerminalBench)
NEW_TERMINAL_FEATURES = [
    TOOLING_COMPLEXITY,          # existing from unified_features.py (replaces integration_complexity)
    SOLUTION_LOCALITY,           # NEW
    SIDE_EFFECT_RISK,            # NEW
    INFORMATION_COMPLETENESS,    # existing from swebench_problem_extended.py
    CODEBASE_SCOPE,              # existing from swebench_problem_extended.py
    ERROR_SPECIFICITY,           # existing from swebench_problem_extended.py
    DEBUGGING_COMPLEXITY,        # existing from swebench_problem_extended.py
]


# =============================================================================
# Output Format Templates
# =============================================================================

OUTPUT_FORMAT_7_FEATURES_CODE = """## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "integration_complexity": <1-5>,
    "solution_locality": <1-3>,
    "side_effect_risk": <1-5>,
    "information_completeness": <1-5>,
    "codebase_scope": <1-5>,
    "error_specificity": <1-5>,
    "debugging_complexity": <1-5>,
    "reasoning": "<2-3 sentence summary of the key difficulty factors>"
}}"""

OUTPUT_FORMAT_7_FEATURES_TERMINAL = """## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "tooling_complexity": <1-5>,
    "solution_locality": <1-3>,
    "side_effect_risk": <1-5>,
    "information_completeness": <1-5>,
    "codebase_scope": <1-5>,
    "error_specificity": <1-5>,
    "debugging_complexity": <1-5>,
    "reasoning": "<2-3 sentence summary of the key difficulty factors>"
}}"""
