"""LLM judge prompt for semantic feature extraction (Experiment A - TerminalBench).

This prompt extracts semantic features from static task information
(instruction, solution.sh) to predict task difficulty for terminal/shell tasks.

The features are adapted from the SWE-bench LLM judge to fit terminal tasks:
- fix_in_description (0-3) -> solution_in_instruction
- problem_clarity (1-5) -> task_clarity
- error_message_provided (0/1) -> dropped (not applicable)
- reproduction_steps (0/1) -> dropped (not applicable)
- fix_locality (1-3) -> solution_size
- domain_knowledge_required (1-5) -> domain_knowledge_required
- fix_complexity (1-5) -> task_complexity
- logical_reasoning_required (1-5) -> logical_reasoning_required
- atypicality (1-5) -> atypicality
- NEW: tooling_complexity (1-5)
"""

# Feature names for TerminalBench semantic extraction
LLM_JUDGE_SEMANTIC_FEATURES = [
    "solution_in_instruction",
    "task_clarity",
    "solution_size",
    "domain_knowledge_required",
    "task_complexity",
    "logical_reasoning_required",
    "atypicality",
    "tooling_complexity",
]

# The semantic-only grading prompt for TerminalBench tasks
LLM_JUDGE_PROMPT = """You are analyzing a TerminalBench terminal/shell task to predict its difficulty.
You will analyze the task instruction and reference solution to evaluate semantic features.

## TASK INFORMATION

**Task ID:** {task_id}
**Category:** {category}
**Tags:** {tags}
**Claimed Difficulty:** {claimed_difficulty}

**Task Instruction:**
{instruction}

**Reference Solution (solution.sh):**
```bash
{solution}
```

## FEATURES TO EVALUATE

Analyze the instruction and solution to evaluate these 8 semantic features.
Be precise and consistent with your ratings.

### 1. Solution Hints in Instruction (solution_in_instruction: 0-3)
Does the instruction contain or hint at the solution approach?
- 0: No hint at the solution at all
- 1: Vague hint or general direction
- 2: Clear description of approach needed
- 3: Exact commands or detailed solution steps provided

### 2. Task Clarity (task_clarity: 1-5)
How clear and well-specified is the task?
- 1: Very vague, unclear what's actually required
- 2: Somewhat clear but missing key details
- 3: Reasonably clear, some ambiguity about requirements
- 4: Clear with good context and success criteria
- 5: Crystal clear with explicit steps and expected outputs

### 3. Solution Size (solution_size: 1-3)
How large/complex is the reference solution script?
- 1: Simple, few commands (1-10 lines)
- 2: Moderate complexity (11-50 lines)
- 3: Large/complex script (>50 lines or multiple components)

### 4. Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed?
- 1: Basic shell commands anyone could use (ls, cd, cat, echo)
- 2: Standard Unix tools (grep, sed, awk, find)
- 3: Specialized tools or configurations (cmake, git internals, network tools)
- 4: Deep understanding of systems (kernel, filesystems, protocols)
- 5: Obscure tools, APIs, or highly specialized domain knowledge

### 5. Task Complexity (task_complexity: 1-5)
How complex is the actual task to complete?
- 1: Trivial (single command, simple file operation)
- 2: Simple (straightforward multi-step task)
- 3: Moderate (requires understanding context, multiple tools)
- 4: Complex (multiple interdependent steps, debugging needed)
- 5: Very complex (architectural changes, cross-system integration)

### 6. Logical Reasoning Required (logical_reasoning_required: 1-5)
How much logical reasoning is needed?
- 1: Mechanical execution, no reasoning needed
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about system behavior, edge cases

### 7. Atypicality (atypicality: 1-5)
How unusual is this type of terminal task?
- 1: Very common task (file manipulation, basic scripting)
- 2: Common task (process management, system configuration)
- 3: Moderately unusual
- 4: Unusual task pattern
- 5: Rare or novel task

### 8. Tooling Complexity (tooling_complexity: 1-5)
How complex is the tooling/environment setup?
- 1: No special tools needed (basic shell)
- 2: Standard development tools (git, make, pip)
- 3: Multiple specialized tools or complex configuration
- 4: Uncommon tools or complex build systems
- 5: Exotic toolchain, legacy systems, or cross-compilation

## OUTPUT FORMAT

Respond with ONLY a JSON object containing all features. No markdown, no extra text.

{{
    "solution_in_instruction": <0-3>,
    "task_clarity": <1-5>,
    "solution_size": <1-3>,
    "domain_knowledge_required": <1-5>,
    "task_complexity": <1-5>,
    "logical_reasoning_required": <1-5>,
    "atypicality": <1-5>,
    "tooling_complexity": <1-5>,
    "reasoning": "<2-3 sentence summary of the key difficulty factors>"
}}
"""


def format_llm_judge_prompt(
    task_id: str,
    instruction: str,
    solution: str,
    category: str = "",
    tags: list = None,
    claimed_difficulty: str = "",
) -> str:
    """Format the LLM judge prompt with task-specific information.

    Args:
        task_id: TerminalBench task ID (e.g., "3d-model-format-legacy")
        instruction: The task instruction from task.yaml
        solution: The reference solution from solution.sh
        category: Task category (e.g., "software-engineering")
        tags: List of tags (e.g., ["coding", "file-operations"])
        claimed_difficulty: Self-reported difficulty (e.g., "hard")

    Returns:
        Formatted prompt string
    """
    if tags is None:
        tags = []

    # Truncate very long fields to avoid context overflow
    instruction = instruction[:12000] if len(instruction) > 12000 else instruction
    solution = solution[:12000] if len(solution) > 12000 else solution

    return LLM_JUDGE_PROMPT.format(
        task_id=task_id,
        category=category or "N/A",
        tags=", ".join(tags) if tags else "N/A",
        claimed_difficulty=claimed_difficulty or "N/A",
        instruction=instruction,
        solution=solution,
    )
