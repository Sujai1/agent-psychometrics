"""
Dummy solver that immediately finishes without making any model calls.

This is used for Experiment C to measure the cost of Lunette grading
without the cost of running a full agent. The solver:
1. Gets access to the sandbox environment (via Lunette)
2. Runs a few bash commands to explore the environment
3. Submits an empty patch (guaranteed to fail)

Usage:
    # Run with lunette eval
    lunette eval swebench --solver lunette_utils/dummy_solver.py:dummy_solver --model mockllm/model --limit 1

    # Or with inspect directly
    inspect eval inspect_evals/swe_bench_verified_mini \
        --solver lunette_utils/dummy_solver.py:dummy_solver \
        --model mockllm/model \
        --sandbox lunette \
        -T sandbox_type=lunette \
        -T build_docker_images=False \
        --limit 1
"""

from inspect_ai.solver import Solver, TaskState, solver
from inspect_ai.tool import bash


@solver
def dummy_solver() -> Solver:
    """A minimal solver that explores the environment but doesn't solve anything.

    This is useful for:
    1. Testing the Lunette sandbox setup
    2. Measuring grading cost without agent cost
    3. Getting environment features for difficulty prediction
    """

    async def solve(state: TaskState, generate) -> TaskState:
        """Explore the environment and immediately give up."""

        # Get the bash tool to interact with sandbox
        bash_tool = bash(timeout=30)

        # Run some exploratory commands to gather environment info
        # (This will be visible in the trajectory for the grader to analyze)

        # 1. Check current directory and list files
        pwd_result = await bash_tool(state, "pwd && ls -la")

        # 2. Find Python files in the repo
        find_result = await bash_tool(state, "find . -name '*.py' -type f | head -20")

        # 3. Check git status
        git_result = await bash_tool(state, "git log --oneline -5 2>/dev/null || echo 'Not a git repo'")

        # 4. Look at the structure
        tree_result = await bash_tool(state, "find . -type d | head -30")

        # Add a message indicating we're giving up
        state.messages.append({
            "role": "assistant",
            "content": "I have explored the repository structure. This is a dummy agent that does not attempt to solve the problem. Submitting empty patch."
        })

        # The solver doesn't submit anything - the task will score as failed
        # which is fine for our purposes (we just want to grade the task difficulty)

        return state

    return solve


@solver
def instant_fail_solver() -> Solver:
    """Even more minimal - just immediately returns without any exploration."""

    async def solve(state: TaskState, generate) -> TaskState:
        """Immediately fail without doing anything."""
        state.messages.append({
            "role": "assistant",
            "content": "Dummy agent: No attempt made."
        })
        return state

    return solve
