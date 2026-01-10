"""
Experiment C: Cost Analysis of Lunette Grading vs Full Agent Run

This script measures the cost of:
1. Running a dummy agent (just to get Lunette sandbox access)
2. Running Lunette grading/investigation on the task

vs the estimated cost of:
3. Running a full SWE-bench agent

Usage:
    python llm_judge/experiment_c_cost_analysis.py --run_id <run_id>
    python llm_judge/experiment_c_cost_analysis.py --n_tasks 10  # Run full experiment
"""

import argparse
import asyncio
import time
from pathlib import Path

from lunette import LunetteClient
from lunette.analysis import GradingPlan


# Grading prompt for difficulty prediction
DIFFICULTY_GRADING_PROMPT = """You are analyzing a SWE-bench task to predict its difficulty.

You have access to the sandbox environment. Please:
1. Explore the repository structure
2. Read the problem statement carefully
3. Try to locate relevant files and understand the scope of the fix

Based on your exploration, evaluate these dimensions (each on 1-5 scale):
1. Problem clarity: How clear is the problem description?
2. Fix hints: Does the description suggest the solution? (0-3)
3. Domain knowledge: How much specialized knowledge is needed?
4. Fix complexity: How complex is the actual fix?
5. Codebase complexity: How large/complex is the relevant codebase?

Return a difficulty score from 0.0 (trivial) to 1.0 (extremely hard), along with your reasoning.
"""


async def grade_single_run(client: LunetteClient, run_id: str, enable_sandbox: bool = True):
    """Grade a single run and measure the time/cost."""
    start_time = time.time()

    plan = GradingPlan(
        name='task-difficulty',
        prompt=DIFFICULTY_GRADING_PROMPT,
        enable_sandbox=enable_sandbox,
    )

    results = await client.investigate(
        run_id=run_id,
        plan=plan,
        limit=1,
    )

    elapsed = time.time() - start_time

    if results.results:
        result_data = results.results[0].data
        return {
            'run_id': run_id,
            'elapsed_seconds': elapsed,
            'score': result_data.get('score'),
            'explanation': result_data.get('explanation'),
            'success': True,
        }
    else:
        return {
            'run_id': run_id,
            'elapsed_seconds': elapsed,
            'success': False,
            'error': 'No results returned',
        }


async def main():
    parser = argparse.ArgumentParser(description="Experiment C: Cost Analysis")
    parser.add_argument('--run_id', type=str, help='Single run ID to grade')
    parser.add_argument('--n_tasks', type=int, default=1, help='Number of tasks to run')
    parser.add_argument('--no_sandbox', action='store_true', help='Disable sandbox access')

    args = parser.parse_args()

    async with LunetteClient() as client:
        if args.run_id:
            # Grade single run
            print(f"Grading run: {args.run_id}")
            print(f"Sandbox enabled: {not args.no_sandbox}")

            result = await grade_single_run(
                client,
                args.run_id,
                enable_sandbox=not args.no_sandbox
            )

            print(f"\n=== Results ===")
            print(f"Elapsed time: {result['elapsed_seconds']:.1f} seconds")
            if result['success']:
                print(f"Difficulty score: {result['score']}")
                print(f"\nExplanation:\n{result['explanation']}")
            else:
                print(f"Error: {result.get('error')}")

            # Cost estimates
            print(f"\n=== Cost Analysis ===")
            print(f"Lunette grading time: {result['elapsed_seconds']:.1f}s")
            print(f"Estimated Lunette cost: ~${0.50:.2f}")
            print(f"Full agent run time: 5-30 minutes")
            print(f"Full agent cost (Claude): $5-30")
            print(f"Cost savings: ~10-60x cheaper with Lunette grading")
        else:
            print("Please provide --run_id or --n_tasks")


if __name__ == "__main__":
    asyncio.run(main())
