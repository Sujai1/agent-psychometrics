"""
Lunette analysis for SWE-bench trajectories.

This script runs Lunette's built-in analysis plans on SWE-bench agent trajectories:
- BottleneckPlan: on passing trajectories (what enabled success?)
- IssueDetectionPlan: on failing trajectories (what went wrong?)
- GradingPlan: on both passing and failing trajectories

Usage:
    python llm_judge/lunette_analysis.py --num_tasks 5
    python llm_judge/lunette_analysis.py --task_id django__django-13658
"""

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Optional

from lunette import LunetteClient
from lunette.analysis import BottleneckPlan, IssueDetectionPlan, GradingPlan
from lunette.models.run import Run
from lunette.models.trajectory import Trajectory, ScalarScore
from lunette.models.messages import SystemMessage, UserMessage, AssistantMessage, ToolMessage, ToolCall


def load_swebench_trajectory(traj_path: Path) -> dict:
    """Load a SWE-bench trajectory file."""
    with open(traj_path) as f:
        return json.load(f)


def convert_to_lunette_trajectory(
    task_id: str,
    swebench_traj: dict,
    resolved: bool,
) -> Trajectory:
    """Convert a SWE-bench trajectory to Lunette format."""
    messages = []
    position = 0

    # Convert history messages to Lunette format
    history = swebench_traj.get('history', [])

    for msg in history:
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        if role == 'system':
            messages.append(SystemMessage(position=position, content=content))
        elif role == 'user':
            messages.append(UserMessage(position=position, content=content))
        elif role == 'assistant':
            # Check if this contains a command (tool call)
            if '<command>' in content and '</command>' in content:
                # Extract the command
                cmd_start = content.find('<command>') + len('<command>')
                cmd_end = content.find('</command>')
                command = content[cmd_start:cmd_end].strip()

                # Create tool call
                tool_call = ToolCall(
                    id=f"call_{position}",
                    function="bash",
                    arguments={"command": command}
                )
                messages.append(AssistantMessage(
                    position=position,
                    content=content,
                    tool_calls=[tool_call]
                ))
            else:
                messages.append(AssistantMessage(position=position, content=content))

        position += 1

    # Get solution (patch) from info
    info = swebench_traj.get('info', {})
    solution = info.get('submission')

    # Create score based on resolution status
    scores = {
        'resolved': ScalarScore(value=1.0 if resolved else 0.0)
    }

    return Trajectory(
        sample=task_id,
        messages=messages,
        scores=scores,
        solution=solution,
        metadata={
            'environment': swebench_traj.get('environment', ''),
            'exit_status': info.get('exit_status', ''),
        }
    )


def get_task_lists(results_path: Path, trajs_dir: Path) -> tuple[list[str], list[str]]:
    """Get lists of resolved and unresolved task IDs."""
    with open(results_path) as f:
        results = json.load(f)

    resolved_set = set(results.get('resolved', []))

    all_tasks = [f.stem for f in trajs_dir.glob('*.traj')]
    resolved = [t for t in all_tasks if t in resolved_set]
    unresolved = [t for t in all_tasks if t not in resolved_set]

    return resolved, unresolved


async def run_analysis(
    client: LunetteClient,
    run_id: str,
    plan_type: str,
    prompt: str,
) -> dict:
    """Run an analysis plan on a saved run."""

    if plan_type == 'bottleneck':
        plan = BottleneckPlan(
            name="swebench-bottleneck",
            prompt=prompt,
        )
    elif plan_type == 'issue':
        plan = IssueDetectionPlan(
            name="swebench-issues",
            prompt=prompt,
        )
    elif plan_type == 'grading':
        plan = GradingPlan(
            name="swebench-grading",
            prompt=prompt,
        )
    else:
        raise ValueError(f"Unknown plan type: {plan_type}")

    results = await client.investigate(
        run_id=run_id,
        plan=plan,
        limit=1,  # We're analyzing one trajectory at a time
    )

    return results


async def analyze_task(
    client: LunetteClient,
    task_id: str,
    traj_path: Path,
    resolved: bool,
    model_name: str = "sweagent_claude3.5sonnet",
) -> dict:
    """Analyze a single task with appropriate plans."""

    print(f"\n{'='*60}")
    print(f"Analyzing: {task_id} ({'PASS' if resolved else 'FAIL'})")
    print('='*60)

    # Load and convert trajectory
    swebench_traj = load_swebench_trajectory(traj_path)
    lunette_traj = convert_to_lunette_trajectory(task_id, swebench_traj, resolved)

    # Create a run with this single trajectory
    run = Run(
        task="swebench-verified",
        model=model_name,
        trajectories=[lunette_traj],
    )

    # Save run to Lunette
    print(f"  Uploading trajectory to Lunette...")
    run_meta = await client.save_run(run)
    run_id = run_meta['run_id']
    print(f"  Run ID: {run_id}")

    results = {
        'task_id': task_id,
        'resolved': resolved,
        'run_id': run_id,
    }

    # Run appropriate analysis plans
    if resolved:
        # For passing trajectories: run Bottleneck and Grading
        print(f"\n  Running BottleneckPlan...")
        try:
            bottleneck_results = await run_analysis(
                client, run_id, 'bottleneck',
                prompt="""Analyze this successful SWE-bench trajectory.
                What was the key insight or action that enabled the agent to solve the task?
                What bottleneck did the agent overcome to succeed?"""
            )
            results['bottleneck'] = {
                'trajectory_count': bottleneck_results.trajectory_count,
                'results': [r.data for r in bottleneck_results.results] if bottleneck_results.results else []
            }
            print(f"    Bottleneck analysis complete")
        except Exception as e:
            print(f"    Bottleneck analysis failed: {e}")
            results['bottleneck'] = {'error': str(e)}

        print(f"\n  Running GradingPlan (passing)...")
        try:
            grading_results = await run_analysis(
                client, run_id, 'grading',
                prompt="""Grade this successful SWE-bench trajectory on:
                1. Efficiency: How directly did the agent find the solution?
                2. Code quality: How clean and correct is the submitted patch?
                3. Process: How systematic was the debugging/exploration approach?"""
            )
            results['grading'] = {
                'trajectory_count': grading_results.trajectory_count,
                'results': [r.data for r in grading_results.results] if grading_results.results else []
            }
            print(f"    Grading analysis complete")
        except Exception as e:
            print(f"    Grading analysis failed: {e}")
            results['grading'] = {'error': str(e)}

    else:
        # For failing trajectories: run IssueDetection and Grading
        print(f"\n  Running IssueDetectionPlan...")
        try:
            issue_results = await run_analysis(
                client, run_id, 'issue',
                prompt="""Analyze this failed SWE-bench trajectory.
                What issues prevented the agent from solving the task?
                Consider: wrong approach, missing context, incorrect fix, etc."""
            )
            results['issues'] = {
                'trajectory_count': issue_results.trajectory_count,
                'results': [r.data for r in issue_results.results] if issue_results.results else []
            }
            print(f"    Issue detection complete")
        except Exception as e:
            print(f"    Issue detection failed: {e}")
            results['issues'] = {'error': str(e)}

        print(f"\n  Running GradingPlan (failing)...")
        try:
            grading_results = await run_analysis(
                client, run_id, 'grading',
                prompt="""Grade this failed SWE-bench trajectory on:
                1. Effort: How much progress did the agent make toward the solution?
                2. Understanding: Did the agent correctly identify the problem?
                3. Approach: Was the debugging strategy reasonable?"""
            )
            results['grading'] = {
                'trajectory_count': grading_results.trajectory_count,
                'results': [r.data for r in grading_results.results] if grading_results.results else []
            }
            print(f"    Grading analysis complete")
        except Exception as e:
            print(f"    Grading analysis failed: {e}")
            results['grading'] = {'error': str(e)}

    return results


async def main():
    parser = argparse.ArgumentParser(description='Run Lunette analysis on SWE-bench trajectories')
    parser.add_argument('--submission', type=str,
                        default='verified/20240620_sweagent_claude3.5sonnet',
                        help='Submission path under evaluation/')
    parser.add_argument('--num_tasks', type=int, default=None,
                        help='Number of tasks to analyze (samples from both pass/fail)')
    parser.add_argument('--task_id', type=str, default=None,
                        help='Analyze a specific task')
    parser.add_argument('--output_path', type=str,
                        default='chris_output/lunette/analysis_results.json',
                        help='Output path for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    args = parser.parse_args()

    # Setup paths (experiments/ is gitignored and contains SWE-bench data)
    experiments_dir = Path(__file__).resolve().parents[1] / 'experiments'
    eval_dir = experiments_dir / 'evaluation' / args.submission
    trajs_dir = eval_dir / 'trajs'
    results_path = eval_dir / 'results' / 'results.json'

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get task lists
    resolved, unresolved = get_task_lists(results_path, trajs_dir)
    print(f"Found {len(resolved)} resolved and {len(unresolved)} unresolved tasks")

    # Select tasks to analyze
    random.seed(args.seed)

    if args.task_id:
        task_ids = [args.task_id]
        is_resolved = {args.task_id: args.task_id in resolved}
    elif args.num_tasks:
        # Sample equally from resolved and unresolved
        n_each = args.num_tasks // 2
        sampled_resolved = random.sample(resolved, min(n_each, len(resolved)))
        sampled_unresolved = random.sample(unresolved, min(n_each, len(unresolved)))
        task_ids = sampled_resolved + sampled_unresolved
        is_resolved = {t: t in resolved for t in task_ids}
    else:
        # Default: analyze 2 of each
        sampled_resolved = random.sample(resolved, min(2, len(resolved)))
        sampled_unresolved = random.sample(unresolved, min(2, len(unresolved)))
        task_ids = sampled_resolved + sampled_unresolved
        is_resolved = {t: t in resolved for t in task_ids}

    print(f"\nAnalyzing {len(task_ids)} tasks:")
    for tid in task_ids:
        print(f"  - {tid} ({'PASS' if is_resolved[tid] else 'FAIL'})")

    # Run analysis
    all_results = []

    async with LunetteClient() as client:
        for task_id in task_ids:
            traj_path = trajs_dir / f"{task_id}.traj"
            if not traj_path.exists():
                print(f"Warning: trajectory not found for {task_id}")
                continue

            result = await analyze_task(
                client,
                task_id,
                traj_path,
                is_resolved[task_id],
            )
            all_results.append(result)

            # Save incrementally
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for result in all_results:
        print(f"\n{result['task_id']} ({'PASS' if result['resolved'] else 'FAIL'}):")

        if 'bottleneck' in result and 'results' in result['bottleneck']:
            for r in result['bottleneck'].get('results', []):
                print(f"  Bottleneck: {r}")

        if 'issues' in result and 'results' in result['issues']:
            for r in result['issues'].get('results', []):
                print(f"  Issues: {r}")

        if 'grading' in result and 'results' in result['grading']:
            for r in result['grading'].get('results', []):
                print(f"  Grading: {r}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
