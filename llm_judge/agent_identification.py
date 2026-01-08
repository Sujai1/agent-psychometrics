"""
Agent Identification from Trajectories using Lunette.

This script uses Lunette's GradingPlan to extract interpretable behavioral features
from SWE-bench trajectories that can distinguish different agents. The goal is to
identify non-spurious behavioral signatures - patterns that reflect genuine
problem-solving strategies rather than artifacts of the model or tool constraints.

Key design decisions to avoid spurious correlations:
1. Strip chat messages, focus only on file edits and their effects
2. Instruct the judge to ignore model-specific artifacts (naming conventions, etc.)
3. Focus on problem-solving strategy features, not stylistic ones

Usage:
    python llm_judge/agent_identification.py --upload_and_grade --agent sweagent_claude3.5sonnet --num_tasks 50
    python llm_judge/agent_identification.py --compare_agents --agents agent1,agent2,agent3
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import pandas as pd

try:
    from lunette import LunetteClient
    from lunette.analysis import GradingPlan
    from lunette.models.run import Run
    from lunette.models.trajectory import Trajectory, ScalarScore
    from lunette.models.messages import SystemMessage, UserMessage, AssistantMessage
    LUNETTE_AVAILABLE = True
except ImportError:
    LUNETTE_AVAILABLE = False
    print("Warning: lunette not installed. Run: pip install lunette-sdk")


# Grading prompt focused on behavioral signatures for agent identification
AGENT_IDENTIFICATION_PROMPT = """You are analyzing a coding trajectory to identify behavioral patterns that reveal the problem-solving strategy of the agent.

## CRITICAL INSTRUCTIONS
You are analyzing ONLY the code edits and their effects. Ignore any:
- Model names or identifying text in messages
- Specific tool commands (these vary by agent framework)
- Stylistic formatting preferences
- Comments that mention specific systems

Focus ONLY on the actual problem-solving approach reflected in the edits.

## Features to Extract (Problem-Solving Strategy)

1. **localization_strategy** (1-5): How did the agent locate the bug?
   - 1: Random/scattered file exploration
   - 2: Keyword search based
   - 3: Systematic directory traversal
   - 4: Following import/call chains
   - 5: Direct navigation to known location

2. **hypothesis_testing** (0-3): Did the agent test hypotheses before editing?
   - 0: No evidence of hypothesis testing
   - 1: Single hypothesis, immediate edit
   - 2: Multiple hypotheses considered
   - 3: Systematic hypothesis elimination

3. **fix_scope** (1-3): How broad was the fix?
   - 1: Single-line surgical fix
   - 2: Multi-line modification in one location
   - 3: Changes across multiple locations/files

4. **incremental_vs_big_bang** (1-3): Was the fix incremental or all-at-once?
   - 1: Multiple small iterative edits
   - 2: Mixed approach
   - 3: Single large edit attempt

5. **error_recovery** (0-3): How did the agent handle errors?
   - 0: No errors encountered or gave up on first error
   - 1: Simple retry with minor changes
   - 2: Changed approach after error
   - 3: Systematic debugging and recovery

6. **test_driven** (0-2): Did the agent write or run tests?
   - 0: No testing
   - 1: Ran existing tests
   - 2: Created new tests

7. **exploration_depth** (1-5): How deep was the codebase exploration?
   - 1: Minimal (1-2 files)
   - 2: Shallow (3-5 files)
   - 3: Moderate (6-10 files)
   - 4: Deep (11-20 files)
   - 5: Very deep (>20 files)

8. **edit_precision** (1-5): How precise were the edits?
   - 1: Many unnecessary changes, reformatting
   - 2: Some unnecessary changes mixed with fix
   - 3: Mostly relevant changes
   - 4: Clean, focused edits
   - 5: Surgically precise, minimal diff

9. **context_awareness** (1-5): Did the agent understand the broader codebase context?
   - 1: Ignored related code, potential for side effects
   - 2: Limited awareness
   - 3: Moderate awareness of immediate context
   - 4: Good awareness of related components
   - 5: Excellent understanding of system architecture

10. **persistence** (1-5): How persistent was the agent?
    - 1: Gave up after first attempt
    - 2: 2-3 attempts
    - 3: 4-5 attempts with variations
    - 4: 6-10 attempts, systematic exploration
    - 5: Very persistent, exhaustive exploration

11. **efficiency** (1-5): How efficient was the problem-solving?
    - 1: Very inefficient, many wasted steps
    - 2: Somewhat inefficient
    - 3: Average efficiency
    - 4: Efficient, few wasted steps
    - 5: Very efficient, direct path to solution

12. **verification_approach** (0-3): How did the agent verify the fix?
    - 0: No verification
    - 1: Visual inspection only
    - 2: Ran tests or reproduction script
    - 3: Multiple verification methods

## Output Format

Respond with ONLY a JSON object:
{
    "localization_strategy": <1-5>,
    "hypothesis_testing": <0-3>,
    "fix_scope": <1-3>,
    "incremental_vs_big_bang": <1-3>,
    "error_recovery": <0-3>,
    "test_driven": <0-2>,
    "exploration_depth": <1-5>,
    "edit_precision": <1-5>,
    "context_awareness": <1-5>,
    "persistence": <1-5>,
    "efficiency": <1-5>,
    "verification_approach": <0-3>,
    "reasoning": "<2-3 sentence summary of the agent's problem-solving approach>"
}
"""


def load_local_trajectory(traj_path: Path) -> dict:
    """Load a local SWE-bench .traj file."""
    with open(traj_path) as f:
        return json.load(f)


def strip_trajectory_to_edits(swebench_traj: dict) -> dict:
    """
    Strip a trajectory to only include edit-related information.
    This removes chat messages and other potentially spurious signals.
    """
    traj_list = swebench_traj.get('trajectory', [])
    stripped_steps = []

    for step in traj_list:
        action = step.get('action', '')
        observation = step.get('observation', '')

        # Only keep steps that are edit-related or show code
        action_lower = action.lower().strip()

        # Include: edits, file opens showing code, test runs
        if any([
            action_lower.startswith('edit '),
            action_lower.startswith('create '),
            action_lower.startswith('open '),
            'File:' in observation,  # Shows file content
            action_lower.startswith('python '),
            action_lower.startswith('pytest'),
            'test' in action_lower,
        ]):
            # Redact potential model-identifying information
            redacted_observation = redact_model_info(observation)

            stripped_steps.append({
                'action': action,
                'observation': redacted_observation,
                # Omit 'thought' and 'response' as they may contain model signatures
            })

    return {
        'trajectory': stripped_steps,
        'info': swebench_traj.get('info', {}),
    }


def redact_model_info(text: str) -> str:
    """Redact potential model-identifying information from text."""
    # Common model names to redact
    model_patterns = [
        r'claude', r'gpt-?4', r'gpt-?3\.?5', r'sonnet', r'opus', r'haiku',
        r'gemini', r'llama', r'mistral', r'qwen', r'deepseek',
        r'sweagent', r'swe-agent', r'autocoderover', r'openhands',
    ]

    import re
    result = text
    for pattern in model_patterns:
        result = re.sub(pattern, '[REDACTED]', result, flags=re.IGNORECASE)

    return result


def convert_to_lunette_trajectory(
    task_id: str,
    swebench_traj: dict,
    resolved: bool,
    strip_to_edits: bool = True
) -> Trajectory:
    """Convert a local SWE-bench trajectory to Lunette format."""
    if strip_to_edits:
        swebench_traj = strip_trajectory_to_edits(swebench_traj)

    messages = []
    traj_list = swebench_traj.get('trajectory', [])

    for i, step in enumerate(traj_list):
        action = step.get('action', '')
        observation = step.get('observation', '')

        # Create alternating user (action) and assistant (observation) messages
        messages.append(UserMessage(position=i*2, content=f"Action: {action}"))
        messages.append(AssistantMessage(position=i*2+1, content=f"Result: {observation[:5000]}"))  # Truncate long observations

    info = swebench_traj.get('info', {})
    solution = info.get('submission')

    scores = {'resolved': ScalarScore(value=1.0 if resolved else 0.0)}

    return Trajectory(
        sample=task_id,
        messages=messages,
        scores=scores,
        solution=solution,
        metadata={
            'exit_status': info.get('exit_status', ''),
        }
    )


def get_local_task_lists(submission_dir: Path) -> tuple[list[str], list[str]]:
    """Get lists of resolved and unresolved task IDs from local trajectories.

    Returns sorted lists to ensure reproducibility with fixed seeds.
    """
    trajs_dir = submission_dir / 'trajs'
    results_path = submission_dir / 'results' / 'results.json'

    with open(results_path) as f:
        results = json.load(f)

    resolved_set = set(results.get('resolved', []))
    all_tasks = sorted([f.stem for f in trajs_dir.glob('*.traj')])  # Sort for reproducibility

    resolved = sorted([t for t in all_tasks if t in resolved_set])
    unresolved = sorted([t for t in all_tasks if t not in resolved_set])

    return resolved, unresolved


async def grade_agent_trajectories(
    client,  # LunetteClient
    submission_dir: Path,
    agent_name: str,
    num_tasks: int,
    output_path: Path,
    strip_to_edits: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Upload trajectories to Lunette and grade them for behavioral features.

    Args:
        seed: Random seed for reproducible task sampling. With the same seed,
              the same tasks will be selected across runs.
    """

    trajs_dir = submission_dir / 'trajs'
    resolved, unresolved = get_local_task_lists(submission_dir)

    print(f"Found {len(resolved)} resolved and {len(unresolved)} unresolved tasks for {agent_name}")

    # Sample tasks with fixed seed for reproducibility
    print(f"Using random seed: {seed}")
    random.seed(seed)
    n_each = num_tasks // 2
    sampled_resolved = random.sample(resolved, min(n_each, len(resolved)))
    sampled_unresolved = random.sample(unresolved, min(n_each, len(unresolved)))
    task_ids = sampled_resolved + sampled_unresolved
    is_resolved = {t: t in resolved for t in task_ids}

    print(f"Sampled {len(task_ids)} tasks ({len(sampled_resolved)} resolved, {len(sampled_unresolved)} unresolved)")
    print(f"First 5 task IDs: {task_ids[:5]}")

    all_results = []

    for i, task_id in enumerate(task_ids):
        print(f"\n[{i+1}/{len(task_ids)}] {task_id} ({'PASS' if is_resolved[task_id] else 'FAIL'})")

        traj_path = trajs_dir / f"{task_id}.traj"
        if not traj_path.exists():
            print(f"  -> Trajectory not found, skipping")
            continue

        try:
            # Load and convert trajectory
            swebench_traj = load_local_trajectory(traj_path)
            lunette_traj = convert_to_lunette_trajectory(
                task_id,
                swebench_traj,
                is_resolved[task_id],
                strip_to_edits=strip_to_edits
            )

            # Create run and upload
            run = Run(
                task="swebench-verified",
                model=agent_name,
                trajectories=[lunette_traj],
            )

            print(f"  -> Uploading to Lunette...")
            run_meta = await client.save_run(run)
            run_id = run_meta['run_id']
            print(f"  -> Run ID: {run_id}")

            # Grade the run
            print(f"  -> Grading behavioral features...")
            results = await client.investigate(
                run_id=run_id,
                plan=GradingPlan(name="agent-identification", prompt=AGENT_IDENTIFICATION_PROMPT),
                limit=1,
            )

            if results.results:
                result_data = results.results[0].data
                result_data["task_id"] = task_id
                result_data["run_id"] = run_id
                result_data["resolved"] = is_resolved[task_id]
                result_data["agent"] = agent_name
                all_results.append(result_data)

                print(f"  -> localization={result_data.get('localization_strategy')}, "
                      f"efficiency={result_data.get('efficiency')}")
            else:
                print(f"  -> No grading results returned")
                all_results.append({"task_id": task_id, "run_id": run_id, "agent": agent_name, "error": "No results"})

            # Save incrementally
            df = pd.DataFrame(all_results)
            df.to_csv(output_path, index=False)

        except Exception as e:
            print(f"  -> Error: {e}")
            all_results.append({"task_id": task_id, "agent": agent_name, "error": str(e)})

    return pd.DataFrame(all_results)


async def compare_agents_behavioral_features(
    agent_features: dict[str, pd.DataFrame],
    output_path: Path
):
    """Compare behavioral features across multiple agents."""
    import matplotlib.pyplot as plt

    feature_cols = [
        'localization_strategy', 'hypothesis_testing', 'fix_scope',
        'incremental_vs_big_bang', 'error_recovery', 'test_driven',
        'exploration_depth', 'edit_precision', 'context_awareness',
        'persistence', 'efficiency', 'verification_approach'
    ]

    # Compute mean features for each agent
    comparison_data = {}
    for agent_name, df in agent_features.items():
        available_cols = [col for col in feature_cols if col in df.columns]
        if available_cols:
            comparison_data[agent_name] = df[available_cols].mean()

    if not comparison_data:
        print("No feature data available for comparison")
        return

    comparison_df = pd.DataFrame(comparison_data).T

    print("\n" + "=" * 80)
    print("AGENT BEHAVIORAL FEATURE COMPARISON")
    print("=" * 80)
    print(comparison_df.round(2).to_string())

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Radar chart data preparation
    available_features = list(comparison_df.columns)
    agents = list(comparison_df.index)

    # Bar chart of key features
    key_features = ['efficiency', 'edit_precision', 'exploration_depth', 'persistence']
    key_features = [f for f in key_features if f in available_features]

    if key_features:
        x = np.arange(len(key_features))
        width = 0.8 / len(agents)

        for i, agent in enumerate(agents):
            values = [comparison_df.loc[agent, f] for f in key_features]
            axes[0, 0].bar(x + i * width, values, width, label=agent)

        axes[0, 0].set_xlabel('Feature')
        axes[0, 0].set_ylabel('Mean Score')
        axes[0, 0].set_title('Key Behavioral Features by Agent')
        axes[0, 0].set_xticks(x + width * (len(agents) - 1) / 2)
        axes[0, 0].set_xticklabels(key_features, rotation=45, ha='right')
        axes[0, 0].legend()

    # Heatmap
    im = axes[0, 1].imshow(comparison_df.values, aspect='auto', cmap='RdYlGn')
    axes[0, 1].set_xticks(range(len(available_features)))
    axes[0, 1].set_xticklabels(available_features, rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_yticks(range(len(agents)))
    axes[0, 1].set_yticklabels(agents)
    axes[0, 1].set_title('Feature Heatmap')
    plt.colorbar(im, ax=axes[0, 1])

    # Scatter: efficiency vs exploration
    if 'efficiency' in available_features and 'exploration_depth' in available_features:
        for agent_name, df in agent_features.items():
            if 'efficiency' in df.columns and 'exploration_depth' in df.columns:
                axes[1, 0].scatter(
                    df['exploration_depth'],
                    df['efficiency'],
                    alpha=0.5,
                    label=agent_name
                )
        axes[1, 0].set_xlabel('Exploration Depth')
        axes[1, 0].set_ylabel('Efficiency')
        axes[1, 0].set_title('Efficiency vs Exploration Depth')
        axes[1, 0].legend()

    # Resolution rate by agent (if available)
    resolution_rates = {}
    for agent_name, df in agent_features.items():
        if 'resolved' in df.columns:
            resolution_rates[agent_name] = df['resolved'].mean()

    if resolution_rates:
        axes[1, 1].bar(resolution_rates.keys(), resolution_rates.values())
        axes[1, 1].set_ylabel('Resolution Rate')
        axes[1, 1].set_title('Task Resolution Rate by Agent')
        axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved comparison visualization to {output_path}")

    return comparison_df


async def main():
    parser = argparse.ArgumentParser(description='Grade SWE-bench trajectories for agent identification')
    parser.add_argument('--agent', type=str, default='20240620_sweagent_claude3.5sonnet',
                        help='Agent submission directory name')
    parser.add_argument('--upload_and_grade', action='store_true',
                        help='Upload trajectories to Lunette and grade them')
    parser.add_argument('--compare_agents', action='store_true',
                        help='Compare multiple agents')
    parser.add_argument('--agents', type=str, default=None,
                        help='Comma-separated list of agents to compare')
    parser.add_argument('--num_tasks', type=int, default=50,
                        help='Number of tasks to grade per agent')
    parser.add_argument('--no_strip', action='store_true',
                        help='Do not strip trajectories to edits only')
    parser.add_argument('--output_dir', type=str, default='chris_output/agent_identification',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments_dir = Path(__file__).resolve().parents[1] / 'experiments'
    verified_dir = experiments_dir / 'evaluation' / 'verified'

    if not LUNETTE_AVAILABLE:
        print("Lunette SDK not available. Please install: pip install lunette-sdk")
        return

    if args.compare_agents:
        # Compare multiple agents
        if args.agents:
            agent_names = [a.strip() for a in args.agents.split(',')]
        else:
            # Try to load existing results
            existing_files = list(output_dir.glob('*_behavioral_features.csv'))
            if existing_files:
                print(f"Loading {len(existing_files)} existing result files...")
                agent_features = {}
                for f in existing_files:
                    agent_name = f.stem.replace('_behavioral_features', '')
                    df = pd.read_csv(f)
                    if len(df) > 0:
                        agent_features[agent_name] = df

                if len(agent_features) >= 2:
                    await compare_agents_behavioral_features(
                        agent_features,
                        output_dir / 'agent_comparison.png'
                    )
                return
            else:
                print("No existing results found. Specify agents with --agents")
                return

        async with LunetteClient() as client:
            agent_features = {}
            for agent_name in agent_names:
                agent_dir = verified_dir / agent_name
                if not (agent_dir / 'trajs').exists():
                    print(f"Skipping {agent_name} - no trajectories found")
                    continue

                output_path = output_dir / f'{agent_name}_behavioral_features.csv'
                df = await grade_agent_trajectories(
                    client=client,
                    submission_dir=agent_dir,
                    agent_name=agent_name,
                    num_tasks=args.num_tasks,
                    output_path=output_path,
                    strip_to_edits=not args.no_strip,
                    seed=args.seed,
                )
                agent_features[agent_name] = df

            if len(agent_features) >= 2:
                await compare_agents_behavioral_features(
                    agent_features,
                    output_dir / 'agent_comparison.png'
                )

    elif args.upload_and_grade:
        # Grade single agent
        agent_dir = verified_dir / args.agent

        if not (agent_dir / 'trajs').exists():
            print(f"No trajectories found at {agent_dir / 'trajs'}")
            print(f"Download them with:")
            print(f"  cd experiments && python -m analysis.download_logs evaluation/verified/{args.agent} --only_trajs")
            return

        output_path = output_dir / f'{args.agent}_behavioral_features.csv'

        async with LunetteClient() as client:
            df = await grade_agent_trajectories(
                client=client,
                submission_dir=agent_dir,
                agent_name=args.agent,
                num_tasks=args.num_tasks,
                output_path=output_path,
                strip_to_edits=not args.no_strip,
                seed=args.seed,
            )

        print(f"\nGraded {len(df)} trajectories")
        print(f"Results saved to {output_path}")

        # Print feature summary
        feature_cols = [
            'localization_strategy', 'hypothesis_testing', 'fix_scope',
            'incremental_vs_big_bang', 'error_recovery', 'test_driven',
            'exploration_depth', 'edit_precision', 'context_awareness',
            'persistence', 'efficiency', 'verification_approach'
        ]
        available_cols = [col for col in feature_cols if col in df.columns]
        if available_cols:
            print("\nFeature Statistics:")
            print(df[available_cols].describe().round(2).to_string())

    else:
        print("Please specify --upload_and_grade or --compare_agents")
        print("\nExample usage:")
        print("  python llm_judge/agent_identification.py --upload_and_grade --agent 20240620_sweagent_claude3.5sonnet")
        print("  python llm_judge/agent_identification.py --compare_agents --agents agent1,agent2")


if __name__ == '__main__':
    asyncio.run(main())
