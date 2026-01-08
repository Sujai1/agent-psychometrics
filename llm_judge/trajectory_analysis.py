"""
Trajectory Analysis for Agent Behavioral Signatures.

This script analyzes SWE-bench trajectories to identify interpretable behavioral
patterns that distinguish different agents. It focuses on:
1. Action sequences and patterns
2. Problem-solving strategies (exploration vs. exploitation)
3. Edit patterns and code modification styles
4. Error recovery behaviors

Usage:
    # Analyze single agent (use full directory name from experiments/evaluation/verified/)
    python llm_judge/trajectory_analysis.py --visualize --agent 20240620_sweagent_claude3.5sonnet

    # Analyze specific task
    python llm_judge/trajectory_analysis.py --task_id django__django-10880 --agent 20240620_sweagent_claude3.5sonnet

    # Compare multiple agents
    python llm_judge/trajectory_analysis.py --compare_agents --agents 20240620_sweagent_claude3.5sonnet,20240728_sweagent_gpt4o

    # With custom seed for reproducibility
    python llm_judge/trajectory_analysis.py --agent 20240620_sweagent_claude3.5sonnet --seed 42 --sample_size 100
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_trajectory(traj_path: Path) -> dict:
    """Load a trajectory file."""
    with open(traj_path) as f:
        return json.load(f)


def extract_actions(trajectory: dict) -> list[dict]:
    """Extract action sequences from a trajectory."""
    actions = []
    traj_list = trajectory.get('trajectory', [])

    for i, step in enumerate(traj_list):
        action_str = step.get('action', '')
        observation = step.get('observation', '')
        thought = step.get('thought', '')

        # Classify action type
        action_type = classify_action(action_str)

        actions.append({
            'step': i,
            'action_raw': action_str,
            'action_type': action_type,
            'observation_length': len(observation),
            'thought_length': len(thought),
            'success': 'error' not in observation.lower() and 'traceback' not in observation.lower(),
        })

    return actions


def classify_action(action_str: str) -> str:
    """Classify an action string into a high-level category."""
    action_lower = action_str.lower().strip()

    # Navigation/exploration
    if action_lower.startswith(('find_file', 'search_file', 'search_dir', 'ls', 'find ')):
        return 'search'
    if action_lower.startswith(('open ', 'goto ', 'scroll_')):
        return 'navigate'
    if action_lower.startswith('grep') or 'grep ' in action_lower:
        return 'search'

    # File modification
    if action_lower.startswith('edit '):
        return 'edit'
    if action_lower.startswith('create '):
        return 'create'

    # Execution/testing
    if action_lower.startswith(('python ', 'pytest', 'make ', 'pip ')):
        return 'execute'
    if 'test' in action_lower:
        return 'test'

    # Git operations
    if action_lower.startswith('git '):
        return 'git'

    # Terminal navigation
    if action_lower.startswith('cd '):
        return 'cd'

    # Submission
    if action_lower.startswith('submit'):
        return 'submit'

    return 'other'


def extract_edits(trajectory: dict) -> list[dict]:
    """Extract all code edits from a trajectory."""
    edits = []
    traj_list = trajectory.get('trajectory', [])

    for i, step in enumerate(traj_list):
        action_str = step.get('action', '')

        if action_str.lower().startswith('edit '):
            # Parse edit command - format is "edit START:END\n<content>\nend_of_edit"
            lines = action_str.split('\n')
            if len(lines) > 1:
                # Extract line range
                match = re.match(r'edit\s+(\d+):(\d+)', lines[0], re.IGNORECASE)
                if match:
                    start_line = int(match.group(1))
                    end_line = int(match.group(2))

                    # Extract edit content (everything between first line and end_of_edit)
                    content_lines = []
                    for line in lines[1:]:
                        if line.strip().lower() == 'end_of_edit':
                            break
                        content_lines.append(line)

                    edits.append({
                        'step': i,
                        'start_line': start_line,
                        'end_line': end_line,
                        'lines_modified': end_line - start_line + 1,
                        'content_lines': len(content_lines),
                        'content': '\n'.join(content_lines),
                    })

    return edits


def compute_behavioral_features(trajectory: dict) -> dict:
    """Compute behavioral features from a trajectory."""
    actions = extract_actions(trajectory)
    edits = extract_edits(trajectory)

    if not actions:
        return {}

    # Action counts by type
    action_counts = Counter(a['action_type'] for a in actions)
    total_actions = len(actions)

    # Compute ratios
    features = {
        'total_actions': total_actions,
        'total_edits': len(edits),

        # Action type ratios
        'search_ratio': action_counts.get('search', 0) / total_actions,
        'navigate_ratio': action_counts.get('navigate', 0) / total_actions,
        'edit_ratio': action_counts.get('edit', 0) / total_actions,
        'execute_ratio': action_counts.get('execute', 0) / total_actions,
        'test_ratio': action_counts.get('test', 0) / total_actions,
        'create_ratio': action_counts.get('create', 0) / total_actions,

        # Error rate
        'error_rate': 1 - sum(a['success'] for a in actions) / total_actions,

        # Exploration vs exploitation
        'exploration_ratio': (action_counts.get('search', 0) + action_counts.get('navigate', 0)) / total_actions,
        'exploitation_ratio': (action_counts.get('edit', 0) + action_counts.get('create', 0)) / total_actions,
    }

    # Edit statistics
    if edits:
        features['avg_edit_size'] = np.mean([e['lines_modified'] for e in edits])
        features['max_edit_size'] = max(e['lines_modified'] for e in edits)
        features['edit_concentration'] = len(set(e['start_line'] for e in edits)) / len(edits)
    else:
        features['avg_edit_size'] = 0
        features['max_edit_size'] = 0
        features['edit_concentration'] = 0

    # Action sequence patterns
    action_sequence = [a['action_type'] for a in actions]

    # First N actions pattern (problem-solving approach)
    first_5_actions = action_sequence[:5] if len(action_sequence) >= 5 else action_sequence
    features['starts_with_search'] = 1 if first_5_actions and first_5_actions[0] == 'search' else 0
    features['starts_with_navigate'] = 1 if first_5_actions and first_5_actions[0] == 'navigate' else 0

    # Iteration patterns (search->edit->test cycles)
    features['test_after_edit'] = sum(
        1 for i in range(len(action_sequence) - 1)
        if action_sequence[i] == 'edit' and action_sequence[i + 1] in ('test', 'execute')
    )

    return features


def analyze_agent_trajectories(
    trajs_dir: Path,
    results_path: Optional[Path] = None,
    sample_size: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Analyze all trajectories for an agent and compute aggregate features.

    Args:
        seed: Random seed for reproducible task sampling. With the same seed,
              the same tasks will be selected across runs.
    """
    import random

    # Load results if available
    resolved_set = set()
    if results_path and results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
            resolved_set = set(results.get('resolved', []))

    # Get trajectory files - sort for reproducibility with fixed seed
    traj_files = sorted(trajs_dir.glob('*.traj'), key=lambda p: p.stem)
    print(f"Using random seed: {seed}")
    random.seed(seed)
    if len(traj_files) > sample_size:
        traj_files = random.sample(traj_files, sample_size)
        print(f"Sampled {sample_size} tasks. First 5: {[f.stem for f in traj_files[:5]]}")

    all_features = []
    for traj_path in traj_files:
        task_id = traj_path.stem
        try:
            traj = load_trajectory(traj_path)
            features = compute_behavioral_features(traj)
            features['task_id'] = task_id
            features['resolved'] = task_id in resolved_set
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {traj_path}: {e}")

    return pd.DataFrame(all_features)


def visualize_action_distribution(df: pd.DataFrame, agent_name: str, output_path: Optional[Path] = None):
    """Visualize the distribution of action types for an agent."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Action type ratios
    action_cols = ['search_ratio', 'navigate_ratio', 'edit_ratio', 'execute_ratio', 'test_ratio', 'create_ratio']
    means = [df[col].mean() for col in action_cols]
    labels = ['Search', 'Navigate', 'Edit', 'Execute', 'Test', 'Create']

    axes[0, 0].bar(labels, means)
    axes[0, 0].set_title(f'Action Type Distribution - {agent_name}')
    axes[0, 0].set_ylabel('Mean Ratio')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Exploration vs Exploitation
    axes[0, 1].hist(df['exploration_ratio'], bins=20, alpha=0.7, label='Exploration')
    axes[0, 1].hist(df['exploitation_ratio'], bins=20, alpha=0.7, label='Exploitation')
    axes[0, 1].set_title('Exploration vs Exploitation')
    axes[0, 1].set_xlabel('Ratio')
    axes[0, 1].legend()

    # Total actions distribution
    axes[1, 0].hist(df['total_actions'], bins=30)
    axes[1, 0].set_title('Distribution of Trajectory Lengths')
    axes[1, 0].set_xlabel('Total Actions')

    # Error rate vs resolution
    if 'resolved' in df.columns:
        resolved_errors = df[df['resolved']]['error_rate']
        unresolved_errors = df[~df['resolved']]['error_rate']
        axes[1, 1].boxplot([resolved_errors, unresolved_errors], labels=['Resolved', 'Unresolved'])
        axes[1, 1].set_title('Error Rate by Resolution Status')
        axes[1, 1].set_ylabel('Error Rate')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def compare_agents(agent_features: dict[str, pd.DataFrame], output_path: Optional[Path] = None):
    """Compare behavioral features across multiple agents."""
    # Compute mean features for each agent
    comparison_data = {}
    for agent_name, df in agent_features.items():
        feature_cols = [col for col in df.columns if col not in ('task_id', 'resolved')]
        comparison_data[agent_name] = df[feature_cols].mean()

    comparison_df = pd.DataFrame(comparison_data).T

    print("\nAgent Comparison (Mean Features):")
    print("=" * 80)
    print(comparison_df.round(3).to_string())

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Feature comparison heatmap
    key_features = ['exploration_ratio', 'exploitation_ratio', 'error_rate', 'avg_edit_size', 'total_actions']
    subset = comparison_df[key_features]

    im = axes[0, 0].imshow(subset.values, aspect='auto', cmap='viridis')
    axes[0, 0].set_xticks(range(len(key_features)))
    axes[0, 0].set_xticklabels(key_features, rotation=45, ha='right')
    axes[0, 0].set_yticks(range(len(agent_features)))
    axes[0, 0].set_yticklabels(list(agent_features.keys()))
    axes[0, 0].set_title('Feature Comparison Heatmap')
    plt.colorbar(im, ax=axes[0, 0])

    # Exploration/Exploitation scatter
    for agent_name, df in agent_features.items():
        axes[0, 1].scatter(
            df['exploration_ratio'],
            df['exploitation_ratio'],
            alpha=0.5,
            label=agent_name
        )
    axes[0, 1].set_xlabel('Exploration Ratio')
    axes[0, 1].set_ylabel('Exploitation Ratio')
    axes[0, 1].set_title('Exploration vs Exploitation by Agent')
    axes[0, 1].legend()

    # Total actions distribution
    for agent_name, df in agent_features.items():
        axes[1, 0].hist(df['total_actions'], bins=30, alpha=0.5, label=agent_name)
    axes[1, 0].set_xlabel('Total Actions')
    axes[1, 0].set_title('Trajectory Length Distribution')
    axes[1, 0].legend()

    # Error rate comparison
    error_rates = [df['error_rate'].mean() for df in agent_features.values()]
    agent_names = list(agent_features.keys())
    axes[1, 1].bar(agent_names, error_rates)
    axes[1, 1].set_ylabel('Mean Error Rate')
    axes[1, 1].set_title('Error Rate by Agent')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved comparison to {output_path}")
    else:
        plt.show()

    return comparison_df


def print_trajectory_summary(traj_path: Path):
    """Print a human-readable summary of a trajectory."""
    traj = load_trajectory(traj_path)
    actions = extract_actions(traj)
    edits = extract_edits(traj)

    print(f"\n{'='*80}")
    print(f"Trajectory: {traj_path.stem}")
    print(f"{'='*80}")

    print(f"\nTotal Steps: {len(actions)}")
    print(f"Total Edits: {len(edits)}")

    action_counts = Counter(a['action_type'] for a in actions)
    print("\nAction Type Distribution:")
    for action_type, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action_type:15s}: {count:3d} ({count/len(actions)*100:.1f}%)")

    print("\nAction Sequence (first 20 steps):")
    for i, action in enumerate(actions[:20]):
        status = "OK" if action['success'] else "ERR"
        action_preview = action['action_raw'][:60].replace('\n', ' ')
        print(f"  {i:2d}. [{action['action_type']:8s}] [{status:3s}] {action_preview}...")

    if len(actions) > 20:
        print(f"  ... ({len(actions) - 20} more steps)")

    if edits:
        print("\nEdit Summary:")
        for edit in edits[:5]:
            print(f"  Step {edit['step']}: Lines {edit['start_line']}-{edit['end_line']} ({edit['lines_modified']} lines)")
        if len(edits) > 5:
            print(f"  ... ({len(edits) - 5} more edits)")


def main():
    parser = argparse.ArgumentParser(description='Analyze SWE-bench trajectories for behavioral signatures')
    parser.add_argument('--agent', type=str, default='20240620_sweagent_claude3.5sonnet',
                        help='Agent submission directory name (full name from experiments/evaluation/verified/)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--compare_agents', action='store_true',
                        help='Compare multiple agents')
    parser.add_argument('--agents', type=str, default=None,
                        help='Comma-separated list of agents to compare')
    parser.add_argument('--sample_size', type=int, default=50,
                        help='Number of trajectories to sample per agent')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--task_id', type=str, default=None,
                        help='Specific task ID to analyze')
    parser.add_argument('--output_dir', type=str, default='chris_output/trajectory_analysis',
                        help='Output directory for results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments_dir = Path(__file__).resolve().parents[1] / 'experiments'
    verified_dir = experiments_dir / 'evaluation' / 'verified'

    if args.task_id:
        # Analyze specific trajectory
        agent_dir = verified_dir / args.agent
        traj_path = agent_dir / 'trajs' / f'{args.task_id}.traj'
        if traj_path.exists():
            print_trajectory_summary(traj_path)
        else:
            print(f"Trajectory not found: {traj_path}")
        return

    if args.compare_agents:
        # Compare multiple agents
        if args.agents:
            agent_names = [a.strip() for a in args.agents.split(',')]
        else:
            # Default to available agents with trajectories
            agent_names = []
            for agent_path in verified_dir.iterdir():
                trajs_dir = agent_path / 'trajs'
                if trajs_dir.exists() and list(trajs_dir.glob('*.traj')):
                    agent_names.append(agent_path.name)
            agent_names = agent_names[:5]  # Limit to 5 for comparison

        agent_features = {}
        for agent_name in agent_names:
            agent_dir = verified_dir / agent_name
            trajs_dir = agent_dir / 'trajs'
            results_path = agent_dir / 'results' / 'results.json'

            if not trajs_dir.exists():
                print(f"Skipping {agent_name} - no trajectories found")
                continue

            print(f"\nAnalyzing {agent_name}...")
            df = analyze_agent_trajectories(trajs_dir, results_path, sample_size=args.sample_size, seed=args.seed)
            if len(df) > 0:
                agent_features[agent_name] = df

        if len(agent_features) >= 2:
            compare_agents(agent_features, output_dir / 'agent_comparison.png')

    else:
        # Analyze single agent
        agent_dir = verified_dir / args.agent
        trajs_dir = agent_dir / 'trajs'
        results_path = agent_dir / 'results' / 'results.json'

        if not trajs_dir.exists():
            print(f"No trajectories found at {trajs_dir}")
            print(f"You may need to download them first:")
            print(f"  cd experiments && python -m analysis.download_logs evaluation/verified/{args.agent} --only_trajs")
            return

        print(f"Analyzing {args.agent}...")
        df = analyze_agent_trajectories(trajs_dir, results_path, sample_size=args.sample_size, seed=args.seed)

        print(f"\nAnalyzed {len(df)} trajectories")
        print("\nFeature Statistics:")
        print(df.describe().round(3).to_string())

        # Save features
        df.to_csv(output_dir / f'{args.agent}_features.csv', index=False)
        print(f"\nSaved features to {output_dir / f'{args.agent}_features.csv'}")

        if args.visualize:
            visualize_action_distribution(df, args.agent, output_dir / f'{args.agent}_actions.png')


if __name__ == '__main__':
    main()
