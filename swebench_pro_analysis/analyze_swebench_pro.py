#!/usr/bin/env python3
"""
Analyze SWE-bench Pro CSV data to extract statistics.
"""

import pandas as pd
from pathlib import Path

def main():
    csv_path = Path("data/swe-bench-pro.csv")

    print("Loading SWE-bench Pro CSV...")
    df = pd.read_csv(csv_path)

    print("\n=== SWE-bench Pro Statistics ===\n")

    # Basic counts
    print(f"Total trajectories: {len(df):,}")
    print(f"Unique agent_run_ids: {df['agent_run_id'].nunique():,}")

    # Agent statistics
    n_agents = df['metadata.model_name'].nunique()
    print(f"\nAgents: {n_agents}")
    print("\nAgent names:")
    for agent in sorted(df['metadata.model_name'].unique()):
        count = len(df[df['metadata.model_name'] == agent])
        resolved = df[df['metadata.model_name'] == agent]['metadata.resolved'].sum()
        rate = resolved / count * 100 if count > 0 else 0
        print(f"  {agent:50s} {count:4d} tasks, {resolved:4d} resolved ({rate:.1f}%)")

    # Problem statistics
    n_problems = df['metadata.instance_id'].nunique()
    print(f"\nProblems: {n_problems}")

    # Resolution statistics
    total_resolved = df['metadata.resolved'].sum()
    resolution_rate = total_resolved / len(df) * 100
    print(f"\nResolved trajectories: {total_resolved:,} / {len(df):,} ({resolution_rate:.1f}%)")

    # Check if matrix is complete
    agent_counts = df['metadata.model_name'].value_counts()
    print(f"\nProblems per agent:")
    print(f"  Min: {agent_counts.min()}")
    print(f"  Max: {agent_counts.max()}")
    print(f"  Median: {agent_counts.median():.0f}")

    if agent_counts.min() == agent_counts.max():
        print(f"  ✓ Complete matrix: All agents evaluated on all {n_problems} problems")
    else:
        print(f"  ✗ Incomplete matrix: Agents evaluated on different numbers of problems")
        print(f"\nAgents with fewer than {n_problems} problems:")
        for agent, count in agent_counts.items():
            if count < n_problems:
                print(f"    {agent}: {count}/{n_problems}")

    # Turn statistics
    print(f"\nTurn statistics:")
    print(f"  Mean: {df['metadata.turns'].mean():.1f}")
    print(f"  Median: {df['metadata.turns'].median():.0f}")
    print(f"  Min: {df['metadata.turns'].min()}")
    print(f"  Max: {df['metadata.turns'].max()}")

    # Date range
    print(f"\nDate range:")
    print(f"  Earliest: {df['created_at'].min()}")
    print(f"  Latest: {df['created_at'].max()}")

    print("\n=== Summary ===")
    print(f"SWE-bench Pro contains:")
    print(f"  • {n_agents} agents")
    print(f"  • {n_problems} problems")
    print(f"  • {len(df):,} agent × problem trajectories")
    print(f"  • {total_resolved:,} resolved ({resolution_rate:.1f}%)")

    # Check for duplicates
    duplicates = df.groupby(['metadata.model_name', 'metadata.instance_id']).size()
    if (duplicates > 1).any():
        print(f"\n⚠ Warning: Found {(duplicates > 1).sum()} agent-problem pairs with multiple trajectories")
    else:
        print(f"\n✓ No duplicate agent-problem pairs")

if __name__ == "__main__":
    main()
