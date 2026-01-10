#!/usr/bin/env python3
"""
Visualize SWE-bench Pro statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_agent_performance():
    """Plot agent resolution rates."""
    csv_path = Path("data/swe-bench-pro.csv")
    df = pd.read_csv(csv_path)

    # Calculate resolution rate per agent
    agent_stats = df.groupby('metadata.model_name').agg({
        'metadata.resolved': ['sum', 'count']
    }).reset_index()
    agent_stats.columns = ['agent', 'resolved', 'total']
    agent_stats['rate'] = agent_stats['resolved'] / agent_stats['total'] * 100
    agent_stats = agent_stats.sort_values('rate', ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot horizontal bars
    y_pos = range(len(agent_stats))
    bars = ax.barh(y_pos, agent_stats['rate'], color='steelblue')

    # Color the best performer
    bars[-1].set_color('darkgreen')

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agent_stats['agent'], fontsize=9)
    ax.set_xlabel('Resolution Rate (%)', fontsize=11)
    ax.set_title('SWE-bench Pro: Agent Resolution Rates', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add percentage labels
    for i, (rate, total, resolved) in enumerate(zip(agent_stats['rate'], agent_stats['total'], agent_stats['resolved'])):
        ax.text(rate + 1, i, f'{rate:.1f}% ({resolved}/{total})',
                va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('chris_output/figures/swebench_pro_agent_performance.png', dpi=300, bbox_inches='tight')
    print("Saved: chris_output/figures/swebench_pro_agent_performance.png")
    plt.close()

def plot_turn_distribution():
    """Plot distribution of conversation turns."""
    csv_path = Path("data/swe-bench-pro.csv")
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df['metadata.turns'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df['metadata.turns'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["metadata.turns"].mean():.1f}')
    axes[0].axvline(df['metadata.turns'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["metadata.turns"].median():.0f}')
    axes[0].set_xlabel('Number of Turns', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Turn Distribution (All Trajectories)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plot by resolution status
    df_plot = df[['metadata.resolved', 'metadata.turns']].copy()
    df_plot['status'] = df_plot['metadata.resolved'].map({True: 'Resolved', False: 'Failed'})

    sns.boxplot(data=df_plot, x='status', y='metadata.turns', ax=axes[1], palette={'Resolved': 'green', 'Failed': 'red'})
    axes[1].set_xlabel('Status', fontsize=11)
    axes[1].set_ylabel('Number of Turns', fontsize=11)
    axes[1].set_title('Turns by Resolution Status', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('chris_output/figures/swebench_pro_turn_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: chris_output/figures/swebench_pro_turn_distribution.png")
    plt.close()

def plot_matrix_completeness():
    """Plot how complete each agent's evaluation is."""
    csv_path = Path("data/swe-bench-pro.csv")
    df = pd.read_csv(csv_path)

    # Count problems per agent
    agent_counts = df.groupby('metadata.model_name').size().reset_index(name='problems')
    agent_counts = agent_counts.sort_values('problems', ascending=True)
    agent_counts['completeness'] = agent_counts['problems'] / 730 * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot horizontal bars
    y_pos = range(len(agent_counts))
    bars = ax.barh(y_pos, agent_counts['completeness'], color='steelblue')

    # Color complete agents
    for i, completeness in enumerate(agent_counts['completeness']):
        if completeness >= 100:
            bars[i].set_color('darkgreen')
        elif completeness >= 99:
            bars[i].set_color('limegreen')

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agent_counts['metadata.model_name'], fontsize=9)
    ax.set_xlabel('Completeness (%)', fontsize=11)
    ax.set_title('SWE-bench Pro: Matrix Completeness by Agent\n(% of 730 problems evaluated)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(75, 101)

    # Add labels
    for i, (completeness, problems) in enumerate(zip(agent_counts['completeness'], agent_counts['problems'])):
        ax.text(completeness + 0.2, i, f'{completeness:.1f}% ({problems}/730)',
                va='center', fontsize=8)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkgreen', label='100% complete'),
        Patch(facecolor='limegreen', label='≥99% complete'),
        Patch(facecolor='steelblue', label='<99% complete')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig('chris_output/figures/swebench_pro_matrix_completeness.png', dpi=300, bbox_inches='tight')
    print("Saved: chris_output/figures/swebench_pro_matrix_completeness.png")
    plt.close()

def plot_comparison_table():
    """Create a comparison visualization between SWE-bench Pro and Verified."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Data for comparison
    data = [
        ['Metric', 'SWE-bench Pro', 'SWE-bench Verified'],
        ['Agents', '14', '123'],
        ['Problems', '730', '500'],
        ['Total Trajectories', '9,729', '61,500'],
        ['Matrix Completeness', '77-100%', '100%'],
        ['Avg Resolution Rate', '27.1%', '~27%'],
        ['Date Range', 'Oct 2025 (9 days)', '2020-2025'],
        ['Matrix Status', 'Incomplete', 'Complete'],
    ]

    # Create table
    table = ax.table(cellText=data, cellLoc='left', loc='center',
                     colWidths=[0.3, 0.3, 0.3])

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style other rows
    for i in range(1, len(data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            table[(i, j)].set_text_props(fontsize=10)

    # Increase cell height
    table.scale(1, 2.5)

    plt.title('SWE-bench Pro vs SWE-bench Verified Comparison',
              fontsize=14, fontweight='bold', pad=20)

    plt.savefig('chris_output/figures/swebench_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: chris_output/figures/swebench_comparison.png")
    plt.close()

def main():
    print("Generating SWE-bench Pro visualizations...\n")

    plot_agent_performance()
    plot_turn_distribution()
    plot_matrix_completeness()
    plot_comparison_table()

    print("\n✓ All visualizations complete!")
    print("\nGenerated files:")
    print("  • chris_output/figures/swebench_pro_agent_performance.png")
    print("  • chris_output/figures/swebench_pro_turn_distribution.png")
    print("  • chris_output/figures/swebench_pro_matrix_completeness.png")
    print("  • chris_output/figures/swebench_comparison.png")

if __name__ == "__main__":
    main()
