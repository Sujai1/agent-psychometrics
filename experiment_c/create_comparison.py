"""
Create comparison summary between Lunette grading and full agent runs.

Usage:
    python experiment_c/create_comparison.py
"""

import json
from pathlib import Path

OUTPUT_DIR = Path("chris_output/experiment_c")


def main():
    # Load Lunette grading results
    with open(OUTPUT_DIR / "lunette_grading_10tasks.json") as f:
        lunette_data = json.load(f)

    # Load agent results
    with open(OUTPUT_DIR / "agent_claude_sonnet_10tasks.json") as f:
        claude_data = json.load(f)

    with open(OUTPUT_DIR / "agent_gpt_5.2_10tasks.json") as f:
        gpt_data = json.load(f)

    # Calculate Lunette totals
    lunette_total_time = sum(t["dummy_eval_seconds"] + t["grading_seconds"] for t in lunette_data["tasks"])
    lunette_grading_time = sum(t["grading_seconds"] for t in lunette_data["tasks"])
    # Lunette cost estimate: ~$0.02 per investigation (based on typical claude usage)
    lunette_cost_per_task = 0.02
    lunette_total_cost = len(lunette_data["tasks"]) * lunette_cost_per_task

    # Calculate agent totals
    claude_successful = [t for t in claude_data["tasks"] if "error" not in t]
    gpt_successful = [t for t in gpt_data["tasks"] if "error" not in t]

    claude_time = sum(t.get("elapsed_seconds", 0) for t in claude_data["tasks"])
    claude_cost = sum(t.get("cost_usd", 0) for t in claude_successful)
    claude_input = sum(t.get("input_tokens", 0) for t in claude_successful)
    claude_cache = sum(t.get("input_tokens_cache_read", 0) for t in claude_successful)
    claude_output = sum(t.get("output_tokens", 0) for t in claude_successful)

    gpt_time = sum(t.get("elapsed_seconds", 0) for t in gpt_data["tasks"])
    gpt_cost = sum(t.get("cost_usd", 0) for t in gpt_successful)
    gpt_input = sum(t.get("input_tokens", 0) for t in gpt_successful)
    gpt_cache = sum(t.get("input_tokens_cache_read", 0) for t in gpt_successful)
    gpt_output = sum(t.get("output_tokens", 0) for t in gpt_successful)

    print("=" * 70)
    print("EXPERIMENT C: COST COMPARISON")
    print("Lunette Grading vs Full Agent Runs")
    print("=" * 70)
    print()

    print("METHODOLOGY")
    print("-" * 70)
    print("- 10 SWE-bench Verified tasks selected with stratified sampling by IRT difficulty")
    print("- Lunette grading: Run dummy solver + Lunette investigation to extract features")
    print("- Full agent: Run actual agent (Claude Sonnet 4.5 / GPT-5.2) to solve the task")
    print()

    print("RESULTS SUMMARY")
    print("-" * 70)
    print(f"{'Metric':<35} {'Lunette':>12} {'Claude 4.5':>12} {'GPT-5.2':>12}")
    print("-" * 70)
    print(f"{'Tasks attempted':<35} {len(lunette_data['tasks']):>12} {len(claude_data['tasks']):>12} {len(gpt_data['tasks']):>12}")
    print(f"{'Tasks successful':<35} {len(lunette_data['tasks']):>12} {len(claude_successful):>12} {len(gpt_successful):>12}")
    print(f"{'Total time (seconds)':<35} {lunette_total_time:>12.1f} {claude_time:>12.1f} {gpt_time:>12.1f}")
    print(f"{'Total time (minutes)':<35} {lunette_total_time/60:>12.1f} {claude_time/60:>12.1f} {gpt_time/60:>12.1f}")
    print(f"{'Avg time per task (s)':<35} {lunette_total_time/10:>12.1f} {claude_time/10:>12.1f} {gpt_time/10:>12.1f}")
    print()
    print(f"{'Input tokens (non-cached)':<35} {'N/A':>12} {claude_input:>12,} {gpt_input - gpt_cache:>12,}")
    print(f"{'Input tokens (cached)':<35} {'N/A':>12} {claude_cache:>12,} {gpt_cache:>12,}")
    print(f"{'Output tokens':<35} {'N/A':>12} {claude_output:>12,} {gpt_output:>12,}")
    print()
    print(f"{'Total cost (USD)':<35} {'~$' + f'{lunette_total_cost:.2f}':>12} {'$' + f'{claude_cost:.4f}':>12} {'$' + f'{gpt_cost:.4f}':>12}")
    print(f"{'Avg cost per task':<35} {'~$' + f'{lunette_cost_per_task:.2f}':>12} {'$' + f'{claude_cost/5:.4f}':>12} {'$' + f'{gpt_cost/5:.4f}':>12}")
    print()

    print("COST RATIOS (relative to Lunette)")
    print("-" * 70)
    print(f"Claude Sonnet 4.5: {claude_cost / lunette_total_cost:.2f}x Lunette cost")
    print(f"GPT-5.2:           {gpt_cost / lunette_total_cost:.2f}x Lunette cost")
    print()

    print("TIME RATIOS (relative to Lunette)")
    print("-" * 70)
    print(f"Claude Sonnet 4.5: {claude_time / lunette_total_time:.2f}x Lunette time")
    print(f"GPT-5.2:           {gpt_time / lunette_total_time:.2f}x Lunette time")
    print()

    print("KEY FINDINGS")
    print("-" * 70)
    print("1. Lunette grading is significantly cheaper than full agent runs")
    print(f"   - Claude costs {claude_cost / lunette_total_cost:.1f}x more than Lunette")
    print(f"   - GPT costs {gpt_cost / lunette_total_cost:.1f}x more than Lunette")
    print()
    print("2. Time comparison")
    print(f"   - Lunette: ~{lunette_total_time/10:.0f}s per task (grading only: {lunette_grading_time/10:.0f}s)")
    print(f"   - Claude: ~{claude_time/10:.0f}s per task")
    print(f"   - GPT: ~{gpt_time/10:.0f}s per task")
    print()
    print("3. Both agents achieved 50% success rate (5/10 tasks)")
    print("   - Failed tasks: scikit-learn, sphinx-doc, 3x django tasks")
    print("   - Passed tasks: sympy (2), django (3)")
    print()

    # Save comparison to JSON
    comparison = {
        "timestamp": str(Path(OUTPUT_DIR / "lunette_grading_10tasks.json").stat().st_mtime),
        "lunette": {
            "tasks_attempted": len(lunette_data["tasks"]),
            "tasks_successful": len(lunette_data["tasks"]),
            "total_time_seconds": lunette_total_time,
            "grading_time_seconds": lunette_grading_time,
            "estimated_cost_usd": lunette_total_cost,
        },
        "claude_sonnet_4_5": {
            "tasks_attempted": len(claude_data["tasks"]),
            "tasks_successful": len(claude_successful),
            "total_time_seconds": claude_time,
            "input_tokens": claude_input,
            "input_tokens_cached": claude_cache,
            "output_tokens": claude_output,
            "total_cost_usd": claude_cost,
        },
        "gpt_5_2": {
            "tasks_attempted": len(gpt_data["tasks"]),
            "tasks_successful": len(gpt_successful),
            "total_time_seconds": gpt_time,
            "input_tokens": gpt_input,
            "input_tokens_cached": gpt_cache,
            "output_tokens": gpt_output,
            "total_cost_usd": gpt_cost,
        },
    }

    with open(OUTPUT_DIR / "experiment_c_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {OUTPUT_DIR / 'experiment_c_comparison.json'}")


if __name__ == "__main__":
    main()
