"""
Experiment C: Rigorous Cost Comparison

Compare the cost of:
1. Lunette grading (dummy agent + investigation)
2. Full Claude Sonnet 4.5 agent run
3. Full GPT-5.2 agent run

Records:
- Wall clock time
- Input/output tokens (for full agent runs)
- Run IDs (for Lunette server-side cost lookup)
- Actual computed costs based on API pricing

Usage:
    # Run full experiment
    python llm_judge/experiment_c_rigorous.py --n_tasks 10

    # Dry run to see selected tasks
    python llm_judge/experiment_c_rigorous.py --n_tasks 10 --dry_run

    # Just run Lunette grading (skip full agent runs)
    python llm_judge/experiment_c_rigorous.py --n_tasks 10 --lunette_only
"""

import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lunette import LunetteClient
from lunette.analysis import GradingPlan

# ============================================================================
# Configuration
# ============================================================================

# Model IDs (dated checkpoints for reproducibility)
# https://docs.claude.com/en/docs/about-claude/models/overview
CLAUDE_SONNET_MODEL = "claude-sonnet-4-5-20250929"
# https://platform.openai.com/docs/models/gpt-5.2
GPT52_MODEL = "gpt-5.2-2025-12-11"

# API Pricing (per million tokens)
# Sources:
# - Claude: https://www.anthropic.com/pricing
# - GPT-5.2: https://openai.com/api/pricing/
PRICING = {
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},  # $3/$15 per 1M
    "gpt-5.2-2025-12-11": {"input": 1.75, "output": 14.0},  # $1.75/$14 per 1M
    "lunette_investigation": {"per_run": 0.50},  # Estimated ~$0.50 per investigation
}

# Output directory
OUTPUT_DIR = Path("chris_output/experiment_c")

# Grading prompt for Lunette
DIFFICULTY_GRADING_PROMPT = """You are analyzing a SWE-bench task to predict its difficulty.

You have access to the sandbox environment. Please:
1. Explore the repository structure
2. Read the problem statement carefully
3. Try to locate relevant files and understand the scope of the fix

Based on your exploration, evaluate:
- Problem clarity (1-5)
- Fix hints in description (0-3)
- Domain knowledge required (1-5)
- Fix complexity (1-5)
- Codebase complexity (1-5)

Return a difficulty score from 0.0 (trivial) to 1.0 (extremely hard), with reasoning.
"""


# ============================================================================
# Task Selection
# ============================================================================

def select_tasks(n_tasks: int, seed: int = 42) -> list[str]:
    """Select a stratified sample of SWE-bench tasks."""
    import random
    from datasets import load_dataset

    random.seed(seed)

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    # Group by repo
    repo_tasks = {}
    for item in ds:
        repo = item["repo"]
        if repo not in repo_tasks:
            repo_tasks[repo] = []
        repo_tasks[repo].append(item["instance_id"])

    # Round-robin selection from repos
    selected = []
    repos = list(repo_tasks.keys())
    random.shuffle(repos)

    repo_idx = 0
    while len(selected) < n_tasks and any(repo_tasks.values()):
        repo = repos[repo_idx % len(repos)]
        if repo_tasks[repo]:
            task = random.choice(repo_tasks[repo])
            repo_tasks[repo].remove(task)
            selected.append(task)
        repo_idx += 1

    return selected[:n_tasks]


# ============================================================================
# Lunette Grading (per task)
# ============================================================================

async def run_single_task_lunette(task_id: str, output_dir: Path) -> dict:
    """Run dummy agent + Lunette investigation on a single task.

    Returns dict with timing and cost data.
    """
    result = {
        "task_id": task_id,
        "method": "Lunette",
        "dummy_agent_time": 0,
        "investigation_time": 0,
        "total_time": 0,
        "run_id": None,
        "investigation_run_id": None,
        "score": None,
        "error": None,
    }

    # Step 1: Run dummy agent on single task
    print(f"  [1/2] Running dummy agent for {task_id}...")
    start_time = time.time()

    cmd = [
        "inspect", "eval",
        "lunette_utils/dummy_swebench_task.py@dummy_swebench",
        "--model", "mockllm/model",
        "--sandbox", "lunette",
        "--sample-id", task_id,
        "--no-score",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        result["dummy_agent_time"] = time.time() - start_time

        if proc.returncode != 0:
            result["error"] = f"Dummy agent failed: {proc.stderr[:500]}"
            return result

    except subprocess.TimeoutExpired:
        result["error"] = "Dummy agent timed out"
        return result

    # Step 2: Find the run ID
    async with LunetteClient() as client:
        import httpx
        async with httpx.AsyncClient(
            base_url=client.base_url,
            headers={"X-API-Key": client.api_key},
            timeout=30
        ) as http:
            r = await http.get("/runs/")
            runs = r.json()

            # Find the most recent dummy_swebench run
            for run in runs:
                if run.get("task") == "dummy_swebench":
                    result["run_id"] = run.get("id")
                    break

        if not result["run_id"]:
            result["error"] = "Could not find run ID"
            return result

        # Step 3: Run investigation
        print(f"  [2/2] Running Lunette investigation...")
        start_time = time.time()

        try:
            plan = GradingPlan(
                name='experiment-c-difficulty',
                prompt=DIFFICULTY_GRADING_PROMPT,
                enable_sandbox=True,
            )

            investigation = await client.investigate(
                run_id=result["run_id"],
                plan=plan,
                limit=1,
            )

            result["investigation_time"] = time.time() - start_time
            result["investigation_run_id"] = investigation.run_id

            if investigation.results:
                result["score"] = investigation.results[0].data.get("score")

        except Exception as e:
            result["investigation_time"] = time.time() - start_time
            result["error"] = f"Investigation failed: {str(e)}"

    result["total_time"] = result["dummy_agent_time"] + result["investigation_time"]
    return result


# ============================================================================
# Full Agent Run (per task)
# ============================================================================

def run_single_task_full_agent(task_id: str, model: str, output_dir: Path) -> dict:
    """Run a full SWE-bench agent on a single task.

    Returns dict with timing and token usage.
    """
    result = {
        "task_id": task_id,
        "method": model,
        "total_time": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "log_path": None,
        "resolved": None,
        "error": None,
    }

    # Determine provider prefix
    if "claude" in model.lower():
        model_spec = f"anthropic/{model}"
    elif "gpt" in model.lower():
        model_spec = f"openai/{model}"
    else:
        model_spec = model

    print(f"  Running {model} on {task_id}...")
    start_time = time.time()

    cmd = [
        "inspect", "eval",
        "inspect_evals/swe_bench",
        "--model", model_spec,
        "--sandbox", "docker",
        "--sample-id", task_id,
        "--message-limit", "50",
        "--max-connections", "1",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per task
        )

        result["total_time"] = time.time() - start_time

        # Parse log path from output
        for line in proc.stdout.split("\n"):
            if "Log:" in line:
                result["log_path"] = line.split("Log:")[-1].strip()
                break

        # Extract token counts from log
        if result["log_path"] and Path(result["log_path"]).exists():
            tokens = parse_inspect_log_for_tokens(result["log_path"])
            result["input_tokens"] = tokens["input_tokens"]
            result["output_tokens"] = tokens["output_tokens"]
            result["resolved"] = tokens.get("resolved")

        if proc.returncode != 0 and not result["log_path"]:
            result["error"] = proc.stderr[:500] if proc.stderr else "Unknown error"

    except subprocess.TimeoutExpired:
        result["total_time"] = 1800
        result["error"] = "Timed out after 30 minutes"

    return result


def parse_inspect_log_for_tokens(log_path: str) -> dict:
    """Parse an inspect log file to extract token counts."""
    try:
        from inspect_ai.log import read_eval_log

        log = read_eval_log(log_path)

        total_input = 0
        total_output = 0
        resolved = None

        if log.samples:
            for sample in log.samples:
                # Get token usage
                if hasattr(sample, 'model_usage') and sample.model_usage:
                    for usage in sample.model_usage.values():
                        total_input += usage.input_tokens or 0
                        total_output += usage.output_tokens or 0

                # Get resolved status
                if hasattr(sample, 'scores') and sample.scores:
                    for score_name, score_val in sample.scores.items():
                        if 'resolved' in score_name.lower():
                            resolved = score_val.value if hasattr(score_val, 'value') else score_val

        # Fallback to eval-level stats
        if hasattr(log, 'stats') and log.stats:
            if hasattr(log.stats, 'model_usage') and log.stats.model_usage:
                for usage in log.stats.model_usage.values():
                    if total_input == 0:
                        total_input = usage.input_tokens or 0
                    if total_output == 0:
                        total_output = usage.output_tokens or 0

        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "resolved": resolved,
        }

    except Exception as e:
        print(f"    Warning: Could not parse log: {e}")
        return {"input_tokens": 0, "output_tokens": 0, "resolved": None}


# ============================================================================
# Cost Calculation
# ============================================================================

def calculate_cost(method: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Calculate API cost based on method and token usage."""
    if method == "Lunette":
        return PRICING["lunette_investigation"]["per_run"]

    for model_id, pricing in PRICING.items():
        if model_id in method or method in model_id:
            if "per_run" in pricing:
                return pricing["per_run"]
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            return input_cost + output_cost

    return 0.0


# ============================================================================
# Visualization
# ============================================================================

def create_comparison_plots(results_df: pd.DataFrame, output_dir: Path):
    """Create comparison plots for cost and time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to successful runs
    valid_df = results_df[results_df["error"].isna()].copy()

    if len(valid_df) == 0:
        print("No successful results to plot")
        return None

    # Calculate costs
    valid_df["cost_usd"] = valid_df.apply(
        lambda r: calculate_cost(r["method"], r.get("input_tokens", 0), r.get("output_tokens", 0)),
        axis=1
    )

    # Aggregate by method
    method_stats = valid_df.groupby("method").agg({
        "total_time": ["mean", "std", "sum", "count"],
        "cost_usd": ["mean", "std", "sum"],
        "input_tokens": "sum",
        "output_tokens": "sum",
    }).reset_index()

    method_stats.columns = [
        "method",
        "time_mean", "time_std", "time_total", "n_tasks",
        "cost_mean", "cost_std", "cost_total",
        "input_tokens", "output_tokens",
    ]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    methods = method_stats["method"].tolist()
    colors = {"Lunette": "#2ecc71", CLAUDE_SONNET_MODEL: "#3498db", GPT52_MODEL: "#e74c3c"}
    bar_colors = [colors.get(m, "#95a5a6") for m in methods]

    # Plot 1: Total Cost
    ax1 = axes[0, 0]
    costs = method_stats["cost_total"].tolist()
    bars = ax1.bar(range(len(methods)), costs, color=bar_colors, edgecolor="black")
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.split("-")[0] if "-" in m else m for m in methods], rotation=45, ha="right")
    ax1.set_ylabel("Total Cost (USD)")
    ax1.set_title("Total Cost Comparison")
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"${cost:.2f}",
                ha="center", va="bottom", fontsize=9)

    # Plot 2: Total Time
    ax2 = axes[0, 1]
    times = [t / 60 for t in method_stats["time_total"].tolist()]
    bars = ax2.bar(range(len(methods)), times, color=bar_colors, edgecolor="black")
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([m.split("-")[0] if "-" in m else m for m in methods], rotation=45, ha="right")
    ax2.set_ylabel("Total Time (minutes)")
    ax2.set_title("Total Time Comparison")
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{t:.1f}m",
                ha="center", va="bottom", fontsize=9)

    # Plot 3: Cost per Task
    ax3 = axes[1, 0]
    costs_per_task = method_stats["cost_mean"].tolist()
    cost_errs = method_stats["cost_std"].fillna(0).tolist()
    bars = ax3.bar(range(len(methods)), costs_per_task, yerr=cost_errs,
                   color=bar_colors, edgecolor="black", capsize=5)
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels([m.split("-")[0] if "-" in m else m for m in methods], rotation=45, ha="right")
    ax3.set_ylabel("Cost per Task (USD)")
    ax3.set_title("Average Cost per Task (±1 std)")
    for bar, cost in zip(bars, costs_per_task):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"${cost:.2f}",
                ha="center", va="bottom", fontsize=9)

    # Plot 4: Time per Task
    ax4 = axes[1, 1]
    times_per_task = [t / 60 for t in method_stats["time_mean"].tolist()]
    time_errs = [t / 60 for t in method_stats["time_std"].fillna(0).tolist()]
    bars = ax4.bar(range(len(methods)), times_per_task, yerr=time_errs,
                   color=bar_colors, edgecolor="black", capsize=5)
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels([m.split("-")[0] if "-" in m else m for m in methods], rotation=45, ha="right")
    ax4.set_ylabel("Time per Task (minutes)")
    ax4.set_title("Average Time per Task (±1 std)")
    for bar, t in zip(bars, times_per_task):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{t:.1f}m",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "cost_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "cost_comparison.pdf", bbox_inches="tight")
    print(f"\nPlots saved to {output_dir}/cost_comparison.png")

    # Save summary CSV
    method_stats.to_csv(output_dir / "summary.csv", index=False)

    return method_stats


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Experiment C: Rigorous Cost Comparison")
    parser.add_argument("--n_tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", help="Just show selected tasks")
    parser.add_argument("--lunette_only", action="store_true", help="Only run Lunette grading")
    parser.add_argument("--skip_lunette", action="store_true", help="Skip Lunette, only run agents")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select tasks
    print("=" * 60)
    print("EXPERIMENT C: Rigorous Cost Comparison")
    print("=" * 60)

    print(f"\nSelecting {args.n_tasks} tasks (seed={args.seed})...")
    task_ids = select_tasks(args.n_tasks, args.seed)

    print(f"\nSelected tasks:")
    for i, task_id in enumerate(task_ids):
        print(f"  {i+1}. {task_id}")

    # Save task list
    with open(output_dir / "selected_tasks.json", "w") as f:
        json.dump({"tasks": task_ids, "seed": args.seed, "timestamp": datetime.now().isoformat()}, f, indent=2)

    if args.dry_run:
        print("\nDry run - not executing experiments")
        return

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ========================================================================
    # Phase 1: Lunette Grading
    # ========================================================================
    if not args.skip_lunette:
        print("\n" + "=" * 60)
        print("PHASE 1: Lunette Grading (Dummy Agent + Investigation)")
        print("=" * 60)

        for i, task_id in enumerate(task_ids):
            print(f"\n[{i+1}/{len(task_ids)}] {task_id}")

            result = await run_single_task_lunette(task_id, output_dir)
            all_results.append(result)

            # Print result
            if result["error"]:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Time: {result['total_time']:.1f}s (agent: {result['dummy_agent_time']:.1f}s, investigation: {result['investigation_time']:.1f}s)")
                print(f"  Run ID: {result['run_id']}")
                print(f"  Score: {result['score']}")

            # Save intermediate results
            pd.DataFrame(all_results).to_csv(output_dir / f"results_{timestamp}.csv", index=False)

    # ========================================================================
    # Phase 2: Full Agent Runs
    # ========================================================================
    if not args.lunette_only:
        for model in [CLAUDE_SONNET_MODEL, GPT52_MODEL]:
            print("\n" + "=" * 60)
            print(f"PHASE 2: Full Agent ({model})")
            print("=" * 60)

            for i, task_id in enumerate(task_ids):
                print(f"\n[{i+1}/{len(task_ids)}] {task_id}")

                result = run_single_task_full_agent(task_id, model, output_dir)
                all_results.append(result)

                # Calculate cost
                cost = calculate_cost(model, result["input_tokens"], result["output_tokens"])

                # Print result
                if result["error"]:
                    print(f"  ERROR: {result['error']}")
                else:
                    print(f"  Time: {result['total_time']:.1f}s ({result['total_time']/60:.1f} min)")
                    print(f"  Tokens: {result['input_tokens']:,} in / {result['output_tokens']:,} out")
                    print(f"  Cost: ${cost:.2f}")
                    print(f"  Resolved: {result['resolved']}")

                # Save intermediate results
                pd.DataFrame(all_results).to_csv(output_dir / f"results_{timestamp}.csv", index=False)

    # ========================================================================
    # Final Analysis
    # ========================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    results_df = pd.DataFrame(all_results)
    results_path = output_dir / f"results_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Create plots
    if len(results_df) > 0:
        stats = create_comparison_plots(results_df, output_dir)

        if stats is not None:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(stats.to_string(index=False))

    print(f"\nExperiment complete. Output directory: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
