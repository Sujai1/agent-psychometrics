"""
Run full SWE-bench agent evaluations with token tracking.

This script runs actual LLM agents (Claude Sonnet 4.5, GPT-5.2) on SWE-bench tasks
and tracks token usage for cost comparison with Lunette grading.

Usage:
    # Run Claude Sonnet 4.5 on selected tasks
    python experiment_c/run_full_agent.py --model anthropic/claude-sonnet-4-5-20250929

    # Run GPT-5.2 on selected tasks
    python experiment_c/run_full_agent.py --model openai/gpt-5.2-2025-12-11

    # Specify tasks file
    python experiment_c/run_full_agent.py --model anthropic/claude-sonnet-4-5-20250929 --tasks_file chris_output/experiment_c/selected_tasks.json
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("chris_output/experiment_c")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pricing per 1M tokens (as of Jan 2026)
# Format: provider/model-name
PRICING = {
    "anthropic/claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "openai/gpt-5.2-2025-12-11": {"input": 1.75, "output": 14.0},
}


def run_agent_on_task(model: str, task_id: str) -> dict:
    """
    Run a full agent on a single SWE-bench task.

    Returns dict with timing and token usage.
    """
    print(f"\n--- Running {model} on {task_id} ---")

    result = {
        "task_id": task_id,
        "model": model,
        "timestamp": datetime.now().isoformat(),
    }

    start_time = time.time()

    # Run inspect eval with SWE-bench task
    # Using inspect_evals/swe_bench with dataset parameter for verified
    cmd = [
        "inspect",
        "eval",
        "inspect_evals/swe_bench",
        "-T",
        "dataset=princeton-nlp/SWE-bench_Verified",
        "--model",
        model,
        "--sandbox",
        "lunette",
        "--sample-id",
        task_id,
        "--message-limit",
        "50",  # Limit messages to control costs
    ]

    print(f"  Command: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800  # 30 min timeout
        )
        result["elapsed_seconds"] = time.time() - start_time
        result["return_code"] = proc.returncode

        # Capture both stdout and stderr for debugging
        result["stdout"] = proc.stdout[:2000] if proc.stdout else ""
        result["stderr"] = proc.stderr[:2000] if proc.stderr else ""

        if proc.returncode != 0:
            result["error"] = proc.stderr[:1000] if proc.stderr else proc.stdout[:1000]
            print(f"  Failed: {(proc.stderr or proc.stdout)[:200]}")
        else:
            print(f"  Completed in {result['elapsed_seconds']:.1f}s")

            # Try to extract token usage from the log
            # Inspect logs token usage in the eval log (.eval files)
            try:
                # Find the most recent log file
                log_dir = Path("logs")
                if log_dir.exists():
                    log_files = sorted(log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if log_files:
                        with open(log_files[0]) as f:
                            log_data = json.load(f)
                            if "stats" in log_data:
                                stats = log_data["stats"]
                                result["input_tokens"] = stats.get("input_tokens", 0)
                                result["output_tokens"] = stats.get("output_tokens", 0)
                                result["total_tokens"] = result["input_tokens"] + result["output_tokens"]

                                # Calculate cost
                                if model in PRICING:
                                    pricing = PRICING[model]
                                    input_cost = (result["input_tokens"] / 1_000_000) * pricing["input"]
                                    output_cost = (result["output_tokens"] / 1_000_000) * pricing["output"]
                                    result["cost_usd"] = input_cost + output_cost
                                    print(f"  Tokens: {result['input_tokens']:,} in / {result['output_tokens']:,} out")
                                    print(f"  Cost: ${result['cost_usd']:.4f}")
            except Exception as e:
                result["token_extraction_error"] = str(e)

    except subprocess.TimeoutExpired:
        result["elapsed_seconds"] = time.time() - start_time
        result["error"] = "Timeout after 30 minutes"
        print(f"  Timeout after 30 minutes")
    except Exception as e:
        result["elapsed_seconds"] = time.time() - start_time
        result["error"] = str(e)
        print(f"  Error: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run full agent on SWE-bench tasks")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use (e.g., claude-sonnet-4-5-20250929, gpt-5.2-2025-12-11)",
    )
    parser.add_argument(
        "--tasks_file",
        type=str,
        default="chris_output/experiment_c/selected_tasks.json",
        help="JSON file with selected tasks",
    )
    args = parser.parse_args()

    # Load tasks
    with open(args.tasks_file) as f:
        tasks_data = json.load(f)
    tasks = tasks_data["tasks"]

    print("=" * 60)
    print(f"Running {args.model} on {len(tasks)} tasks")
    print("=" * 60)
    print(f"Tasks: {tasks}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "tasks_file": args.tasks_file,
        "tasks": [],
    }

    # Run each task sequentially
    for i, task_id in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Task {i+1}/{len(tasks)}: {task_id}")
        print("=" * 60)

        task_result = run_agent_on_task(args.model, task_id)
        results["tasks"].append(task_result)

        # Save intermediate results
        # Extract model name: "anthropic/claude-sonnet..." -> "claude_sonnet"
        model_short = args.model.split("/")[-1].split("-")[0] + "_" + args.model.split("/")[-1].split("-")[1]
        output_file = OUTPUT_DIR / f"agent_{model_short}_{len(tasks)}tasks.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [t for t in results["tasks"] if "error" not in t]
    total_time = sum(t.get("elapsed_seconds", 0) for t in results["tasks"])
    total_cost = sum(t.get("cost_usd", 0) for t in successful)
    total_input = sum(t.get("input_tokens", 0) for t in successful)
    total_output = sum(t.get("output_tokens", 0) for t in successful)

    print(f"Model: {args.model}")
    print(f"Tasks attempted: {len(results['tasks'])}")
    print(f"Tasks successful: {len(successful)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total tokens: {total_input:,} in / {total_output:,} out")
    print(f"Total cost: ${total_cost:.4f}")
    if successful:
        print(f"Avg per task: {total_time/len(successful):.1f}s, ${total_cost/len(successful):.4f}")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
