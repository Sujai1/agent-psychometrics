"""
Backfill token usage into results files from eval logs.

Usage:
    python experiment_c/backfill_tokens.py
"""

import json
import zipfile
from pathlib import Path
from datetime import datetime

PRICING = {
    "anthropic/claude-sonnet-4-5-20250929": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,  # 10% of input price for cached tokens
    },
    "openai/gpt-5.2-2025-12-11": {
        "input": 1.75,
        "output": 14.0,
        "cache_read": 0.875,  # 50% of input price for cached tokens
    },
}

OUTPUT_DIR = Path("chris_output/experiment_c")
LOGS_DIR = Path("logs")


def extract_info_from_eval(eval_path: Path) -> dict:
    """Extract task info and token usage from a .eval ZIP file."""
    with zipfile.ZipFile(eval_path, 'r') as zf:
        with zf.open('header.json') as f:
            data = json.load(f)

    # Get task ID from samples
    task_id = None
    try:
        with zipfile.ZipFile(eval_path, 'r') as zf:
            # Look for sample files to get task ID
            sample_files = [n for n in zf.namelist() if n.startswith('samples/') and n.endswith('.json')]
            if sample_files:
                with zf.open(sample_files[0]) as f:
                    sample_data = json.load(f)
                    task_id = sample_data.get('id', None)
    except:
        pass

    result = {
        "eval_file": eval_path.name,
        "task_id": task_id,
        "completed_at": data.get("stats", {}).get("completed_at"),
    }

    if "stats" in data and "model_usage" in data["stats"]:
        model_usage = data["stats"]["model_usage"]
        for model, usage in model_usage.items():
            result["model"] = model
            result["input_tokens"] = usage.get("input_tokens", 0)
            result["output_tokens"] = usage.get("output_tokens", 0)
            result["total_tokens"] = usage.get("total_tokens", 0)
            result["input_tokens_cache_read"] = usage.get("input_tokens_cache_read", 0)

            # Calculate cost with cache pricing
            if model in PRICING:
                pricing = PRICING[model]
                if "anthropic" in model:
                    # Claude: input_tokens are non-cached, cache_read are cached
                    input_cost = (result["input_tokens"] / 1_000_000) * pricing["input"]
                    cache_cost = (result["input_tokens_cache_read"] / 1_000_000) * pricing["cache_read"]
                    output_cost = (result["output_tokens"] / 1_000_000) * pricing["output"]
                    result["cost_usd"] = input_cost + cache_cost + output_cost
                else:
                    # GPT: input_tokens includes all, cache_read is subset
                    non_cached = result["input_tokens"] - result["input_tokens_cache_read"]
                    input_cost = (non_cached / 1_000_000) * pricing["input"]
                    cache_cost = (result["input_tokens_cache_read"] / 1_000_000) * pricing["cache_read"]
                    output_cost = (result["output_tokens"] / 1_000_000) * pricing["output"]
                    result["cost_usd"] = input_cost + cache_cost + output_cost

    return result


def main():
    # Get all swe-bench eval files
    eval_files = list(LOGS_DIR.glob("*_swe-bench_*.eval"))
    print(f"Found {len(eval_files)} swe-bench eval files")

    # Extract info from each
    eval_info = []
    for ef in eval_files:
        try:
            info = extract_info_from_eval(ef)
            eval_info.append(info)
        except Exception as e:
            print(f"  Error processing {ef.name}: {e}")

    # Group by model
    by_model = {}
    for info in eval_info:
        model = info.get("model", "unknown")
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(info)

    # Process each results file
    for results_file in OUTPUT_DIR.glob("agent_*_10tasks.json"):
        print(f"\nProcessing {results_file.name}")

        with open(results_file) as f:
            results = json.load(f)

        model = results["model"]
        model_evals = by_model.get(model, [])
        print(f"  Found {len(model_evals)} eval logs for {model}")

        # Create mapping from task_id to eval info
        task_to_eval = {e["task_id"]: e for e in model_evals if e.get("task_id")}

        # Update each task result
        updated = 0
        for task in results["tasks"]:
            task_id = task["task_id"]
            if task_id in task_to_eval and "error" not in task:
                eval_data = task_to_eval[task_id]
                task["input_tokens"] = eval_data.get("input_tokens", 0)
                task["output_tokens"] = eval_data.get("output_tokens", 0)
                task["total_tokens"] = eval_data.get("total_tokens", 0)
                task["input_tokens_cache_read"] = eval_data.get("input_tokens_cache_read", 0)
                task["cost_usd"] = eval_data.get("cost_usd", 0)
                updated += 1
                print(f"    Updated {task_id}: {task['input_tokens']:,} in / {task['output_tokens']:,} out / ${task['cost_usd']:.4f}")

        print(f"  Updated {updated} tasks")

        # Save updated results
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for results_file in OUTPUT_DIR.glob("agent_*_10tasks.json"):
        with open(results_file) as f:
            results = json.load(f)

        successful = [t for t in results["tasks"] if "error" not in t]
        total_time = sum(t.get("elapsed_seconds", 0) for t in results["tasks"])
        total_cost = sum(t.get("cost_usd", 0) for t in successful)
        total_input = sum(t.get("input_tokens", 0) for t in successful)
        total_output = sum(t.get("output_tokens", 0) for t in successful)
        total_cache = sum(t.get("input_tokens_cache_read", 0) for t in successful)

        print(f"\n{results['model']}:")
        print(f"  Tasks: {len(results['tasks'])} attempted, {len(successful)} successful")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Total tokens: {total_input:,} in / {total_output:,} out")
        print(f"  Cache reads: {total_cache:,}")
        print(f"  Total cost: ${total_cost:.4f}")
        if successful:
            print(f"  Avg per successful task: {total_time/len(successful):.1f}s, ${total_cost/len(successful):.4f}")


if __name__ == "__main__":
    main()
