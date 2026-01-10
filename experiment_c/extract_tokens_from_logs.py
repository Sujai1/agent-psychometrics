"""
Extract token usage from completed eval logs and update results files.

This script retroactively extracts token counts from .eval ZIP files
and updates the agent results JSON files.

Usage:
    python experiment_c/extract_tokens_from_logs.py
"""

import json
import zipfile
from pathlib import Path
from datetime import datetime

# Pricing per 1M tokens
PRICING = {
    "anthropic/claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "openai/gpt-5.2-2025-12-11": {"input": 1.75, "output": 14.0},
}

OUTPUT_DIR = Path("chris_output/experiment_c")
LOGS_DIR = Path("logs")


def extract_token_usage_from_eval(eval_path: Path) -> dict:
    """Extract token usage from a .eval ZIP file."""
    with zipfile.ZipFile(eval_path, 'r') as zf:
        with zf.open('header.json') as f:
            data = json.load(f)

    result = {
        "eval_file": eval_path.name,
        "task": data.get("eval", {}).get("task", "unknown"),
    }

    if "stats" in data and "model_usage" in data["stats"]:
        model_usage = data["stats"]["model_usage"]
        for model, usage in model_usage.items():
            result["model"] = model
            result["input_tokens"] = usage.get("input_tokens", 0)
            result["output_tokens"] = usage.get("output_tokens", 0)
            result["total_tokens"] = usage.get("total_tokens", 0)
            result["input_tokens_cache_read"] = usage.get("input_tokens_cache_read", 0)

            if model in PRICING:
                pricing = PRICING[model]
                input_cost = (result["input_tokens"] / 1_000_000) * pricing["input"]
                output_cost = (result["output_tokens"] / 1_000_000) * pricing["output"]
                result["cost_usd"] = input_cost + output_cost

    return result


def main():
    # Get all swe-bench eval files (not dummy-swebench)
    eval_files = sorted(
        [f for f in LOGS_DIR.glob("*_swe-bench_*.eval")],
        key=lambda p: p.stat().st_mtime
    )

    print(f"Found {len(eval_files)} swe-bench eval files")

    # Extract token usage from each
    all_usage = []
    for eval_file in eval_files:
        try:
            usage = extract_token_usage_from_eval(eval_file)
            all_usage.append(usage)
            print(f"  {eval_file.name}: {usage.get('model', 'unknown')} - {usage.get('input_tokens', 0):,} in / {usage.get('output_tokens', 0):,} out")
        except Exception as e:
            print(f"  {eval_file.name}: ERROR - {e}")

    # Group by model
    by_model = {}
    for usage in all_usage:
        model = usage.get("model", "unknown")
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(usage)

    # Print summary by model
    print("\n" + "=" * 60)
    print("SUMMARY BY MODEL")
    print("=" * 60)

    for model, usages in by_model.items():
        total_input = sum(u.get("input_tokens", 0) for u in usages)
        total_output = sum(u.get("output_tokens", 0) for u in usages)
        total_cache = sum(u.get("input_tokens_cache_read", 0) for u in usages)
        total_cost = sum(u.get("cost_usd", 0) for u in usages)

        print(f"\n{model}:")
        print(f"  Tasks: {len(usages)}")
        print(f"  Total tokens: {total_input:,} in / {total_output:,} out")
        print(f"  Cache reads: {total_cache:,} ({100*total_cache/total_input:.1f}% of input)")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Avg per task: ${total_cost/len(usages):.4f}")

    # Save detailed results
    output_file = OUTPUT_DIR / "token_usage_extracted.json"
    with open(output_file, "w") as f:
        json.dump({
            "extracted_at": datetime.now().isoformat(),
            "eval_files_processed": len(eval_files),
            "by_model": by_model,
            "all_usage": all_usage,
        }, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    main()
