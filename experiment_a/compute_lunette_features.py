"""Compute Lunette-based features for SWE-bench tasks (Experiment A).

This script uses Lunette's sandbox to extract difficulty-prediction features
from SWE-bench Verified tasks. It runs a dummy solver to get environment access,
then uses the GradingPlan API to extract structured features.

Usage:
    # Dry run to see execution plan
    python -m experiment_a.compute_lunette_features --dry_run

    # Run on single task for validation
    python -m experiment_a.compute_lunette_features --limit 1

    # Run on 50 tasks for pilot
    python -m experiment_a.compute_lunette_features --limit 50

    # Full run on all 500 tasks
    python -m experiment_a.compute_lunette_features
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a.lunette_grading_prompt import (
    LUNETTE_FEATURE_NAMES,
    format_grading_prompt,
)

# Try to import Lunette
try:
    from lunette import LunetteClient
    from lunette.analysis import GradingPlan
    from lunette.models.run import Run
    from lunette.models.trajectory import Trajectory, ScalarScore
    from lunette.models.messages import AssistantMessage

    HAS_LUNETTE = True
except ImportError:
    HAS_LUNETTE = False
    print("Warning: lunette-sdk not installed. Run: pip install lunette-sdk")


# Output directory
OUTPUT_DIR = ROOT / "chris_output" / "experiment_a" / "lunette_features"


def load_swebench_verified() -> List[dict]:
    """Load SWE-bench Verified dataset from HuggingFace."""
    from datasets import load_dataset

    print("Loading SWE-bench Verified dataset...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    tasks = []
    for item in ds:
        tasks.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "version": item["version"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "test_patch": item["test_patch"],
            "hints_text": item["hints_text"],
            "base_commit": item["base_commit"],
            "FAIL_TO_PASS": item["FAIL_TO_PASS"],
            "PASS_TO_PASS": item["PASS_TO_PASS"],
        })

    print(f"Loaded {len(tasks)} tasks")
    return tasks


def create_dummy_trajectory(task_id: str) -> Trajectory:
    """Create a minimal trajectory that immediately fails.

    The dummy trajectory just has one message saying the agent didn't attempt.
    This is enough to get Lunette sandbox access for the grading judge.
    """
    return Trajectory(
        sample=task_id,
        messages=[
            AssistantMessage(
                position=0,
                content="Dummy agent: No attempt made. This trajectory exists only to enable Lunette sandbox access for feature extraction.",
            )
        ],
        scores={"resolved": ScalarScore(value=0.0)},  # Always fails
        solution=None,
        metadata={"dummy": True},
    )


def parse_feature_response(text: str) -> Optional[Dict]:
    """Parse JSON response from Lunette grading.

    Args:
        text: Response text from Lunette GradingPlan

    Returns:
        Parsed feature dict or None if parsing failed
    """
    if not text:
        return None

    # Try to extract JSON from response
    # First try: look for ```json block
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Second try: look for raw JSON object
        json_match = re.search(r"\{[^{}]*\"repo_file_count\"[^{}]*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        else:
            # Third try: find any JSON-like object
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

    try:
        data = json.loads(text)

        # Validate that we have some expected keys
        if "repo_file_count" not in data and "fix_in_description" not in data:
            return None

        return data
    except json.JSONDecodeError:
        return None


async def extract_features_for_task(
    client: LunetteClient,
    task: dict,
    output_dir: Path,
) -> Optional[Dict]:
    """Extract features for a single task using Lunette.

    Args:
        client: Lunette client
        task: Task metadata dict
        output_dir: Directory to save individual results

    Returns:
        Feature dict or None if failed
    """
    task_id = task["instance_id"]

    # Create output file path
    output_file = output_dir / f"{task_id}.json"
    if output_file.exists():
        print(f"  -> Already computed, loading from cache")
        with open(output_file) as f:
            return json.load(f)

    try:
        # 1. Create dummy trajectory
        trajectory = create_dummy_trajectory(task_id)

        # 2. Create run with SWE-bench metadata
        run = Run(
            task="swebench-verified",
            model="dummy_solver",
            trajectories=[trajectory],
            metadata={
                "repo": task["repo"],
                "patch": task["patch"],
                "test_patch": task.get("test_patch", ""),
                "FAIL_TO_PASS": task.get("FAIL_TO_PASS", "[]"),
                "PASS_TO_PASS": task.get("PASS_TO_PASS", "[]"),
                "version": task.get("version", ""),
                "base_commit": task.get("base_commit", ""),
            },
        )

        # 3. Upload run to Lunette
        print(f"  -> Uploading dummy trajectory...")
        run_meta = await client.save_run(run)
        run_id = run_meta["run_id"]
        print(f"  -> Run ID: {run_id}")

        # 4. Format grading prompt
        grading_prompt = format_grading_prompt(
            instance_id=task_id,
            repo=task["repo"],
            version=task.get("version", "unknown"),
            problem_statement=task["problem_statement"],
            patch=task["patch"],
            fail_to_pass=task.get("FAIL_TO_PASS", "[]"),
            pass_to_pass=task.get("PASS_TO_PASS", "[]"),
            hints_text=task.get("hints_text", ""),
        )

        # 5. Grade using Lunette GradingPlan
        print(f"  -> Running Lunette grading (this takes ~1-2 minutes)...")
        results = await client.investigate(
            run_id=run_id,
            plan=GradingPlan(name="task-difficulty-features", prompt=grading_prompt),
            limit=1,
        )

        if not results.results:
            print(f"  -> No results returned from Lunette")
            return None

        # 6. Parse response
        result_data = results.results[0].data
        print(f"  -> Got response, parsing features...")

        # The data might be the parsed JSON directly or in a text field
        if isinstance(result_data, dict):
            if "explanation" in result_data:
                # GradingPlan returns {name, score, explanation}
                features = parse_feature_response(result_data["explanation"])
            else:
                features = result_data
        else:
            features = parse_feature_response(str(result_data))

        if features is None:
            print(f"  -> Failed to parse features from response")
            # Save raw response for debugging
            debug_file = output_dir / f"{task_id}_raw.json"
            with open(debug_file, "w") as f:
                json.dump({"run_id": run_id, "raw_response": result_data}, f, indent=2)
            return None

        # 7. Add metadata
        features["_instance_id"] = task_id
        features["_run_id"] = run_id
        features["_extracted_at"] = datetime.now().isoformat()

        # 8. Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(features, f, indent=2)

        print(f"  -> Saved features to {output_file.name}")
        return features

    except Exception as e:
        print(f"  -> Error: {e}")
        # Save error for debugging
        error_file = output_dir / f"{task_id}_error.json"
        with open(error_file, "w") as f:
            json.dump({"error": str(e), "task_id": task_id}, f, indent=2)
        return None


async def main():
    parser = argparse.ArgumentParser(
        description="Extract Lunette-based features for SWE-bench tasks"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show execution plan without running",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to process",
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help="Comma-separated list of specific task IDs to process",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip tasks with existing features",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for features",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent tasks to process (default: 5)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    tasks = load_swebench_verified()

    # Filter to specific task IDs if provided
    if args.task_ids:
        task_id_set = set(args.task_ids.split(","))
        tasks = [t for t in tasks if t["instance_id"] in task_id_set]
        print(f"Filtered to {len(tasks)} specified tasks")

    # Apply limit
    if args.limit:
        tasks = tasks[: args.limit]
        print(f"Limited to {args.limit} tasks")

    # Skip existing
    if args.skip_existing:
        original_count = len(tasks)
        tasks = [
            t
            for t in tasks
            if not (output_dir / f"{t['instance_id']}.json").exists()
        ]
        skipped = original_count - len(tasks)
        if skipped > 0:
            print(f"Skipping {skipped} tasks with existing features")

    print(f"\nTasks to process: {len(tasks)}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"\nOutput directory: {output_dir}")
        print(f"\nSample tasks (first 5):")
        for task in tasks[:5]:
            print(f"  - {task['instance_id']} ({task['repo']})")
        if len(tasks) > 5:
            print(f"  ... and {len(tasks) - 5} more")

        # Estimate cost
        cost_per_task = 0.15  # ~$0.10-0.20 for thorough exploration
        print(f"\nEstimated cost: ~${len(tasks) * cost_per_task:.2f}")
        print(f"  ({len(tasks)} tasks × ${cost_per_task}/task)")
        return

    if not HAS_LUNETTE:
        print("\nError: lunette-sdk not installed")
        print("Run: pip install lunette-sdk")
        return

    # Process tasks
    stats = {
        "total": len(tasks),
        "success": 0,
        "failed": 0,
        "cached": 0,
    }

    all_features = []

    # Determine concurrency level
    concurrency = args.concurrency
    print(f"Concurrency level: {concurrency}")
    semaphore = asyncio.Semaphore(concurrency)

    async def process_with_semaphore(client, task, output_dir, idx, total):
        """Process a task with concurrency limiting."""
        async with semaphore:
            task_id = task["instance_id"]
            print(f"\n[{idx + 1}/{total}] {task_id}")
            return await extract_features_for_task(client, task, output_dir)

    async with LunetteClient() as client:
        # Process tasks concurrently with semaphore limiting
        coroutines = [
            process_with_semaphore(client, task, output_dir, i, len(tasks))
            for i, task in enumerate(tasks)
        ]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        for features in results:
            if isinstance(features, Exception):
                print(f"  -> Exception: {features}")
                stats["failed"] += 1
            elif features:
                all_features.append(features)
                if features.get("_cached"):
                    stats["cached"] += 1
                else:
                    stats["success"] += 1
            else:
                stats["failed"] += 1

    # Aggregate to CSV
    if all_features:
        df = pd.DataFrame(all_features)
        csv_path = output_dir.parent / "lunette_features.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n\nAggregated features saved to: {csv_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tasks: {stats['total']}")
    print(f"Successful: {stats['success']}")
    print(f"From cache: {stats['cached']}")
    print(f"Failed: {stats['failed']}")

    # Save stats
    stats_file = output_dir / f"compute_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {stats_file}")


if __name__ == "__main__":
    asyncio.run(main())
