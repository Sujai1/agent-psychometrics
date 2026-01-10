"""Augment Lunette upload tracking files with task-to-run mappings.

This script queries the Lunette API once to build task_id -> run_id mappings
and stores them in the _lunette_uploads.json files. This avoids needing to
query the API every time you want to work with the uploaded trajectories.

Usage:
    # Augment all agents
    python trajectory_upload/lunette_augment_mappings.py

    # Augment specific agents
    python trajectory_upload/lunette_augment_mappings.py --agents 20240620_sweagent_claude3.5sonnet

    # Dry run to see what would be done
    python trajectory_upload/lunette_augment_mappings.py --dry_run
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from lunette import LunetteClient


def load_upload_tracking(agent_dir: Path) -> Dict:
    """Load existing upload tracking file."""
    tracking_file = agent_dir / "_lunette_uploads.json"
    if not tracking_file.exists():
        return None

    with open(tracking_file) as f:
        return json.load(f)


def save_upload_tracking(agent_dir: Path, upload_data: Dict):
    """Save updated upload tracking file."""
    tracking_file = agent_dir / "_lunette_uploads.json"
    with open(tracking_file, "w") as f:
        json.dump(upload_data, f, indent=2)


async def build_task_to_run_mapping(
    client: LunetteClient,
    run_ids: List[str],
) -> Dict[str, str]:
    """Build mapping from task_id to run_id by querying Lunette API.

    Args:
        client: LunetteClient instance.
        run_ids: List of run IDs to check.

    Returns:
        Dict mapping task_id -> run_id.
    """
    task_to_run = {}

    for run_id in run_ids:
        try:
            run = await client.get_run(run_id)
            # Each trajectory has a 'sample' field which is the task_id
            for traj in run.trajectories:
                task_to_run[traj.sample] = run_id
        except Exception as e:
            print(f"    Warning: Failed to fetch run {run_id[:16]}...: {e}")
            continue

    return task_to_run


async def main():
    parser = argparse.ArgumentParser(
        description="Augment Lunette upload tracking files with task-to-run mappings"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="Specific agents to augment (default: all)",
    )
    parser.add_argument(
        "--trajectories_dir",
        type=str,
        default="trajectory_data/unified_trajs",
        help="Base directory containing agent folders",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing mappings even if already present",
    )

    args = parser.parse_args()

    trajectories_dir = Path(args.trajectories_dir)

    if not trajectories_dir.exists():
        print(f"Error: Directory not found: {trajectories_dir}")
        return

    # Find agents to process
    if args.agents:
        agent_dirs = [trajectories_dir / a for a in args.agents]
        agent_dirs = [d for d in agent_dirs if d.exists()]
    else:
        agent_dirs = sorted([
            d for d in trajectories_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith("_")
            and (d / "_lunette_uploads.json").exists()
        ])

    print(f"=== Lunette Upload Mapping Augmentation ===")
    print(f"Found {len(agent_dirs)} agents to process")

    if args.dry_run:
        print("DRY RUN - no files will be modified\n")

    stats = {
        "total": len(agent_dirs),
        "augmented": 0,
        "skipped": 0,
        "failed": 0,
    }

    async with LunetteClient() as client:
        for i, agent_dir in enumerate(agent_dirs):
            agent = agent_dir.name
            print(f"\n[{i+1}/{len(agent_dirs)}] {agent}")

            # Load existing tracking
            upload_data = load_upload_tracking(agent_dir)
            if not upload_data:
                print("  No upload tracking file found")
                stats["failed"] += 1
                continue

            # Check if already has mappings
            if "task_to_run_map" in upload_data and not args.force:
                print(f"  Already has task-to-run mapping ({len(upload_data['task_to_run_map'])} tasks)")
                stats["skipped"] += 1
                continue

            run_ids = upload_data.get("run_ids", [upload_data.get("run_id")])
            if not run_ids or run_ids[0] is None:
                print("  No valid run IDs")
                stats["failed"] += 1
                continue

            # Query Lunette API
            print(f"  Querying {len(run_ids)} run(s)...")

            try:
                task_to_run = await build_task_to_run_mapping(client, run_ids)

                if not task_to_run:
                    print("  No tasks mapped (all queries failed)")
                    stats["failed"] += 1
                    continue

                print(f"  Mapped {len(task_to_run)} tasks")

                # Build reverse mapping: run_id -> list of task_ids
                run_to_tasks = {}
                for task_id, run_id in task_to_run.items():
                    if run_id not in run_to_tasks:
                        run_to_tasks[run_id] = []
                    run_to_tasks[run_id].append(task_id)

                # Augment upload data
                upload_data["task_to_run_map"] = task_to_run
                upload_data["run_to_tasks_map"] = run_to_tasks
                upload_data["task_to_run_map_updated_at"] = datetime.now().isoformat()

                # Also augment each trajectory with its run_id for convenience
                trajectories = upload_data.get("trajectories", [])
                for traj in trajectories:
                    task_id = traj["task_id"]
                    if task_id in task_to_run:
                        traj["run_id"] = task_to_run[task_id]

                if not args.dry_run:
                    save_upload_tracking(agent_dir, upload_data)
                    print("  ✓ Saved updated tracking file")
                else:
                    print("  Would save updated tracking file (dry run)")

                stats["augmented"] += 1

            except Exception as e:
                print(f"  Error: {e}")
                stats["failed"] += 1

    print(f"\n=== Summary ===")
    print(f"Total agents: {stats['total']}")
    print(f"Augmented: {stats['augmented']}")
    print(f"Skipped (already mapped): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")

    if args.dry_run:
        print("\nDRY RUN - no changes were made")
        print("Run without --dry_run to apply changes")


if __name__ == "__main__":
    asyncio.run(main())
