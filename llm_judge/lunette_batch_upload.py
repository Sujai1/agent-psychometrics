"""
Batch upload all converted trajectories to Lunette.

Uploads ALL trajectories for each agent in a SINGLE run (not one run per trajectory).

Usage:
    # Upload all agents
    python llm_judge/lunette_batch_upload.py

    # Upload specific agents
    python llm_judge/lunette_batch_upload.py --agents 20240620_sweagent_claude3.5sonnet 20240728_sweagent_gpt4o

    # Dry run
    python llm_judge/lunette_batch_upload.py --dry_run
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from lunette import LunetteClient
from lunette.models.run import Run

from lunette_upload import (
    load_results_for_agent,
    load_converted_trajectory,
    convert_to_lunette_format,
)


def load_existing_upload(agent_dir: Path) -> dict | None:
    """Load existing upload tracking file."""
    tracking_file = agent_dir / '_lunette_uploads.json'
    if tracking_file.exists():
        with open(tracking_file) as f:
            return json.load(f)
    return None


def save_upload_tracking(agent_dir: Path, upload_info: dict):
    """Save upload tracking info."""
    tracking_file = agent_dir / '_lunette_uploads.json'
    with open(tracking_file, 'w') as f:
        json.dump(upload_info, f, indent=2)


async def upload_agent_batch(
    client: LunetteClient,
    agent_dir: Path,
    agent_name: str,
    dry_run: bool = False,
) -> dict:
    """Upload ALL trajectories for an agent in a single Run."""

    # Check if already uploaded
    existing = load_existing_upload(agent_dir)
    if existing and existing.get('run_id'):
        print(f"  {agent_name}: Already uploaded (run_id: {existing['run_id'][:8]}...)")
        return {'skipped': True, 'existing': existing}

    results = load_results_for_agent(agent_name)

    # Find all trajectory JSON files
    json_files = sorted(agent_dir.glob('*.json'))
    json_files = [f for f in json_files if not f.name.startswith('_')]

    if not json_files:
        print(f"  {agent_name}: No trajectory files found")
        return {'error': 'No trajectories'}

    print(f"  {agent_name}: Converting {len(json_files)} trajectories...")

    # Convert all trajectories
    trajectories = []
    trajectory_info = []

    for file_path in json_files:
        task_id = file_path.stem
        resolved = results.get(task_id, False)

        try:
            unified = load_converted_trajectory(file_path)
            lunette_traj = convert_to_lunette_format(
                unified,
                resolved=resolved,
                model_name=agent_name
            )
            trajectories.append(lunette_traj)
            trajectory_info.append({
                'task_id': task_id,
                'resolved': resolved,
                'message_count': len(unified.get('messages', [])),
            })
        except Exception as e:
            print(f"    Error converting {task_id}: {e}")

    if not trajectories:
        print(f"  {agent_name}: No valid trajectories after conversion")
        return {'error': 'No valid trajectories'}

    if dry_run:
        print(f"  {agent_name}: Would upload {len(trajectories)} trajectories in 1 run")
        return {'dry_run': True, 'trajectory_count': len(trajectories)}

    # Create single Run with ALL trajectories
    print(f"  {agent_name}: Uploading {len(trajectories)} trajectories in single run...")

    run = Run(
        task="swebench-verified",
        model=agent_name,
        trajectories=trajectories,
    )

    try:
        run_meta = await client.save_run(run)
        run_id = run_meta['run_id']
        traj_ids = run_meta.get('trajectory_ids', [])

        print(f"  {agent_name}: SUCCESS - run_id: {run_id[:8]}... ({len(traj_ids)} trajectories)")

        # Save tracking info
        upload_info = {
            'agent': agent_name,
            'uploaded_at': datetime.now().isoformat(),
            'run_id': run_id,
            'trajectory_count': len(traj_ids),
            'trajectory_ids': traj_ids,
            'trajectories': [
                {**info, 'trajectory_id': traj_ids[i] if i < len(traj_ids) else None}
                for i, info in enumerate(trajectory_info)
            ],
        }
        save_upload_tracking(agent_dir, upload_info)

        return {'success': True, 'run_id': run_id, 'trajectory_count': len(traj_ids)}

    except Exception as e:
        print(f"  {agent_name}: FAILED - {e}")
        return {'error': str(e)}


async def main():
    parser = argparse.ArgumentParser(description='Batch upload trajectories to Lunette (one run per agent)')
    parser.add_argument('--agents', nargs='+', help='Specific agents to upload (default: all)')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be uploaded')
    parser.add_argument('--input_dir', type=str, default='trajectory_data/unified_trajs',
                        help='Base directory containing agent folders')
    parser.add_argument('--output', type=str, help='Output path for batch summary')

    args = parser.parse_args()

    input_base = Path(args.input_dir)
    if not input_base.exists():
        print(f"Error: Input directory not found: {input_base}")
        return

    # Find all agent directories
    if args.agents:
        agent_dirs = [input_base / a for a in args.agents]
        agent_dirs = [d for d in agent_dirs if d.exists()]
    else:
        agent_dirs = sorted([
            d for d in input_base.iterdir()
            if d.is_dir() and not d.name.startswith('_')
        ])

    print(f"=== Batch Upload to Lunette (1 run per agent) ===")
    print(f"Found {len(agent_dirs)} agents to process")
    if args.dry_run:
        print("DRY RUN - no uploads will be made\n")

    batch_summary = {
        'started': datetime.now().isoformat(),
        'total_agents': len(agent_dirs),
        'successful': 0,
        'skipped': 0,
        'failed': 0,
        'agents': {},
    }

    async with LunetteClient() as client:
        for i, agent_dir in enumerate(agent_dirs):
            agent_name = agent_dir.name
            print(f"\n[{i+1}/{len(agent_dirs)}] {agent_name}")

            try:
                result = await upload_agent_batch(
                    client=client,
                    agent_dir=agent_dir,
                    agent_name=agent_name,
                    dry_run=args.dry_run,
                )

                batch_summary['agents'][agent_name] = result

                if result.get('success'):
                    batch_summary['successful'] += 1
                elif result.get('skipped'):
                    batch_summary['skipped'] += 1
                elif result.get('dry_run'):
                    batch_summary['successful'] += 1
                else:
                    batch_summary['failed'] += 1

            except Exception as e:
                print(f"  {agent_name}: ERROR - {e}")
                batch_summary['agents'][agent_name] = {'error': str(e)}
                batch_summary['failed'] += 1

    batch_summary['completed'] = datetime.now().isoformat()

    print(f"\n=== BATCH UPLOAD COMPLETE ===")
    print(f"Successful: {batch_summary['successful']}")
    print(f"Skipped (already uploaded): {batch_summary['skipped']}")
    print(f"Failed: {batch_summary['failed']}")

    # Save batch summary
    output_path = Path(args.output) if args.output else input_base / '_batch_upload_summary.json'
    with open(output_path, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    print(f"Summary saved to: {output_path}")


if __name__ == '__main__':
    asyncio.run(main())
