"""
Batch upload FILTERED trajectories to Lunette.

Uploads ALL filtered trajectories for each agent in a SINGLE run (not one run per trajectory).

This uploads from trajectory_data/filtered_unified/ (filtered by trajectory_filter.py)
as opposed to lunette_batch_upload.py which uploads from trajectory_data/unified_trajs/ (unfiltered).

Key differences from lunette_batch_upload.py:
- Input directory: trajectory_data/filtered_unified/
- Model name suffix: _filtered (e.g., "20240620_sweagent_claude3.5sonnet_filtered")
- Tracking file: _lunette_filtered_uploads.json (not _lunette_uploads.json)
- Batch summary: _batch_filtered_upload_summary.json

Usage:
    # Upload all filtered agents
    python llm_judge/lunette_filtered_upload.py

    # Upload specific agents
    python llm_judge/lunette_filtered_upload.py --agents 20240620_sweagent_claude3.5sonnet

    # Dry run (show what would be uploaded)
    python llm_judge/lunette_filtered_upload.py --dry_run
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from lunette import LunetteClient
from lunette.models.run import Run
from lunette.models.trajectory import Trajectory, ScalarScore
from lunette.models.messages import SystemMessage, UserMessage, AssistantMessage


# Different tracking file name to avoid conflicts with non-filtered uploads
TRACKING_FILE_NAME = '_lunette_filtered_uploads.json'
BATCH_SUMMARY_NAME = '_batch_filtered_upload_summary.json'
MODEL_SUFFIX = '_filtered'


def load_filtered_trajectory(file_path: Path) -> dict:
    """Load a filtered trajectory JSON file."""
    with open(file_path) as f:
        return json.load(f)


def load_results_for_agent(agent_name: str) -> dict[str, bool]:
    """Load results.json to get resolved/unresolved status for each task."""
    experiments_dir = Path(__file__).resolve().parents[1] / 'experiments'
    results_path = experiments_dir / 'evaluation' / 'verified' / agent_name / 'results' / 'results.json'

    if not results_path.exists():
        print(f"    Warning: results.json not found at {results_path}")
        return {}

    with open(results_path) as f:
        results = json.load(f)

    resolved_set = set(results.get('resolved', []))
    return {task: task in resolved_set for task in resolved_set}


def convert_to_lunette_format(
    filtered_traj: dict,
    resolved: bool = False,
    model_name: str = "unknown",
) -> Trajectory:
    """Convert filtered trajectory format to Lunette Trajectory object."""
    messages = []

    for i, msg in enumerate(filtered_traj.get('messages', [])):
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        position = i

        if role == 'system':
            messages.append(SystemMessage(position=position, content=content))
        elif role == 'assistant':
            messages.append(AssistantMessage(position=position, content=content))
        else:
            messages.append(UserMessage(position=position, content=content))

    task_id = filtered_traj.get('task_id', 'unknown')

    # Get filtering metadata
    metadata = filtered_traj.get('metadata', {})
    original_messages = metadata.get('_original_messages', 0)
    filtered_messages = metadata.get('_filtered_messages', len(messages))

    scores = {'resolved': ScalarScore(value=1.0 if resolved else 0.0)}

    return Trajectory(
        sample=task_id,
        messages=messages,
        scores=scores,
        solution='',
        metadata={
            'filtered': True,
            'original_messages': original_messages,
            'filtered_messages': filtered_messages,
            'reduction_pct': round(100 * (1 - filtered_messages / original_messages), 1) if original_messages > 0 else 0,
        }
    )


def load_existing_upload(agent_dir: Path) -> dict | None:
    """Load existing filtered upload tracking file."""
    tracking_file = agent_dir / TRACKING_FILE_NAME
    if tracking_file.exists():
        with open(tracking_file) as f:
            return json.load(f)
    return None


def save_upload_tracking(agent_dir: Path, upload_info: dict):
    """Save upload tracking info."""
    tracking_file = agent_dir / TRACKING_FILE_NAME
    with open(tracking_file, 'w') as f:
        json.dump(upload_info, f, indent=2)


async def upload_agent_batch_filtered(
    client: LunetteClient,
    agent_dir: Path,
    agent_name: str,
    dry_run: bool = False,
) -> dict:
    """Upload ALL filtered trajectories for an agent in a single Run."""

    model_name = f"{agent_name}{MODEL_SUFFIX}"

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

    print(f"  {agent_name}: Converting {len(json_files)} filtered trajectories...")

    # Convert all trajectories
    trajectories = []
    trajectory_info = []

    for file_path in json_files:
        task_id = file_path.stem
        resolved = results.get(task_id, False)

        try:
            filtered = load_filtered_trajectory(file_path)
            lunette_traj = convert_to_lunette_format(
                filtered,
                resolved=resolved,
                model_name=model_name,
            )
            trajectories.append(lunette_traj)

            metadata = filtered.get('metadata', {})
            trajectory_info.append({
                'task_id': task_id,
                'resolved': resolved,
                'message_count': len(filtered.get('messages', [])),
                'original_message_count': metadata.get('_original_messages', 0),
            })
        except Exception as e:
            print(f"    Error converting {task_id}: {e}")

    if not trajectories:
        print(f"  {agent_name}: No valid trajectories after conversion")
        return {'error': 'No valid trajectories'}

    if dry_run:
        print(f"  {agent_name}: Would upload {len(trajectories)} filtered trajectories as {model_name}")
        return {'dry_run': True, 'trajectory_count': len(trajectories), 'model_name': model_name}

    # Create single Run with ALL trajectories
    print(f"  {agent_name}: Uploading {len(trajectories)} trajectories as {model_name}...")

    run = Run(
        task="swebench-verified",
        model=model_name,
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
            'model_name': model_name,
            'type': 'filtered',
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

        return {'success': True, 'run_id': run_id, 'trajectory_count': len(traj_ids), 'model_name': model_name}

    except Exception as e:
        print(f"  {agent_name}: FAILED - {e}")
        return {'error': str(e)}


async def main():
    parser = argparse.ArgumentParser(
        description='Batch upload FILTERED trajectories to Lunette (one run per agent)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This uploads from trajectory_data/filtered_unified/ (filtered by trajectory_filter.py).

Filtered trajectories have planning/setup/commentary removed, keeping only
actual codebase interactions (commands, outputs, edits).

Model names in Lunette have '_filtered' suffix to distinguish from non-filtered.
        """,
    )
    parser.add_argument('--agents', nargs='+', help='Specific agents to upload (default: all)')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be uploaded')
    parser.add_argument('--input_dir', type=str, default='trajectory_data/filtered_unified',
                        help='Base directory containing filtered agent folders')
    parser.add_argument('--output', type=str, help='Output path for batch summary')

    args = parser.parse_args()

    input_base = Path(args.input_dir)
    if not input_base.exists():
        print(f"Error: Input directory not found: {input_base}")
        print(f"\nHave you run the filter? Try:")
        print(f"  python llm_judge/trajectory_filter.py --all_unified")
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

    print(f"=== Batch Upload FILTERED Trajectories to Lunette (1 run per agent) ===")
    print(f"Input: {input_base}")
    print(f"Found {len(agent_dirs)} agents to process")
    print(f"Model suffix: {MODEL_SUFFIX}")
    if args.dry_run:
        print("DRY RUN - no uploads will be made\n")

    batch_summary = {
        'type': 'filtered',
        'started': datetime.now().isoformat(),
        'input_dir': str(input_base),
        'model_suffix': MODEL_SUFFIX,
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
                result = await upload_agent_batch_filtered(
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
    output_path = Path(args.output) if args.output else input_base / BATCH_SUMMARY_NAME
    with open(output_path, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    print(f"Summary saved to: {output_path}")


if __name__ == '__main__':
    asyncio.run(main())
