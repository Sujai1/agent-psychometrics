#!/usr/bin/env python3
"""
Check the format consistency of SWE-bench Pro trajectories.
"""

import json
from pathlib import Path
from collections import Counter
import random

def check_trajectory_format(trajectory_dir: Path):
    """Check the format of trajectories across all agents."""

    agent_dirs = [d for d in trajectory_dir.iterdir() if d.is_dir()]

    print(f"Found {len(agent_dirs)} agent directories\n")

    # Sample files from each agent
    samples_per_agent = 3

    formats = []
    errors = []

    for agent_dir in agent_dirs:
        agent_name = agent_dir.name
        files = list(agent_dir.glob("*.json"))

        if not files:
            print(f"⚠️  {agent_name}: No files found")
            continue

        # Sample random files
        sample_files = random.sample(files, min(samples_per_agent, len(files)))

        for file in sample_files:
            try:
                with open(file) as f:
                    data = json.load(f)

                # Determine format
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], dict):
                        keys = tuple(sorted(data[0].keys()))
                        formats.append({
                            'agent': agent_name,
                            'file': file.name,
                            'type': 'list',
                            'length': len(data),
                            'first_item_keys': keys
                        })
                    else:
                        formats.append({
                            'agent': agent_name,
                            'file': file.name,
                            'type': 'list (empty or non-dict)',
                            'length': len(data)
                        })
                elif isinstance(data, dict):
                    keys = tuple(sorted(data.keys()))
                    formats.append({
                        'agent': agent_name,
                        'file': file.name,
                        'type': 'dict',
                        'keys': keys
                    })
                else:
                    formats.append({
                        'agent': agent_name,
                        'file': file.name,
                        'type': type(data).__name__
                    })

            except Exception as e:
                errors.append({
                    'agent': agent_name,
                    'file': file.name,
                    'error': str(e)
                })

    # Analyze results
    print("=== Format Analysis ===\n")

    # Count format types
    type_counts = Counter(f['type'] for f in formats)
    print("Format types:")
    for fmt_type, count in type_counts.most_common():
        print(f"  {fmt_type}: {count} files")

    # Check consistency of list formats
    list_formats = [f for f in formats if f['type'] == 'list']
    if list_formats:
        print(f"\n=== List Format Details ({len(list_formats)} files) ===\n")

        # Check list lengths
        lengths = Counter(f['length'] for f in list_formats)
        print("List lengths:")
        for length, count in lengths.most_common():
            print(f"  {length} items: {count} files")

        # Check key consistency
        key_sets = Counter(f['first_item_keys'] for f in list_formats if 'first_item_keys' in f)
        print(f"\nUnique key sets: {len(key_sets)}")

        if len(key_sets) == 1:
            print("✓ All list items have the same keys")
            keys = list(key_sets.keys())[0]
            print(f"\nStandard keys ({len(keys)}):")
            for key in keys:
                print(f"  - {key}")
        else:
            print("⚠️  Multiple key sets found:")
            for keys, count in key_sets.most_common():
                print(f"\n  Keys ({count} files): {', '.join(keys[:5])}...")

    # Check dict formats
    dict_formats = [f for f in formats if f['type'] == 'dict']
    if dict_formats:
        print(f"\n=== Dict Format Details ({len(dict_formats)} files) ===\n")

        key_sets = Counter(f['keys'] for f in dict_formats)
        print(f"Unique key sets: {len(key_sets)}")

        if len(key_sets) == 1:
            print("✓ All dicts have the same keys")
            keys = list(key_sets.keys())[0]
            print(f"\nStandard keys ({len(keys)}):")
            for key in keys:
                print(f"  - {key}")
        else:
            print("⚠️  Multiple key sets found:")
            for keys, count in key_sets.most_common(3):
                print(f"\n  Keys ({count} files): {', '.join(keys[:5])}...")

    # Report errors
    if errors:
        print(f"\n=== Errors ({len(errors)}) ===\n")
        for err in errors[:10]:
            print(f"  {err['agent']}/{err['file']}: {err['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("\n✓ No parsing errors")

    # Sample a full structure
    if formats:
        print("\n=== Sample Structure ===\n")
        sample = random.choice(formats)
        sample_file = Path(trajectory_dir) / sample['agent'] / sample['file']

        with open(sample_file) as f:
            data = json.load(f)

        print(f"File: {sample['agent']}/{sample['file']}")
        print(f"Type: {sample['type']}")

        if isinstance(data, list) and len(data) > 0:
            print(f"\nFirst item structure:")
            print(json.dumps(data[0], indent=2, default=str)[:1000] + "...")
        elif isinstance(data, dict):
            print(f"\nStructure:")
            print(json.dumps(data, indent=2, default=str)[:1000] + "...")


if __name__ == "__main__":
    trajectory_dir = Path("trajectory_data/swebench_pro")
    check_trajectory_format(trajectory_dir)
