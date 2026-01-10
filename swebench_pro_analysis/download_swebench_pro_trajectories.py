#!/usr/bin/env python3
"""
Download full trajectories from SWE-bench Pro using the Docent API.

This script downloads all agent trajectories by calling the Docent REST API
directly, which is much faster than browser automation.
"""

import json
import argparse
import asyncio
from pathlib import Path
import pandas as pd
import httpx
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm


COLLECTION_ID = "032fb63d-4992-4bfc-911d-3b7dafcb931f"
API_BASE = "https://api.docent.transluce.org/rest"


async def fetch_trajectory(
    client: httpx.AsyncClient,
    agent_run_id: str,
    full_tree: bool = True
) -> dict | None:
    """
    Fetch a single trajectory from the Docent API.

    Args:
        client: HTTP client
        agent_run_id: The agent run ID
        full_tree: Whether to fetch the full transcript tree

    Returns:
        Trajectory data dict or None on failure
    """
    url = f"{API_BASE}/{COLLECTION_ID}/agent_run_with_tree"
    params = {
        "agent_run_id": agent_run_id,
        "apply_base_where_clause": "false",
        "full_tree": "true" if full_tree else "false"
    }

    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


async def estimate_storage(csv_path: Path, sample_size: int = 10, session_cookie: str = None) -> tuple[int, float]:
    """
    Estimate storage by downloading a sample of trajectories.

    Returns (total_bytes, avg_bytes_per_trajectory)
    """
    df = pd.read_csv(csv_path)
    sample_ids = df.sample(n=min(sample_size, len(df)), random_state=42)

    print(f"Sampling {len(sample_ids)} trajectories to estimate storage...")

    sizes = []
    temp_dir = Path("temp_estimate")
    temp_dir.mkdir(exist_ok=True)

    # Set up cookies if provided
    cookies = {}
    if session_cookie:
        cookies["docent_session"] = session_cookie

    async with httpx.AsyncClient(timeout=30.0, cookies=cookies) as client:
        for _, row in tqdm(sample_ids.iterrows(), total=len(sample_ids), desc="Sampling"):
            agent_run_id = row['agent_run_id']
            temp_file = temp_dir / f"{agent_run_id}.json"

            data = await fetch_trajectory(client, agent_run_id)

            if data:
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)
                sizes.append(temp_file.stat().st_size)
                temp_file.unlink()  # Clean up
            else:
                print(f"  Warning: Failed to fetch {agent_run_id}")

    # Clean up temp directory
    try:
        temp_dir.rmdir()
    except:
        pass

    if not sizes:
        raise RuntimeError("Failed to download any trajectories for estimation")

    avg_size = sum(sizes) / len(sizes)
    total_size = avg_size * len(df)

    return int(total_size), avg_size


async def download_all_trajectories(
    csv_path: Path,
    output_dir: Path,
    resume: bool = False,
    batch_size: int = 10,
    full_tree: bool = True,
    session_cookie: str = None
) -> None:
    """
    Download all trajectories from SWE-bench Pro.

    Args:
        csv_path: Path to swe-bench-pro.csv
        output_dir: Directory to save trajectories (organized by agent)
        resume: Skip already downloaded trajectories
        batch_size: Number of concurrent downloads
        full_tree: Whether to download full transcript trees
        session_cookie: Docent session cookie for authentication
    """
    df = pd.read_csv(csv_path)

    print(f"\n=== Downloading {len(df)} trajectories ===\n")
    print(f"Batch size: {batch_size} concurrent downloads")
    print(f"Full tree: {full_tree}")
    print(f"Authentication: {'✓ Using session cookie' if session_cookie else '✗ No authentication (will likely fail)'}")

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by agent
    agents = df['metadata.model_name'].unique()
    print(f"Found {len(agents)} agents\n")

    # Track statistics
    stats = {
        'downloaded': 0,
        'skipped': 0,
        'failed': 0,
        'total_bytes': 0
    }

    # Set up cookies if provided
    cookies = {}
    if session_cookie:
        cookies["docent_session"] = session_cookie

    async with httpx.AsyncClient(timeout=60.0, cookies=cookies) as client:
        for agent in agents:
            # Create agent directory
            agent_slug = agent.lower().replace(" ", "_").replace(".", "_").replace("-", "_")
            agent_dir = output_dir / agent_slug
            agent_dir.mkdir(parents=True, exist_ok=True)

            # Get trajectories for this agent
            agent_df = df[df['metadata.model_name'] == agent]

            print(f"[{agent}] - {len(agent_df)} trajectories")

            # Prepare download tasks
            tasks = []
            for _, row in agent_df.iterrows():
                agent_run_id = row['agent_run_id']
                instance_id = row['metadata.instance_id']
                output_file = agent_dir / f"{instance_id}.json"

                # Skip if already exists (resume mode)
                if resume and output_file.exists():
                    stats['skipped'] += 1
                    continue

                tasks.append((agent_run_id, output_file))

            # Download in batches with progress bar
            if not tasks:
                print(f"  All {len(agent_df)} trajectories already downloaded (skipped)")
                continue

            pbar = tqdm(total=len(tasks), desc=f"  {agent_slug}")

            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]

                # Download batch concurrently
                batch_results = await asyncio.gather(
                    *[fetch_trajectory(client, agent_run_id, full_tree)
                      for agent_run_id, _ in batch],
                    return_exceptions=True
                )

                # Save results
                for (agent_run_id, output_file), data in zip(batch, batch_results):
                    if isinstance(data, Exception):
                        stats['failed'] += 1
                    elif data is None:
                        stats['failed'] += 1
                    else:
                        try:
                            with open(output_file, 'w') as f:
                                json.dump(data, f, indent=2)
                            stats['downloaded'] += 1
                            stats['total_bytes'] += output_file.stat().st_size
                        except Exception as e:
                            stats['failed'] += 1
                            print(f"    Error saving {agent_run_id}: {e}")

                    pbar.update(1)

            pbar.close()

    # Print summary
    print("\n=== Download Complete ===\n")
    print(f"Downloaded: {stats['downloaded']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total size: {format_bytes(stats['total_bytes'])}")
    print(f"Output directory: {output_dir}")


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="Download SWE-bench Pro trajectories using Docent API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate storage requirements
  python download_swebench_pro_trajectories.py --estimate

  # Estimate with larger sample
  python download_swebench_pro_trajectories.py --estimate --sample-size 50

  # Download all trajectories (recommended settings)
  python download_swebench_pro_trajectories.py --download --batch-size 20

  # Resume interrupted download
  python download_swebench_pro_trajectories.py --download --resume

  # Download without full transcript trees (smaller files, faster)
  python download_swebench_pro_trajectories.py --download --no-full-tree

Notes:
  - Uses Docent REST API (no browser automation needed)
  - Much faster than web scraping
  - Adjust --batch-size based on your network connection
  - Use --resume to continue after interruption (Ctrl+C safe)
"""
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/swe-bench-pro.csv"),
        help="Path to SWE-bench Pro CSV file"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trajectory_data/swebench_pro"),
        help="Output directory for trajectories"
    )

    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate storage requirements without downloading all"
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download all trajectories"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted download (skip existing files)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of concurrent downloads (default: 10, recommended: 20)"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of trajectories to sample for estimation (default: 10)"
    )

    parser.add_argument(
        "--no-full-tree",
        action="store_true",
        help="Don't download full transcript trees (smaller files)"
    )

    parser.add_argument(
        "--session-cookie",
        type=str,
        help="Docent session cookie for authentication (required for API access)"
    )

    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        return

    if args.estimate:
        if not args.session_cookie:
            print("Warning: No session cookie provided. This will likely fail.")
            print("Provide --session-cookie or it will return 401 Unauthorized")
            print()

        try:
            total_bytes, avg_bytes = asyncio.run(estimate_storage(
                args.csv,
                args.sample_size,
                args.session_cookie
            ))

            print("\n=== Storage Estimate ===\n")
            print(f"Total trajectories: {len(pd.read_csv(args.csv)):,}")
            print(f"Sample size: {args.sample_size}")
            print(f"Avg trajectory size: {format_bytes(int(avg_bytes))}")
            print(f"Estimated total: {format_bytes(total_bytes)}")
            print(f"\nRecommended free space: {format_bytes(int(total_bytes * 1.2))} (with 20% buffer)")

            # Time estimate
            total_trajectories = len(pd.read_csv(args.csv))
            time_per_trajectory = 0.1  # seconds (much faster with API)
            total_time = (total_trajectories * time_per_trajectory) / args.batch_size
            minutes = total_time / 60
            print(f"\nEstimated download time: {minutes:.1f} minutes with batch_size={args.batch_size}")

        except Exception as e:
            print(f"Error during estimation: {e}")
            import traceback
            traceback.print_exc()
            return

    elif args.download:
        if not args.session_cookie:
            print("Error: --session-cookie is required for downloading")
            print("\nTo get your session cookie:")
            print("  1. Log into https://docent.transluce.org in your browser")
            print("  2. Open DevTools (F12) → Application → Cookies")
            print("  3. Copy the 'docent_session' cookie value")
            print("  4. Run with: --session-cookie YOUR_COOKIE_VALUE")
            return

        try:
            asyncio.run(download_all_trajectories(
                csv_path=args.csv,
                output_dir=args.output,
                resume=args.resume,
                batch_size=args.batch_size,
                full_tree=not args.no_full_tree,
                session_cookie=args.session_cookie
            ))
        except KeyboardInterrupt:
            print("\n\nDownload interrupted by user. Use --resume to continue later.")
        except Exception as e:
            print(f"Error during download: {e}")
            import traceback
            traceback.print_exc()
            return

    else:
        parser.print_help()
        print("\nError: Must specify --estimate or --download")


if __name__ == "__main__":
    main()
