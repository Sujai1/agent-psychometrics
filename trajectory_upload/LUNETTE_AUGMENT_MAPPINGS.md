# Lunette Upload Mapping Augmentation

This script pre-computes task-to-run mappings by querying the Lunette API once and storing the results in `_lunette_uploads.json` files. This avoids repeated API queries when working with uploaded trajectories.

## Purpose

When trajectories are uploaded to Lunette, they may be split across multiple runs:
- **Batched uploads**: Agents with >100 trajectories are split (e.g., 5 runs × 100 trajectories)
- **Single-run uploads**: All trajectories in one run

To grade or analyze specific trajectories, you need to know which run each task belongs to. This script:
1. Queries `client.get_run()` for each run_id
2. Extracts the task list from each run
3. Stores the `task_id -> run_id` mapping in the upload tracking file
4. Also augments each trajectory object with its `run_id` for convenience

## Usage

```bash
source .venv/bin/activate

# Augment all agents
python trajectory_upload/lunette_augment_mappings.py

# Augment specific agents
python trajectory_upload/lunette_augment_mappings.py --agents 20240620_sweagent_claude3.5sonnet

# Dry run to see what would be done
python trajectory_upload/lunette_augment_mappings.py --dry_run

# Force re-augmentation (overwrite existing mappings)
python trajectory_upload/lunette_augment_mappings.py --force
```

## Output

The script updates `trajectory_data/unified_trajs/<agent>/_lunette_uploads.json` with:

### New Top-Level Fields

```json
{
  "agent": "20240620_sweagent_claude3.5sonnet",
  "run_ids": ["4b42140c-...", "c5dd5b11-...", ...],
  "task_to_run_map": {
    "django__django-11728": "4b42140c-d28b-4bcd-b6ad-5f09e7c8e785",
    "django__django-11815": "4b42140c-d28b-4bcd-b6ad-5f09e7c8e785",
    "astropy__astropy-12907": "c5dd5b11-8b03-4082-b20b-ec0c1f012c74",
    ...
  },
  "run_to_tasks_map": {
    "4b42140c-d28b-4bcd-b6ad-5f09e7c8e785": [
      "django__django-11728",
      "django__django-11815",
      ...
    ],
    "c5dd5b11-8b03-4082-b20b-ec0c1f012c74": [
      "astropy__astropy-12907",
      ...
    ]
  },
  "task_to_run_map_updated_at": "2026-01-08T14:30:52.123456",
  "trajectories": [...]
}
```

### Augmented Trajectory Objects

Each trajectory in the `trajectories` array gets a `run_id` field:

```json
{
  "task_id": "django__django-11728",
  "trajectory_id": "0f77e97b-0dd2-406d-8b4f-54c123c6e139",
  "resolved": false,
  "message_count": 91,
  "run_id": "4b42140c-d28b-4bcd-b6ad-5f09e7c8e785"
}
```

## Integration with Other Scripts

### lunette_batch_grading.py

The batch grading script automatically uses stored mappings if available:

```
Building trajectory-to-run mapping...
  20240620_sweagent_claude3.5sonnet: Using stored mapping (500 tasks, updated 2026-01-08)
```

If mappings are not available, it falls back to querying the API (slower):

```
Building trajectory-to-run mapping...
  agent_without_mapping: No stored mapping, will query Lunette API

Querying Lunette API for 1 agent(s)...
(Tip: Run 'python trajectory_upload/lunette_augment_mappings.py' to pre-compute all mappings)
```

### Other Scripts

Any script that needs to know which run a task belongs to can now:

```python
import json
from pathlib import Path

# Load augmented tracking file
with open("trajectory_data/unified_trajs/agent_name/_lunette_uploads.json") as f:
    data = json.load(f)

# Forward mapping: task_id -> run_id
task_to_run = data["task_to_run_map"]
run_id = task_to_run["django__django-11728"]

# Reverse mapping: run_id -> list of task_ids
run_to_tasks = data["run_to_tasks_map"]
tasks_in_run = run_to_tasks["4b42140c-d28b-4bcd-b6ad-5f09e7c8e785"]
print(f"Run has {len(tasks_in_run)} tasks")

# Or iterate through trajectories (each has run_id already)
for traj in data["trajectories"]:
    print(f"{traj['task_id']} -> {traj['run_id']}")
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--agents` | None | Specific agents to augment (default: all) |
| `--trajectories_dir` | trajectory_data/unified_trajs | Base directory containing agent folders |
| `--dry_run` | False | Show what would be done without making changes |
| `--force` | False | Overwrite existing mappings even if already present |

## Example Output

```bash
$ python trajectory_upload/lunette_augment_mappings.py --agents 20240620_sweagent_claude3.5sonnet

=== Lunette Upload Mapping Augmentation ===
Found 1 agents to process

[1/1] 20240620_sweagent_claude3.5sonnet
  Querying 5 run(s)...
  Mapped 500 tasks
  ✓ Saved updated tracking file

=== Summary ===
Total agents: 1
Augmented: 1
Skipped (already mapped): 0
Failed: 0
```

## When to Run

Run this script:
- **Once** after uploading trajectories with `lunette_batch_upload.py`
- **Again** if you force-reupload an agent (use `--force` flag)
- **Periodically** for all agents: `python trajectory_upload/lunette_augment_mappings.py`

## Performance

- **API calls**: 1 call per run (e.g., 5 calls for a batched agent)
- **Time**: ~2-5 seconds per agent
- **Cost**: Free (read-only API calls)
- **Benefit**: Eliminates repeated API queries in analysis scripts

## Verification

To verify the mappings are correct:

```bash
# Check forward mapping exists and has correct count
cat trajectory_data/unified_trajs/agent_name/_lunette_uploads.json | jq '.task_to_run_map | keys | length'
# Expected: 500

# Check reverse mapping shows task distribution per run
cat trajectory_data/unified_trajs/agent_name/_lunette_uploads.json | jq '.run_to_tasks_map | to_entries | map({run_id: .key[:16], task_count: (.value | length)})'
# Expected: [{"run_id": "...", "task_count": 100}, ...]

# Check trajectory objects have run_id
cat trajectory_data/unified_trajs/agent_name/_lunette_uploads.json | jq '.trajectories[0] | keys'
# Expected: ["message_count", "resolved", "run_id", "task_id", "trajectory_id"]
```
