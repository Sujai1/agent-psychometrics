# SWE-bench Pro Trajectory Format

## Summary

✅ **All 9,729 trajectories follow the same format**

- **Format:** JSON list with 2 items
- **Consistent across all 14 agents**
- **Total size:** 2.0 GB
- **Location:** `trajectory_data/swebench_pro/`

## File Structure

Each trajectory file is a JSON list with exactly **2 items**:

```json
[
  {/* Item 0: Agent Run Data */},
  {/* Item 1: Tree Structure */}
]
```

### Item 0: Agent Run Data

Contains the main agent run information including transcript and metadata.

**Keys:**
- `id` (string): Unique agent run ID
- `name` (null): Agent run name (always null)
- `description` (null): Description (always null)
- `transcripts` (list): List of transcript objects (usually 1 item)
- `transcript_groups` (list): Transcript groupings
- `metadata` (dict): Run metadata

#### Metadata Structure

```json
{
  "turns": 77,                    // Number of conversation turns
  "resolved": true,               // Whether task was resolved
  "model_name": "Claude 4.5 Sonnet - 10132025",  // Agent/model name
  "instance_id": "instance_..."   // Problem instance ID
}
```

#### Transcript Structure

Each transcript contains:

```json
{
  "id": "uuid",
  "name": null,
  "description": null,
  "transcript_group_id": null,
  "created_at": "2025-10-13T22:44:07.807384",
  "messages": [/* List of messages */],
  "metadata": {/* Transcript-level metadata */}
}
```

#### Message Structure

Messages contain the actual conversation:

```json
{
  "id": null,
  "content": "Observation: ...",
  "role": "user" | "assistant" | "system",
  "created_at": "timestamp",
  "metadata": {/* Message-level metadata */}
}
```

### Item 1: Tree Structure

Contains the conversation tree structure for branching/exploration.

**Keys:**
- `nodes` (list): Tree nodes representing conversation states
- `transcript_id_to_idx` (dict): Maps transcript IDs to indices
- `parent_map` (dict): Parent-child relationships
- `otel_message_ids_by_transcript_id` (dict): OpenTelemetry message IDs

This structure enables:
- Branching conversations
- Multiple exploration paths
- Conversation tree visualization
- Parent-child message relationships

## File Naming Convention

Files are named: `instance_{task_id}.json`

Examples:
- `instance_NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac-vnan.json`
- `instance_ansible__ansible-379058e10f3dbc0fdcaf80394bd09b18927e7d33-v1055803c3a812189a1133297f7f5468579283f86.json`
- `instance_qutebrowser__qutebrowser-99029144b5109bb1b2a53964a7c129e009980cd9-va0fd88aac89cde702ec1ba84877234da33adce8a.json`

Pattern: `instance_{repo}__{repo}-{commit_hash}-{version_hash}.json`

## Directory Organization

```
trajectory_data/swebench_pro/
├── claude_4_5_haiku____10222025/        (731 files)
├── claude_4_5_sonnet___10132025/        (732 files)
├── claude_4_sonnet___10132025/          (564 files)
├── claude_opus_4_1___paper/             (665 files)
├── claude_sonnet_4___paper/             (628 files)
├── gemini_2_5_pro_preview____debug_oct22/ (730 files)
├── gemini_2_5_pro_preview___paper/      (721 files)
├── glm_4_5____10222025/                 (731 files)
├── gpt_4o___paper/                      (621 files)
├── gpt_5___10132025/                    (731 files)
├── gpt_5_codex____debug_oct22/          (710 files)
├── gpt_5_high___paper/                  (732 files)
├── gpt_oss___paper/                     (730 files)
└── kimi___paper/                        (731 files)
```

Agent names are slugified:
- Spaces → underscores
- Periods → underscores
- Hyphens → underscores

## Key Features for Analysis

### Available Metadata

From each trajectory, you can extract:

1. **Task-level:**
   - `instance_id`: Problem identifier
   - `resolved`: Success/failure
   - `turns`: Conversation length
   - `model_name`: Agent/model used

2. **Conversation-level:**
   - Full message history
   - Message roles (user/assistant/system)
   - Timestamps
   - Tool calls and outputs (in message content)

3. **Tree-level:**
   - Branching structure
   - Exploration paths
   - Parent-child relationships

### Use Cases

**For IRT Analysis:**
- Extract `resolved` (0/1) as binary response
- Use `instance_id` as item identifier
- Use `model_name` as subject identifier
- Use `turns` as auxiliary information

**For LLM Judge Analysis:**
- Extract full conversation from `messages`
- Parse tool outputs from message content
- Correlate trajectory features with `resolved`
- Predict difficulty from conversation patterns

**For Trajectory Analysis:**
- Analyze branching/exploration patterns
- Study tool usage patterns
- Examine error recovery strategies
- Compare agent behaviors across models

## Sample Access Code

### Python

```python
import json
from pathlib import Path

# Load a trajectory
trajectory_file = Path("trajectory_data/swebench_pro/claude_4_5_sonnet___10132025/instance_....json")
with open(trajectory_file) as f:
    agent_run, tree = json.load(f)

# Extract metadata
metadata = agent_run["metadata"]
resolved = metadata["resolved"]
turns = metadata["turns"]
model = metadata["model_name"]
instance_id = metadata["instance_id"]

# Get conversation messages
transcript = agent_run["transcripts"][0]
messages = transcript["messages"]

print(f"Task: {instance_id}")
print(f"Agent: {model}")
print(f"Resolved: {resolved}")
print(f"Turns: {turns}")
print(f"Messages: {len(messages)}")
```

### For IRT Format

```python
import json
import pandas as pd
from pathlib import Path

# Convert to IRT format (JSONL)
trajectory_dir = Path("trajectory_data/swebench_pro")
irt_data = []

for agent_dir in trajectory_dir.iterdir():
    if not agent_dir.is_dir():
        continue

    agent_responses = {}

    for traj_file in agent_dir.glob("*.json"):
        with open(traj_file) as f:
            agent_run, _ = json.load(f)

        instance_id = agent_run["metadata"]["instance_id"]
        resolved = agent_run["metadata"]["resolved"]

        agent_responses[instance_id] = 1 if resolved else 0

    irt_data.append({
        "subject_id": agent_dir.name,
        "responses": agent_responses
    })

# Save as JSONL
with open("swebench_pro_irt.jsonl", "w") as f:
    for record in irt_data:
        f.write(json.dumps(record) + "\n")
```

## Format Validation

**Verified properties:**
- ✅ All files are valid JSON
- ✅ All files are lists with exactly 2 items
- ✅ Item 0 has keys: `id`, `name`, `description`, `transcripts`, `transcript_groups`, `metadata`
- ✅ Item 1 has keys: `nodes`, `transcript_id_to_idx`, `parent_map`, `otel_message_ids_by_transcript_id`
- ✅ Metadata contains: `turns`, `resolved`, `model_name`, `instance_id`
- ✅ Format is consistent across all 14 agents
- ✅ No parsing errors in 42 sampled files (3 per agent)

## Comparison to SWE-bench Verified Format

SWE-bench Pro trajectories are **different** from SWE-bench Verified:

| Feature | SWE-bench Pro | SWE-bench Verified |
|---------|---------------|-------------------|
| Format | List with 2 items (run + tree) | Custom trajectory format |
| Tree structure | ✓ Included (item 1) | ✗ Not included |
| Branching support | ✓ Full tree structure | ✗ Linear only |
| Message format | Docent format | Agent-specific format |
| Metadata location | Nested in item 0 | Top-level |
| File organization | By agent directory | By agent directory |

The SWE-bench Pro format is more structured and includes branching/exploration support, making it suitable for analyzing agent exploration strategies and conversation trees.
