# Auditor Agent

LLM-based agent that explores SWE-bench task environments and rates them on 6 difficulty-related axes.

## Features Assessed

| Feature | Description | Scale |
|---------|-------------|-------|
| `test_runability` | Can the test suite be executed? | 1-5 |
| `error_reproducibility` | Can the reported issue be triggered reliably? | 1-5 |
| `entry_point_clarity` | How easy is it to find where the bug manifests? | 1-5 |
| `code_organization` | How well-organized is the relevant code area? | 1-5 |
| `change_blast_radius` | How many components would be affected by changes? | 1-5 |
| `test_feedback_quality` | How informative are the test failure messages? | 1-5 |

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Test on a single instance
inspect eval experiment_a/auditor_agent/inspect_task.py@auditor_task \
    --model anthropic/claude-opus-4-5-20251101 \
    --sample-id django__django-11099 \
    --log-dir chris_output/auditor_test

# Run on all 500 tasks (with batching and Docker cleanup)
python -m experiment_a.auditor_agent.run_auditor

# Run on specific tasks (comma-separated)
python -m experiment_a.auditor_agent.run_auditor \
    --sample_ids "django__django-11099,pytest-dev__pytest-5840"

# Run on tasks from a file (one per line)
python -m experiment_a.auditor_agent.run_auditor \
    --sample_ids_file sample_ids.txt

# Parse logs and create CSV
python -m experiment_a.auditor_agent.parse_outputs \
    --log_dir chris_output/auditor_features/swebench_verified \
    --validate
```

## Files

| File | Purpose |
|------|---------|
| `prompts.py` | Feature definitions and system prompt |
| `inspect_task.py` | Inspect AI task definition with basic_agent |
| `run_auditor.py` | Batch orchestration with Docker cleanup |
| `parse_outputs.py` | Extract features from logs to CSV |
| `verify_commands.py` | Verify agent sees correct command outputs |

## Integration with Experiment A

```python
from experiment_ab_shared.feature_source import CSVFeatureSource, GroupedFeatureSource

auditor_features = CSVFeatureSource(
    Path("chris_output/auditor_features/swebench_verified/auditor_features.csv"),
    name="Auditor"
)

# Combine with other features
grouped = GroupedFeatureSource([
    embedding_source,
    llm_judge_source,
    env_features_source,
    auditor_features,  # NEW
])
```

## Verification

To verify the agent correctly sees command outputs:

```bash
# Run manual + agent verification on same task
python -m experiment_a.auditor_agent.verify_commands \
    --instance_id django__django-11099 \
    --model anthropic/claude-opus-4-5-20251101
```

This runs the same commands both manually (via Docker) and through the agent, comparing results.
