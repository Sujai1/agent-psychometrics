# Auditor Agent

LLM-based agent (Claude Opus 4.5) that explores SWE-bench task environments via bash shell access and rates them on difficulty-related axes.

## Feature Versions

### V3 (Recommended) - 5 Features

Final feature set with strong correlation to IRT difficulty:

| Feature | Description | Correlation | Direction |
|---------|-------------|-------------|-----------|
| `fix_localization` | How spread out is the likely fix? | -0.587 | Higher = easier |
| `entry_point_clarity` | How easy is it to find where the bug manifests? | -0.502 | Higher = easier |
| `change_blast_radius` | How many components would be affected by changes? | +0.502 | Higher = harder |
| `debugging_setup_ease` | How easy is it to set up a debugging workflow? | -0.305 | Higher = easier |
| `test_feedback_quality` | How informative are the test failure messages? | -0.301 | Higher = easier |

### V2 - 6 Features (Experimental)

Added 3 new features, but `test_specificity` showed weak correlation (-0.065) and was dropped in V3.

### V1 - 6 Features (Original)

Original feature set including `test_runability`, `error_reproducibility`, `code_organization`. These showed weaker correlations and limited variance (e.g., `code_organization` was uniformly 4.0 across tasks).

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run V3 auditor on all 500 tasks (with batching and Docker cleanup)
python -m experiment_a.auditor_agent.run_auditor \
    --task auditor_task_v3 \
    --log_dir chris_output/auditor_features/swebench_verified_v3

# Run on specific tasks (comma-separated)
python -m experiment_a.auditor_agent.run_auditor \
    --task auditor_task_v3 \
    --sample_ids "django__django-11099,pytest-dev__pytest-5840"

# Run on tasks from a file (one per line)
python -m experiment_a.auditor_agent.run_auditor \
    --task auditor_task_v3 \
    --sample_ids_file sample_ids.txt

# Parse V3 logs and create CSV
python -m experiment_a.auditor_agent.parse_outputs \
    --log_dir chris_output/auditor_features/swebench_verified_v3 \
    --version 3 \
    --validate
```

## Batching

The auditor uses batching with Docker cleanup to handle storage constraints:

- **Batch size**: 10 tasks (configurable via `--batch_size`)
- **Max connections**: 5 parallel containers (configurable via `--max_connections`)
- **Docker cleanup**: Runs between batches to free storage (disable with `--skip_cleanup`)
- **Resume support**: Automatically skips already-completed tasks

```bash
# Custom batch configuration
python -m experiment_a.auditor_agent.run_auditor \
    --task auditor_task_v3 \
    --batch_size 10 \
    --max_connections 5 \
    --log_dir chris_output/auditor_features/swebench_verified_v3
```

## Files

| File | Purpose |
|------|---------|
| `prompts.py` | V1 feature definitions and system prompt |
| `prompts_v2.py` | V2 feature definitions (6 features) |
| `prompts_v3.py` | V3 feature definitions (5 features, recommended) |
| `inspect_task.py` | Inspect AI task definitions (`auditor_task`, `auditor_task_v2`, `auditor_task_v3`) |
| `run_auditor.py` | Batch orchestration with Docker cleanup |
| `parse_outputs.py` | Extract features from logs to CSV (supports `--version 1/2/3`) |
| `verify_commands.py` | Verify agent sees correct command outputs |

## Integration with Experiment A

```python
from experiment_ab_shared.feature_source import CSVFeatureSource, GroupedFeatureSource

auditor_features = CSVFeatureSource(
    Path("chris_output/auditor_features/swebench_verified_v3/auditor_features.csv"),
    name="Auditor"
)

# Combine with other features
grouped = GroupedFeatureSource([
    embedding_source,
    llm_judge_source,
    env_features_source,
    auditor_features,
])
```

## Correlation Analysis

Correlations computed on 48-50 task samples against IRT difficulty (higher β = harder):

| Feature | V1 Corr | V2 Corr | Notes |
|---------|---------|---------|-------|
| `fix_localization` | - | -0.587 | **Strongest predictor** (new in V2) |
| `entry_point_clarity` | -0.438 | -0.502 | Improved in V2 |
| `change_blast_radius` | +0.382 | +0.502 | Improved in V2 |
| `debugging_setup_ease` | - | -0.305 | New in V2 |
| `test_feedback_quality` | -0.330 | -0.301 | Consistent |
| `test_specificity` | - | -0.065 | Dropped in V3 (weak) |
| `code_organization` | -0.046 | - | Dropped (no variance) |
| `error_reproducibility` | -0.041 | - | Dropped (weak) |
| `test_runability` | -0.188 | - | Dropped |

## Verification

To verify the agent correctly sees command outputs:

```bash
# Run manual + agent verification on same task
python -m experiment_a.auditor_agent.verify_commands \
    --instance_id django__django-11099 \
    --model anthropic/claude-opus-4-5-20251101
```

This runs the same commands both manually (via Docker) and through the agent, comparing results.
