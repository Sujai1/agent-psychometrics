# Lunette Batch Grading Pipeline

This pipeline grades uploaded Lunette trajectories to evaluate how well difficulty features predict IRT parameters. It uses the same fixed evaluation prompt from [lunette_analysis.py](lunette_analysis.py) but operates on already-uploaded trajectories.

## Overview

**Goal:** Grade a stratified sample of ~50 tasks across multiple agents to compute correlations between LLM-extracted features and IRT difficulty parameters.

**Key difference from existing pipelines:**
- `lunette_analysis.py`: Uploads local trajectories one-by-one and grades them
- `lunette_batch_grading.py`: Uses trajectories already uploaded by `lunette_batch_upload.py`

## Setup

All trajectories should be uploaded and augmented with task-to-run mappings:

```bash
# 1. Upload trajectories (if not already done)
python trajectory_upload/lunette_batch_upload.py

# 2. Augment with task-to-run mappings (recommended for performance)
python trajectory_upload/lunette_augment_mappings.py

# This queries the Lunette API once to build mappings and stores them
# in _lunette_uploads.json files. The batch grading script will use
# these stored mappings, avoiding repeated API queries.
```

See [LUNETTE_AUGMENT_MAPPINGS.md](LUNETTE_AUGMENT_MAPPINGS.md) for details.

## Usage

### Dry Run (Recommended First)

```bash
source .venv/bin/activate

# See execution plan without running
python llm_judge/lunette_batch_grading.py --dry_run --n_tasks 50 --n_agents 3
```

**Example output:**
```
Found 44 agents with uploaded trajectories

Selected 3 agents:
  - 20240620_sweagent_claude3.5sonnet
  - 20240728_sweagent_gpt4o
  - 20250616_Skywork-SWE-32B

=== Grading Plan ===
Total trajectories to grade: 150
Unique tasks: 150

DRY RUN - not executing grading

Estimated cost (assuming ~$0.50 per grading):
  $75.00
```

### Run Grading

```bash
# Grade 50 tasks across 3 agents
python llm_judge/lunette_batch_grading.py --n_tasks 50 --n_agents 3

# Grade specific agents
python llm_judge/lunette_batch_grading.py \
  --agents 20240620_sweagent_claude3.5sonnet 20240728_sweagent_gpt4o \
  --n_tasks 50

# Use different random seed
python llm_judge/lunette_batch_grading.py --n_tasks 50 --n_agents 3 --seed 123
```

## Grading Features

The fixed grading prompt extracts **12 features** per trajectory:

### Task-Intrinsic Features (7)
1. **fix_in_description** (0-3): Does the problem statement hint at the fix?
2. **problem_clarity** (1-5): How clear is the problem specification?
3. **error_message_provided** (0/1): Is an error message included?
4. **reproduction_steps** (0/1): Are reproduction steps provided?
5. **fix_locality** (1-3): How localized is the fix?
6. **domain_knowledge_required** (1-5): How much specialized knowledge is needed?
7. **fix_complexity** (1-5): How complex is the fix?

### Trajectory-Based Signals (5)
8. **agent_declared_success_wrongly** (0/1): False positive completion claim?
9. **agent_looping** (0/1): Stuck in a loop?
10. **agent_expressed_uncertainty** (0/1): Agent expressed confusion?
11. **agent_wrong_file_focus** (0/1): Spent time on irrelevant files?
12. **agent_gave_up_early** (0/1): Stopped before exhausting options?

## Output

Results are saved to `chris_output/lunette_grading/`:

```
chris_output/lunette_grading/
├── grading_results_20260108_143052.csv    # Raw feature scores per trajectory
├── correlations_20260108_143052.csv       # Correlation with IRT difficulty
└── discrimination_20260108_143052.csv     # Feature discrimination analysis
```

### grading_results_TIMESTAMP.csv

Columns:
- `task_id`: SWE-bench instance ID
- `agent`: Agent name
- `run_id`: Lunette run ID
- `difficulty`: IRT b parameter
- `resolved`: Whether agent solved the task
- `fix_in_description`, `problem_clarity`, ... (12 feature scores)
- `reasoning`: LLM explanation
- `error`: Error message (if grading failed)

### correlations_TIMESTAMP.csv

Measures how well each feature predicts IRT task difficulty.

Columns:
- `feature`: Feature name
- `correlation`: Pearson r with IRT difficulty
- `p_value`: Statistical significance
- `n`: Number of valid observations

**High correlation** means the feature is a good predictor of task difficulty.

### discrimination_TIMESTAMP.csv

Measures how well each feature differentiates between agents (agent signatures).

Columns:
- `feature`: Feature name
- `overall_mean`: Mean value across all trajectories
- `overall_std`: Standard deviation across all trajectories
- `within_agent_var`: Average variance within each agent (consistency)
- `between_agent_var`: Variance of agent means (differentiation)
- `discrimination_ratio`: `between_var / within_var` (higher = better discriminator)

**High discrimination ratio** means the feature shows stable differences between agents:
- Low `within_agent_var`: Each agent is consistent across tasks
- High `between_agent_var`: Agents differ substantially in their means
- Example: If Agent A consistently scores 4-5 on a feature while Agent B consistently scores 1-2, the discrimination ratio will be high

## Existing Results

Previous Lunette grading results (from `lunette_analysis.py`, n=39):

| Feature | Correlation | p-value |
|---------|-------------|---------|
| fix_in_description | -0.57 | 0.001 *** |
| problem_clarity | -0.38 | 0.025 * |
| fix_complexity | +0.34 | 0.046 * |
| reproduction_steps | +0.29 | 0.11 |
| domain_knowledge_required | +0.24 | 0.18 |

**Key finding:** `fix_in_description` is the strongest predictor. Tasks where the problem statement hints at the fix are significantly easier.

## Two Types of Analysis

This script performs two complementary analyses:

### 1. Difficulty Prediction (Correlation Analysis)

**Question:** Which features predict how hard a task is for agents in general?

**Metric:** Pearson correlation with IRT difficulty parameter `b`

**Interpretation:**
- High positive correlation: Feature increases with task difficulty (e.g., `fix_complexity`)
- High negative correlation: Feature decreases with task difficulty (e.g., `fix_in_description`)
- Near-zero correlation: Feature doesn't predict difficulty

**Use case:** Estimating task difficulty from problem statements without running agents

### 2. Agent Differentiation (Discrimination Analysis)

**Question:** Which features reveal stable behavioral differences between agents?

**Metric:** Discrimination ratio = `between_agent_var / within_agent_var`

**Interpretation:**
- High ratio: Agents have distinct "signatures" on this feature
  - Example: Agent A always explores deeply (4-5), Agent B explores shallowly (1-2)
- Low ratio: Feature is noisy or task-dependent, not agent-specific

**Use case:** Identifying agent signatures for behavioral clustering or classification

### Example Scenarios

| Feature | High Correlation | High Discrimination | Interpretation |
|---------|------------------|---------------------|----------------|
| `fix_complexity` | ✓ | ✗ | Predicts difficulty but all agents perceive it similarly |
| `agent_looping` | ✓ | ✓ | Hard tasks cause loops, some agents loop more than others |
| `error_message_provided` | ✗ | ✗ | Neither predicts difficulty nor differentiates agents |
| `exploration_depth` | ✗ | ✓ | Agent-specific behavior not related to task difficulty |

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--n_tasks` | 50 | Tasks to grade per agent |
| `--n_agents` | 3 | Number of agents to sample |
| `--agents` | None | Specific agents (default: random) |
| `--items_path` | chris_output/.../1d/items.csv | IRT parameters file |
| `--trajectories_dir` | trajectory_data/unified_trajs | Uploaded trajectories |
| `--output_dir` | chris_output/lunette_grading | Output directory |
| `--seed` | 42 | Random seed |
| `--dry_run` | False | Show plan without executing |

## Implementation Details

### Sampling Strategy

1. **Agent selection**: Random sample of n_agents from available agents
2. **Task selection**: Per agent, random sample of n_tasks from valid trajectories
3. **Valid trajectories**: Must exist in IRT items.csv (500 SWE-bench Verified tasks)

### Grading Process

1. Load IRT difficulty parameters from items.csv
2. Query Lunette API to build task-to-run mappings:
   - For each agent, load upload tracking file (`_lunette_uploads.json`)
   - Call `client.get_run()` for each run_id to retrieve trajectory list
   - Build mapping: `task_id -> run_id` (handles both batched and non-batched uploads)
3. Sample n_tasks from available trajectories per agent
4. For each trajectory:
   - Look up correct run_id from task-to-run mapping
   - Call Lunette `investigate()` API with GradingPlan
   - Extract 12 features + reasoning
   - Save incrementally to CSV
5. Compute Pearson correlations with IRT difficulty
6. Compute discrimination ratios (if multiple agents)
7. Report statistically significant features (p < 0.05)

**Note on run mapping:** The script queries the Lunette API to determine which run each task belongs to. This works for both:
- **Batched uploads**: Agents with >100 trajectories split across multiple runs
- **Single-run uploads**: Agents with all 500 trajectories in one run

The script automatically detects and handles both cases.

### Cost Estimation

- ~$0.50 per trajectory (Lunette grading call with Claude)
- 50 tasks × 3 agents = 150 trajectories = **~$75**
- For full run (100 tasks × 5 agents): **~$250**

## Relation to Evolutionary Pipeline

This script is **independent** of the evolutionary feature discovery system in `llm_judge/evolutionary/`. Key differences:

| Aspect | Batch Grading | Evolutionary |
|--------|---------------|--------------|
| Features | Fixed 12 features | Auto-discovered & evolved |
| Evaluation | Lunette trajectories | Direct LLM on problem text |
| Cost | ~$0.50/trajectory | ~$0.01/task |
| Environment | Full sandbox access | Problem + patch only |

The evolutionary system could potentially be adapted to use Lunette grading in the future, but the current batch grading script is simpler and sufficient for evaluating the fixed feature set.

## Next Steps

1. **Run initial grading**: 50 tasks × 3 agents (~$75)
2. **Analyze results**: Which features correlate with difficulty?
3. **Compare to existing results**: Do trajectory signals add predictive power?
4. **Expand if promising**: Run larger sample (100+ tasks, 5+ agents)
5. **Iterate on prompt**: Refine feature definitions based on results
