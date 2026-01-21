# Experiment B: Frontier Task Difficulty Prediction

Predict difficulty of **frontier tasks** (tasks only solvable by newer models) using methods that do NOT have access to post-frontier agents.

## Overview

**Research Question**: Can we predict the difficulty of tasks that are currently beyond the capability of existing models, using only information available before those models were released?

**Setting**:
- **Date-based split**: Pre-frontier vs Post-frontier agents (by release date)
- **No data leakage**: Predictions made using only pre-frontier information; post-frontier agents only used for evaluation

## Evaluation Metrics

The two primary metrics reported are:
1. **ROC-AUC**: Ability to rank (agent, task) pairs by solve probability on frontier tasks
2. **MAE (days)**: Mean absolute error in predicting when tasks become solvable

## Frontier Task Definitions

Two definitions of "frontier task" are supported (both evaluated by default):

1. **Pass-rate based** (`passrate`): Tasks with ≤10% pre-frontier pass rate AND >10% post-frontier pass rate
2. **IRT-based** (`irt`): Tasks where NO pre-frontier agent has ≥30% solve probability under IRT

## Quick Start

```bash
source .venv/bin/activate

# Run on SWE-bench (default) - evaluates both frontier definitions
python -m experiment_b.compare_methods

# Run on TerminalBench
python -m experiment_b.compare_methods --dataset terminalbench

# Run with only one frontier definition
python -m experiment_b.compare_methods --frontier_definitions passrate

# Disable date forecasting (faster)
python -m experiment_b.compare_methods --no_forecast_dates

# Save results to CSV
python -m experiment_b.compare_methods --output_csv results.csv
```

## Results (2026-01-20)

### SWE-bench Verified

**Cutoff**: 2025-05-01 | **Pre-frontier agents**: 76 | **Post-frontier agents**: 55

#### Pass-rate Definition (47 frontier tasks)

| Method | ROC-AUC | MAE (days) |
|--------|---------|------------|
| Oracle (upper bound) | 0.8439 | 39.3 |
| SAD-IRT (best) | 0.8036 | 111.8 |
| Feature-IRT (Embedding) | 0.7744 | 107.8 |
| Baseline IRT (pre-frontier only) | 0.7654 | 95.8 |
| LLM Judge + Ridge | 0.7482 | 248.4 |
| Embedding + Ridge | 0.7476 | 229.9 |

#### IRT Definition (30 frontier tasks)

| Method | ROC-AUC | MAE (days) |
|--------|---------|------------|
| Oracle (upper bound) | 0.7412 | N/A |
| SAD-IRT (best) | 0.6998 | N/A |
| Feature-IRT (Embedding) | 0.6974 | N/A |
| Baseline IRT (pre-frontier only) | 0.6868 | N/A |

### TerminalBench

**Cutoff**: 2025-11-17 | **Pre-frontier agents**: 52 | **Post-frontier agents**: 31

#### Pass-rate Definition (18 frontier tasks)

| Method | ROC-AUC | MAE (days) |
|--------|---------|------------|
| Oracle (upper bound) | 0.8130 | 20.3 |
| Feature-IRT (Embedding) | 0.7378 | 39.4 |
| LLM Judge + Ridge | 0.7366 | 43.9 |
| Baseline IRT (pre-frontier only) | 0.6988 | 39.0 |

#### IRT Definition (4 frontier tasks)

| Method | ROC-AUC | MAE (days) |
|--------|---------|------------|
| Oracle (upper bound) | 0.7725 | N/A |
| Feature-IRT (Embedding) | 0.7542 | N/A |
| Baseline IRT (pre-frontier only) | 0.6740 | N/A |

**Key observations**:
- **Feature-IRT (Embedding) consistently outperforms Baseline IRT**
- **SAD-IRT** (using trajectory information) achieves strong results on SWE-bench
- Ridge-based predictors have high MAE, suggesting they don't generalize well to frontier tasks

## Methods Compared

| Method | Description | Training Data |
|--------|-------------|---------------|
| **Oracle** | True IRT difficulties (upper bound) | All agents |
| **Baseline IRT** | IRT trained on pre-frontier agents only | Pre-frontier responses |
| **Feature-IRT** | Joint IRT + feature learning (see below) | All tasks + pre-frontier responses |
| **Embedding + Ridge** | Task embeddings → Ridge regression | Non-frontier tasks + baseline IRT β |
| **LLM Judge + Ridge** | LLM semantic features → Ridge regression | Non-frontier tasks + baseline IRT β |

### Feature-IRT (New)

Feature-IRT learns task difficulties as a linear function of features, jointly with agent abilities:

```
b_i = w^T f_i + bias + r_i    (task difficulty)
θ_j learned jointly           (agent ability)
P(success) = sigmoid(θ_j - b_i)
```

**Key differences from Ridge predictors**:
- Learns from response patterns (IRT likelihood), not just baseline IRT difficulties
- Trains on ALL tasks (including frontier), since it uses pre-frontier agent responses
- Per-task residuals (r_i) with strong L2 regularization encourage feature-based predictions
- Ridge warm-start initialization for feature weights

**Hyperparameters** (grid search available via `--grid_search`):
- `l2_weight`: Regularization on feature weights (default: 0.01)
- `l2_residual`: Regularization on per-task residuals (default: 10.0)
- `use_residuals`: Whether to include per-task residuals (default: True)

## Evaluation Methodology

### 1. Spearman Correlation
- Compare predicted vs oracle difficulty ranking on frontier tasks
- No scale alignment needed (rank-based metric)

### 2. ROC-AUC with Scale Alignment
1. **Identify anchor tasks**: Tasks with 10-90% pass rate in BOTH pre- and post-frontier groups
2. **Fit affine transformation**: `oracle_β = slope × predicted_β + intercept` on anchors
3. **Compute probabilities**: For each (post-frontier agent, frontier task): `P(success) = sigmoid(θ_oracle - β_shifted)`
4. **Calculate ROC-AUC**: Compare predicted probabilities to actual responses

### 3. Date Forecasting (Optional)

Predict **when** tasks will become solvable with 50% probability, based on the linear relationship between difficulty and time discovered in Experiment D.

**Key insight**: From IRT, `P(success) = sigmoid(θ - β) = 0.5` when `θ = β`. Combined with Experiment D's finding that frontier ability grows linearly over time, we can predict when a task with difficulty β will become solvable.

**Methodology**:
1. **Compute ground truth**: For each task, find the earliest agent where `θ_agent >= β_oracle` (using Oracle IRT)
2. **Split tasks**: Pre-cutoff tasks (first capable agent before cutoff) for training; post-cutoff tasks for evaluation
3. **Fit linear model**: `days = slope × predicted_β + intercept` on pre-cutoff tasks
4. **Evaluate**: MAE, Pearson correlation on post-cutoff task predictions

```bash
# Run with date forecasting
python -m experiment_b.compare_methods --forecast_dates
```

**Example output** (TerminalBench):
```
DATE FORECASTING: PREDICT WHEN TASKS BECOME SOLVABLE
==========================================================================================

Data Summary:
  Post-cutoff tasks (eval set): 10
  Tasks without any capable agent: 12

Method                                          MAE (days)    Pearson r    R²(fit)      n
--------------------------------------------------------------------------------------
Oracle (upper bound)                                  20.3       0.7047     0.5106     10
Baseline IRT (pre-frontier only)                      39.0       0.1718     0.2845     10
Feature-IRT (Embedding)                               39.4       0.2748     0.2912     10
```

## Data Leakage Constraints

**Critical**: Oracle data and post-frontier agent data must NEVER be exposed during training.

- **Training ground truth**: Always from baseline IRT (pre-frontier agents only)
- **Oracle IRT**: Used ONLY for evaluation metrics (ROC-AUC alignment, date forecasting ground truth)
- **All methods train on ALL tasks**: Ground truth difficulties come from baseline IRT, which is trained only on pre-frontier agents

This constraint ensures a realistic simulation of predicting difficulty for tasks beyond current model capabilities.

## Architecture

Uses the shared `shared/` infrastructure for predictors:

```python
from shared.feature_source import EmbeddingFeatureSource, CSVFeatureSource
from shared.feature_predictor import FeatureBasedPredictor
from experiment_b.shared.feature_irt_predictor import FeatureIRTPredictor

# Ridge-based predictor
source = EmbeddingFeatureSource(embeddings_path)
predictor = FeatureBasedPredictor(source)

# Feature-IRT predictor (joint learning)
predictor = FeatureIRTPredictor(source, use_residuals=True)
predictor.fit(task_ids, ground_truth_b, responses)  # responses = pre-frontier only
```

### Directory Structure

```
experiment_b/
├── compare_methods.py        # Main entry point
├── shared/
│   ├── data_splits.py        # Agent/task splitting utilities
│   ├── evaluate.py           # Evaluation metrics (Spearman, AUC, alignment)
│   ├── baseline_irt.py       # Baseline IRT training with caching
│   ├── feature_irt_predictor.py  # Feature-IRT predictor
│   └── date_forecasting.py   # Date forecasting utilities (NEW)
└── datasets/                 # Dataset-specific configurations
    ├── base.py               # DatasetConfig base class
    ├── swebench.py           # SWE-bench config
    └── terminalbench.py      # TerminalBench config
```

### Key Functions

**Data Splitting** (`data_splits.py`):
- `split_agents_by_dates()`: Split by date cutoff
- `identify_frontier_tasks_irt()`: IRT-based frontier definition (no pre-frontier agent with theta >= beta)
- `identify_frontier_tasks()`: Pass-rate based definition (use `--frontier_definition passrate`)
- `identify_nontrivial_tasks()`: Anchor tasks (10-90% pass rate)

**Evaluation** (`evaluate.py`):
- `compute_frontier_difficulty_metrics()`: Spearman correlation
- `compute_scale_offset()`: Fit alignment transformation
- `compute_frontier_auc()`: ROC-AUC on frontier tasks

## Data Paths

### SWE-bench

| File | Purpose |
|------|---------|
| `clean_data/swebench_verified_20251120_full/1d_1pl/items.csv` | Oracle IRT difficulties |
| `clean_data/swebench_verified/swebench_verified_20251120_full.jsonl` | Response matrix |
| `chris_output/experiment_a/embeddings/` | Task embeddings |
| `chris_output/experiment_a/llm_judge_features/` | LLM judge features |

### TerminalBench

| File | Purpose |
|------|---------|
| `chris_output/terminal_bench_2.0/1d/items.csv` | Oracle IRT difficulties |
| `data/terminal_bench/terminal_bench_2.0.jsonl` | Response matrix |
| `chris_output/experiment_a_terminalbench/embeddings/` | Task embeddings |
| `chris_output/experiment_a_terminalbench/llm_judge_features/` | LLM judge features (4 pre-selected) |

## Configuration

Dataset configs in `datasets/`:

```python
@dataclass
class DatasetConfig:
    responses_path: Path
    oracle_irt_path: Path
    embeddings_path: Path
    llm_judge_path: Path
    cutoff_date: str
    pre_threshold: float = 0.1   # Max pass rate for pre-frontier
    post_threshold: float = 0.1  # Min pass rate for post-frontier

    @property
    def llm_judge_feature_cols(self) -> List[str]:
        # SWE-bench: 9 features
        # TerminalBench: 4 pre-selected features
```

## Command Line Options

```
--dataset              Dataset to use: swebench (default) or terminalbench
--frontier_definitions Space-separated list: 'passrate' 'irt' (default: both)
--no_forecast_dates    Disable date forecasting evaluation (faster, but no MAE metric)
--output_csv           Save results to CSV file
--grid_search          Run grid search over Feature-IRT hyperparameters
--verbose              Show alignment parameters and training progress
--cutoff_date          Override default cutoff date (YYYYMMDD format)
```

## Caches

| Cache | Location | When to Clear |
|-------|----------|---------------|
| **Baseline IRT** | `chris_output/experiment_b/{dataset}/baseline_irt/` | Auto-invalidated when training data changes |
| **Embeddings** | `chris_output/experiment_a{_terminalbench}/embeddings/` | When changing backbone |
| **LLM Features** | `chris_output/experiment_a{_terminalbench}/llm_judge_features/` | When re-extracting |

## Related Experiments

- **Experiment A**: Prior validation - tests how well static task features predict difficulty on held-out tasks
- **Experiment SAD-IRT**: Uses trajectory information for frontier difficulty prediction

## References

- [IRT Models Documentation](../docs/IRT_MODELS.md)
- [Research Proposal](../chris%20proposal.md) - Section 3.2
