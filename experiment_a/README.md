# Experiment A: Prior Validation (IRT AUC)

Evaluates how well a difficulty predictor can predict agent success on held-out tasks using the 1PL IRT model.

## Overview

**Goal**: Validate that predicted task difficulties are useful for forecasting agent performance without running agents on new tasks.

**Core Idea**: Given a predicted difficulty β̂_i and known agent ability θ_j, compute:

```
P(success) = sigmoid(θ_j - β̂_i)
```

Then measure AUC by comparing these predicted probabilities to actual binary outcomes.

This corresponds to **Section 3.1** in the [research proposal](../chris%20proposal.md).

## Quick Start

```bash
source .venv/bin/activate

# Run with baselines only (no embeddings required)
python -m experiment_a.train_evaluate

# Run with pre-computed embeddings
python -m experiment_a.train_evaluate --embeddings_path /path/to/embeddings.npz

# Run with Lunette features
python -m experiment_a.run_evaluation_v2

# Dry run to check config
python -m experiment_a.train_evaluate --dry_run
```

## Evaluation Protocol

1. **Split tasks** (not agents) into train/test sets using deterministic hash-based splitting
2. **Train difficulty predictor** on train tasks (e.g., embeddings → Ridge regression → difficulty)
3. **Predict difficulty** for test tasks
4. **Compute IRT probabilities**: For each (agent, task) pair, compute P(success) = sigmoid(θ - β̂)
5. **Calculate AUC**: Compare predicted probabilities to actual 0/1 outcomes

## Results (2026-01-13)

| Method | AUC | Description |
|--------|-----|-------------|
| Oracle (true b) | 0.9447 | Upper bound using ground truth IRT difficulty |
| **Embedding** | **0.8333** | Qwen3-VL-8B embeddings + Ridge |
| **Lunette v2** | **0.7522** | 24 features with Lasso selection |
| Constant baseline | 0.7176 | Predict mean difficulty for all tasks |
| Agent-only | 0.7178 | Use agent's overall success rate |
| Task-only | 0.5000 | Use mean pass rate (no discrimination) |

## Feature Sources

### 1. Embeddings (Best Performance)

Run Daria's pipeline on the Engaging cluster:

```bash
sbatch predict_question_difficulty_engaging.sh
```

This produces:
```
out/prior_qwen3vl8b/
├── embeddings__Qwen__Qwen3-VL-8B-Instruct__*.npz  # Input for experiment_a
├── predictions.csv     # Per-task predictions
└── metrics.json        # Train/test R², Pearson r
```

### 2. Lunette Features (24 total)

Extracts features using Lunette's sandbox environment:

**Environment-based (15)**: repo_file_count, repo_line_count, patch_file_count, patch_line_count, test_file_count, related_file_count, import_count, class_count_in_file, function_count_in_file, test_count_fail_to_pass, test_count_pass_to_pass, git_commit_count, directory_depth, has_conftest, has_init

**Semantic (9)**: fix_in_description (0-3), problem_clarity (1-5), error_message_provided (0/1), reproduction_steps (0/1), fix_locality (1-3), domain_knowledge_required (1-5), fix_complexity (1-5), logical_reasoning_required (1-5), atypicality (1-5)

**Running extraction:**
```bash
# Dry run
python -m experiment_a.overnight_lunette_extraction --dry_run

# Full extraction (~$75 for 500 tasks)
nohup python -m experiment_a.overnight_lunette_extraction --concurrency 5 &> overnight.log &

# Resume after interruption
python -m experiment_a.overnight_lunette_extraction --resume --concurrency 5

# Monitor progress
tail -f overnight.log
cat chris_output/experiment_a/lunette_features_v2/progress.json
```

**Output:** `chris_output/experiment_a/lunette_features_v2/`

**Top Lunette coefficients (Ridge on standardized features):**

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| domain_knowledge_required | +0.89 | More domain knowledge → harder |
| has_init | -0.73 | Package structure → easier |
| test_count_fail_to_pass | +0.70 | More failing tests → harder |
| directory_depth | -0.60 | Deeper directories → easier |
| fix_complexity | +0.58 | Complex fixes → harder |

## Module Structure

```
experiment_a/
├── __init__.py                    # Module exports
├── config.py                      # ExperimentAConfig dataclass
├── data_loader.py                 # Load IRT params, responses; create splits
├── difficulty_predictor.py        # DifficultyPredictor protocol + implementations
├── irt_evaluation.py              # AUC computation using 1PL IRT formula
├── baselines.py                   # Agent-only, task-only baselines
├── train_evaluate.py              # Main pipeline
├── lunette_grading_prompt.py      # 24-feature extraction prompt
├── overnight_lunette_extraction.py # Robust overnight extraction
├── postprocess_lunette_features.py # Parse features from Lunette responses
└── run_evaluation_v2.py           # Run evaluation with v2 features
```

## Output

Results saved to `chris_output/experiment_a/experiment_a_results.json`:

```json
{
  "config": {...},
  "data_summary": {"n_agents": 130, "n_tasks_total": 500, "n_train": 400, "n_test": 100},
  "oracle": {"auc": 0.9447},
  "embedding_predictor": {"auc_result": {"auc": 0.XX}, "difficulty_metrics": {...}},
  "constant_baseline": {"auc": 0.7176},
  "agent_only_baseline": {"auc": 0.7178},
  "task_only_baseline": {"auc": 0.5000}
}
```

## Command Line Options

```
--test_fraction     Fraction of tasks for test set (default: 0.2)
--split_seed        Random seed for train/test split (default: 0)
--embeddings_path   Path to pre-computed embeddings .npz file
--ridge_alpha       Ridge regression alpha (default: 10000.0)
--output_dir        Output directory (default: chris_output/experiment_a)
--dry_run           Show configuration without running
```

## Known Issues

**Train/Test Split Bias:** The hash-based split has statistically significant bias (p=0.025):
- Train tasks: mean b = +0.61
- Test tasks: mean b = -0.27
- Effect size: Cohen's d = 0.25 (small)

For fully reliable results, run Lunette extraction on all 500 tasks.

## References

- [Truong et al. (2025)](https://arxiv.org/pdf/2503.13335) - Amortized model-based evaluation
- IRT formula: `P = sigmoid(theta - beta)` matches py_irt's 1PL implementation
