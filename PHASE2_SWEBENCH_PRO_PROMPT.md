# Phase 2: SWE-bench Pro LLM Judge Feature Extraction

## Context

Phase 1 is complete: SWE-bench Pro is now integrated into Experiment B. The dataset config, IRT model, and LLM judge infrastructure are all set up.

**Current results (Experiment B, pass-rate frontier definition, 60 frontier tasks):**
- Oracle: 0.7975 ROC-AUC
- Embedding + Ridge: 0.7412 ROC-AUC
- Baseline IRT: 0.6500 ROC-AUC
- LLM Judge: Not yet available (this is what we're building)

## Your Task

Extract LLM judge features for SWE-bench Pro tasks, validate their discriminativeness, and select the best features for difficulty prediction.

## Step 1: Run Feature Extraction on 100 Task Sample (~$6)

```bash
source .venv/bin/activate

# First, dry run to verify setup
python -m experiment_ab_shared.llm_judge extract --dataset swebench_pro --dry-run \
    --output-dir chris_output/experiment_a_swebench_pro/llm_judge_features

# Extract features on 100 task sample
python -m experiment_ab_shared.llm_judge extract --dataset swebench_pro --limit 100 \
    --output-dir chris_output/experiment_a_swebench_pro/llm_judge_features

# Aggregate to CSV
python -m experiment_ab_shared.llm_judge aggregate --dataset swebench_pro \
    --output-dir chris_output/experiment_a_swebench_pro/llm_judge_features
```

## Step 2: Validate Features Against IRT Difficulty

After extracting features, analyze their correlation with oracle IRT difficulty (β):

1. Load the features CSV: `chris_output/experiment_a_swebench_pro/llm_judge_features/llm_judge_features.csv`
2. Load oracle IRT difficulties: `chris_output/swebench_pro_irt/1d/items.csv` (column `b`)
3. For each of the 9 features, compute:
   - Pearson correlation with difficulty
   - Check coefficient sign (higher feature value should correlate with higher/lower difficulty as expected)
4. Run Ridge regression on all 9 features, check R² and coefficients
5. Run Lasso to identify which features can be dropped

**Expected features (same as SWE-bench Verified):**
- fix_in_description (0-3): Lower → harder
- problem_clarity (1-5): Lower → harder
- error_message_provided (0-1): 0 → harder
- reproduction_steps (0-1): 0 → harder
- fix_locality (1-3): Higher → harder
- domain_knowledge_required (1-5): Higher → harder
- fix_complexity (1-5): Higher → harder
- logical_reasoning_required (1-5): Higher → harder
- atypicality (1-5): Higher → harder

## Step 3: Feature Selection

Based on the analysis:
1. Identify which features have significant correlation with difficulty
2. Check if any features are redundant (high multicollinearity)
3. If needed, update `experiment_b/swebench_pro/config.py` to use only the selected features in `llm_judge_feature_cols`

TerminalBench went from 8 → 4 features using this process. SWE-bench Verified uses all 9.

## Step 4: Full Run (If Features Look Good)

If the 100-task sample shows features are discriminative:

```bash
# Full extraction (~$44 with Opus 4.5, or cheaper with Sonnet)
python -m experiment_ab_shared.llm_judge extract --dataset swebench_pro \
    --output-dir chris_output/experiment_a_swebench_pro/llm_judge_features

# Or use a cheaper model (Sonnet 4)
python -m experiment_ab_shared.llm_judge extract --dataset swebench_pro \
    --model claude-sonnet-4-20250514 \
    --output-dir chris_output/experiment_a_swebench_pro/llm_judge_features

# Aggregate final CSV
python -m experiment_ab_shared.llm_judge aggregate --dataset swebench_pro \
    --output-dir chris_output/experiment_a_swebench_pro/llm_judge_features
```

## Step 5: Re-run Experiment B with LLM Judge

```bash
python -m experiment_b.compare_methods --dataset swebench_pro --frontier_definitions passrate
```

## Key Files

| Purpose | Path |
|---------|------|
| Dataset config | `experiment_b/swebench_pro/config.py` |
| LLM judge prompt | `experiment_ab_shared/llm_judge/prompts/swebench_pro.py` |
| Oracle IRT difficulties | `chris_output/swebench_pro_irt/1d/items.csv` |
| Feature output dir | `chris_output/experiment_a_swebench_pro/llm_judge_features/` |
| Task data source | HuggingFace `ScaleAI/SWE-bench_Pro` (731 tasks) |

## Task ID Mapping

The task IDs need to match between LLM judge features and IRT items:
- HuggingFace `instance_id`: `"instance_flipt-io__flipt-cd18e54a..."`
- IRT `item_id`: `"flipt-io__flipt-cd18e54a..."` (strip `instance_` prefix, strip `-v...` suffix)

Check `swebench_irt/prep_swebench_pro.py` for the exact cleaning logic if there are mismatches.

## Notes

- Use Opus 4.5 for best quality, or Sonnet 4 for lower cost (~3x cheaper)
- The LLM judge's release date should be before the frontier cutoff (2025-09-01) to avoid data leakage - Opus 4.5 was released 2025-11-24 so this is actually a concern. Consider using an older model like Claude Sonnet 4 (2025-05-22) or Claude Opus 4.1 (2025-08-05) for strict no-leakage.
- 730 tasks total, but HuggingFace has 731 (one may be filtered during IRT training)
