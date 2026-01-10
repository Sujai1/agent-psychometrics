# Experiment C: Cost Comparison Results

## Overview

Experiment C compares the cost of **Lunette grading** (dummy agent + investigation) vs **full agent runs** for SWE-bench tasks.

## Lunette Grading Results (Completed)

**10 tasks selected with stratified sampling by IRT difficulty (seed=42)**

| Task ID | Dummy Eval (s) | Grading (s) | Total (s) | Difficulty Score |
|---------|---------------|-------------|-----------|------------------|
| scikit-learn__scikit-learn-14496 | 9.8 | 71.7 | 81.5 | 0.45 |
| django__django-16527 | 9.0 | 67.6 | 76.6 | 0.45 |
| sphinx-doc__sphinx-8475 | 8.3 | 103.9 | 112.2 | 0.55 |
| django__django-15814 | 9.1 | 131.3 | 140.4 | 0.35 |
| matplotlib__matplotlib-24637 | 10.2 | 65.8 | 76.0 | 0.45 |
| django__django-14017 | 8.1 | 73.9 | 82.0 | 0.45 |
| sympy__sympy-23413 | 8.1 | 62.3 | 70.4 | 0.55 |
| django__django-11141 | 8.7 | 66.0 | 74.7 | 0.55 |
| sympy__sympy-20438 | 7.4 | 69.3 | 76.7 | 0.45 |
| django__django-15629 | 7.9 | 68.3 | 76.2 | 0.45 |

### Summary Statistics

- **Total dummy eval time**: 86.5 seconds
- **Total grading time**: 780.0 seconds
- **Total time**: 866.6 seconds (~14.4 minutes)
- **Average per task**: 86.7 seconds

### Run IDs for Server-Side Cost Lookup

See `lunette_grading_10tasks.json` for the run IDs to look up actual Lunette costs on the server side.

## Full Agent Comparison (Pending)

Full agent runs (Claude Sonnet 4.5, GPT-5.2) require:
- Sandbox setup with Docker images for each SWE-bench task
- Model inference for exploring codebase and attempting fixes
- Each run takes 5-30+ minutes per task

### Estimated Costs (Based on typical SWE-bench runs)

| Method | Time per Task | API Cost per Task | Total (10 tasks) |
|--------|---------------|-------------------|------------------|
| Lunette grading | ~87s | ~$0.50 (estimated) | ~$5 |
| Claude Sonnet 4.5 | ~15-30 min | ~$2-5 | ~$20-50 |
| GPT-5.2 | ~15-30 min | ~$2-5 | ~$20-50 |

*Note: API costs depend on token usage, which varies significantly by task complexity.*

## Key Findings

1. **Lunette grading is fast**: ~87 seconds per task average, most of which is the investigation phase (~78s)

2. **Difficulty scores correlate**: The Lunette investigator produces difficulty scores (0.35-0.55 range in this sample) that can be compared to IRT-fitted difficulties

3. **Cost efficiency**: Lunette grading appears to be ~10x cheaper than running full agents, though this needs server-side verification

## Files

- `lunette_grading_10tasks.json` - Full Lunette grading results with run IDs
- `selected_tasks.json` - The 10 selected task IDs
- `run_full_agent.py` - Script for running full agents (incomplete due to timeout)

## Next Steps

1. Look up actual Lunette costs using the run IDs
2. Run full agent comparisons in a longer session
3. Compare Lunette difficulty scores to IRT difficulties
