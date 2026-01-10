# Experiment C: Cost Comparison Results

## Overview

Experiment C compares the cost of **Lunette grading** (dummy agent + investigation) vs **full agent runs** for SWE-bench tasks.

## Results (5 tasks that ran successfully)

Note: 5 of 10 tasks failed due to missing Docker images, not model failures.

### Cost & Time Comparison

| Metric | Lunette | Claude Sonnet 4.5 | GPT-5.2 |
|--------|---------|-------------------|---------|
| Tasks | 5 | 5 | 5 |
| Total time (s) | 379.9 | 716.7 | 742.5 |
| Total time (min) | 6.3 | 11.9 | 12.4 |
| Avg time per task (s) | 76.0 | 143.3 | 148.5 |
| Input tokens (non-cached) | N/A | 29 | 37,630 |
| Input tokens (cached) | N/A | 116,170 | 309,632 |
| Output tokens | N/A | 2,863 | 2,957 |
| Total cost (USD) | Unknown | $0.0779 | $0.3782 |
| Avg cost per task | Unknown | $0.0156 | $0.0756 |

### Cost Comparison

- **GPT vs Claude**: 4.86x more expensive

## Key Findings

1. **Lunette grading is ~2x faster** than full agent runs (76s vs 143s per task)

2. **Claude Sonnet 4.5 is very cheap** due to aggressive prompt caching:
   - 116K cached tokens vs only 29 non-cached tokens
   - $0.016 per task average

3. **GPT-5.2 is ~5x more expensive than Claude** despite similar output tokens:
   - Less efficient caching (37K non-cached vs Claude's 29)
   - $0.076 per task average

4. **Both models ran on identical 5 tasks**:
   - django__django-14017
   - sympy__sympy-23413
   - django__django-11141
   - sympy__sympy-20438
   - django__django-15629

## Lunette Grading Details (All 10 tasks)

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

**Total**: 866.6 seconds (~14.4 minutes) for 10 tasks

## Files

- `lunette_grading_10tasks.json` - Full Lunette grading results with run IDs
- `agent_claude_sonnet_10tasks.json` - Claude Sonnet 4.5 results
- `agent_gpt_5.2_10tasks.json` - GPT-5.2 results
- `token_usage_extracted.json` - Detailed token usage from eval logs
- `experiment_c_comparison.json` - Structured comparison data
- `selected_tasks.json` - The 10 selected task IDs

## Next Steps

1. Get actual Lunette costs from server-side billing
2. Re-run failed tasks after fixing Docker image issues
3. Compare Lunette difficulty scores to IRT difficulties
