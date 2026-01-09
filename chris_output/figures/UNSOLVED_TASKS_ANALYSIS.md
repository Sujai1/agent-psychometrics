# SWE-bench Verified: Unsolved Tasks Analysis

**Analysis Date:** 2026-01-09
**Dataset:** SWE-bench Verified (500 tasks, 123 agents)
**IRT Model:** 1D 2PL (cutoff date: 2025-09-30)

---

## Key Findings

### 1. Completely Unsolved Tasks (All 123 Agents)

**Result:** **37 tasks** (7.4% of all tasks) remain completely unsolved by all 123 agents.

These tasks represent the hardest problems in the benchmark, with:
- **Mean difficulty:** b = 4.67
- **Range:** b = 4.55 to 5.04
- **All have high discrimination** (a > 1.8), indicating they reliably distinguish ability levels

**Hardest task:** `django__django-10999` (b = 5.038, a = 2.055)

<details>
<summary>View all 37 completely unsolved tasks</summary>

| Task ID | Difficulty (b) | Discrimination (a) |
|---------|----------------|---------------------|
| django__django-10999 | 5.038 | 2.055 |
| astropy__astropy-13977 | 4.824 | 1.997 |
| django__django-13513 | 4.841 | 1.978 |
| django__django-13344 | 4.755 | 1.815 |
| pydata__xarray-7229 | 4.754 | 1.953 |
| pydata__xarray-6992 | 4.920 | 2.057 |
| pylint-dev__pylint-4551 | 4.947 | 1.970 |
| matplotlib__matplotlib-25479 | 4.745 | 2.057 |
| sphinx-doc__sphinx-9461 | 4.866 | 1.920 |
| sympy__sympy-20438 | 4.791 | 2.142 |
| sympy__sympy-21930 | 4.810 | 2.054 |

(showing top 11 of 37)

</details>

---

### 2. Optimal Threshold for 100+ Unsolved Tasks

**Question:** If we only consider agents below a certain ability level, what's the highest threshold that still leaves at least 100 tasks unsolved?

**Answer:** **θ ≤ -0.427** (46 agents)

This threshold provides:
- **100 unsolved tasks** (20% of benchmark)
- **46 agents included** (37% of all agents)
- **Last agent included:** `20240920_solver` (θ = -0.427)

---

### 3. Unsolved Task Characteristics at Optimal Threshold

At the optimal threshold (θ ≤ -0.427), the 100 unsolved tasks have:

| Statistic | Difficulty (b) |
|-----------|----------------|
| Mean | 3.385 |
| Std Dev | 1.287 |
| Min | 0.549 |
| Max | 5.038 |
| Median | 3.619 |
| 25th percentile | 2.400 |
| 75th percentile | 4.659 |

**Key insight:** The unsolved tasks span a wide difficulty range, not just the hardest tasks. This suggests that even "medium difficulty" tasks (b ≈ 2-3) can be challenging for lower-ability agents.

---

### 4. The 46 Agents Included

Agents with θ ≤ -0.427, sorted by ability (highest to lowest):

| Rank | Agent | Ability (θ) | Std Error |
|------|-------|-------------|-----------|
| 1 | 20240920_solver | -0.427 | ±0.094 |
| 2 | 20250118_codeshellagent_gemini_2.0_flash_experimental | -0.463 | ±0.074 |
| 3 | 20250806_SWE-Exp_DeepSeek-V3 | -0.509 | ±0.092 |
| 4 | 20250214_agentless_lite_o3_mini | -0.536 | ±0.093 |
| 5 | 20250527_amazon.nova-premier-v1.0 | -0.551 | ±0.076 |
| ... | ... | ... | ... |
| 46 | 20231010_rag_gpt35 | -5.438 | ±0.531 |

<details>
<summary>View all 46 agents</summary>

1. 20240920_solver (θ = -0.427)
2. 20250118_codeshellagent_gemini_2.0_flash_experimental (θ = -0.463)
3. 20250806_SWE-Exp_DeepSeek-V3 (θ = -0.509)
4. 20250214_agentless_lite_o3_mini (θ = -0.536)
5. 20250527_amazon.nova-premier-v1.0 (θ = -0.551)
6. 20250629_deepswerl_r2eagent (θ = -0.554)
7. 20241030_nfactorial (θ = -0.571)
8. 20241016_composio_swekit (θ = -0.615)
9. 20240820_honeycomb (θ = -0.621)
10. 20241022_tools_claude-3-5-haiku (θ = -0.636)
11. 20250112_ugaiforge (θ = -0.644)
12. 20250226_swerl_llama3_70b (θ = -0.649)
13. 20241028_agentless-1.5_gpt4o (θ = -0.655)
14. 20241029_epam-ai-run-claude-3-5-sonnet (θ = -0.656)
15. 20250511_sweagent_lm_32b (θ = -0.663)
16. 20241113_nebius-search-open-weight-models-11-24 (θ = -0.687)
17. 20240628_autocoderover-v20240620 (θ = -0.767)
18. 20250725_sweagent_devstral_small_2507 (θ = -0.770)
19. 20240721_amazon-q-developer-agent-20240719-dev (θ = -0.812)
20. 20250616_Skywork-SWE-32B (θ = -0.832)
21. 20240617_factory_code_droid (θ = -0.896)
22. 20240620_sweagent_claude3.5sonnet (θ = -0.995)
23. 20241120_artemis_agent (θ = -1.055)
24. 20250306_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor (θ = -1.070)
25. 20240612_MASAI_gpt4o (θ = -1.150)
26. 20241007_nfactorial (θ = -1.198)
27. 20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128 (θ = -1.263)
28. 20241002_lingma-agent_lingma-swe-gpt-72b (θ = -1.298)
29. 20241016_epam-ai-run-gpt-4o (θ = -1.471)
30. 20240509_amazon-q-developer-agent-20240430-dev (θ = -1.544)
31. 20240918_lingma-agent_lingma-swe-gpt-72b (θ = -1.548)
32. 20240615_appmap-navie_gpt4o (θ = -1.568)
33. 20240820_epam-ai-run-gpt-4o (θ = -1.575)
34. 20250627_agentless_MCTS-Refine-7B (θ = -1.627)
35. 20241001_nfactorial (θ = -1.634)
36. 20240728_sweagent_gpt4o (θ = -1.664)
37. 20240402_sweagent_gpt4 (θ = -1.864)
38. 20241002_lingma-agent_lingma-swe-gpt-7b (θ = -1.925)
39. 20240402_sweagent_claude3opus (θ = -2.308)
40. 20240918_lingma-agent_lingma-swe-gpt-7b (θ = -2.643)
41. 20240402_rag_claude3opus (θ = -3.119)
42. 20231010_rag_claude2 (θ = -3.723)
43. 20240402_rag_gpt4 (θ = -4.043)
44. 20231010_rag_swellama7b (θ = -4.521)
45. 20231010_rag_swellama13b (θ = -4.619)
46. 20231010_rag_gpt35 (θ = -5.438)

</details>

---

### 5. Threshold Progression

How the number of unsolved tasks changes with ability threshold:

| Ability Threshold (θ) | # Agents | Unsolved Tasks |
|------------------------|----------|----------------|
| ≤ -5.438 | 1 | 498 (99.6%) |
| ≤ -1.664 | 11 | 305 (61.0%) |
| ≤ -1.198 | 21 | 176 (35.2%) |
| ≤ -0.687 | 31 | 136 (27.2%) |
| ≤ -0.554 | 41 | 113 (22.6%) |
| **≤ -0.427** | **46** | **100 (20.0%)** ← Optimal |
| ≤ -0.308 | 51 | 93 (18.6%) |
| ≤ -0.010 | 61 | 84 (16.8%) |
| ≤ 0.302 | 71 | 80 (16.0%) |
| ≤ 0.710 | 81 | 76 (15.2%) |
| ≤ 1.109 | 91 | 64 (12.8%) |
| ≤ 1.529 | 101 | 52 (10.4%) |
| ≤ 1.820 | 111 | 46 (9.2%) |
| ≤ 2.409 | 121 | 38 (7.6%) |
| ≤ 2.578 | 123 | 37 (7.4%) |

**Observation:** The relationship is roughly exponential - excluding higher-ability agents rapidly increases the number of unsolved tasks.

---

## Visualizations

Generated figures:
1. `unsolved_tasks_threshold_analysis.png` - 4-panel analysis showing:
   - Unsolved tasks vs ability threshold
   - Number of agents vs threshold
   - Unsolved tasks vs number of agents
   - Difficulty distribution of unsolved tasks at optimal threshold

2. `agent_abilities_threshold.png` - Bar chart showing all 123 agents with the optimal threshold highlighted

---

## Implications for Benchmarking

1. **For difficulty estimation:** The 100 tasks unsolved by agents with θ ≤ -0.427 provide a good "challenging subset" for training difficulty prediction models. These tasks span a wide difficulty range (b = 0.5 to 5.0) but remain unsolved by lower-ability agents.

2. **For benchmark design:** 7.4% of tasks are completely unsolved even by the best agents. These could be candidates for:
   - Manual review (are they solvable?)
   - Future work (track when they first get solved)
   - Difficulty ceiling validation

3. **For agent evaluation:** The threshold analysis shows that agent ability has a strong exponential relationship with task completion. Small improvements in ability (Δθ ≈ 0.5) can unlock significant numbers of previously unsolved tasks.

---

## Files Generated

- `analyze_unsolved_tasks.py` - Analysis script
- `visualize_threshold_analysis.py` - Visualization script
- `chris_output/figures/unsolved_tasks_threshold_analysis.png`
- `chris_output/figures/agent_abilities_threshold.png`
- `chris_output/figures/UNSOLVED_TASKS_ANALYSIS.md` (this file)
