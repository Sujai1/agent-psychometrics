# Trajectory Filter Verification Log

This file tracks verification of filtered trajectories for each agent.

## Verification Status

| Agent | Status | Reduction | Notes |
|-------|--------|-----------|-------|
| 20240402_sweagent_claude3opus | | | |
| 20240402_sweagent_gpt4 | | | |
| 20240612_MASAI_gpt4o | | | |
| 20240620_sweagent_claude3.5sonnet | | | |
| 20240721_amazon-q-developer-agent-20240719-dev | | | |
| 20240728_sweagent_gpt4o | | | |
| 20240820_epam-ai-run-gpt-4o | | | |
| 20240820_honeycomb | | | |
| 20240918_lingma-agent_lingma-swe-gpt-72b | | | |
| 20241007_nfactorial | | | |
| 20241016_composio_swekit | | | |
| 20241016_epam-ai-run-gpt-4o | | | |
| 20241025_composio_swekit | | | |
| 20241028_agentless-1.5_gpt4o | | | |
| 20241029_OpenHands-CodeAct-2.1-sonnet-20241022 | | | |
| 20241029_epam-ai-run-claude-3-5-sonnet | | | |
| 20241106_navie-2-gpt4o-sonnet | | | |
| 20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022 | | | |
| 20241108_devlo | | | |
| 20241113_nebius-search-open-weight-models-11-24 | | | |
| 20241125_enginelabs | | | |
| 20241125_marscode-agent-dev | | | |
| 20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128 | | | |
| 20241202_agentless-1.5_claude-3.5-sonnet-20241022 | | | |
| 20241212_epam-ai-run-claude-3-5-sonnet | | | |
| 20241213_devlo | | | |
| 20241221_codestory_midwit_claude-3-5-sonnet_swe-search | | | |
| 20250110_blackboxai_agent_v1.1 | | | |
| 20250110_learn_by_interact_claude3.5 | | | |
| 20250117_wandb_programmer_o1_crosscheck5 | | | |
| 20250118_codeshellagent_gemini_2.0_flash_experimental | | | |
| 20250203_openhands_4x_scaled | | | |
| 20250226_swerl_llama3_70b | | | |
| 20250228_epam-ai-run-claude-3-5-sonnet | | | |
| 20250306_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor | | | |
| 20250410_cortexa | | | |
| 20250415_openhands | | | |
| 20250511_sweagent_lm_32b | | | |
| 20250515_Refact_Agent | | | |
| 20250519_devlo | | | |
| 20250519_trae | | | |
| 20250520_openhands_devstral_small | | | |
| 20250522_sweagent_claude-4-sonnet-20250514 | | | |
| 20250524_openhands_claude_4_sonnet | | | |
| 20250527_amazon.nova-premier-v1.0 | | | |
| 20250603_Refact_Agent_claude-4-sonnet | | | |
| 20250612_trae | | | |
| 20250616_Skywork-SWE-32B | | | |
| 20250616_Skywork-SWE-32B+TTS_Bo8 | | | |
| 20250710_bloop | | | |
| 20250716_openhands_kimi_k2 | | | |
| 20250728_zai_glm4-5 | | | |
| 20250804_codesweep_sweagent_kimi_k2_instruct | | | |
| 20250804_epam-ai-run-claude-4-sonnet | | | |
| 20250807_openhands_gpt5 | | | |
| 20250901_entroPO_R2E_QwenCoder30BA3B | | | |
| 20250901_entroPO_R2E_QwenCoder30BA3B_tts | | | |
| 20250924_artemis_agent_v2 | | | |
| 20250928_trae_doubao_seed_code | | | |
| 20250929_Prometheus_v1.2_gpt5 | | | |
| 20250930_zai_glm4-6 | | | |
| 20251015_Prometheus_v1.2.1_gpt5 | | | |
| 20251103_SalesforceAIResearch_SAGE_OpenHands | | | |
| 20251103_sonar-foundation-agent_claude-sonnet-4-5 | | | |
| 20251110_frogboss-32b | | | |
| 20251110_frogmini-14b | | | |

## Verification Criteria

A filtered trajectory is "reasonable" if:
1. It keeps actual codebase interactions (file views, edits, command execution, test runs)
2. It filters out pure thinking/planning without actions
3. It filters out setup context and environment reminders
4. The remaining content shows behavioral choices the agent made

## Detailed Notes

(Add detailed notes for any agent that needs investigation)
