"""Configuration for trajectory feature extraction.

Selected agents span IRT ability from -1.26 to +2.24, all with trajectories < 120K tokens.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AgentInfo:
    """Information about a selected agent."""
    name: str
    theta: float  # IRT ability
    avg_tokens: int  # Average trajectory tokens
    max_tokens: int  # Maximum trajectory tokens


# Selected 20 agents spanning the ability spectrum
# All have trajectories that fit in 200K context window
SELECTED_AGENTS: List[AgentInfo] = [
    AgentInfo("20250928_trae_doubao_seed_code", 2.24, 41000, 79000),
    AgentInfo("20250804_epam-ai-run-claude-4-sonnet", 2.19, 28000, 55000),
    AgentInfo("20251103_sonar-foundation-agent_claude-sonnet-4-5", 2.07, 33000, 54000),
    AgentInfo("20250915_JoyCode", 2.04, 20000, 52000),
    AgentInfo("20251103_SalesforceAIResearch_SAGE_OpenHands", 1.88, 23000, 53000),
    AgentInfo("20250807_openhands_gpt5", 1.71, 32000, 72000),
    AgentInfo("20250522_sweagent_claude-4-sonnet-20250514", 1.19, 47000, 87000),
    AgentInfo("20250415_openhands", 1.11, 35000, 55000),
    AgentInfo("20250117_wandb_programmer_o1_crosscheck5", 0.98, 10000, 53000),
    AgentInfo("20250228_epam-ai-run-claude-3-5-sonnet", 0.79, 17000, 55000),
    AgentInfo("20250901_entroPO_R2E_QwenCoder30BA3B_tts", 0.68, 54000, 72000),
    AgentInfo("20251110_frogboss-32b", 0.40, 54000, 91000),
    AgentInfo("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", 0.26, 13000, 55000),
    AgentInfo("20241125_enginelabs", 0.16, 10000, 54000),
    AgentInfo("20250520_openhands_devstral_small", -0.04, 32000, 55000),
    AgentInfo("20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022", -0.12, 6000, 10000),
    AgentInfo("20250527_amazon.nova-premier-v1.0", -0.47, 10000, 54000),
    AgentInfo("20240620_sweagent_claude3.5sonnet", -0.81, 33000, 71000),
    AgentInfo("20241016_epam-ai-run-gpt-4o", -1.21, 22000, 55000),
    AgentInfo("20240918_lingma-agent_lingma-swe-gpt-72b", -1.26, 8000, 29000),
]

AGENT_NAMES = [a.name for a in SELECTED_AGENTS]


@dataclass
class FeatureSpec:
    """Specification for a trajectory feature."""
    name: str
    description: str
    scale: str  # "0-5", "bool", "count"
    expected_direction: str  # "positive" (higher = harder), "negative", "unknown"


# Features to extract from trajectories
TRAJECTORY_FEATURES: List[FeatureSpec] = [
    FeatureSpec(
        "loop_detection",
        "Did the model repeat similar actions/mistakes?",
        "0-5",
        "positive"  # More loops = harder task
    ),
    FeatureSpec(
        "premature_cutoff",
        "Did the model terminate before completing the task?",
        "bool",
        "positive"  # Early cutoff = harder task
    ),
    FeatureSpec(
        "localization_quality",
        "Did the model correctly identify the problem location?",
        "0-5",
        "negative"  # Better localization = easier task
    ),
    FeatureSpec(
        "debugging_cycles",
        "Number of debug-fix cycles (attempts to fix after errors)",
        "count",
        "positive"  # More cycles = harder task
    ),
    FeatureSpec(
        "error_recovery",
        "Did the model successfully recover from errors?",
        "0-5",
        "negative"  # Better recovery = easier task
    ),
    FeatureSpec(
        "exploration_breadth",
        "How many files/approaches did the model explore?",
        "count",
        "positive"  # More exploration = harder task
    ),
    FeatureSpec(
        "focus_drift",
        "Did the model stay on task or get distracted?",
        "0-5",
        "positive"  # More drift = harder task
    ),
    FeatureSpec(
        "solution_completeness",
        "How complete was the attempted solution?",
        "0-5",
        "negative"  # More complete = easier task
    ),
]

FEATURE_NAMES = [f.name for f in TRAJECTORY_FEATURES]


# Model configuration
DEFAULT_MODEL_EXPLORATION = "claude-sonnet-4-5-20250929"
DEFAULT_MODEL_FINAL = "claude-opus-4-5-20251101"
