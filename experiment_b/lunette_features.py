"""Extract trajectory features using Lunette as LLM judge.

This module provides rich semantic features extracted from agent trajectories,
combining insights from:
- AgentDiagnose (EMNLP 2025): Agentic competency ratings
- SWE-bench Failures paper: Failure mode taxonomy
- SWE-bench Pro: Additional trajectory signals
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Feature names for the Lunette-extracted feature vector
LUNETTE_FEATURE_NAMES = [
    # Agentic competencies (1-4 scale, from AgentDiagnose)
    "backtracking_exploration",
    "task_decomposition",
    "observation_reading",
    "self_verification",
    # Failure mode indicators (0-1 binary, from SWE-bench Failures)
    "localization_failure",
    "strategy_defect",
    "implementation_defect",
    "incomplete_repair",
    "verification_failure",
    # Trajectory signals (0-1 binary)
    "agent_looping",
    "agent_gave_up_early",
    "agent_wrong_focus",
    "context_overflow",
]

# Grading prompt for Lunette
TRAJECTORY_GRADING_PROMPT = """You are analyzing a SWE-bench agent trajectory to extract signals about task difficulty.

The trajectory shows an agent attempting to solve a software engineering task. Your job is to rate the agent's behavior and identify failure patterns.

## AGENTIC COMPETENCIES (score 1-4)

Rate how well the agent exhibited each competency:

1. **backtracking_exploration**: Did the agent try alternative approaches when stuck?
   - 1 = Never explored alternatives, stuck in one approach
   - 2 = Rarely tried alternatives, mostly persisted with failing approach
   - 3 = Sometimes explored alternatives when clearly stuck
   - 4 = Systematically explored alternatives, good breadth-first search

2. **task_decomposition**: Did the agent break down the problem into subproblems?
   - 1 = Never decomposed, jumped straight to implementation
   - 2 = Minimal decomposition, mostly ad-hoc
   - 3 = Some structured decomposition into steps
   - 4 = Systematic decomposition with clear subgoals

3. **observation_reading**: Did the agent correctly understand command outputs and errors?
   - 1 = Ignored or misread most outputs
   - 2 = Often misinterpreted outputs
   - 3 = Usually understood outputs correctly
   - 4 = Accurately interpreted all outputs and errors

4. **self_verification**: Did the agent verify its progress and test its fixes?
   - 1 = Never verified, declared success without testing
   - 2 = Minimal verification, often skipped tests
   - 3 = Usually ran tests, but sometimes ignored failures
   - 4 = Thorough verification, ran tests and interpreted results correctly

## FAILURE MODE DETECTION (0 or 1)

Indicate whether each failure pattern occurred (0=no, 1=yes):

- **localization_failure**: Agent searched in wrong files/locations, failed to find the relevant code
- **strategy_defect**: Fix was superficial (only handles specific case, suppresses error without fixing root cause)
- **implementation_defect**: Algorithm logic, control flow, or boundary handling errors in the fix
- **incomplete_repair**: Multiple components needed changes but agent only fixed some
- **verification_failure**: Failed to create working test, misread test output, or abandoned verification

## TRAJECTORY SIGNALS (0 or 1)

Indicate whether each signal is present (0=no, 1=yes):

- **agent_looping**: Agent repeated similar actions multiple times without meaningful progress
- **agent_gave_up_early**: Agent stopped trying before exhausting reasonable approaches
- **agent_wrong_focus**: Agent fixated on irrelevant code or wrong part of the codebase
- **context_overflow**: Agent appeared to lose track of earlier context or findings

## OUTPUT FORMAT

Return a JSON object with exactly these 13 keys:
{
    "backtracking_exploration": <1-4>,
    "task_decomposition": <1-4>,
    "observation_reading": <1-4>,
    "self_verification": <1-4>,
    "localization_failure": <0 or 1>,
    "strategy_defect": <0 or 1>,
    "implementation_defect": <0 or 1>,
    "incomplete_repair": <0 or 1>,
    "verification_failure": <0 or 1>,
    "agent_looping": <0 or 1>,
    "agent_gave_up_early": <0 or 1>,
    "agent_wrong_focus": <0 or 1>,
    "context_overflow": <0 or 1>
}

Analyze the trajectory carefully and return ONLY the JSON object, no other text.
"""


@dataclass
class LunetteFeatures:
    """Features extracted from a trajectory using Lunette."""

    # Agentic competencies (1-4 scale)
    backtracking_exploration: float
    task_decomposition: float
    observation_reading: float
    self_verification: float

    # Failure mode indicators (0-1)
    localization_failure: float
    strategy_defect: float
    implementation_defect: float
    incomplete_repair: float
    verification_failure: float

    # Trajectory signals (0-1)
    agent_looping: float
    agent_gave_up_early: float
    agent_wrong_focus: float
    context_overflow: float

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector in standard order."""
        return np.array([
            self.backtracking_exploration,
            self.task_decomposition,
            self.observation_reading,
            self.self_verification,
            self.localization_failure,
            self.strategy_defect,
            self.implementation_defect,
            self.incomplete_repair,
            self.verification_failure,
            self.agent_looping,
            self.agent_gave_up_early,
            self.agent_wrong_focus,
            self.context_overflow,
        ])

    @classmethod
    def from_dict(cls, d: Dict) -> "LunetteFeatures":
        """Create from dictionary (e.g., parsed JSON)."""
        return cls(
            backtracking_exploration=float(d.get("backtracking_exploration", 2.5)),
            task_decomposition=float(d.get("task_decomposition", 2.5)),
            observation_reading=float(d.get("observation_reading", 2.5)),
            self_verification=float(d.get("self_verification", 2.5)),
            localization_failure=float(d.get("localization_failure", 0)),
            strategy_defect=float(d.get("strategy_defect", 0)),
            implementation_defect=float(d.get("implementation_defect", 0)),
            incomplete_repair=float(d.get("incomplete_repair", 0)),
            verification_failure=float(d.get("verification_failure", 0)),
            agent_looping=float(d.get("agent_looping", 0)),
            agent_gave_up_early=float(d.get("agent_gave_up_early", 0)),
            agent_wrong_focus=float(d.get("agent_wrong_focus", 0)),
            context_overflow=float(d.get("context_overflow", 0)),
        )

    @classmethod
    def default(cls) -> "LunetteFeatures":
        """Return default/neutral features."""
        return cls(
            backtracking_exploration=2.5,
            task_decomposition=2.5,
            observation_reading=2.5,
            self_verification=2.5,
            localization_failure=0.0,
            strategy_defect=0.0,
            implementation_defect=0.0,
            incomplete_repair=0.0,
            verification_failure=0.0,
            agent_looping=0.0,
            agent_gave_up_early=0.0,
            agent_wrong_focus=0.0,
            context_overflow=0.0,
        )


def load_lunette_features(
    task_id: str,
    agent: str,
    features_dir: Path,
) -> Optional[LunetteFeatures]:
    """Load pre-computed Lunette features for a task-agent pair.

    Args:
        task_id: Task instance ID
        agent: Agent name
        features_dir: Base directory for features

    Returns:
        LunetteFeatures or None if not found
    """
    feature_file = features_dir / agent / f"{task_id}.json"
    if not feature_file.exists():
        return None

    try:
        with open(feature_file) as f:
            data = json.load(f)
        return LunetteFeatures.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_lunette_features_for_task(
    task_id: str,
    agents: List[str],
    features_dir: Path,
) -> Dict[str, LunetteFeatures]:
    """Load Lunette features for a task across multiple agents.

    Args:
        task_id: Task instance ID
        agents: List of agent names
        features_dir: Base directory for features

    Returns:
        Dict mapping agent -> LunetteFeatures
    """
    result = {}
    for agent in agents:
        features = load_lunette_features(task_id, agent, features_dir)
        if features is not None:
            result[agent] = features
    return result


def aggregate_lunette_features(features: Dict[str, LunetteFeatures]) -> np.ndarray:
    """Aggregate Lunette features across multiple trajectories.

    Returns averaged feature vector across all agents.
    """
    if not features:
        return np.zeros(len(LUNETTE_FEATURE_NAMES))

    vectors = [f.to_vector() for f in features.values()]
    return np.mean(vectors, axis=0)


def load_and_aggregate_lunette_features(
    task_ids: List[str],
    agents: List[str],
    features_dir: Path,
) -> Dict[str, np.ndarray]:
    """Load and aggregate Lunette features for multiple tasks.

    Args:
        task_ids: List of task IDs
        agents: List of agent names whose features to use
        features_dir: Base directory for features

    Returns:
        Dict mapping task_id -> aggregated feature vector
    """
    result = {}
    for task_id in task_ids:
        features = load_lunette_features_for_task(task_id, agents, features_dir)
        if features:
            result[task_id] = aggregate_lunette_features(features)
    return result
