"""LLM judge features V5: Features observable from FAILING trajectories.

V5 focuses on HOW the agent failed, not just effort/complexity ratios.
These features are designed to correlate with embedding prior residuals:
- High residual (harder than predicted) → inefficient navigation, failed reproduction
- Low residual (easier than predicted) → efficient navigation, successful reproduction

Key insight: M1 agents mostly FAIL on D_train tasks, so features must be
observable from failing trajectories.

Features use INTEGER scales (1-5) for better LLM reliability.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# V5 Features - focused on failure patterns
LLM_JUDGE_V5_FEATURE_NAMES = [
    "navigation_efficiency",      # 1-5: did agent find relevant code quickly?
    "reproduction_success",       # 1-5: could agent reproduce the issue?
    "location_vs_fix_alignment",  # 1-5: found right location but couldn't fix?
    "exploration_breadth",        # 1-5: was agent focused or scattered?
]


V5_PROMPT = """You are analyzing a FAILING SWE-bench agent trajectory to understand HOW the agent failed.

## CONTEXT

We want to predict task difficulty from agent failure patterns. Key insight:
- EASY tasks (that seem hard from text): Agent finds right area but makes simple mistake
- HARD tasks (that seem easy from text): Agent wanders extensively, can't find approach

You are comparing FAILING trajectories to understand what made the task hard or easy.

## TASK METADATA

**Task ID:** {instance_id}
**Repository:** {repo}
**Problem Statement ({problem_len} chars):**
{problem_statement}

**Gold Patch ({patch_len} chars):**
```diff
{patch}
```

{hints_section}

## AGENT TRAJECTORY (FAILED)

{trajectory_text}

**Outcome:** {resolved_status}

## FEATURES TO EXTRACT (all 1-5 integers)

### 1. NAVIGATION EFFICIENCY (1-5)
How efficiently did the agent find relevant code?

- **1**: Agent found right file/function within first few searches
  - Example: sphinx-doc__sphinx-10673 - Agent found `index.rst` and `toctree` code quickly
  - This suggests: task is EASY, navigation is straightforward
- **2**: Agent found right area after a few wrong turns
- **3**: Agent found right area after moderate exploration
- **4**: Agent spent significant time searching before finding relevant code
- **5**: Agent wandered extensively without finding relevant code
  - Example: Heavy search-explore loops, never converging on right area
  - This suggests: task is HARD, codebase is confusing or problem is vague

COUNT: Number of searches/file opens before finding the file where gold patch applies.

### 2. REPRODUCTION SUCCESS (1-5)
Could the agent reproduce the issue described in the problem?

- **1**: Agent successfully reproduced exact issue with test script
  - Example: Agent created reproduce.py, saw exact error/warning described
  - This suggests: task is EASY, clear reproduction path
- **2**: Agent reproduced issue but not perfectly (similar but not exact error)
- **3**: Agent partially reproduced or had unclear reproduction
- **4**: Agent attempted reproduction but failed to see expected behavior
- **5**: Agent could not reproduce issue or didn't attempt
  - Example: Visual outputs, environment-specific behavior, complex setup
  - This suggests: task is HARD, problem is hard to observe

LOOK FOR: Did agent create test scripts? Did they see the expected error messages?

### 3. LOCATION VS FIX ALIGNMENT (1-5)
Did agent find the right location but fail to implement the correct fix?

- **1**: Agent never found the right location (wandering failure)
  - Agent failed by not discovering WHERE to fix
- **2**: Agent found same file as gold patch but wrong section
- **3**: Agent found right file but wrong function/class
- **4**: Agent found right function but wrong lines
- **5**: Agent found EXACT right location but couldn't implement fix
  - Example: django-12774 - Edited same line (695) 4+ times without solving
  - This suggests: task is HARD due to hidden knowledge, not location

COMPARE: Agent's edits vs gold patch location. Same file? Same function? Same lines?

### 4. EXPLORATION BREADTH (1-5)
How many different files/approaches did agent try?

- **1**: Agent focused narrowly on 1-2 files
  - Could be successful localization OR gave up early
- **2**: Agent explored 3-5 files in focused way
- **3**: Agent explored moderate number of files (5-10)
- **4**: Agent explored many files (10+) with some patterns
- **5**: Agent scattered across many files without focus
  - Example: Heavy test-edit looping across multiple unrelated areas
  - This suggests: confusion about where problem lies

COUNT: Number of distinct files opened/searched/edited.

## Response Format

Respond with ONLY a JSON object (integers 1-5, not floats):
{{
    "navigation_efficiency": <1-5>,
    "reproduction_success": <1-5>,
    "location_vs_fix_alignment": <1-5>,
    "exploration_breadth": <1-5>,
    "reasoning": "<2-3 sentences: How did this agent's failure pattern reveal task difficulty?>"
}}

IMPORTANT: Use integers 1-5 only. A failing trajectory on an EASY task should show
low navigation_efficiency (1-2) and high reproduction_success (1-2).
"""


@dataclass
class LLMJudgeV5Features:
    """V5 features - failure pattern analysis."""

    navigation_efficiency: int      # 1-5 (1=efficient, 5=wandering)
    reproduction_success: int       # 1-5 (1=reproduced, 5=couldn't reproduce)
    location_vs_fix_alignment: int  # 1-5 (1=never found, 5=found but couldn't fix)
    exploration_breadth: int        # 1-5 (1=focused, 5=scattered)

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector (normalized to 0-1 range)."""
        # Normalize from 1-5 to 0-1
        return np.array([
            (self.navigation_efficiency - 1) / 4.0,
            (self.reproduction_success - 1) / 4.0,
            (self.location_vs_fix_alignment - 1) / 4.0,
            (self.exploration_breadth - 1) / 4.0,
        ])

    def to_raw_vector(self) -> np.ndarray:
        """Convert to feature vector (raw 1-5 values)."""
        return np.array([
            float(self.navigation_efficiency),
            float(self.reproduction_success),
            float(self.location_vs_fix_alignment),
            float(self.exploration_breadth),
        ])

    @classmethod
    def from_dict(cls, d: Dict) -> "LLMJudgeV5Features":
        """Create from dict (JSON response)."""
        return cls(
            navigation_efficiency=int(d.get("navigation_efficiency", 3)),
            reproduction_success=int(d.get("reproduction_success", 3)),
            location_vs_fix_alignment=int(d.get("location_vs_fix_alignment", 3)),
            exploration_breadth=int(d.get("exploration_breadth", 3)),
        )

    @classmethod
    def default(cls) -> "LLMJudgeV5Features":
        """Return default features (middle values)."""
        return cls(
            navigation_efficiency=3,
            reproduction_success=3,
            location_vs_fix_alignment=3,
            exploration_breadth=3,
        )


def load_llm_judge_v5_features(
    task_id: str,
    agent: str,
    features_dir: Path,
) -> Optional[LLMJudgeV5Features]:
    """Load V5 features for a task-agent pair."""
    feature_file = features_dir / agent / f"{task_id}.json"
    if not feature_file.exists():
        return None

    try:
        with open(feature_file) as f:
            data = json.load(f)
        return LLMJudgeV5Features.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_llm_judge_v5_features_for_task(
    task_id: str,
    agents: List[str],
    features_dir: Path,
) -> Dict[str, LLMJudgeV5Features]:
    """Load V5 features for a task across agents."""
    result = {}
    for agent in agents:
        features = load_llm_judge_v5_features(task_id, agent, features_dir)
        if features is not None:
            result[agent] = features
    return result


def aggregate_llm_judge_v5_features(features: Dict[str, LLMJudgeV5Features]) -> np.ndarray:
    """Aggregate V5 features across agents (mean of normalized features)."""
    if not features:
        return LLMJudgeV5Features.default().to_vector()

    vectors = [f.to_vector() for f in features.values()]
    return np.mean(vectors, axis=0)
