"""Initial feature generation from difficulty extremes."""

import random
from typing import List

from .config import EvolutionConfig
from .data_loader import DataLoader, Task
from .feature_store import Feature
from .llm_client import LLMClient


GENERATION_PROMPT = """You are an expert at analyzing software engineering tasks and predicting their difficulty.

I will show you examples of EASY tasks (low difficulty) and HARD tasks (high difficulty) from the SWE-bench benchmark. Your goal is to hypothesize {n_features} features that might distinguish easy tasks from hard tasks.

## EASY TASKS (Low IRT Difficulty)

{easy_tasks}

## HARD TASKS (High IRT Difficulty)

{hard_tasks}

## Your Task

Based on these examples, hypothesize {n_features} features that could predict task difficulty. Each feature should:
1. Be measurable from the problem statement and/or the gold patch
2. Produce a score from 1 to 5
3. Have a clear hypothesis about whether higher scores correlate with higher or lower difficulty

{thinking_style}

Respond with a JSON array of exactly {n_features} features. Each feature should have:
- "name": Short identifier (snake_case)
- "description": What the feature measures (1-2 sentences)
- "scale_low": What a score of 1 means
- "scale_high": What a score of 5 means
- "hypothesis": Your hypothesis about correlation direction (e.g., "Higher values correlate with higher difficulty because...")
- "extraction_prompt": The exact prompt to use when extracting this feature from a task

Example format:
```json
[
  {{
    "name": "code_complexity",
    "description": "How complex is the code change required to fix this issue?",
    "scale_low": "Simple one-liner or parameter change",
    "scale_high": "Complex multi-file refactoring with intricate logic",
    "hypothesis": "Higher complexity correlates with higher difficulty because complex changes require more understanding and are harder to get right.",
    "extraction_prompt": "Rate the complexity of the code change required to solve this issue:\\n1 = Simple one-liner or parameter change\\n5 = Complex multi-file refactoring with intricate logic\\n\\nRespond with JSON: {{\"score\": <1-5>, \"reasoning\": \"<brief explanation>\"}}"
  }}
]
```

Return ONLY the JSON array with {n_features} features.
"""


def format_task(task: Task, include_patch: bool = True) -> str:
    """Format a task for display in prompts."""
    result = f"""### {task.task_id} (difficulty: {task.difficulty:.2f})
**Repository:** {task.repo}

**Problem Statement:**
{task.problem_statement[:2000]}{"..." if len(task.problem_statement) > 2000 else ""}
"""

    if include_patch:
        result += f"""
**Gold Patch:**
```diff
{task.patch[:1500]}{"..." if len(task.patch) > 1500 else ""}
```
"""

    return result


class FeatureGenerator:
    """Generate initial features from difficulty extremes."""

    def __init__(
        self,
        config: EvolutionConfig,
        data_loader: DataLoader,
        llm_client: LLMClient,
    ):
        """Initialize feature generator.

        Args:
            config: Evolution configuration.
            data_loader: Data loader for tasks.
            llm_client: LLM client for API calls.
        """
        self.config = config
        self.data_loader = data_loader
        self.llm_client = llm_client

    def generate_initial_features(
        self,
        n_features: int,
        generation: int = 0,
    ) -> List[Feature]:
        """Generate initial features by analyzing difficulty extremes.

        Args:
            n_features: Number of features to generate.
            generation: Generation number for feature IDs.

        Returns:
            List of Feature objects.
        """
        # Get difficulty extremes
        easy_tasks, hard_tasks = self.data_loader.get_difficulty_extremes(
            percentile=self.config.difficulty_percentile,
            n_per_group=5,  # Show 5 examples of each
        )

        # Format tasks for prompt
        easy_text = "\n\n".join(format_task(t) for t in easy_tasks)
        hard_text = "\n\n".join(format_task(t) for t in hard_tasks)

        # Select a random thinking style
        thinking_style = random.choice(self.config.thinking_styles)

        # Build prompt
        prompt = GENERATION_PROMPT.format(
            n_features=n_features,
            easy_tasks=easy_text,
            hard_tasks=hard_text,
            thinking_style=f"**Thinking approach:** {thinking_style}",
        )

        # Call LLM
        response = self.llm_client.call_json(prompt, max_tokens=4096, temperature=0.8)

        # Parse features
        features = []
        for i, feat_data in enumerate(response):
            feature = Feature(
                id=f"gen{generation}_feat{i}",
                name=feat_data["name"],
                description=feat_data["description"],
                extraction_prompt=feat_data["extraction_prompt"],
                scale_low=feat_data["scale_low"],
                scale_high=feat_data["scale_high"],
                hypothesis=feat_data["hypothesis"],
                generation=generation,
                mutation_type="initial",
            )
            features.append(feature)

        return features

    def generate_with_context(
        self,
        n_features: int,
        existing_features: List[Feature],
        generation: int,
    ) -> List[Feature]:
        """Generate novel features given existing ones (zero-order mutation).

        Args:
            n_features: Number of new features to generate.
            existing_features: List of existing Feature objects.
            generation: Generation number for feature IDs.

        Returns:
            List of new Feature objects.
        """
        # Get difficulty extremes
        easy_tasks, hard_tasks = self.data_loader.get_difficulty_extremes(
            percentile=self.config.difficulty_percentile,
            n_per_group=3,
        )

        # Format existing features
        existing_summary = "\n".join(
            f"- {f.name}: {f.description} (hypothesis: {f.hypothesis[:100]}...)"
            for f in existing_features[:10]
        )

        # Format tasks
        easy_text = "\n\n".join(format_task(t, include_patch=False) for t in easy_tasks)
        hard_text = "\n\n".join(format_task(t, include_patch=False) for t in hard_tasks)

        thinking_style = random.choice(self.config.thinking_styles)

        prompt = f"""You are generating NEW features for predicting SWE-bench task difficulty.

## Existing Features (do NOT duplicate these)

{existing_summary}

## EASY TASKS (Low Difficulty)

{easy_text}

## HARD TASKS (High Difficulty)

{hard_text}

## Your Task

Generate {n_features} NEW features that are DIFFERENT from the existing ones. Focus on aspects not yet covered. Each feature should produce a 1-5 score.

{thinking_style}

Respond with a JSON array of {n_features} features with: name, description, scale_low, scale_high, hypothesis, extraction_prompt.
"""

        response = self.llm_client.call_json(prompt, max_tokens=4096, temperature=0.9)

        features = []
        for i, feat_data in enumerate(response):
            feature = Feature(
                id=f"gen{generation}_novel{i}",
                name=feat_data["name"],
                description=feat_data["description"],
                extraction_prompt=feat_data["extraction_prompt"],
                scale_low=feat_data["scale_low"],
                scale_high=feat_data["scale_high"],
                hypothesis=feat_data["hypothesis"],
                generation=generation,
                mutation_type="zero_order",
            )
            features.append(feature)

        return features
