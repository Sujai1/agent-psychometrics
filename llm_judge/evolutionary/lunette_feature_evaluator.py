"""Feature evaluation via Lunette grading (trajectory-based scoring).

This evaluator uses Lunette's GradingPlan API to extract feature scores from
agent trajectories. Unlike the standard FeatureEvaluator which only sees the
problem statement and patch, this evaluator has access to the full trajectory
and can run commands in the sandbox environment.
"""

from typing import Dict, List, Optional, Tuple
import asyncio
import json

import numpy as np
from scipy import stats

from lunette import LunetteClient
from lunette.analysis import GradingPlan

from .feature_store import Feature, FeatureEvaluation


LUNETTE_EXTRACTION_TEMPLATE = """Analyze this SWE-bench task trajectory to rate it on the following feature.

You have access to the agent's full trajectory and can examine the codebase in the sandbox.

## Feature to Evaluate

**{feature_name}**: {feature_description}

{extraction_prompt}

## Instructions

1. Review the task description and gold patch
2. Examine the agent's trajectory to understand what they attempted
3. Use the sandbox to inspect code, run tests, or verify hypotheses
4. Rate the task on the feature using a 1-5 scale

Respond with ONLY a JSON object:
{{
    "score": <1-5>,
    "reasoning": "<2-3 sentence explanation>"
}}
"""


class LunetteFeatureEvaluator:
    """Evaluate features using Lunette's trajectory grading."""

    def __init__(self, verbose: bool = False):
        """Initialize Lunette feature evaluator.

        Args:
            verbose: Whether to print progress.
        """
        self.verbose = verbose
        self._client: Optional[LunetteClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._client = await LunetteClient().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def extract_score(
        self,
        feature: Feature,
        run_id: str,
        trajectory_id: str,
        task_id: str,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Extract feature score for a single trajectory.

        Args:
            feature: Feature to evaluate.
            run_id: Lunette run ID containing the trajectory.
            trajectory_id: Specific trajectory to grade.
            task_id: Task ID (for logging).

        Returns:
            Tuple of (score, reasoning) or (None, error_message) on failure.
        """
        if not self._client:
            return None, "LunetteClient not initialized (use async with)"

        prompt = LUNETTE_EXTRACTION_TEMPLATE.format(
            feature_name=feature.name,
            feature_description=feature.description,
            extraction_prompt=feature.extraction_prompt,
        )

        try:
            results = await self._client.investigate(
                run_id=run_id,
                plan=GradingPlan(
                    name=f"feature-{feature.id}",
                    prompt=prompt,
                ),
                limit=1,
                # Filter to specific trajectory if multiple in run
                trajectory_filters={"trajectory_id": trajectory_id} if trajectory_id else None,
            )

            if not results.results:
                return None, "No grading results returned"

            result_data = results.results[0].data

            score = result_data.get("score")
            reasoning = result_data.get("reasoning", "")

            # Validate score
            if score is None:
                return None, "No score in response"

            score = float(score)
            if not 1 <= score <= 5:
                # Clamp to valid range
                score = max(1, min(5, score))

            return score, reasoning

        except Exception as e:
            return None, str(e)

    async def evaluate_feature(
        self,
        feature: Feature,
        task_trajectories: List[Dict],  # [{task_id, run_id, trajectory_id, difficulty}, ...]
    ) -> FeatureEvaluation:
        """Evaluate a feature on a set of trajectories.

        Args:
            feature: Feature to evaluate.
            task_trajectories: List of dicts with keys:
                - task_id: SWE-bench instance ID
                - run_id: Lunette run ID
                - trajectory_id: Lunette trajectory ID (optional)
                - difficulty: IRT b parameter

        Returns:
            FeatureEvaluation with correlation and task scores.
        """
        scores = {}
        difficulties = {}
        errors = {}

        for i, traj_info in enumerate(task_trajectories):
            task_id = traj_info["task_id"]
            run_id = traj_info["run_id"]
            trajectory_id = traj_info.get("trajectory_id")
            difficulty = traj_info["difficulty"]

            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Evaluating trajectory {i + 1}/{len(task_trajectories)}...")

            score, reasoning = await self.extract_score(
                feature=feature,
                run_id=run_id,
                trajectory_id=trajectory_id,
                task_id=task_id,
            )

            if score is not None:
                scores[task_id] = score
                difficulties[task_id] = difficulty
            else:
                # Use neutral score (3) for failures
                scores[task_id] = 3.0
                difficulties[task_id] = difficulty
                if self.verbose:
                    print(f"  Warning: Failed to score {task_id}: {reasoning}")

        # Compute correlation
        score_array = np.array([scores[tid] for tid in scores])
        diff_array = np.array([difficulties[tid] for tid in scores])

        if len(score_array) > 2 and np.std(score_array) > 0:
            correlation, p_value = stats.pearsonr(score_array, diff_array)
        else:
            correlation = 0.0

        # Compute prediction errors (for refinement)
        if np.std(score_array) > 0:
            norm_scores = (score_array - np.mean(score_array)) / np.std(score_array)
            norm_scores = norm_scores * np.std(diff_array) + np.mean(diff_array)
        else:
            norm_scores = np.full_like(score_array, np.mean(diff_array))

        for tid, norm_score, actual_diff in zip(scores.keys(), norm_scores, diff_array):
            errors[tid] = abs(norm_score - actual_diff)

        return FeatureEvaluation(
            feature_id=feature.id,
            correlation=float(correlation),
            abs_correlation=abs(float(correlation)),
            mean_score=float(np.mean(score_array)),
            std_score=float(np.std(score_array)),
            n_tasks=len(scores),
            task_scores=scores,
            task_errors=errors,
        )

    async def evaluate_features(
        self,
        features: List[Feature],
        task_trajectories: List[Dict],
    ) -> List[FeatureEvaluation]:
        """Evaluate multiple features on the same trajectories.

        Args:
            features: List of features to evaluate.
            task_trajectories: List of trajectory info dicts.

        Returns:
            List of FeatureEvaluation objects.
        """
        evaluations = []

        for i, feature in enumerate(features):
            if self.verbose:
                print(f"\nEvaluating feature {i + 1}/{len(features)}: {feature.name}")

            evaluation = await self.evaluate_feature(feature, task_trajectories)
            evaluations.append(evaluation)

            if self.verbose:
                print(f"  Correlation: {evaluation.correlation:+.3f}")

        return evaluations

    def get_failure_cases(
        self,
        feature: Feature,
        evaluation: FeatureEvaluation,
        task_trajectories: List[Dict],
        n_failures: int = 5,
    ) -> List[Dict]:
        """Get trajectories where the feature prediction was worst.

        Args:
            feature: The feature being analyzed.
            evaluation: Evaluation results.
            task_trajectories: Original trajectory info list.
            n_failures: Number of failure cases to return.

        Returns:
            List of dicts with trajectory info and error details.
        """
        task_map = {t["task_id"]: t for t in task_trajectories}

        # Sort by prediction error
        sorted_errors = sorted(
            evaluation.task_errors.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        failures = []
        for task_id, error in sorted_errors[:n_failures]:
            if task_id not in task_map:
                continue

            traj_info = task_map[task_id]
            score = evaluation.task_scores.get(task_id, 3.0)

            failures.append({
                "task_id": task_id,
                "run_id": traj_info["run_id"],
                "trajectory_id": traj_info.get("trajectory_id"),
                "actual_difficulty": traj_info["difficulty"],
                "predicted_score": score,
                "error": error,
            })

        return failures
