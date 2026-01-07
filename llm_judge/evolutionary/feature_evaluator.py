"""Feature evaluation via score extraction and correlation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .data_loader import Task
from .feature_store import Feature, FeatureEvaluation
from .llm_client import LLMClient


EXTRACTION_TEMPLATE = """Analyze this software engineering task and rate it on the following feature.

## Task Information

**Task ID:** {task_id}
**Repository:** {repo}

**Problem Statement:**
{problem_statement}

**Gold Patch (the correct solution):**
```diff
{patch}
```

## Feature to Evaluate

**{feature_name}**: {feature_description}

{extraction_prompt}
"""


class FeatureEvaluator:
    """Evaluate features by extracting scores and computing correlations."""

    def __init__(self, llm_client: LLMClient):
        """Initialize feature evaluator.

        Args:
            llm_client: LLM client for API calls.
        """
        self.llm_client = llm_client

    def extract_score(
        self,
        feature: Feature,
        task: Task,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Extract feature score for a single task.

        Args:
            feature: Feature to evaluate.
            task: Task to score.

        Returns:
            Tuple of (score, reasoning) or (None, error_message) on failure.
        """
        prompt = EXTRACTION_TEMPLATE.format(
            task_id=task.task_id,
            repo=task.repo,
            problem_statement=task.problem_statement[:6000],
            patch=task.patch[:3000],
            feature_name=feature.name,
            feature_description=feature.description,
            extraction_prompt=feature.extraction_prompt,
        )

        try:
            response = self.llm_client.call_json(
                prompt,
                max_tokens=256,
                temperature=0.3,  # Lower temperature for consistent scoring
            )

            score = response.get("score")
            reasoning = response.get("reasoning", "")

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

    def evaluate_feature(
        self,
        feature: Feature,
        tasks: List[Task],
        verbose: bool = False,
    ) -> FeatureEvaluation:
        """Evaluate a feature on a set of tasks.

        Args:
            feature: Feature to evaluate.
            tasks: List of tasks to score.
            verbose: Whether to print progress.

        Returns:
            FeatureEvaluation with correlation and task scores.
        """
        scores = {}
        difficulties = {}
        errors = {}

        for i, task in enumerate(tasks):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluating task {i + 1}/{len(tasks)}...")

            score, reasoning = self.extract_score(feature, task)

            if score is not None:
                scores[task.task_id] = score
                difficulties[task.task_id] = task.difficulty
            else:
                # Use neutral score (3) for failures
                scores[task.task_id] = 3.0
                difficulties[task.task_id] = task.difficulty
                if verbose:
                    print(f"  Warning: Failed to score {task.task_id}: {reasoning}")

        # Compute correlation
        score_array = np.array([scores[tid] for tid in scores])
        diff_array = np.array([difficulties[tid] for tid in scores])

        if len(score_array) > 2 and np.std(score_array) > 0:
            correlation, p_value = stats.pearsonr(score_array, diff_array)
        else:
            correlation = 0.0

        # Compute prediction errors (for refinement)
        # Normalize scores to difficulty scale for error computation
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

    def evaluate_features(
        self,
        features: List[Feature],
        tasks: List[Task],
        verbose: bool = False,
    ) -> List[FeatureEvaluation]:
        """Evaluate multiple features on the same tasks.

        Args:
            features: List of features to evaluate.
            tasks: List of tasks to score.
            verbose: Whether to print progress.

        Returns:
            List of FeatureEvaluation objects.
        """
        evaluations = []

        for i, feature in enumerate(features):
            if verbose:
                print(f"Evaluating feature {i + 1}/{len(features)}: {feature.name}")

            evaluation = self.evaluate_feature(feature, tasks, verbose=verbose)
            evaluations.append(evaluation)

            if verbose:
                print(f"  Correlation: {evaluation.correlation:+.3f}")

        return evaluations

    def get_failure_cases(
        self,
        feature: Feature,
        evaluation: FeatureEvaluation,
        tasks: List[Task],
        n_failures: int = 5,
    ) -> List[Dict]:
        """Get tasks where the feature prediction was worst.

        Args:
            feature: The feature being analyzed.
            evaluation: Evaluation results.
            tasks: Original task list.
            n_failures: Number of failure cases to return.

        Returns:
            List of dicts with task info and error details.
        """
        task_map = {t.task_id: t for t in tasks}

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

            task = task_map[task_id]
            score = evaluation.task_scores.get(task_id, 3.0)

            failures.append({
                "task_id": task_id,
                "repo": task.repo,
                "actual_difficulty": task.difficulty,
                "predicted_score": score,
                "error": error,
                "problem_summary": task.problem_statement[:500],
            })

        return failures
