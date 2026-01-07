"""Selection and diversity filtering for features."""

import random
from typing import Dict, List, Tuple

import numpy as np

from .feature_store import Feature, FeatureEvaluation


class FeatureSelector:
    """Select top features with diversity filtering."""

    def __init__(
        self,
        top_k: int = 10,
        redundancy_threshold: float = 0.8,
        diversity_threshold: float = 0.85,
    ):
        """Initialize feature selector.

        Args:
            top_k: Number of top features to keep.
            redundancy_threshold: Max inter-feature score correlation.
            diversity_threshold: Max embedding cosine similarity.
        """
        self.top_k = top_k
        self.redundancy_threshold = redundancy_threshold
        self.diversity_threshold = diversity_threshold

    def compute_score_correlation(
        self,
        eval_a: FeatureEvaluation,
        eval_b: FeatureEvaluation,
    ) -> float:
        """Compute correlation between two features' task scores.

        Args:
            eval_a: First feature's evaluation.
            eval_b: Second feature's evaluation.

        Returns:
            Pearson correlation coefficient.
        """
        # Get common tasks
        common_tasks = set(eval_a.task_scores.keys()) & set(eval_b.task_scores.keys())
        if len(common_tasks) < 3:
            return 0.0

        scores_a = np.array([eval_a.task_scores[t] for t in common_tasks])
        scores_b = np.array([eval_b.task_scores[t] for t in common_tasks])

        if np.std(scores_a) == 0 or np.std(scores_b) == 0:
            return 0.0

        return float(np.corrcoef(scores_a, scores_b)[0, 1])

    def compute_text_similarity(
        self,
        feature_a: Feature,
        feature_b: Feature,
    ) -> float:
        """Compute text similarity between features (simple approach).

        Args:
            feature_a: First feature.
            feature_b: Second feature.

        Returns:
            Similarity score (0-1).
        """
        # Simple word overlap similarity
        words_a = set(feature_a.description.lower().split())
        words_b = set(feature_b.description.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0

    def binary_tournament_selection(
        self,
        features: List[Feature],
        evaluations: List[FeatureEvaluation],
        n_select: int,
    ) -> List[Tuple[Feature, FeatureEvaluation]]:
        """Select features using binary tournament (PromptBreeder-style).

        Args:
            features: List of candidate features.
            evaluations: Corresponding evaluations.
            n_select: Number of features to select.

        Returns:
            List of (feature, evaluation) tuples.
        """
        eval_map = {e.feature_id: e for e in evaluations}
        feature_eval_pairs = [(f, eval_map[f.id]) for f in features if f.id in eval_map]

        if len(feature_eval_pairs) <= n_select:
            return feature_eval_pairs

        selected = []
        remaining = list(feature_eval_pairs)

        while len(selected) < n_select and len(remaining) >= 2:
            # Randomly select two contestants
            idx1, idx2 = random.sample(range(len(remaining)), 2)
            pair1, pair2 = remaining[idx1], remaining[idx2]

            # Winner is the one with higher absolute correlation
            if pair1[1].abs_correlation >= pair2[1].abs_correlation:
                winner = pair1
                loser_idx = idx2
            else:
                winner = pair2
                loser_idx = idx1

            selected.append(winner)

            # Remove both from remaining to avoid re-selection
            remaining = [r for i, r in enumerate(remaining) if i not in [idx1, idx2]]

        # If we need more, take from remaining by rank
        if len(selected) < n_select:
            remaining.sort(key=lambda x: x[1].abs_correlation, reverse=True)
            selected.extend(remaining[:n_select - len(selected)])

        return selected

    def remove_redundant(
        self,
        features: List[Feature],
        evaluations: List[FeatureEvaluation],
    ) -> Tuple[List[Feature], List[FeatureEvaluation]]:
        """Remove redundant features based on score correlation.

        Args:
            features: List of features.
            evaluations: Corresponding evaluations.

        Returns:
            Filtered (features, evaluations) tuple.
        """
        eval_map = {e.feature_id: e for e in evaluations}

        # Sort by absolute correlation (best first)
        sorted_features = sorted(
            features,
            key=lambda f: eval_map.get(f.id, FeatureEvaluation(f.id, 0, 0, 0, 0, 0)).abs_correlation,
            reverse=True,
        )

        kept_features = []
        kept_evals = []

        for feature in sorted_features:
            if feature.id not in eval_map:
                continue

            eval_f = eval_map[feature.id]

            # Check redundancy with already kept features
            is_redundant = False
            for kept_f in kept_features:
                kept_eval = eval_map[kept_f.id]

                # Check score correlation
                score_corr = abs(self.compute_score_correlation(eval_f, kept_eval))
                if score_corr > self.redundancy_threshold:
                    is_redundant = True
                    break

                # Check text similarity
                text_sim = self.compute_text_similarity(feature, kept_f)
                if text_sim > self.diversity_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                kept_features.append(feature)
                kept_evals.append(eval_f)

        return kept_features, kept_evals

    def select(
        self,
        features: List[Feature],
        evaluations: List[FeatureEvaluation],
        use_tournament: bool = True,
    ) -> Tuple[List[Feature], List[FeatureEvaluation]]:
        """Select top-K features with diversity filtering.

        Args:
            features: List of candidate features.
            evaluations: Corresponding evaluations.
            use_tournament: Whether to use binary tournament selection.

        Returns:
            Tuple of (selected_features, selected_evaluations).
        """
        if use_tournament:
            # Binary tournament selection
            selected_pairs = self.binary_tournament_selection(
                features, evaluations, self.top_k * 2
            )
            pre_features = [p[0] for p in selected_pairs]
            pre_evals = [p[1] for p in selected_pairs]
        else:
            # Simple top-K by correlation
            eval_map = {e.feature_id: e for e in evaluations}
            sorted_features = sorted(
                features,
                key=lambda f: eval_map.get(f.id, FeatureEvaluation(f.id, 0, 0, 0, 0, 0)).abs_correlation,
                reverse=True,
            )
            pre_features = sorted_features[:self.top_k * 2]
            pre_evals = [eval_map[f.id] for f in pre_features if f.id in eval_map]

        # Remove redundant features
        filtered_features, filtered_evals = self.remove_redundant(
            pre_features, pre_evals
        )

        # Limit to top_k
        return filtered_features[:self.top_k], filtered_evals[:self.top_k]

    def get_diverse_pairs(
        self,
        features: List[Feature],
        evaluations: List[FeatureEvaluation],
        n_pairs: int = 5,
    ) -> List[Tuple[Feature, Feature]]:
        """Get diverse pairs of features for crossover.

        Args:
            features: List of features.
            evaluations: Corresponding evaluations.
            n_pairs: Number of pairs to return.

        Returns:
            List of (feature_a, feature_b) tuples.
        """
        if len(features) < 2:
            return []

        eval_map = {e.feature_id: e for e in evaluations}
        pairs = []

        for _ in range(n_pairs * 3):  # Try more to get diverse pairs
            if len(pairs) >= n_pairs:
                break

            f1, f2 = random.sample(features, 2)

            # Check diversity
            if f1.id in eval_map and f2.id in eval_map:
                score_corr = abs(self.compute_score_correlation(
                    eval_map[f1.id], eval_map[f2.id]
                ))
                if score_corr < self.redundancy_threshold:
                    pairs.append((f1, f2))

        return pairs
