"""Baseline predictors and adapters for Experiment A evaluation.

All predictors implement the CVPredictor protocol from cross_validation.py,
allowing them to be used with the unified cross-validation framework.

This module provides:
1. AgentOnlyPredictor - Predicts based on agent's empirical success rate
2. ConstantPredictor - Predicts using mean training difficulty
3. OraclePredictor - Uses true IRT difficulties (upper bound)
4. DifficultyPredictorAdapter - Wraps any DifficultyPredictorBase to implement CVPredictor
5. FeatureIRTCVPredictor - Joint feature + IRT learning predictor
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.dataset import ExperimentData
from experiment_ab_shared.evaluator import compute_irt_probability
from experiment_ab_shared.feature_source import TaskFeatureSource
from experiment_ab_shared.predictor_base import DifficultyPredictorBase


class DifficultyPredictorAdapter:
    """Adapts any DifficultyPredictorBase to the CVPredictor protocol.

    Takes a difficulty-based predictor and wraps it to provide predicted
    probabilities using the IRT formula: P(success) = sigmoid(θ - β).
    """

    def __init__(
        self,
        predictor: DifficultyPredictorBase,
        use_full_abilities: bool = False,
    ):
        """Initialize the adapter.

        Args:
            predictor: Any predictor implementing DifficultyPredictorBase
            use_full_abilities: If True, use full IRT abilities (oracle only)
        """
        self._predictor = predictor
        self._use_full_abilities = use_full_abilities
        self._predicted_difficulties: Dict[str, float] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the underlying difficulty predictor."""
        ground_truth_b = data.train_items.loc[train_task_ids, "b"].values
        self._predictor.fit(train_task_ids, ground_truth_b)
        # Clear cached predictions for new fold
        self._predicted_difficulties = {}

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict success probability using IRT formula."""
        # Lazily predict difficulty for this task if not already done
        if task_id not in self._predicted_difficulties:
            # Predict for all test tasks at once for efficiency
            test_tasks = data.test_tasks
            predictions = self._predictor.predict(test_tasks)
            self._predicted_difficulties.update(predictions)

        if task_id not in self._predicted_difficulties:
            raise ValueError(f"No predicted difficulty for task {task_id}")

        beta = self._predicted_difficulties[task_id]
        abilities = data.full_abilities if self._use_full_abilities else data.train_abilities

        if agent_id not in abilities.index:
            raise ValueError(f"Agent {agent_id} not found in abilities")

        theta = abilities.loc[agent_id, "ability"]
        return compute_irt_probability(theta, beta)


class AgentOnlyPredictor:
    """Baseline that predicts based on agent success rates.

    For each agent, computes their success rate on training tasks.
    At prediction time, ignores task difficulty and returns the
    agent's empirical success rate.
    """

    def __init__(self) -> None:
        self._agent_success_rates: Dict[str, float] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Compute agent success rates on training tasks."""
        self._agent_success_rates = {}

        for agent_id in data.train_abilities.index:
            if agent_id not in data.responses:
                continue

            # Compute success rate using expand_for_auc to handle binary/binomial
            all_outcomes: List[int] = []
            for task_id in train_task_ids:
                if task_id in data.responses[agent_id]:
                    outcomes, _ = data.expand_for_auc(agent_id, task_id, 0.0)
                    all_outcomes.extend(outcomes)

            if all_outcomes:
                self._agent_success_rates[agent_id] = float(np.mean(all_outcomes))
            else:
                self._agent_success_rates[agent_id] = 0.5

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Return agent's empirical success rate (ignores task)."""
        if agent_id not in self._agent_success_rates:
            raise ValueError(f"No success rate computed for agent {agent_id}")
        return self._agent_success_rates[agent_id]


class ConstantPredictor:
    """Baseline that predicts the mean training difficulty for all tasks.

    Uses IRT formula: P(success) = sigmoid(θ - β_mean)
    """

    def __init__(self) -> None:
        self._mean_difficulty: float = 0.0

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Compute mean difficulty from training tasks."""
        difficulties = data.get_train_difficulties()
        self._mean_difficulty = float(np.mean(difficulties))

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Return P(success) using mean difficulty and agent's ability."""
        from experiment_ab_shared.evaluator import compute_irt_probability

        theta = data.train_abilities.loc[agent_id, "ability"]
        return compute_irt_probability(theta, self._mean_difficulty)


class OraclePredictor:
    """Oracle baseline that uses true IRT difficulties from full model.

    This represents the best possible performance given the IRT framework.
    Uses the full IRT model (trained on all data) for evaluation.
    """

    def __init__(self) -> None:
        pass

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """No training needed - uses oracle difficulties."""
        pass

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Return P(success) using true difficulty and oracle ability."""
        from experiment_ab_shared.evaluator import compute_irt_probability

        # Use full (oracle) abilities and difficulties
        theta = data.full_abilities.loc[agent_id, "ability"]
        beta = data.full_items.loc[task_id, "b"]
        return compute_irt_probability(theta, beta)


class FeatureIRTCVPredictor:
    """Feature-IRT predictor that jointly learns feature weights and abilities.

    This predictor maximizes IRT log-likelihood:
        L = sum_ij log P(y_ij | theta_j, b_i)

    where b_i = w^T f_i + bias (no residual term).

    After training on train tasks, it can predict difficulty for unseen test tasks
    using only the learned feature weights, enabling cross-validation.

    Supports both:
    - Binary data (Bernoulli likelihood): y_ij ~ Bernoulli(sigmoid(theta_j - b_i))
    - Binomial data (Binomial likelihood): k_ij ~ Binomial(n_ij, sigmoid(theta_j - b_i))
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        l2_weight: float = 0.01,
        l2_ability: float = 0.01,
        lr: float = 0.1,
        max_iter: int = 500,
        tol: float = 1e-5,
        verbose: bool = False,
    ):
        """Initialize Feature-IRT CV predictor.

        Args:
            source: TaskFeatureSource providing features for tasks.
            l2_weight: L2 regularization on feature weights.
            l2_ability: L2 regularization on mean(abilities)^2 for identifiability.
            lr: Learning rate for L-BFGS optimizer.
            max_iter: Maximum optimization iterations.
            tol: Convergence tolerance.
            verbose: Print training progress.
        """
        self.source = source
        self.l2_weight = l2_weight
        self.l2_ability = l2_ability
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Model state (set after fit())
        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[float] = None
        self._scaler: Optional[StandardScaler] = None
        self._learned_abilities: Optional[Dict[str, float]] = None
        self._is_fitted: bool = False
        self._predicted_difficulties: Dict[str, float] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit by maximizing IRT log-likelihood jointly with abilities.

        Automatically detects binary vs binomial data and uses appropriate likelihood.

        Args:
            data: ExperimentData containing responses and agent information
            train_task_ids: Task IDs to train on
        """
        try:
            import torch
            from torch.optim import LBFGS
            from torch.distributions import Bernoulli, Binomial
        except ImportError:
            raise ImportError("PyTorch is required for FeatureIRTCVPredictor")

        # Clear cached predictions for new fold
        self._predicted_difficulties = {}

        # Detect if data is binomial
        from experiment_ab_shared.dataset import BinomialExperimentData
        is_binomial = isinstance(data, BinomialExperimentData)

        # Extract responses from data.responses for train tasks
        agent_ids = list(data.train_abilities.index)

        if is_binomial:
            # Binomial: store (successes, trials) per agent-task
            responses_counts: Dict[str, Dict[str, tuple]] = {}
            for agent_id in agent_ids:
                if agent_id not in data.responses:
                    continue
                agent_responses = {}
                for task_id in train_task_ids:
                    if task_id in data.responses[agent_id]:
                        resp = data.responses[agent_id][task_id]
                        agent_responses[task_id] = (resp["successes"], resp["trials"])
                if agent_responses:
                    responses_counts[agent_id] = agent_responses
            agent_ids_with_responses = list(responses_counts.keys())
        else:
            # Binary: store 0/1 per agent-task
            responses_binary: Dict[str, Dict[str, int]] = {}
            for agent_id in agent_ids:
                if agent_id not in data.responses:
                    continue
                agent_responses = {}
                for task_id in train_task_ids:
                    if task_id in data.responses[agent_id]:
                        agent_responses[task_id] = data.responses[agent_id][task_id]
                if agent_responses:
                    responses_binary[agent_id] = agent_responses
            agent_ids_with_responses = list(responses_binary.keys())

        n_agents = len(agent_ids_with_responses)
        n_tasks = len(train_task_ids)

        if n_agents == 0:
            raise ValueError("No agents with responses on training tasks")
        if n_tasks == 0:
            raise ValueError("No training tasks")

        # Get features from source
        features = self.source.get_features(train_task_ids)
        feature_dim = features.shape[1]

        # Standardize features
        self._scaler = StandardScaler()
        features_scaled = self._scaler.fit_transform(features)

        # Build response tensors
        device = "cpu"

        if is_binomial:
            # Build counts and trials matrices
            counts_matrix = np.full((n_agents, n_tasks), np.nan)
            trials_matrix = np.full((n_agents, n_tasks), np.nan)
            for i, agent_id in enumerate(agent_ids_with_responses):
                for j, task_id in enumerate(train_task_ids):
                    if task_id in responses_counts.get(agent_id, {}):
                        k, n = responses_counts[agent_id][task_id]
                        counts_matrix[i, j] = k
                        trials_matrix[i, j] = n

            counts_tensor = torch.tensor(counts_matrix, dtype=torch.float32, device=device)
            trials_tensor = torch.tensor(trials_matrix, dtype=torch.float32, device=device)
            mask = ~torch.isnan(counts_tensor)

            # Compute empirical task pass rate for initialization
            eps = 1e-3
            task_difficulty_init = np.zeros(n_tasks)
            for j, task_id in enumerate(train_task_ids):
                total_successes = 0
                total_trials = 0
                for agent_id in agent_ids_with_responses:
                    if task_id in responses_counts.get(agent_id, {}):
                        k, n = responses_counts[agent_id][task_id]
                        total_successes += k
                        total_trials += n
                if total_trials > 0:
                    acc = max(eps, min(1 - eps, total_successes / total_trials))
                    task_difficulty_init[j] = -np.log(acc / (1 - acc))

            # Compute empirical agent ability
            theta_init = np.zeros(n_agents, dtype=np.float32)
            for i, agent_id in enumerate(agent_ids_with_responses):
                agent_resp = responses_counts.get(agent_id, {})
                total_successes = sum(k for k, n in agent_resp.values())
                total_trials = sum(n for k, n in agent_resp.values())
                if total_trials > 0:
                    acc = max(eps, min(1 - eps, total_successes / total_trials))
                    theta_init[i] = np.log(acc / (1 - acc))
        else:
            # Binary data
            response_matrix = np.full((n_agents, n_tasks), np.nan)
            for i, agent_id in enumerate(agent_ids_with_responses):
                for j, task_id in enumerate(train_task_ids):
                    if task_id in responses_binary.get(agent_id, {}):
                        response_matrix[i, j] = responses_binary[agent_id][task_id]

            response_tensor = torch.tensor(response_matrix, dtype=torch.float32, device=device)
            mask = ~torch.isnan(response_tensor)

            # Compute empirical task pass rate for initialization
            eps = 1e-3
            task_difficulty_init = np.zeros(n_tasks)
            for j, task_id in enumerate(train_task_ids):
                successes = sum(
                    1 for r in responses_binary.values() if r.get(task_id, None) == 1
                )
                total = sum(1 for r in responses_binary.values() if task_id in r)
                if total > 0:
                    acc = max(eps, min(1 - eps, successes / total))
                    task_difficulty_init[j] = -np.log(acc / (1 - acc))

            # Compute empirical agent ability
            theta_init = np.zeros(n_agents, dtype=np.float32)
            for i, agent_id in enumerate(agent_ids_with_responses):
                agent_resp = responses_binary.get(agent_id, {})
                correct = sum(agent_resp.values())
                total = len(agent_resp)
                if total > 0:
                    acc = max(eps, min(1 - eps, correct / total))
                    theta_init[i] = np.log(acc / (1 - acc))

        # Warm-start feature weights via Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(features_scaled, task_difficulty_init)
        w_init = ridge.coef_.astype(np.float32)
        bias_init = float(ridge.intercept_)

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32, device=device)

        # Initialize learnable parameters
        w = torch.tensor(w_init, requires_grad=True, device=device)
        b = torch.tensor([bias_init], requires_grad=True, device=device)
        theta = torch.tensor(theta_init, requires_grad=True, device=device)

        params = [w, b, theta]

        # L-BFGS optimizer
        optim = LBFGS(
            params,
            lr=self.lr,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        l2_w = self.l2_weight
        l2_a = self.l2_ability

        if is_binomial:
            def closure():
                optim.zero_grad()

                # Compute difficulties: b_i = features @ w + bias
                diff = torch.matmul(features_tensor, w) + b

                # P(success) = sigmoid(theta - diff)
                # logits = theta - diff
                logits = theta[:, None] - diff[None, :]

                # Binomial negative log-likelihood
                # k ~ Binomial(n, sigmoid(logits))
                nll = -Binomial(total_count=trials_tensor[mask], logits=logits[mask]).log_prob(
                    counts_tensor[mask]
                ).mean()

                # Regularization
                weight_reg = l2_w * torch.sum(w**2)
                ability_reg = l2_a * (theta.mean() ** 2)

                loss = nll + weight_reg + ability_reg
                loss.backward()

                return loss
        else:
            def closure():
                optim.zero_grad()

                # Compute difficulties: b_i = features @ w + bias
                diff = torch.matmul(features_tensor, w) + b

                # P(success) = sigmoid(theta - diff)
                probs = torch.sigmoid(theta[:, None] - diff[None, :])

                # Bernoulli negative log-likelihood
                nll = -Bernoulli(probs=probs[mask]).log_prob(response_tensor[mask]).mean()

                # Regularization
                weight_reg = l2_w * torch.sum(w**2)
                ability_reg = l2_a * (theta.mean() ** 2)

                loss = nll + weight_reg + ability_reg
                loss.backward()

                return loss

        # Training loop
        if self.verbose:
            data_type = "binomial" if is_binomial else "binary"
            print(f"   Feature-IRT Training ({data_type}): {n_tasks} tasks, {n_agents} agents")
            print(f"   Feature dim: {feature_dim}")
            print(f"   Valid response pairs: {mask.sum().item()}")

        for iteration in range(self.max_iter):
            if iteration > 0:
                previous_loss = loss.clone().detach()

            loss = optim.step(closure)

            if iteration > 0:
                d_loss = abs((previous_loss - loss).item())

                if self.verbose and (iteration % 50 == 0 or iteration < 5):
                    print(
                        f"     Iter {iteration}: loss={loss.item():.6f}, d_loss={d_loss:.2e}"
                    )

                # Check convergence
                if d_loss < self.tol:
                    if self.verbose:
                        print(f"   Converged at iteration {iteration}")
                    break

        # Store learned parameters
        self._weights = w.detach().cpu().numpy()
        self._bias = b.detach().cpu().item()

        # Store learned abilities as dict
        abilities_np = theta.detach().cpu().numpy()
        self._learned_abilities = {
            agent_id: float(abilities_np[i])
            for i, agent_id in enumerate(agent_ids_with_responses)
        }

        self._is_fitted = True

        if self.verbose:
            print(f"   Final loss: {loss.item():.6f}")

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict success probability using IRT formula.

        Uses learned feature weights to predict difficulty, and jointly-learned
        abilities for theta.

        Args:
            data: ExperimentData (used for test_tasks list)
            agent_id: Agent identifier
            task_id: Task identifier

        Returns:
            Predicted probability of success
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before calling predict_probability()")

        # Lazily compute difficulties for test tasks
        if task_id not in self._predicted_difficulties:
            test_tasks = data.test_tasks
            features = self.source.get_features(test_tasks)
            features_scaled = self._scaler.transform(features)
            preds = features_scaled @ self._weights + self._bias
            self._predicted_difficulties.update({
                t: float(p) for t, p in zip(test_tasks, preds)
            })

        if task_id not in self._predicted_difficulties:
            raise ValueError(f"No predicted difficulty for task {task_id}")

        beta = self._predicted_difficulties[task_id]

        # Use jointly-learned abilities
        if agent_id not in self._learned_abilities:
            raise ValueError(
                f"Agent {agent_id} not found in learned abilities. "
                f"This agent may not have responses on training tasks."
            )

        theta = self._learned_abilities[agent_id]
        return compute_irt_probability(theta, beta)
