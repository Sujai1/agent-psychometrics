"""MLP predictor that directly predicts P(success) from (agent, task) pairs.

Unlike difficulty-based predictors that predict task difficulty (beta) and then
use the IRT formula P = sigmoid(theta - beta), this predictor directly learns
to map (agent_one_hot, task_features) -> P(success).

Architecture:
    Input: [agent_one_hot (n_agents) | task_features (feature_dim)]
    -> Linear(input_dim, hidden_size)
    -> ReLU
    -> Linear(hidden_size, 1)
    -> Sigmoid
    Output: P(success) in [0, 1]
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.dataset import BinomialExperimentData, ExperimentData
from experiment_ab_shared.feature_source import TaskFeatureSource


def build_input_vector(
    agent_idx: int,
    n_agents: int,
    task_features: np.ndarray,
) -> np.ndarray:
    """Build input vector for MLP: [agent_one_hot | task_features].

    Args:
        agent_idx: Index of the agent in the agent list.
        n_agents: Total number of agents.
        task_features: Scaled task feature vector.

    Returns:
        Concatenated input vector of shape (n_agents + feature_dim,).
    """
    agent_one_hot = np.zeros(n_agents, dtype=np.float32)
    agent_one_hot[agent_idx] = 1.0
    return np.concatenate([agent_one_hot, task_features])


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP with sigmoid output."""

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)


class MLPPredictor:
    """MLP that directly predicts P(success) from (agent, task) pairs.

    This predictor learns to predict success probability directly without
    going through IRT difficulty. It implements the CVPredictor protocol.

    Input: [agent_one_hot | task_features]
    Output: sigmoid(MLP(...)) = P(success)

    Training uses binary cross-entropy loss with Adam optimizer and
    weight decay (L2 regularization).
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 200,
        verbose: bool = False,
    ):
        """Initialize MLP predictor.

        Args:
            source: TaskFeatureSource providing features for tasks.
            hidden_size: Number of hidden units in the MLP.
            learning_rate: Learning rate for Adam optimizer.
            weight_decay: L2 regularization strength (Adam weight_decay).
            n_epochs: Number of training epochs.
            verbose: Print training progress.
        """
        self.source = source
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.verbose = verbose

        # Model state (set after fit())
        self._model: Optional[SimpleMLP] = None
        self._scaler: Optional[StandardScaler] = None
        self._agent_to_idx: Optional[Dict[str, int]] = None
        self._n_agents: int = 0
        self._is_fitted: bool = False

        # Training diagnostics
        self._training_losses: List[float] = []

        # Prediction cache
        self._task_feature_cache: Dict[str, np.ndarray] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the MLP on training data.

        Args:
            data: ExperimentData containing responses and agent information.
            train_task_ids: List of task IDs to train on.
        """
        # Clear caches
        self._task_feature_cache = {}
        self._training_losses = []

        # Build agent-to-index mapping
        all_agents = data.get_all_agents()
        self._agent_to_idx = {agent: i for i, agent in enumerate(all_agents)}
        self._n_agents = len(all_agents)

        # Get task features and fit scaler
        task_features = self.source.get_features(train_task_ids)
        self._scaler = StandardScaler()
        task_features_scaled = self._scaler.fit_transform(task_features)

        # Build task_id -> scaled features mapping for train tasks
        task_to_features = {
            task_id: task_features_scaled[i]
            for i, task_id in enumerate(train_task_ids)
        }

        # Build training data: (agent_one_hot + task_features, response)
        X_list: List[np.ndarray] = []
        y_list: List[float] = []

        is_binomial = isinstance(data, BinomialExperimentData)

        for task_id in train_task_ids:
            task_feat = task_to_features[task_id]

            for agent_id in all_agents:
                if agent_id not in data.responses:
                    continue
                if task_id not in data.responses[agent_id]:
                    continue

                # Build input vector using helper
                agent_idx = self._agent_to_idx[agent_id]
                x = build_input_vector(agent_idx, self._n_agents, task_feat)

                # Get response
                response = data.responses[agent_id][task_id]

                if is_binomial:
                    # Expand binomial to individual observations
                    k = response["successes"]
                    n = response["trials"]
                    # Add k success observations
                    for _ in range(k):
                        X_list.append(x)
                        y_list.append(1.0)
                    # Add (n-k) failure observations
                    for _ in range(n - k):
                        X_list.append(x)
                        y_list.append(0.0)
                else:
                    # Binary response
                    X_list.append(x)
                    y_list.append(float(response))

        if len(X_list) == 0:
            raise ValueError("No training examples found")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        if self.verbose:
            print(f"   Training MLP: {len(X)} samples, {X.shape[1]} features")
            print(f"   Agent one-hot dim: {self._n_agents}, task feature dim: {task_features_scaled.shape[1]}")

        # Create model
        input_dim = X.shape[1]
        self._model = SimpleMLP(input_dim, self.hidden_size)

        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        if self.verbose and device.type == "cuda":
            print(f"   Using GPU: {torch.cuda.get_device_name(0)}")

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCELoss()

        # Training loop (full-batch)
        self._model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()

            # Forward pass
            y_pred = self._model(X_tensor)
            loss = criterion(y_pred, y_tensor)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss per iteration
            loss_val = loss.item()
            self._training_losses.append(loss_val)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"      Epoch {epoch + 1}/{self.n_epochs}: Loss = {loss_val:.6f}")

        self._is_fitted = True

        if self.verbose:
            print(f"   Final loss: {self._training_losses[-1]:.6f}")

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict success probability for a specific (agent, task) pair.

        Args:
            data: ExperimentData (used for test_tasks list).
            agent_id: Agent identifier.
            task_id: Task identifier.

        Returns:
            Predicted probability of success (0 to 1).
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before calling predict_probability()")

        # Lazily cache test task features
        if task_id not in self._task_feature_cache:
            self._cache_test_task_features(data.test_tasks)

        if task_id not in self._task_feature_cache:
            raise ValueError(f"No features for task {task_id}")

        if agent_id not in self._agent_to_idx:
            raise ValueError(f"Unknown agent {agent_id}")

        # Build input using helper
        agent_idx = self._agent_to_idx[agent_id]
        task_feat = self._task_feature_cache[task_id]
        x = build_input_vector(agent_idx, self._n_agents, task_feat)

        # Get device
        device = next(self._model.parameters()).device

        # Forward pass
        self._model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            prob = self._model(x_tensor).item()

        return prob

    def _cache_test_task_features(self, test_tasks: List[str]) -> None:
        """Cache scaled features for test tasks."""
        features = self.source.get_features(test_tasks)
        features_scaled = self._scaler.transform(features)

        for i, task_id in enumerate(test_tasks):
            self._task_feature_cache[task_id] = features_scaled[i]

    def get_training_losses(self) -> List[float]:
        """Return list of loss values per iteration for convergence plots."""
        return self._training_losses.copy()

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return f"MLP ({self.source.name})"
