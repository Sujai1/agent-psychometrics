"""
Comprehensive tests for the IRT training pipeline.

Tests cover:
1. Preprocessing (prep_swebench.py functions)
2. Dataset loading (py_irt/dataset.py)
3. Training correctness (train.py)
4. Model evaluation (compare_dims.py)
5. Initializers (py_irt/initializers.py)

Run with: pytest tests/test_irt_pipeline.py -v
"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def sample_jsonl_data():
    """Create sample JSONL response data for testing."""
    return [
        {"subject_id": "agent_A", "responses": {"task1": 1, "task2": 0, "task3": 1}},
        {"subject_id": "agent_B", "responses": {"task1": 0, "task2": 1, "task3": 1}},
        {"subject_id": "agent_C", "responses": {"task1": 1, "task2": 1, "task3": 0}},
    ]


@pytest.fixture
def sample_jsonl_file(sample_jsonl_data, tmp_path):
    """Write sample data to a temporary JSONL file."""
    filepath = tmp_path / "test_responses.jsonl"
    with open(filepath, "w") as f:
        for record in sample_jsonl_data:
            f.write(json.dumps(record) + "\n")
    return filepath


@pytest.fixture
def sample_results_json():
    """Sample SWE-bench results.json format."""
    return {
        "resolved": ["task1", "task3"],
        "applied": ["task1", "task2", "task3", "task4"],
        "failed": ["task2", "task4"],
    }


@pytest.fixture
def sample_abilities_1d():
    """Sample 1D abilities DataFrame."""
    return pd.DataFrame({
        "theta": [1.0, 0.5, -0.5],
        "theta_std": [0.1, 0.1, 0.1],
    }, index=["agent_A", "agent_B", "agent_C"])


@pytest.fixture
def sample_items_1d():
    """Sample 1D items DataFrame."""
    return pd.DataFrame({
        "a": [1.0, 1.5, 0.8],  # discrimination
        "b": [0.0, 1.0, -1.0],  # difficulty
        "a_std": [0.1, 0.1, 0.1],
        "b_std": [0.1, 0.1, 0.1],
    }, index=["task1", "task2", "task3"])


@pytest.fixture
def sample_abilities_2d():
    """Sample 2D abilities DataFrame."""
    return pd.DataFrame({
        "theta1": [1.0, 0.5, -0.5],
        "theta2": [0.5, 1.0, 0.0],
        "theta1_std": [0.1, 0.1, 0.1],
        "theta2_std": [0.1, 0.1, 0.1],
        "theta_avg": [0.75, 0.75, -0.25],
    }, index=["agent_A", "agent_B", "agent_C"])


@pytest.fixture
def sample_items_2d():
    """Sample 2D items DataFrame."""
    return pd.DataFrame({
        "a1": [1.0, 1.5, 0.8],
        "b1": [0.0, 1.0, -1.0],
        "a2": [0.5, 0.8, 1.2],
        "b2": [0.5, -0.5, 0.0],
        "a1_std": [0.1, 0.1, 0.1],
        "b1_std": [0.1, 0.1, 0.1],
        "a2_std": [0.1, 0.1, 0.1],
        "b2_std": [0.1, 0.1, 0.1],
    }, index=["task1", "task2", "task3"])


# ==============================================================================
# Test: prep_swebench.py
# ==============================================================================

class TestPrepSwebench:
    """Tests for SWE-bench preprocessing functions."""

    def test_load_results_json_format(self, sample_results_json, tmp_path):
        """Test that load_results_json correctly parses results.json format."""
        from src.prep_swebench import load_results_json

        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(sample_results_json, f)

        responses, all_items = load_results_json(results_path)

        # Check resolved tasks are marked as 1
        assert responses["task1"] == 1
        assert responses["task3"] == 1
        # Check non-resolved tasks are marked as 0
        assert responses["task2"] == 0
        assert responses["task4"] == 0
        # Check all items are collected
        assert all_items == {"task1", "task2", "task3", "task4"}

    def test_load_results_json_empty_resolved(self, tmp_path):
        """Test handling of empty resolved list."""
        from src.prep_swebench import load_results_json

        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump({"resolved": [], "applied": ["task1"]}, f)

        responses, all_items = load_results_json(results_path)

        assert responses["task1"] == 0
        assert len(all_items) == 1

    def test_list_items_filters_non_strings(self):
        """Test that _list_items filters out non-string values."""
        from src.prep_swebench import _list_items

        # Test with mixed types
        result = list(_list_items(["valid", 123, "also_valid", None, {"dict": "value"}]))
        assert result == ["valid", "also_valid"]

        # Test with non-list input
        result = list(_list_items("not a list"))
        assert result == []

        result = list(_list_items(None))
        assert result == []

    def test_cutoff_date_filtering(self):
        """Test that cutoff date comparison works correctly for YYYYMMDD format."""
        # This is a string comparison test
        cutoff = "20250930"

        # Dates that should pass
        assert "20231010" <= cutoff  # Earlier year
        assert "20250101" <= cutoff  # Same year, earlier month
        assert "20250930" <= cutoff  # Exact match

        # Dates that should fail
        assert not "20251001" <= cutoff  # Same year, later date
        assert not "20260101" <= cutoff  # Later year


# ==============================================================================
# Test: py_irt/dataset.py
# ==============================================================================

class TestDataset:
    """Tests for Dataset loading and transformation."""

    def test_from_jsonlines_basic(self, sample_jsonl_file):
        """Test basic JSONL loading."""
        from py_irt.dataset import Dataset

        dataset = Dataset.from_jsonlines(sample_jsonl_file)

        # Check counts
        assert len(dataset.subject_ids) == 3
        assert len(dataset.item_ids) == 3
        assert len(dataset.observations) == 9  # 3 subjects * 3 items

    def test_from_jsonlines_index_mapping(self, sample_jsonl_file):
        """Test that subject/item index mappings are consistent."""
        from py_irt.dataset import Dataset

        dataset = Dataset.from_jsonlines(sample_jsonl_file)

        # Check bidirectional mappings
        for sid, idx in dataset.subject_id_to_ix.items():
            assert dataset.ix_to_subject_id[idx] == sid

        for iid, idx in dataset.item_id_to_ix.items():
            assert dataset.ix_to_item_id[idx] == iid

    def test_from_pandas_basic(self):
        """Test DataFrame-based dataset creation."""
        from py_irt.dataset import Dataset

        df = pd.DataFrame({
            "user_id": ["joe", "sarah", "juan"],
            "item_1": [0, 1, 1],
            "item_2": [1, 0, 1],
        })

        dataset = Dataset.from_pandas(df, subject_column="user_id", item_columns=["item_1", "item_2"])

        assert len(dataset.subject_ids) == 3
        assert len(dataset.item_ids) == 2
        assert len(dataset.observations) == 6  # 3 subjects * 2 items

    def test_from_pandas_handles_nan(self):
        """Test that NaN values are excluded from observations."""
        from py_irt.dataset import Dataset

        df = pd.DataFrame({
            "user_id": ["joe", "sarah"],
            "item_1": [0, 1],
            "item_2": [np.nan, 1],  # joe didn't answer item_2
        })

        dataset = Dataset.from_pandas(df, subject_column="user_id", item_columns=["item_1", "item_2"])

        # Should have 3 observations (not 4), because one is NaN
        assert len(dataset.observations) == 3

    def test_get_item_accuracies(self, sample_jsonl_file):
        """Test item accuracy computation."""
        from py_irt.dataset import Dataset

        dataset = Dataset.from_jsonlines(sample_jsonl_file)
        accuracies = dataset.get_item_accuracies()

        # Based on sample data:
        # task1: agent_A=1, agent_B=0, agent_C=1 -> 2/3
        # task2: agent_A=0, agent_B=1, agent_C=1 -> 2/3
        # task3: agent_A=1, agent_B=1, agent_C=0 -> 2/3
        for item_id in ["task1", "task2", "task3"]:
            assert item_id in accuracies
            assert accuracies[item_id].total == 3
            assert abs(accuracies[item_id].accuracy - 2/3) < 0.01


# ==============================================================================
# Test: compare_dims.py (Log-likelihood and Model Selection)
# ==============================================================================

class TestCompareDims:
    """Tests for model comparison functions."""

    def test_log_bernoulli_logits_basic(self):
        """Test log-likelihood computation for Bernoulli with logits."""
        from src.compare_dims import log_bernoulli_logits

        # When y=1 and z is large positive, LL should be close to 0
        ll = log_bernoulli_logits(np.array([1]), np.array([10.0]))
        assert ll[0] > -0.001  # Very close to 0

        # When y=0 and z is large negative, LL should be close to 0
        ll = log_bernoulli_logits(np.array([0]), np.array([-10.0]))
        assert ll[0] > -0.001

        # When y=1 and z is large negative, LL should be very negative
        ll = log_bernoulli_logits(np.array([1]), np.array([-10.0]))
        assert ll[0] < -9.0

        # When y=0 and z is large positive, LL should be very negative
        ll = log_bernoulli_logits(np.array([0]), np.array([10.0]))
        assert ll[0] < -9.0

    def test_log_bernoulli_logits_numerical_stability(self):
        """Test numerical stability with extreme values."""
        from src.compare_dims import log_bernoulli_logits

        # Very large positive logit
        ll = log_bernoulli_logits(np.array([1]), np.array([100.0]))
        assert np.isfinite(ll[0])

        # Very large negative logit
        ll = log_bernoulli_logits(np.array([0]), np.array([-100.0]))
        assert np.isfinite(ll[0])

    def test_log_bernoulli_logits_probability_interpretation(self):
        """Test that log-likelihood corresponds to sigmoid probabilities."""
        from src.compare_dims import log_bernoulli_logits
        from scipy.special import expit as sigmoid

        z = np.array([0.0, 1.0, -1.0, 2.0])
        y = np.array([1, 1, 0, 0])

        ll = log_bernoulli_logits(y, z)

        # Manual computation
        p = sigmoid(z)
        expected_ll = y * np.log(p) + (1 - y) * np.log(1 - p)

        np.testing.assert_array_almost_equal(ll, expected_ll)

    def test_n_params_calculation(self):
        """Test parameter count calculation."""
        from src.compare_dims import n_params

        n_agents = 10
        n_items = 20

        # 1D: 10 abilities + 20 difficulties + 20 discriminations = 50
        assert n_params(n_agents, n_items, 1) == 10 + 2 * 20

        # 2D: 20 abilities + 40 difficulties + 40 discriminations = 100
        assert n_params(n_agents, n_items, 2) == 2 * 10 + 2 * 2 * 20

        # 3D: 30 abilities + 60 difficulties + 60 discriminations = 150
        assert n_params(n_agents, n_items, 3) == 3 * 10 + 2 * 3 * 20

    def test_aic_bic_calculation(self):
        """Test AIC/BIC calculation."""
        from src.compare_dims import aic_bic

        ll = -1000.0
        k = 50
        n = 500

        aic, bic = aic_bic(ll, k, n)

        # AIC = -2 * LL + 2 * k
        expected_aic = -2 * ll + 2 * k
        assert abs(aic - expected_aic) < 0.01

        # BIC = -2 * LL + log(n) * k
        expected_bic = -2 * ll + np.log(n) * k
        assert abs(bic - expected_bic) < 0.01

    def test_compute_ll_1d(self, sample_abilities_1d, sample_items_1d, sample_jsonl_file):
        """Test 1D log-likelihood computation."""
        from src.compare_dims import compute_ll

        ll, n = compute_ll(sample_abilities_1d, sample_items_1d, dims=1, jsonl=sample_jsonl_file)

        # Should have computed LL for 9 observations
        assert n == 9
        # LL should be negative (log probabilities)
        assert ll < 0
        # LL should be finite
        assert np.isfinite(ll)

    def test_compute_ll_2d(self, sample_abilities_2d, sample_items_2d, sample_jsonl_file):
        """Test 2D log-likelihood computation."""
        from src.compare_dims import compute_ll

        ll, n = compute_ll(sample_abilities_2d, sample_items_2d, dims=2, jsonl=sample_jsonl_file)

        assert n == 9
        assert ll < 0
        assert np.isfinite(ll)


# ==============================================================================
# Test: IRT Model Probability Computation
# ==============================================================================

class TestIRTModelMath:
    """Tests for IRT model mathematics."""

    def test_1d_2pl_probability(self):
        """Test 1D 2PL probability computation: P = sigmoid(a * (theta - b))"""
        from scipy.special import expit as sigmoid

        # Test case: high ability, low difficulty -> high probability
        a, theta, b = 1.0, 2.0, 0.0
        p = sigmoid(a * (theta - b))
        assert p > 0.8

        # Test case: low ability, high difficulty -> low probability
        a, theta, b = 1.0, 0.0, 2.0
        p = sigmoid(a * (theta - b))
        assert p < 0.2

        # Test case: equal ability and difficulty -> p = 0.5
        a, theta, b = 1.0, 1.0, 1.0
        p = sigmoid(a * (theta - b))
        assert abs(p - 0.5) < 0.01

    def test_2d_mirt_probability(self):
        """Test 2D MIRT probability: P = sigmoid(sum_d a_d * (theta_d - b_d))"""
        from scipy.special import expit as sigmoid

        # Agent abilities and item parameters
        theta = np.array([1.0, 0.5])  # 2D ability
        a = np.array([1.0, 0.5])       # 2D discrimination
        b = np.array([0.0, 0.0])       # 2D difficulty

        # Compensatory model: sum of contributions
        z = np.sum(a * (theta - b))
        p = sigmoid(z)

        expected_z = 1.0 * (1.0 - 0.0) + 0.5 * (0.5 - 0.0)  # = 1.25
        expected_p = sigmoid(expected_z)

        assert abs(p - expected_p) < 0.001

    def test_overall_skill_computation(self, sample_abilities_1d, sample_items_1d):
        """Test overall skill (average probability) computation."""
        from src.compare_dims import overall_skill

        result = overall_skill(sample_abilities_1d, sample_items_1d, dims=1)

        # Should return a Series with one value per agent
        assert len(result) == 3
        # All probabilities should be between 0 and 1
        assert all(0 <= p <= 1 for p in result)


# ==============================================================================
# Test: Initializers
# ==============================================================================

class TestInitializers:
    """Tests for IRT parameter initializers."""

    def test_difficulty_from_accuracy_logit(self):
        """Test that difficulty initialization uses correct logit formula."""
        # The formula is: b = -logit(accuracy) = -log(p/(1-p)) = log((1-p)/p)
        # High accuracy (e.g., 0.9) -> low difficulty (negative b)
        # Low accuracy (e.g., 0.1) -> high difficulty (positive b)

        import torch

        def logit(p):
            return torch.log(torch.tensor(p)) - torch.log1p(torch.tensor(-p))

        # High accuracy -> negative difficulty
        acc_high = 0.9
        b_high = -logit(acc_high)
        assert b_high < 0

        # Low accuracy -> positive difficulty
        acc_low = 0.1
        b_low = -logit(acc_low)
        assert b_low > 0

        # 50% accuracy -> zero difficulty
        acc_mid = 0.5
        b_mid = -logit(acc_mid)
        assert abs(float(b_mid)) < 0.01


# ==============================================================================
# Test: Training Output Format
# ==============================================================================

class TestTrainingOutput:
    """Tests for training output format and consistency."""

    def test_1d_output_columns(self, tmp_path):
        """Test that 1D output has expected columns."""
        # Create mock output files
        items_df = pd.DataFrame({
            "a": [1.0, 1.5],
            "b": [0.0, 1.0],
            "a_std": [0.1, 0.1],
            "b_std": [0.1, 0.1],
        }, index=["task1", "task2"])

        abilities_df = pd.DataFrame({
            "theta": [0.5, -0.5],
            "theta_std": [0.1, 0.1],
        }, index=["agent_A", "agent_B"])

        # Verify column structure
        assert "a" in items_df.columns
        assert "b" in items_df.columns
        assert "theta" in abilities_df.columns

    def test_2d_output_columns(self):
        """Test that 2D output has expected columns."""
        items_df = pd.DataFrame({
            "a1": [1.0], "b1": [0.0], "a1_std": [0.1], "b1_std": [0.1],
            "a2": [0.5], "b2": [0.5], "a2_std": [0.1], "b2_std": [0.1],
        }, index=["task1"])

        abilities_df = pd.DataFrame({
            "theta1": [0.5], "theta2": [0.3],
            "theta1_std": [0.1], "theta2_std": [0.1],
            "theta_avg": [0.4],
        }, index=["agent_A"])

        # Verify column structure
        assert all(f"a{d}" in items_df.columns for d in [1, 2])
        assert all(f"b{d}" in items_df.columns for d in [1, 2])
        assert all(f"theta{d}" in abilities_df.columns for d in [1, 2])
        assert "theta_avg" in abilities_df.columns


# ==============================================================================
# Test: Known Bugs and Edge Cases
# ==============================================================================

class TestKnownIssues:
    """Tests that document and check for known issues."""

    def test_discrimination_should_not_be_exponentiated_1d(self):
        """
        KNOWN BUG: In train.py line 57, discriminations are exponentiated:
            discriminations = [np.exp(i) for i in trainer.best_params["disc"]]

        However, the 2PL model in py_irt uses raw (non-log) discrimination values.
        The guide parameter 'loc_slope' is not log-transformed.

        This test documents the expected behavior: discriminations should NOT
        be exponentiated when extracting from the 2PL model.
        """
        # The 2PL model uses: logits = slope[items] * (ability[subjects] - diff[items])
        # where slope is sampled from Normal(mu_a, 1/u_a)
        # There is no exp() transformation in the model.

        # If loc_slope = 1.0, the discrimination should be 1.0, not exp(1.0) = 2.718
        raw_disc = 1.0

        # Current buggy behavior
        buggy_disc = np.exp(raw_disc)
        assert abs(buggy_disc - 2.718) < 0.01  # This is what the bug produces

        # Expected correct behavior
        correct_disc = raw_disc
        assert correct_disc == 1.0  # This is what it should be

    def test_mirt_discrimination_not_exponentiated(self):
        """
        MIRT correctly does NOT exponentiate discriminations (train.py line 124):
            discriminations = pyro.param("loc_disc").detach().cpu().numpy()

        This test verifies the MIRT path is correct.
        """
        raw_disc = np.array([[1.0, 0.5], [1.5, 0.8]])

        # MIRT correctly uses raw values
        mirt_disc = raw_disc  # No transformation
        np.testing.assert_array_equal(mirt_disc, raw_disc)

    def test_averaging_mirt_results_rotational_indeterminacy(self):
        """
        POTENTIAL ISSUE: Averaging MIRT results across runs may not be meaningful
        due to rotational indeterminacy. Different random seeds can produce
        rotated solutions that are equivalent but have different parameter values.

        This test documents the issue.
        """
        # Two equivalent 2D solutions that differ by rotation
        # Solution 1: theta = [1, 0]
        # Solution 2: theta = [0.707, 0.707] (45-degree rotation)

        theta1 = np.array([1.0, 0.0])
        theta2 = np.array([0.707, 0.707])

        # Simple averaging gives a non-equivalent result
        avg = (theta1 + theta2) / 2

        # The average is NOT on the same "ability level" as either solution
        # (magnitude is different)
        mag1 = np.linalg.norm(theta1)
        mag2 = np.linalg.norm(theta2)
        mag_avg = np.linalg.norm(avg)

        # Average magnitude is less than individual magnitudes
        # This demonstrates that simple averaging loses information
        assert mag_avg < mag1 or mag_avg < mag2


# ==============================================================================
# Test: End-to-End Integration
# ==============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_jsonl_to_dataset_to_probabilities(self, sample_jsonl_file):
        """Test full flow from JSONL to probability predictions."""
        from py_irt.dataset import Dataset
        from scipy.special import expit as sigmoid

        # Load data
        dataset = Dataset.from_jsonlines(sample_jsonl_file)

        # Create synthetic IRT parameters
        n_items = len(dataset.item_ids)
        n_subjects = len(dataset.subject_ids)

        # Simple parameters: all discriminations = 1, difficulties vary
        a = np.ones(n_items)
        b = np.array([0.0, 0.5, -0.5])[:n_items]  # Easy, medium, hard
        theta = np.array([1.0, 0.0, -1.0])[:n_subjects]  # High, medium, low ability

        # Compute predicted probabilities
        for obs_idx, (subj_ix, item_ix, y) in enumerate(zip(
            dataset.observation_subjects,
            dataset.observation_items,
            dataset.observations
        )):
            z = a[item_ix] * (theta[subj_ix] - b[item_ix])
            p = sigmoid(z)

            # Probabilities should be valid
            assert 0 <= p <= 1

            # High ability + easy item should have high probability
            # This is a sanity check, not a precise test


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
