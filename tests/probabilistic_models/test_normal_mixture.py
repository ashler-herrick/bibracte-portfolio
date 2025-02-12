import pytest
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import TensorDataset

from bet_edge.probabilistic_models.models.normal_mixture import DeepNormalMixture


@pytest.fixture
def synthetic_data():
    """Generates synthetic data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    return X, y


@pytest.fixture
def model():
    """Initializes the DeepNormalMixture model."""
    return DeepNormalMixture(
        n_inputs=10,
        n_hidden=20,
        n_dist=3,
        p_dropout=0.1,
        optimizer_class=Adam,
        learning_rate=1e-3,
        batch_size=16,
        use_cuda=False,
    )


def test_model_initialization(model):
    """Test if the model initializes correctly."""
    assert model.network is not None, "Network not initialized."
    assert model.mix_linear.out_features == model.n_dist, "Incorrect number of mixture components in mix_linear."
    assert model.mean_linear.out_features == model.n_dist, "Incorrect number of mixture components in mean_linear."
    assert model.scale_linear.out_features == model.n_dist, "Incorrect number of mixture components in scale_linear."


def test_forward_pass(model):
    """Test the forward pass of the model."""
    x = torch.randn(5, 10)
    dist = model(x)
    assert isinstance(dist, torch.distributions.MixtureSameFamily), "Output is not MixtureSameFamily."
    assert dist.mixture_distribution.logits.shape == (5, model.n_dist), "Incorrect shape for mixture logits."
    assert dist.component_distribution.mean.shape == (5, model.n_dist), "Incorrect shape for component means."
    assert dist.component_distribution.stddev.shape == (5, model.n_dist), "Incorrect shape for component stddev."


def test_training_step(model, synthetic_data):
    """Test a single training step."""
    X, y = synthetic_data

    # Create TensorDatasets
    train_dataset = TensorDataset(
        torch.tensor(X[:80], dtype=torch.float32),
        torch.tensor(y[:80], dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X[80:], dtype=torch.float32),
        torch.tensor(y[80:], dtype=torch.float32),
    )

    # Train the model
    model.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=1,
        early_stopping=False,
    )
    # After training, check if losses are recorded
    assert len(model.train_loss_arr) == 1, "Training loss not recorded."
    assert len(model.val_loss_arr) == 1, "Validation loss not recorded."


def test_prediction(model, synthetic_data):
    """Test the prediction functionality."""
    X, y = synthetic_data

    # Create TensorDatasets
    train_dataset = TensorDataset(
        torch.tensor(X[:80], dtype=torch.float32),
        torch.tensor(y[:80], dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X[80:], dtype=torch.float32),
        torch.tensor(y[80:], dtype=torch.float32),
    )

    # Train the model
    model.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=10,
        early_stopping=False,
    )
    preds = model.predict(val_dataset)
    assert preds.shape == y[80:].shape, "Predicted shape does not match target shape."
    assert isinstance(preds, np.ndarray), "Predictions are not a NumPy array."


def test_pred_dist(model, synthetic_data):
    """Test the distribution prediction."""
    X, y = synthetic_data

    # Create TensorDatasets
    train_dataset = TensorDataset(
        torch.tensor(X[:80], dtype=torch.float32),
        torch.tensor(y[:80], dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X[80:], dtype=torch.float32),
        torch.tensor(y[80:], dtype=torch.float32),
    )

    # Train the model
    model.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=10,
        early_stopping=False,
    )
    dist = model.pred_dist(val_dataset)
    print(dist)
    assert isinstance(dist, torch.distributions.MixtureSameFamily), "Distribution is not MixtureSameFamily."
