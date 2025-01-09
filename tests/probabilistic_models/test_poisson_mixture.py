# tests/test_deep_poisson_mixture.py

import pytest
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from bet_edge.probabilistic_models.models.poisson_mixture import DeepPoissonMixture

# Configure logger for the test module (if needed)
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO or DEBUG as needed
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

@pytest.fixture
def synthetic_data():
    """Generates synthetic count data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.poisson(lam=3.0, size=100).astype(np.float32)
    return X, y

@pytest.fixture
def model():
    """Initializes the DeepPoissonMixture model."""
    return DeepPoissonMixture(
        n_inputs=10,
        n_hidden=20,
        optimizer_class=Adam,
        learning_rate=1e-3,
        p_dropout=0.1,
        n_dist=3,
        batch_size=16,
        use_cuda=False,
    )

def test_model_initialization(model):
    """Test if the model initializes correctly."""
    assert model.hidden is not None, "Hidden layers not initialized."
    assert model.rate_linear.out_features == 3, "Incorrect number of mixture components in rate_linear."
    assert model.mix_linear.out_features == 3, "Incorrect number of mixture components in mix_linear."

def test_forward_pass(model):
    """Test the forward pass of the model."""
    x = torch.randn(5, 10)
    dist = model(x)
    assert isinstance(dist, torch.distributions.MixtureSameFamily), "Output is not MixtureSameFamily."
    assert dist.mixture_distribution.logits.shape == (5, 3), "Incorrect shape for mixture logits."
    assert dist.component_distribution.rate.shape == (5, 3), "Incorrect shape for component rates."

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

    # Create DataLoaders (optional)
    # train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=model.batch_size, shuffle=False)

    model.fit(
        train_data=train_dataset,
        val_data=val_dataset,
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

    model.fit(
        train_data=train_dataset,
        val_data=val_dataset,
        epochs=10,
        early_stopping=False,
    )
    preds = model.predict(X[80:])
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

    model.fit(
        train_data=train_dataset,
        val_data=val_dataset,
        epochs=10,
        early_stopping=False,
    )
    dist = model.pred_dist(X[80:])
    assert isinstance(dist, torch.distributions.MixtureSameFamily), "Predicted distribution is incorrect."
