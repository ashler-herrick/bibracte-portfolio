# tests/test_kfold_dpm.py

import pytest
import torch
import numpy as np
from torch.optim import Adam

from bet_edge.probabilistic_models.models.normal_mixture import DeepNormalMixture
from bet_edge.probabilistic_models.kfold_dpm import KFoldDPM


@pytest.fixture
def synthetic_data():
    """Generates synthetic data for K-Fold testing."""
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    return X, y


@pytest.fixture
def base_model():
    """Initializes a base DeepNormalMixture model."""
    return DeepNormalMixture(
        n_inputs=10,
        n_hidden=20,
        optimizer_class=Adam,
        learning_rate=1e-3,
        p_dropout=0.1,
        n_dist=3,
        batch_size=16,
        use_cuda=False,
    )


@pytest.fixture
def kfold_dpm(base_model):
    """Initializes the KFoldDPM with a base model."""
    return KFoldDPM(torch_dpm=base_model)


def test_kfold_initialization(kfold_dpm):
    """Test if KFoldDPM initializes correctly."""
    assert kfold_dpm.torch_dpm is not None, "Base model not assigned."
    assert isinstance(kfold_dpm.models, list), "Models list not initialized."
    assert kfold_dpm.fold_nll == [], "fold_nll should be empty initially."
    assert kfold_dpm.fold_mae == [], "fold_mae should be empty initially."


def test_train_kfold(kfold_dpm, synthetic_data):
    """Test the k-fold training process."""
    X, y = synthetic_data
    kfold_dpm.train_kfold(
        X=X,
        y=y,
        n_splits=5,
        epochs=10,
        patience=5,
    )
    assert len(kfold_dpm.models) == 5, "Not all folds were trained."
    assert len(kfold_dpm.fold_mae) == 5, "MAE not recorded for all folds."
    assert len(kfold_dpm.fold_nll) == 5, "NLL not recorded for all folds."
    assert kfold_dpm.average_mae > 0, "Average MAE not computed correctly."
    assert kfold_dpm.average_nll > 0, "Average NLL not computed correctly."


def test_kfold_predict(kfold_dpm, synthetic_data):
    """Test prediction after k-fold training."""
    X, y = synthetic_data
    kfold_dpm.train_kfold(
        X=X,
        y=y,
        n_splits=5,
        epochs=10,
        patience=5,
    )
    preds = kfold_dpm.predict(X)
    assert preds.shape == y.shape, "Predicted shape does not match target shape."
    assert isinstance(preds, np.ndarray), "Predictions are not a NumPy array."


def test_kfold_pred_dist(kfold_dpm, synthetic_data):
    """Test distribution prediction after k-fold training."""
    X, y = synthetic_data
    kfold_dpm.train_kfold(
        X=X,
        y=y,
        n_splits=5,
        epochs=10,
        patience=5,
    )
    dist = kfold_dpm.pred_dist(X)
    assert isinstance(dist, torch.distributions.MixtureSameFamily), "Aggregated distribution is incorrect."
