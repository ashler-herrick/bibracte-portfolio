import pytest
import torch
import numpy as np
from bet_edge.probabilistic_models.models.normal_mixture import DeepNormalMixture
from torch.optim import Adam


@pytest.fixture
def cpu_model():
    """
    Fixture to initialize the CPU version of the DeepNormalMixture model.
    """
    model = DeepNormalMixture(
        n_inputs=1,
        n_hidden=16,
        optimizer_class=Adam,
        use_cuda=False,  # Ensure the model is on CPU
    )
    return model


@pytest.fixture
def gpu_model():
    """
    Fixture to initialize the GPU version of the DeepNormalMixture model.
    """
    if torch.cuda.is_available():
        model = DeepNormalMixture(
            n_inputs=1,
            n_hidden=16,
            optimizer_class=Adam,
            use_cuda=True,  # Ensure the model is on GPU
        )
        return model
    else:
        pytest.skip("No GPU available for testing")


def test_model_runs_on_cpu(cpu_model):
    # Simple training data
    X_train = np.random.randn(1000, 1)
    y_train = np.random.randn(1000)

    X_test = np.random.randn(200, 1)
    y_test = np.random.randn(200)

    cpu_model.fit((X_train, X_test), (y_train, y_test), epochs=5, early_stopping=False)
    preds = cpu_model.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]


def test_model_runs_on_gpu(gpu_model):
    # Simple training data
    X_train = np.random.randn(1000, 1)
    y_train = np.random.randn(1000)

    X_test = np.random.randn(200, 1)
    y_test = np.random.randn(200)

    gpu_model.fit((X_train, X_test), (y_train, y_test), epochs=5, early_stopping=False)
    preds = gpu_model.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]
    # Check if the model is on GPU
    for param in gpu_model.parameters():
        assert param.device.type == "cuda"


def test_mixture_distribution(cpu_model):
    X_sample = np.array([[0.0]])
    dist = cpu_model.pred_dist(X_sample)
    # Check if it's a MixtureSameFamily distribution
    from torch.distributions import MixtureSameFamily

    assert isinstance(dist, MixtureSameFamily)


def test_no_data_shuffle_in_validation(cpu_model):
    # Just ensure that no error is raised and that predictions are the same order
    X_train = np.random.randn(100, 1)
    y_train = np.random.randn(100)

    X_test = np.random.randn(20, 1)
    y_test = np.random.randn(20)

    cpu_model.fit((X_train, X_test), (y_train, y_test), epochs=2, early_stopping=False)
    preds = cpu_model.predict(X_test)
    # If the dataloader for test was shuffled, there's no direct error, but at least we can ensure same shape and no error.
    assert preds.shape[0] == X_test.shape[0]
