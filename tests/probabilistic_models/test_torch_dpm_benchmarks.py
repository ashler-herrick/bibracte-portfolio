import logging

import pytest
import torch
import numpy as np
from bet_edge.probabilistic_models.models.normal_mixture import DeepNormalMixture
from torch.optim import Adam


# Configure logger for the test module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO or DEBUG as needed
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

N_DIMS = 1024
TRAIN_SIZE = 250000
TEST_SIZE = 50000
BATCH_SIZE = 64

# Generate synthetic training and testing data
X_train = np.random.randn(TRAIN_SIZE, N_DIMS).astype(np.float32)
y_train = np.random.randn(TRAIN_SIZE).astype(np.float32)
X_test = np.random.randn(TEST_SIZE, N_DIMS).astype(np.float32)
y_test = np.random.randn(TEST_SIZE).astype(np.float32)


@pytest.fixture
def cpu_model():
    """
    Fixture to initialize the CPU version of the DeepNormalMixture model.
    """
    model = DeepNormalMixture(
        n_inputs=N_DIMS,
        n_hidden=N_DIMS,
        optimizer_class=Adam,
        use_cuda=False,  # Ensure the model is on CPU
        batch_size=BATCH_SIZE,
    )
    return model


@pytest.fixture
def gpu_model():
    """
    Fixture to initialize the GPU version of the DeepNormalMixture model.
    """
    if torch.cuda.is_available():
        model = DeepNormalMixture(
            n_inputs=N_DIMS,
            n_hidden=N_DIMS,
            optimizer_class=Adam,
            use_cuda=True,  # Ensure the model is on GPU
            batch_size=BATCH_SIZE,
        )
        return model
    else:
        pytest.skip("No GPU available for testing")


def test_cpu_efficiency(cpu_model, benchmark):
    """
    Benchmark the training efficiency of the CPU model.
    """

    # Benchmark CPU training
    benchmark.pedantic(
        lambda: cpu_model.fit((X_train, X_test), (y_train, y_test), epochs=5, early_stopping=False),
        iterations=1,
        rounds=1,
    )


def test_gpu_efficiency(gpu_model, benchmark):
    """
    Benchmark the training efficiency of the GPU model.
    """

    # Benchmark GPU training
    benchmark.pedantic(
        lambda: gpu_model.fit((X_train, X_test), (y_train, y_test), epochs=5, early_stopping=False),
        iterations=1,
        rounds=1,
    )
