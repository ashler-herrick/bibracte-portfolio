import logging

import torch
from torch.optim.optimizer import Optimizer
from typing import Callable
import torch.nn.functional as F

from bet_edge.probabilistic_models.dpm import DeepProbabilisticModel


logger = logging.getLogger(__name__)


class DeepNormalMixture(DeepProbabilisticModel):
    """
    Fits a mixture of normal distributions using a neural network.

    This model is suitable for modeling continuous target variables.
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_dist: int = 3,
        p_dropout: float = 0.1,
        optimizer_class: Callable[..., Optimizer] = torch.optim.Adam,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        seed: int = 42,
        use_cuda: bool = False,
    ):
        """
        Initializes the DeepNormalMixture model.

        Args:
            n_inputs (int): Number of input features.
            n_hidden (int): Number of network units.
            n_dist (int, optional): Number of mixture components. Defaults to 3.
            p_dropout (float, optional): Dropout probability. Defaults to 0.1.
            optimizer_class (Callable, optional): Optimizer class. Defaults to torch.optim.Adam.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            use_cuda (bool, optional): Whether to use CUDA if available. Defaults to False.
        """
        super().__init__(
            optimizer_class=optimizer_class,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
            use_cuda=use_cuda,
        )
        self.n_dist = n_dist
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_hidden),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(n_hidden, n_hidden),
        )
        self.mean_linear = torch.nn.Linear(n_hidden, n_dist)
        self.scale_linear = torch.nn.Linear(n_hidden, n_dist)
        self.mix_linear = torch.nn.Linear(n_hidden, n_dist)

        self.to(self.device)

        logger.info("Initialized DeepNormalMixture model architecture.")

    def forward(self, x: torch.Tensor) -> torch.distributions.MixtureSameFamily:
        """
        Defines the forward pass for the DeepNormalMixture model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.distributions.MixtureSameFamily: Output mixture distribution.
        """
        outputs = self.network(x)
        outputs = torch.tanh(outputs)

        mean = self.mean_linear(outputs)
        # Use softplus for scale to ensure positivity
        scale = F.softplus(self.scale_linear(outputs))
        # Use logits directly in the Categorical distribution
        mix_logits = self.mix_linear(outputs)

        mix = torch.distributions.Categorical(logits=mix_logits)
        comp = torch.distributions.Normal(mean, scale)
        mixture = torch.distributions.MixtureSameFamily(mix, comp)

        logger.debug("Forward pass completed for DeepNormalMixture.")
        return mixture
