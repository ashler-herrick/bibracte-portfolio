import logging

import torch
from torch.optim.optimizer import Optimizer
from typing import Callable
import torch.nn.functional as F

from bet_edge.probabilistic_models.dpm import DeepProbabilisticModel

logger = logging.getLogger(__name__)


class DeepPoissonMixture(DeepProbabilisticModel):
    """
    Fits a mixture of Poisson distributions using a neural network.

    This model is suitable for modeling discrete target variables.
    """

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        optimizer_class: Callable[..., Optimizer],
        learning_rate: float = 1e-4,
        p_dropout: float = 0.5,
        n_dist: int = 5,
        batch_size: int = 64,
        use_cuda: bool = False,
    ):
        """
        Initializes the DeepPoissonMixture model.

        Args:
            n_inputs (int): Number of input features.
            n_hidden (int): Number of hidden units per layer.
            optimizer_class (Callable): Optimizer class from torch.optim.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            p_dropout (float, optional): Dropout probability. Defaults to 0.5.
            n_dist (int, optional): Number of mixture components. Defaults to 5.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            use_cuda (bool, optional): Whether to use CUDA if available. Defaults to False.
        """
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer_class=optimizer_class,
            use_cuda=use_cuda,
        )

        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_hidden),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.Dropout(p_dropout),
            torch.nn.Linear(n_hidden, n_hidden),
        )
        self.rate_linear = torch.nn.Linear(n_hidden, n_dist)
        self.mix_linear = torch.nn.Linear(n_hidden, n_dist)
        self.to(self.device)

        logger.info("Initialized DeepPoissonMixture model architecture.")

    def forward(self, x: torch.Tensor) -> torch.distributions.MixtureSameFamily:
        """
        Defines the forward pass for the DeepPoissonMixture model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.distributions.MixtureSameFamily: Output mixture distribution.
        """
        outputs = self.hidden(x)
        outputs = torch.tanh(outputs)

        # Use softplus for rate to ensure positivity
        rate = F.softplus(self.rate_linear(outputs))
        mix_logits = self.mix_linear(outputs)

        mix = torch.distributions.Categorical(logits=mix_logits)
        comp = torch.distributions.Poisson(rate)
        mixture = torch.distributions.MixtureSameFamily(mix, comp)

        logger.debug("Forward pass completed for DeepPoissonMixture.")
        return mixture
