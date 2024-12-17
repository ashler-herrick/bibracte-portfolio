"""
DPM: Deep Probabilistic Models

This module provides a template class for constructing probabilistic neural networks using PyTorch.
The `DeepProbabilisticModel` class is an abstract base class designed for training and evaluating
probabilistic models with support for customizable optimizers, early stopping, and CUDA compatibility.

Classes:
    DeepProbabilisticModel: Template class for probabilistic neural networks.
"""

import torch
import random
import numpy as np
import logging
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.optimizer import Optimizer
from abc import ABC, abstractmethod
from typing import Tuple, Union, Callable, Sized

logger = logging.getLogger(__name__)


class DeepProbabilisticModel(torch.nn.Module, ABC):
    """
    Abstract base class serving as a template for constructing probabilistic neural networks.

    This class serves as a template for implementing probabilistic neural networks
    with a defined negative log-likelihood loss, training, validation, early stopping,
    and prediction functionalities.
    """

    def __init__(
        self,
        optimizer_class: Callable[..., Optimizer],
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        seed: int = 42,
        use_cuda: bool = False,
    ):
        """
        Initializes the DeepProbabilisticModel.

        Args:
            optimizer_class (Callable): Optimizer class from `torch.optim`.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            use_cuda (bool, optional): Whether to use CUDA if available. Defaults to False.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.batch_size = batch_size
        self.train_loss_arr = []
        self.val_loss_arr = []
        self.best_val_loss = 0
        self.best_val_iter = 0
        self.patience_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.set_seed(seed)
        logger.info(f"Initialized model on device: {self.device}")

        # Log device of all parameters
        for name, param in self.named_parameters():
            assert param.device == self.device, f"Parameter {name} is on {param.device}, expected {self.device}"
            logger.debug(f"Parameter '{name}' is on device: {param.device}")

    def set_seed(self, seed: int = 42):
        """
        Sets the random seed for reproducibility.

        Args:
            seed (int, optional): Seed value. Defaults to 42.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        logger.debug(f"Random seed set to: {seed}")

    def _nll(self, y_hat: torch.distributions.Distribution, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the Negative Log-Likelihood (NLL) loss.

        Args:
            y_hat (torch.distributions.Distribution): Predicted distribution.
            y (torch.Tensor): True target values.

        Returns:
            torch.Tensor: Mean NLL loss.
        """
        negloglik = -y_hat.log_prob(y)
        return torch.mean(negloglik)

    def _train_iter(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Performs a single training iteration on a batch.

        Args:
            X (torch.Tensor): Input features.
            y (torch.Tensor): Target values.

        Returns:
            float: Loss value for the batch.
        """
        y_hat = self(X)
        loss = self._nll(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logger.debug(f"Training iteration loss: {loss.item()}")
        return loss.item()

    def _val_iter(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Performs a single validation iteration on a batch.

        Args:
            X (torch.Tensor): Input features.
            y (torch.Tensor): Target values.

        Returns:
            float: Validation loss for the batch.
        """
        with torch.no_grad():
            y_hat = self(X)
            val_loss = self._nll(y_hat, y).item()
        logger.debug(f"Validation iteration loss: {val_loss}")
        return val_loss

    def _batch_train(self, dl_train: DataLoader) -> float:
        """
        Trains the model over all batches in the training DataLoader.

        Args:
            dl_train (DataLoader): DataLoader for training data.

        Returns:
            float: Average training loss over all batches.
        """
        total_loss = 0.0
        self.train()
        logger.info("Starting training batch loop.")
        for batch_idx, (X, y) in enumerate(dl_train):
            # Move data to device once per batch
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            loss = self._train_iter(X, y)
            total_loss += loss
            if batch_idx % 100 == 0:
                logger.debug(f"Batch {batch_idx}: Loss = {loss}")
        average_loss = total_loss / len(dl_train)
        logger.info(f"Average training loss: {average_loss}")
        return average_loss

    def _batch_val(self, dl_test: DataLoader) -> float:
        """
        Validates the model over all batches in the validation DataLoader.

        Args:
            dl_test (DataLoader): DataLoader for validation data.

        Returns:
            float: Average validation loss over all batches.
        """
        if not isinstance(dl_test.dataset, Sized):
            raise TypeError("The dataset in DataLoader must implement the Sized protocol.")

        self.eval()
        size = len(dl_test.dataset)
        val_loss = 0.0
        logger.info("Starting validation batch loop.")
        with torch.no_grad():
            for X, y in dl_test:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                _test_loss = self._val_iter(X, y)
                val_loss += _test_loss * len(X)
        val_loss /= size
        logger.info(f"Average validation loss: {val_loss}")
        return val_loss

    def _early_stopping(self, patience: int) -> bool:
        """
        Checks whether early stopping criteria have been met.

        Args:
            patience (int): Number of epochs to wait for improvement.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        current_best_val_loss = np.min(self.val_loss_arr)
        self.best_val_loss = current_best_val_loss
        self.best_val_iter = int(np.argmin(self.val_loss_arr))

        if len(self.val_loss_arr) > 1 and self.val_loss_arr[-1] > self.best_val_loss:
            self.patience_counter += 1
            logger.debug(f"Patience counter increased to {self.patience_counter}")
        else:
            self.patience_counter = 0
            self.best_model_state = self.state_dict()
            logger.debug("Best model updated and patience counter reset.")

        if self.patience_counter >= patience:
            logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement.")
            return True

        return False

    def fit(
        self,
        X: Tuple[np.ndarray, np.ndarray],
        y: Tuple[np.ndarray, np.ndarray],
        epochs: int = 100,
        early_stopping: bool = False,
        patience: int = 10,
    ) -> None:
        """
        Trains the model using the provided training and validation data.

        Args:
            X (Tuple[np.ndarray, np.ndarray]): Tuple containing training and validation feature arrays.
            y (Tuple[np.ndarray, np.ndarray]): Tuple containing training and validation target arrays.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            patience (int, optional): Patience for early stopping. Defaults to 10.
        """
        self.train()
        self.patience_counter = 0
        self.optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        X_train, X_test = X
        y_train, y_test = y

        # Convert to CPU tensors here, no extra dimension insertion
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # CPU datasets
        td_train = TensorDataset(X_train_tensor, y_train_tensor)
        td_test = TensorDataset(X_test_tensor, y_test_tensor)

        # Enable pin_memory if on GPU
        dl_train = DataLoader(
            td_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        dl_test = DataLoader(
            td_test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

        logger.info(f"Starting training for {epochs} epochs.")
        for t in range(epochs):
            logger.debug(f"Epoch {t+1}/{epochs} started.")
            train_loss = self._batch_train(dl_train)
            val_loss = self._batch_val(dl_test)
            self.train_loss_arr.append(train_loss)
            self.val_loss_arr.append(val_loss)

            if t % 25 == 0:
                logger.info(f"Epoch {t+1}/{epochs} -> Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

            if early_stopping:
                early_stop = self._early_stopping(patience)
                if early_stop:
                    logger.info(f"Early stopping after {t+1} epochs.")
                    self.load_state_dict(self.best_model_state)
                    logger.info(f"Best Model: Epoch {self.best_val_iter + 1}, Loss: {self.best_val_loss:.6f}")
                    break

        logger.info("Training completed.")

    def pred_dist(self, x: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Distribution:
        """
        Predicts the distribution for the given input data.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input features.

        Returns:
            torch.distributions.Distribution: Predicted distribution.
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self.device)
            y_hat = self(x)
        logger.debug("Predicted distribution for input data.")
        return y_hat

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predicts the mean of the distribution for the given input data.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input features.

        Returns:
            np.ndarray: Predicted mean values.
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self.device)
            y_hat = self(x)
        preds = y_hat.mean.cpu().numpy()
        logger.debug("Computed mean predictions for input data.")
        return preds

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.distributions.Distribution: Output distribution.
        """
        pass
