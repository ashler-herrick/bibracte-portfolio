import torch
import random
import numpy as np
import logging
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from abc import ABC, abstractmethod
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class DeepProbabilisticModel(torch.nn.Module, ABC):
    """
    Abstract base class serving as a template for constructing probabilistic neural networks.
    This class supports training and evaluating probabilistic models, with customizable
    optimizers, early stopping, and CUDA compatibility.
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
        self.best_val_loss = np.inf
        self.best_val_iter = 0
        self.patience_counter = 0
        self.best_model_state = None

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.to(self.device)
        self.set_seed(seed)
        logger.info(f"Initialized model on device: {self.device}")

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

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 100,
        early_stopping: bool = False,
        patience: int = 10,
    ) -> None:
        """
        Trains the model using the provided training and validation datasets.

        Args:
            train_dataset (Dataset): The training dataset.
            val_dataset (Optional[Dataset]): The validation dataset. Defaults to None.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            patience (int, optional): Patience for early stopping. Defaults to 10.
        """
        self.train()
        self.patience_counter = 0
        self.best_val_loss = np.inf
        self.best_val_iter = 0
        self.best_model_state = None

        # Initialize the optimizer
        self.optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)

        # Create DataLoaders
        dl_train = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )

        dl_val = None
        if val_dataset is not None:
            dl_val = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=self.device.type == "cuda",
            )

        logger.info(f"Starting training for {epochs} epochs.")
        for epoch in range(1, epochs + 1):
            logger.debug(f"Epoch {epoch}/{epochs} started.")
            train_loss = self._train_epoch(dl_train)
            if dl_val is not None:
                val_loss = self._validate_epoch(dl_val)
            else:
                val_loss = float('nan')

            self.train_loss_arr.append(train_loss)
            self.val_loss_arr.append(val_loss)

            logger.info(
                f"Epoch {epoch}/{epochs} -> "
                f"Train Loss: {train_loss:.6f}, "
                f"Validation Loss: {val_loss:.6f}"
            )

            if early_stopping and dl_val is not None:
                if self._check_early_stopping(val_loss, patience):
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    if self.best_model_state is not None:
                        self.load_state_dict(self.best_model_state)
                        logger.info(
                            f"Loaded best model from epoch {self.best_val_iter} with "
                            f"Validation Loss: {self.best_val_loss:.6f}"
                        )
                    break

        logger.info("Training completed.")

    def _train_epoch(self, dl_train: DataLoader) -> float:
        """
        Performs one full training epoch.

        Args:
            dl_train (DataLoader): DataLoader for training data.

        Returns:
            float: Average training loss for the epoch.
        """
        self.train()
        total_loss = 0.0
        for batch_idx, (X, y) in enumerate(dl_train, 1):
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            y_hat = self(X)
            loss = self._nll(y_hat, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                logger.debug(f"Batch {batch_idx}: Loss = {loss.item()}")

        average_loss = total_loss / len(dl_train)
        logger.debug(f"Epoch Training Loss: {average_loss}")
        return average_loss

    def _validate_epoch(self, dl_val: DataLoader) -> float:
        """
        Performs one full validation epoch.

        Args:
            dl_val (DataLoader): DataLoader for validation data.

        Returns:
            float: Average validation loss for the epoch.
        """
        self.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for X, y in dl_val:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                y_hat = self(X)
                loss = self._nll(y_hat, y)

                batch_size = X.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        average_loss = total_loss / total_samples
        logger.debug(f"Epoch Validation Loss: {average_loss}")
        return average_loss

    def _check_early_stopping(self, current_val_loss: float, patience: int) -> bool:
        """
        Checks if early stopping criteria have been met.

        Args:
            current_val_loss (float): Current epoch's validation loss.
            patience (int): Number of epochs to wait for improvement.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_val_iter = len(self.val_loss_arr) - 1
            self.patience_counter = 0
            self.best_model_state = self.state_dict()
            logger.debug("Validation loss improved. Resetting patience counter.")
        else:
            self.patience_counter += 1
            logger.debug(f"No improvement in validation loss. Patience counter: {self.patience_counter}")

        return self.patience_counter >= patience

    def pred_dist(self, dataset: Dataset) -> torch.distributions.Distribution:
        """
        Predicts the distribution for each data point in the dataset and aggregates them into a single distribution.

        Args:
            dataset (Dataset): Input dataset for which to predict distributions.

        Returns:
            torch.distributions.Distribution: Aggregated distribution representing all data points.
        """
        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )
        self.eval()
        all_logits = []
        all_means = []
        all_scales = []
        logger.info("Starting prediction distribution batch loop.")
        with torch.no_grad():
            for X, _ in dl:
                X = X.to(self.device, non_blocking=True)
                y_hat = self(X)  # MixtureSameFamily distribution
                # Extract logits, means, scales
                logits = y_hat.mixture_distribution.logits  # [batch_size, n_dist]
                means = y_hat.component_distribution.mean  # [batch_size, n_dist]
                scales = y_hat.component_distribution.stddev  # [batch_size, n_dist]
                all_logits.append(logits.cpu())
                all_means.append(means.cpu())
                all_scales.append(scales.cpu())

        # Concatenate across all batches
        concatenated_logits = torch.cat(all_logits, dim=0)  # [num_samples, n_dist]
        concatenated_means = torch.cat(all_means, dim=0)    # [num_samples, n_dist]
        concatenated_scales = torch.cat(all_scales, dim=0)  # [num_samples, n_dist]

        # Create new mixture distribution
        mixture_distribution = torch.distributions.Categorical(logits=concatenated_logits)
        component_distribution = torch.distributions.Normal(loc=concatenated_means, scale=concatenated_scales)
        aggregated_distribution = torch.distributions.MixtureSameFamily(mixture_distribution, component_distribution)

        logger.debug("Completed prediction distribution aggregation.")
        return aggregated_distribution

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the point estimate (e.g., mean) for each data point in the dataset.

        Args:
            dataset (Dataset): Input dataset for which to predict point estimates.

        Returns:
            np.ndarray: Predicted point estimates.
        """
        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )
        self.eval()
        preds_list = []
        logger.info("Starting prediction batch loop.")
        with torch.no_grad():
            for X, _ in dl:
                X = X.to(self.device, non_blocking=True)
                y_hat = self(X)
                preds = y_hat.mean.cpu().numpy()
                preds_list.append(preds)
        ensemble_preds = np.concatenate(preds_list, axis=0)
        logger.debug("Completed point predictions.")
        return ensemble_preds

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
