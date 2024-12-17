"""
K-Fold Deep Probabilistic Model Module

This module provides functionality for training and evaluating probabilistic models
using k-fold cross-validation. The `KFoldDPM` class generalizes a `DeepProbabilisticModel`
to train across multiple data splits, allowing for robust model evaluation.

Classes:
    KFoldDPM: Facilitates k-fold cross-validation for probabilistic models.

Functions:
    _average_tensors: Computes the average of a list of tensors.
    _average_dist_params: Aggregates distribution parameters across a list of distributions.
"""

import torch
import numpy as np
import logging
import copy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from typing import List, Union, cast

from bet_edge.probabilistic_models.dpm import DeepProbabilisticModel


logger = logging.getLogger(__name__)


class KFoldDPM:
    """
    Generalizes a DeepProbabilisticModel to train across k folds using K-Fold Cross-Validation.

    This class handles the training, validation, and aggregation of multiple models
    trained on different data splits.
    """

    def __init__(self, torch_dpm: DeepProbabilisticModel):
        """
        Initializes the KFoldDPM with a given DeepProbabilisticModel.

        Args:
            torch_dpm (DeepProbabilisticModel): The probabilistic model to be trained.
        """
        self.torch_dpm = torch_dpm
        self.models: List[DeepProbabilisticModel] = []
        self.X: Union[np.ndarray, None] = None
        self.y: Union[np.ndarray, None] = None
        self.fold_nll = []
        self.fold_mae = []

        logger.info("Initialized KFoldDPM for cross-validation.")

    def _clone_model(self) -> DeepProbabilisticModel:
        """
        Creates a deep copy of the original model for each fold.

        Returns:
            DeepProbabilisticModel: Cloned model instance.
        """
        cloned_model = copy.deepcopy(self.torch_dpm)
        logger.debug("Cloned model for new fold.")
        return cloned_model

    def train_kfold(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, epochs: int = 2000, patience: int = 50):
        """
        Trains the DeepProbabilisticModel using k-fold cross-validation.

        Args:
            X (np.ndarray): Feature matrix for training and validation.
            y (np.ndarray): Target vector for training and validation.
            n_splits (int, optional): Number of folds for K-Fold. Defaults to 5.
            epochs (int, optional): Maximum number of training epochs per fold. Defaults to 2000.
            patience (int, optional): Patience for early stopping. Defaults to 50.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.oof_training_preds = np.zeros((y.shape[0]))
        logger.info(f"Starting k-fold training with {n_splits} splits.")

        for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
            logger.info(f"Starting Fold {fold}/{n_splits}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = self._clone_model()
            model.fit(
                X=(X_train, X_test),
                y=(y_train, y_test),
                epochs=epochs,
                early_stopping=True,
                patience=patience,
            )

            y_preds = model.predict(X_test)
            y_dist = model.pred_dist(X_test)

            mae = mean_absolute_error(y_test, y_preds)
            self.fold_mae.append(mae)
            nll = model._nll(y_dist, torch.tensor(y_test, dtype=torch.float32)).item()
            self.fold_nll.append(nll)

            self.oof_training_preds[test_index] = y_preds.squeeze()
            self.models.append(model)

            logger.info(f"Fold {fold} completed: MAE = {mae:.6f}, NLL = {nll:.6f}")

        self.average_nll = np.mean(self.fold_nll)
        self.average_mae = np.mean(self.fold_mae)

        logger.info(f"Average MAE across all folds: {self.average_mae:.6f}")
        logger.info(f"Average NLL across all folds: {self.average_nll:.6f}")

    def pred_dist(self, x: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Distribution:
        """
        Returns an evenly weighted distribution object across all trained models.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input features.

        Returns:
            torch.distributions.Distribution: Aggregated distribution from all models.
        """
        return self._get_pred_dist(x)

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Returns the mean of an evenly weighted distribution object across all trained models.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input features.

        Returns:
            np.ndarray: Predicted mean values.
        """
        return self._get_pred_dist(x).mean.cpu().numpy()

    def _get_pred_dist(self, x: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Distribution:
        """
        Internal method for aggregating prediction distributions from all models.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input features.

        Returns:
            torch.distributions.Distribution: Aggregated distribution.
        """
        dists = []
        for idx, model in enumerate(self.models, 1):
            dist = model.pred_dist(x)
            dists.append(dist)
            logger.debug(f"Aggregated distribution from model {idx}.")

        mixture = _average_dist_params(dists)
        logger.debug("Aggregated all model distributions into a single mixture distribution.")
        return mixture


def _average_tensors(tensors):
    """
    Computes the average of a list of tensors.

    Args:
        tensors (List[torch.Tensor]): List of tensors to average.

    Returns:
        torch.Tensor: Averaged tensor.
    """
    return torch.mean(torch.stack(tensors), dim=0)


def _average_dist_params(
    dist_array: List[torch.distributions.Distribution],
) -> torch.distributions.Distribution:
    """
    Aggregates distribution parameters across a list of distributions.

    Args:
        dist_array (List[torch.distributions.Distribution]): List of distributions to aggregate.

    Returns:
        torch.distributions.Distribution: Aggregated distribution.

    Raises:
        ValueError: If the supplied distribution type is not supported.
    """
    if isinstance(dist_array[0], torch.distributions.MixtureSameFamily):
        # Cast dist_array to the specific type for type checkers
        mixture_list = cast(List[torch.distributions.MixtureSameFamily], dist_array)
        comp = _average_dist_params([x.component_distribution for x in mixture_list])
        mix = _average_dist_params([x.mixture_distribution for x in mixture_list])
        return torch.distributions.MixtureSameFamily(mix, comp)

    elif isinstance(dist_array[0], torch.distributions.Normal):
        normal_list = cast(List[torch.distributions.Normal], dist_array)
        loc = _average_tensors([x.loc for x in normal_list])
        scale = _average_tensors([x.scale for x in normal_list])
        return torch.distributions.Normal(loc=loc, scale=scale)

    elif isinstance(dist_array[0], torch.distributions.Gamma):
        gamma_list = cast(List[torch.distributions.Gamma], dist_array)
        concentration = _average_tensors([x.concentration for x in gamma_list])
        rate = _average_tensors([x.rate for x in gamma_list])
        return torch.distributions.Gamma(concentration=concentration, rate=rate)

    elif isinstance(dist_array[0], torch.distributions.Categorical):
        cat_list = cast(List[torch.distributions.Categorical], dist_array)
        probs = _average_tensors([x.probs for x in cat_list])
        return torch.distributions.Categorical(probs=probs)

    else:
        raise ValueError(f"Supplied distribution {dist_array[0]} is not implemented in '_average_dist_params' method.")
