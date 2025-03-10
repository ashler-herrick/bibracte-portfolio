import torch
import numpy as np
import logging
import copy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from typing import List
from torch.utils.data import Subset, Dataset

from bet_edge.probabilistic_models.dpm import DeepProbabilisticModel

logger = logging.getLogger(__name__)


def gather_features_and_targets(dataset: Dataset):
    """
    Helper function to gather all (features, targets) from a dataset into tensors.
    Useful for metrics like MAE and NLL.

    Args:
        dataset (Dataset): The dataset to gather from.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors of features and targets.
    """
    X_list = []
    y_list = []
    for i in range(len(dataset)):
        X_item, y_item = dataset[i]
        X_list.append(X_item)
        y_list.append(y_item)
    X_tensor = torch.stack(X_list, dim=0)
    y_tensor = torch.stack(y_list, dim=0)
    return X_tensor, y_tensor


class KFoldDPM:
    """
    Manages k-fold cross-validation for DeepProbabilisticModel instances.
    Trains multiple models (one per fold) and aggregates their predictions.
    """

    def __init__(self, torch_dpm: DeepProbabilisticModel):
        """
        Initializes the KFoldDPM with a base DeepProbabilisticModel.

        Args:
            torch_dpm (DeepProbabilisticModel): The base model to clone for each fold.
        """
        self.torch_dpm = torch_dpm
        self.models: List[DeepProbabilisticModel] = []
        self.fold_mae: List[float] = []
        self.fold_nll: List[float] = []

        logger.info("Initialized KFoldDPM for cross-validation.")

    def _clone_model(self) -> DeepProbabilisticModel:
        """
        Creates a deep copy of the base model for a new fold.

        Returns:
            DeepProbabilisticModel: Cloned model instance.
        """
        cloned_model = copy.deepcopy(self.torch_dpm)
        logger.debug("Cloned model for a new fold.")
        return cloned_model

    def _compute_nll_for_distributions(
        self, distributions: List[torch.distributions.Distribution], y_vals: torch.Tensor
    ) -> float:
        """
        Computes the Negative Log-Likelihood (NLL) for a list of distributions against true targets.

        Args:
            distributions (List[torch.distributions.Distribution]): Predicted distributions.
            y_vals (torch.Tensor): True target values.

        Returns:
            float: Average NLL over the dataset.
        """
        total_nll = 0.0
        total_samples = 0
        for dist_obj in distributions:
            # Assuming dist_obj is a batched distribution
            nll = -dist_obj.log_prob(y_vals)
            total_nll += torch.sum(nll).item()
            total_samples += y_vals.size(0)
        average_nll = total_nll / total_samples if total_samples > 0 else float("nan")
        return average_nll

    def train_kfold(
        self,
        dataset: Dataset,
        n_splits: int = 5,
        epochs: int = 20,
        patience: int = 5,
    ):
        """
        Trains the model across k folds using the provided dataset.

        Args:
            dataset (Dataset): The dataset to perform k-fold cross-validation on.
            n_splits (int, optional): Number of folds. Defaults to 5.
            epochs (int, optional): Number of epochs per fold. Defaults to 20.
            patience (int, optional): Patience for early stopping. Defaults to 5.
        """
        logger.info(f"Starting k-fold training with {n_splits} splits.")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Reset previous state
        self.models.clear()
        self.fold_mae.clear()
        self.fold_nll.clear()

        indices = np.arange(len(dataset))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
            logger.info(f"--- Fold {fold_idx}/{n_splits} ---")
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # Clone the base model
            model = self._clone_model()

            # Train the model on the current fold
            model.fit(
                train_dataset=train_subset,
                val_dataset=val_subset,
                epochs=epochs,
                early_stopping=True,
                patience=patience,
            )
            self.models.append(model)

            # Gather validation data
            X_val, y_val = gather_features_and_targets(val_subset)

            # Compute predictions and distributions
            preds = model.predict(val_subset)  # np.ndarray
            distributions = model.pred_dist(val_subset)  # torch.distributions.Distribution

            # Compute MAE
            mae = mean_absolute_error(y_val.cpu().numpy(), preds)
            self.fold_mae.append(mae)

            # Compute NLL
            nll = self._compute_nll_for_distributions([distributions], y_val)
            self.fold_nll.append(nll)

            logger.info(f"Fold {fold_idx}: MAE = {mae:.6f}, NLL = {nll:.6f}")

        # Summarize cross-validation results
        average_mae = np.mean(self.fold_mae) if self.fold_mae else float("nan")
        average_nll = np.mean(self.fold_nll) if self.fold_nll else float("nan")
        logger.info(f"K-Fold Completed. Average MAE = {average_mae:.6f}, " f"Average NLL = {average_nll:.6f}")

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts point estimates by averaging predictions across all trained fold models.

        Args:
            dataset (Dataset): The dataset to make predictions on.

        Returns:
            np.ndarray: Averaged predictions.
        """
        if not self.models:
            raise ValueError("No trained models found. Please train using train_kfold first.")

        all_preds = []
        logger.info("Starting ensemble prediction across all folds.")
        for fold_idx, model in enumerate(self.models, start=1):
            preds = model.predict(dataset)  # np.ndarray
            all_preds.append(preds)
            logger.debug(f"Fold {fold_idx}: Collected predictions.")

        # Stack and average
        all_preds_stack = np.stack(all_preds, axis=0)  # Shape: (n_folds, n_samples)
        ensemble_preds = np.mean(all_preds_stack, axis=0)  # Shape: (n_samples,)
        logger.debug("Ensemble predictions computed by averaging across folds.")
        return ensemble_preds

    def pred_dist(self, dataset: Dataset) -> torch.distributions.Distribution:
        """
        Aggregates distributions from all trained fold models into a single distribution.

        Args:
            dataset (Dataset): The dataset to make distribution predictions on.

        Returns:
            torch.distributions.Distribution: Aggregated distribution representing all data points.
        """
        if not self.models:
            raise ValueError("No trained models found. Please train using train_kfold first.")

        aggregated_logits = []
        aggregated_means = []
        aggregated_scales = []
        logger.info("Starting ensemble distribution prediction across all folds.")
        for fold_idx, model in enumerate(self.models, start=1):
            dist_obj = model.pred_dist(dataset)  # torch.distributions.Distribution
            # Assuming dist_obj is MixtureSameFamily
            if not isinstance(dist_obj, torch.distributions.MixtureSameFamily):
                raise TypeError("Expected MixtureSameFamily distribution.")

            logits = dist_obj.mixture_distribution.logits  # [num_samples, n_dist]
            means = dist_obj.component_distribution.mean  # [num_samples, n_dist]
            scales = dist_obj.component_distribution.stddev  # [num_samples, n_dist]

            aggregated_logits.append(logits.cpu())
            aggregated_means.append(means.cpu())
            aggregated_scales.append(scales.cpu())

            logger.debug(f"Fold {fold_idx}: Collected distribution parameters.")

        # Average the parameters across folds
        concatenated_logits = torch.stack(aggregated_logits, dim=0)  # [n_folds, num_samples, n_dist]
        concatenated_means = torch.stack(aggregated_means, dim=0)  # [n_folds, num_samples, n_dist]
        concatenated_scales = torch.stack(aggregated_scales, dim=0)  # [n_folds, num_samples, n_dist]

        averaged_logits = torch.mean(concatenated_logits, dim=0)  # [num_samples, n_dist]
        averaged_means = torch.mean(concatenated_means, dim=0)  # [num_samples, n_dist]
        averaged_scales = torch.mean(concatenated_scales, dim=0)  # [num_samples, n_dist]

        # Create a new aggregated mixture distribution
        mixture_distribution = torch.distributions.Categorical(logits=averaged_logits)
        component_distribution = torch.distributions.Normal(loc=averaged_means, scale=averaged_scales)
        aggregated_distribution = torch.distributions.MixtureSameFamily(mixture_distribution, component_distribution)

        logger.debug("Completed ensemble distribution aggregation.")
        return aggregated_distribution
