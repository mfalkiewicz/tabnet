from typing import TYPE_CHECKING, Union, List, Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from .tab_network import TabNet
from .utils import (
    PredictDataset,
    check_input,
    check_target,
    define_device,
    ComplexEncoder,
    SparsePredictDataset, 
    filter_weights,
    create_explain_matrix
)
from .metrics import MetricContainer, check_metrics
from .abstract_model import TabModel
from .multiclass_utils import infer_output_dim, check_output_dim
from .callbacks import (
    CallbackContainer,
    History,
    EarlyStopping,
    LRSchedulerCallback,
)
from pyspark.sql import DataFrame
from numpy.typing import NDArray
from scipy.special import softmax, expit
from torch.utils.data import DataLoader
import scipy
import pandas as pd
from torch import Tensor

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from .spark_utils import SparkDataset
else:
    DataFrame = None
    SparkDataset = None

try:
    from .spark_utils import SparkDataset
    from pyspark.sql import DataFrame
    _HAVE_PYSPARK = True
except ImportError:
    SparkDataset = None
    DataFrame = None
    _HAVE_PYSPARK = False


class TabNetClassifier(TabModel):
    def __init__(self, n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 cat_idxs=[], cat_dims=[], cat_emb_dim=1,
                 n_independent=2, n_shared=2, epsilon=1e-15,
                 momentum=0.02, lambda_sparse=1e-3, seed=0,
                 clip_value=1, verbose=1,
                 optimizer_fn=torch.optim.Adam,
                 optimizer_params=dict(lr=2e-2),
                 scheduler_fn=None, scheduler_params=None,
                 mask_type="sparsemax",
                 input_dim=None, output_dim=None,
                 device_name="auto", n_shared_decoder=1, n_indep_decoder=1):
        super(TabNetClassifier, self).__init__(
            n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
            cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
            n_independent=n_independent, n_shared=n_shared, epsilon=epsilon,
            momentum=momentum, lambda_sparse=lambda_sparse, seed=seed,
            clip_value=clip_value, verbose=verbose,
            optimizer_fn=optimizer_fn, optimizer_params=optimizer_params,
            scheduler_fn=scheduler_fn, scheduler_params=scheduler_params,
            mask_type=mask_type, input_dim=input_dim, output_dim=output_dim,
            device_name=device_name,
            n_shared_decoder=n_shared_decoder,
            n_indep_decoder=n_indep_decoder
        )
        self.class_weights = None

    def __post_init__(self):
        super(TabNetClassifier, self).__post_init__()
        self._task = "classification"
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = "accuracy"
        self.max_epochs = None
        self.patience = None
        self.batch_size = 1024
        self.virtual_batch_size = 128
        self.preds_mapper = None  # Initialize preds_mapper
        self.classes_ = None
        self.output_dim = None
        self.input_dim = None
        self.group_attention_matrix = None  # Initialize group_attention_matrix

    def _prepare_input(
        self, X: Union[DataFrame, np.ndarray], target_col: Optional[str] = None
    ) -> Union[DataFrame, np.ndarray]:
        """Prepare input data.

        Parameters
        ----------
        X : Union[np.ndarray, DataFrame]
            Input data
        target_col : Optional[str]
            Target column name for Spark DataFrame

        Returns
        -------
        Union[np.ndarray, DataFrame]
            Prepared input data
        """
        if _HAVE_PYSPARK and isinstance(X, DataFrame):
            feature_cols = [col for col in X.columns if col != target_col]
            self.input_dim = len(feature_cols)
            return X
        else:
            return super()._prepare_input(X)

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        """
        Update fit parameters based on input data

        Parameters
        ----------
        X_train : np.ndarray or pd.DataFrame or torch.Tensor
            Training data
        y_train : np.ndarray or pd.DataFrame or torch.Tensor
            Target values
        eval_set : list of tuples
            List of (X, y) tuple pairs for evaluation
        weights : np.ndarray or None
            Sample weights

        Returns
        -------
        dict
            Updated fit parameters
        """
        # Check input shapes and types
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        elif isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
            
        if len(X_train.shape) != 2:
            raise ValueError(f"Expected 2D input array, got shape {X_train.shape}")
            
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.values
        elif isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()
            
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train and y_train have different number of samples: {X_train.shape[0]} vs {y_train.shape[0]}")

        # Determine number of unique classes
        if len(y_train.shape) == 1 or y_train.shape[1] == 1:
            self.classes_ = np.unique(y_train)
            n_classes = len(self.classes_)
        else:
            self.classes_ = np.arange(y_train.shape[1])
            n_classes = y_train.shape[1]

        updated_params = {
            "input_dim": X_train.shape[1],
            "output_dim": n_classes,
            "weights": weights
        }

        return updated_params

    def prepare_target(self, y):
        """Prepare target data.

        Parameters
        ----------
        y : array-like
            Target data

        Returns
        -------
        array-like
            Prepared target data
        """
        if isinstance(y, pd.DataFrame):
            y = y.values
        elif isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
            
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        return y

    def compute_loss(self, y_score: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Parameters
        ----------
        y_score : torch.Tensor
            Predicted values
        y_true : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # CrossEntropyLoss expects class indices as targets, not one-hot encoding
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = torch.argmax(y_true, dim=1)
        elif len(y_true.shape) > 1:
            y_true = y_true.squeeze(1)
            
        # Ensure y_true is properly formatted for CrossEntropyLoss
        y_true = y_true.long()
        
        # Ensure y_score has shape (N, C) where C matches number of classes
        if y_score.shape[1] != self.output_dim:
            raise ValueError(f"Network output dimension {y_score.shape[1]} does not match expected {self.output_dim}")
            
        return self.loss_fn(y_score, y_true)

    def predict_func(self, outputs):
        """
        Make predictions based on network outputs
        """
        return torch.softmax(outputs, dim=1)

    def predict_proba(self, X):
        """
        Make probability predictions
        """
        self.network.eval()
        
        if isinstance(X, DataFrame):  # PySpark DataFrame
            dataloader = DataLoader(
                SparkDataset(X), batch_size=self.batch_size, shuffle=False
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X), batch_size=self.batch_size, shuffle=False
            )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()

            output, _ = self.network(data)
            output = self.predict_func(output)
            results.append(output.cpu().detach().numpy())
            
        results = np.vstack(results)
        return results

    def predict(self, X):
        """
        Make predictions for the input samples X.
        """
        predictions = self.predict_proba(X)
        if self.output_dim == 2:
            # For binary classification, take the argmax instead of thresholding
            predictions = predictions.argmax(axis=1)
        else:
            predictions = predictions.argmax(axis=1)
            
        # Map predictions back to original classes
        if self.preds_mapper is not None:
            predictions = np.vectorize(self.preds_mapper.get)(predictions)
            
        return predictions

    def stack_batches(self, list_y_true, list_y_score):
        """Stack batches for prediction.

        Parameters
        ----------
        list_y_true : list
            List of true values
        list_y_score : list
            List of predicted values

        Returns
        -------
        tuple
            Stacked true values and predicted values
        """
        if not list_y_true or not list_y_score:
            return np.array([]), np.array([])

        # Convert torch tensors to numpy arrays
        list_y_true = [y.cpu().detach().numpy() if isinstance(y, torch.Tensor) else y for y in list_y_true]
        list_y_score = [y.cpu().detach().numpy() if isinstance(y, torch.Tensor) else y for y in list_y_score]

        # Stack predictions
        y_score = np.vstack(list_y_score)
        
        # Handle y_true stacking based on shape
        if len(list_y_true[0].shape) == 1:
            y_true = np.concatenate(list_y_true)
        else:
            y_true = np.vstack(list_y_true)

        # Ensure y_true is properly shaped for binary classification
        if self.output_dim == 2:
            y_true = y_true.reshape(-1)
            if len(y_score.shape) > 1 and y_score.shape[1] == 2:
                y_score = y_score[:, 1]  # Take probability of positive class
        
        return y_true, y_score

    def explain(
        self, X: Union[np.ndarray, DataFrame], normalize: bool = True
    ) -> Tuple[NDArray[np.float32], Dict[str, NDArray[np.float32]]]:
        """Generate explanations for the model's predictions"""
        self.network.eval()

        # Handle different DataFrame types
        if isinstance(X, DataFrame):  # PySpark DataFrame
            X_proc = X
            if "target" in X_proc.columns:
                X_proc = X_proc.drop("target")
        else:
            X_proc = X.copy() if isinstance(X, np.ndarray) else X

        # Create appropriate dataloader
        if isinstance(X_proc, DataFrame):
            dataloader = DataLoader(
                SparkDataset(X_proc), batch_size=self.batch_size, shuffle=False
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X_proc), batch_size=self.batch_size, shuffle=False
            )

        res_explain: List[NDArray[np.float32]] = []
        res_masks = {}

        for _, data in enumerate(dataloader):
            data = data.to(self.device).float()
            M_explain, masks = self.network.forward_masks(data)
            res_explain.append(M_explain.cpu().detach().numpy())
            
            # Initialize res_masks on first batch
            if not res_masks:
                res_masks = {k: [] for k in masks.keys()}
                
            # Append each mask
            for k, v in masks.items():
                res_masks[k].append(v.cpu().detach().numpy())

        res_explain_array = np.vstack(res_explain)
        
        # Stack all masks
        for k in res_masks.keys():
            res_masks[k] = np.vstack(res_masks[k])

        if normalize:
            res_explain_array = (res_explain_array - res_explain_array.min()) / (
                res_explain_array.max() - res_explain_array.min()
            )

        return res_explain_array, res_masks

    def _set_output_dim(self, y):
        """Set output dimension based on y."""
        if len(y.shape) == 1:
            y_flat = y
        else:
            y_flat = y.ravel() if y.shape[1] == 1 else y

        self.classes_ = np.unique(y_flat)
        self.output_dim = len(self.classes_)
        
        # Always create preds_mapper
        self.preds_mapper = {idx: val for idx, val in enumerate(self.classes_)}

    def _initialize_network(self) -> None:
        """Initialize the network."""
        if self.input_dim is None:
            raise ValueError("Input dimension must be set before initializing network")

        # Set output dimension if not already set
        if self.output_dim is None:
            if hasattr(self, 'classes_') and self.classes_ is not None:
                self.output_dim = len(self.classes_)
            else:
                raise ValueError("Output dimension not set and no classes found")

        self._set_network()
        # For both binary and multiclass classification, use Softmax
        self.predict_func = torch.nn.Softmax(dim=1)
        # Initialize loss function as CrossEntropyLoss
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def fit(
        self,
        X_train,
        y_train,
        eval_set=None,
        eval_name=None,
        eval_metric=None,
        loss_fn=None,
        weights=0,
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        callbacks=None,
        pin_memory=True,
        from_unsupervised=None,
        warm_start=False,
        augmentations=None,
        compute_importance=True,
        from_epoch=0,
    ):
        # Store class weights if using class balancing
        if isinstance(weights, (int, float)) and weights == 1:
            unique_classes = np.unique(y_train)
            class_counts = np.bincount(y_train.ravel())
            self.class_weights = {i: 1.0 / count for i, count in enumerate(class_counts)}
        elif isinstance(weights, dict):
            self.class_weights = weights
        elif isinstance(weights, np.ndarray):
            # Sample weights provided directly
            pass
        
        # Ensure batch_size is at least 2 for BatchNorm and not larger than dataset
        batch_size = max(2, min(batch_size, len(X_train)))
        
        # Set drop_last to True if last batch would be size 1
        if len(X_train) % batch_size == 1:
            drop_last = True
        
        # Update fit params based on input data
        fit_params = self.update_fit_params(X_train, y_train, eval_set, weights)
        self.input_dim = fit_params["input_dim"]
        self.output_dim = fit_params["output_dim"]
        self.updated_weights = fit_params["weights"]
        
        # Initialize network if not already done
        if not hasattr(self, "network") or self.network is None:
            self._initialize_network()

        # Call parent fit method
        return super().fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            eval_name=eval_name,
            eval_metric=eval_metric,
            loss_fn=loss_fn,
            weights=weights,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            callbacks=callbacks,
            pin_memory=pin_memory,
            from_unsupervised=from_unsupervised,
            warm_start=warm_start,
            augmentations=augmentations,
            compute_importance=compute_importance,
            from_epoch=from_epoch
        )


class TabNetRegressor(TabModel):
    def __post_init__(self):
        super(TabNetRegressor, self).__post_init__()
        self._task = "regression"
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = "mse"
        self.preds_mapper = None  # Initialize preds_mapper

    def _prepare_input(
        self, X: Union[DataFrame, np.ndarray], target_col: Optional[str] = None
    ) -> Union[DataFrame, np.ndarray]:
        """Prepare input data.

        Parameters
        ----------
        X : Union[np.ndarray, DataFrame]
            Input data
        target_col : Optional[str]
            Target column name for Spark DataFrame

        Returns
        -------
        Union[np.ndarray, DataFrame]
            Prepared input data
        """
        if _HAVE_PYSPARK and isinstance(X, DataFrame):
            feature_cols = [col for col in X.columns if col != target_col]
            self.input_dim = len(feature_cols)
            return X
        else:
            return super()._prepare_input(X)


    def prepare_target(self, y):
        """Prepare target data.

        Parameters
        ----------
        y : array-like
            Target data

        Returns
        -------
        array-like
            Prepared target data
        """
        if isinstance(y, pd.DataFrame):
            y = y.values
        elif isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
            
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        return y

    def compute_loss(self, y_score: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Parameters
        ----------
        y_score : torch.Tensor
            Score matrix
        y_true : torch.Tensor
            Target matrix

        Returns
        -------
        torch.Tensor
            Loss value
        """
        return self.loss_fn(y_score, y_true)

    def predict_func(self, outputs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Return outputs.

        Parameters
        ----------
        outputs : Union[np.ndarray, torch.Tensor]
            Output matrix

        Returns
        -------
        np.ndarray
            Prediction values
        """
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().detach().numpy()
        return outputs

    def stack_batches(self, list_y_true, list_y_score):
        """Stack batches for prediction.

        Parameters
        ----------
        list_y_true : list
            List of true values
        list_y_score : list
            List of predicted values

        Returns
        -------
        tuple
            Stacked true values and predicted values
        """
        if not list_y_true or not list_y_score:
            return np.array([]), np.array([])

        # Convert torch tensors to numpy arrays
        list_y_true = [y.cpu().detach().numpy() if isinstance(y, torch.Tensor) else y for y in list_y_true]
        list_y_score = [y.cpu().detach().numpy() if isinstance(y, torch.Tensor) else y for y in list_y_score]

        if len(list_y_true[0].shape) == 1:
            # Binary classification or regression
            y_true = np.hstack(list_y_true)
            y_score = np.vstack(list_y_score).reshape(-1) if len(list_y_score[0].shape) == 1 else np.vstack(list_y_score)
        else:
            # Multiclass classification
            y_true = np.vstack(list_y_true)
            y_score = np.vstack(list_y_score)

        return y_true, y_score

    def update_fit_params(
        self,
        X_train: Union[np.ndarray, DataFrame],
        y_train: Union[np.ndarray, str],
        eval_set: Optional[List[Tuple[Union[np.ndarray, DataFrame], Union[np.ndarray, str]]]] = None,
        weights: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        """Update fit parameters based on input data.

        Parameters
        ----------
        X_train : Union[np.ndarray, DataFrame]
            Training data
        y_train : Union[np.ndarray, str]
            Target values
        eval_set : Optional[List[Tuple[Union[np.ndarray, DataFrame], Union[np.ndarray, str]]]], optional
            A list of (X, y) tuple pairs to use as validation sets
        weights : Optional[Union[float, np.ndarray]], optional
            Sample weights for training
        """
        self.updated_weights = weights
        self._set_output_dim(y_train)
        self._initialize_network()

    def _set_output_dim(self, y):
        """Set output dimension."""
        if len(y.shape) == 1:
            self.output_dim = 1
        else:
            if self.output_dim is None:
                self.output_dim = y.shape[1]
            else:
                assert y.shape[1] == self.output_dim, (
                    f"Output dimension mismatch: {y.shape[1]} != {self.output_dim}"
                )

    def _initialize_network(self) -> None:
        """Initialize the network."""
        if self.input_dim is None or self.output_dim is None:
            raise ValueError("Input and output dimensions must be set before initializing network")

        self.network = TabNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
        ).to(self.device)

    def fit(
        self,
        X_train,
        y_train,
        eval_set=None,
        eval_name=None,
        eval_metric=None,
        loss_fn=None,
        weights=0,
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=True,
        callbacks=None,
        pin_memory=True,
        from_unsupervised=None,
        warm_start=False,
        augmentations=None,
        compute_importance=True,
    ):
        # Convert input data
        X_train = self._prepare_input(X_train)
        if eval_set is not None:
            new_eval_set = []
            for X_val, y_val in eval_set:
                X_val = self._prepare_input(X_val)
                new_eval_set.append((X_val, y_val))
            eval_set = new_eval_set

        return super().fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            eval_name=eval_name,
            eval_metric=eval_metric,
            loss_fn=loss_fn,
            weights=weights,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            callbacks=callbacks,
            pin_memory=pin_memory,
            from_unsupervised=from_unsupervised,
            warm_start=warm_start,
            augmentations=augmentations,
            compute_importance=compute_importance,
        )
