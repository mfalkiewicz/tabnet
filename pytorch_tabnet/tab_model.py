import torch
import numpy as np
from scipy.special import softmax
from pytorch_tabnet.utils import SparsePredictDataset, PredictDataset, filter_weights
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim
from torch.utils.data import DataLoader
import scipy
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from .spark_utils import SparkPredictDataset
from typing import Union, Dict, Optional, List, Any, Tuple, Sequence
from numpy.typing import NDArray
import pandas as pd
from torch import Tensor


class TabNetClassifier(TabModel):
    def __post_init__(self):
        super(TabNetClassifier, self).__post_init__()
        self._task = 'classification'
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = 'accuracy'
        self.max_epochs = None
        self.patience = None
        self.batch_size = 1024
        self.virtual_batch_size = 128

    def _prepare_input(self, X: Union[DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Convert input data to appropriate format"""
        if isinstance(X, DataFrame):
            self.input_dim = len(X.columns)
            return X.toPandas()
        return X

    def weight_updater(self, weights: Union[int, Dict[int, float], None]) -> Union[int, Dict[int, float]]:
        """Updates weights dictionary according to target_mapper."""
        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        else:
            return weights

    def prepare_target(self, y: NDArray) -> NDArray:
        return np.vectorize(self.target_mapper.get)(y)

    def compute_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.loss_fn(y_pred, y_true.long())

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self._default_metric = ('auc' if self.output_dim == 2 else 'accuracy')
        self.classes_ = train_labels
        self.target_mapper = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper = {
            str(index): class_label for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights = self.weight_updater(weights)

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.hstack(list_y_true)
        y_score = np.vstack(list_y_score)
        y_score = softmax(y_score, axis=1)
        return y_true, y_score

    def predict_func(self, outputs):
        outputs = np.argmax(outputs, axis=1)
        return np.vectorize(self.preds_mapper.get)(outputs.astype(str))

    def predict_proba(self, X: Union[np.ndarray, DataFrame]) -> np.ndarray:
        """Make predictions for classification"""
        self.network.eval()
        
        if isinstance(X, DataFrame):
            dataloader = DataLoader(
                SparkPredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        elif scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()
            output, M_loss = self.network(data)
            predictions = torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return res

    def explain(
        self, 
        X: Union[np.ndarray, DataFrame], 
        normalize: bool = True
    ) -> Tuple[NDArray[np.float32], Dict[str, NDArray[np.float32]]]:
        """Generate explanations for the model's predictions"""
        self.network.eval()

        # Handle different DataFrame types
        if isinstance(X, DataFrame):  # PySpark DataFrame
            X_proc = X
            if 'target' in X_proc.columns:
                X_proc = X_proc.drop('target')
        else:
            X_proc = X.copy() if isinstance(X, np.ndarray) else X

        # Create appropriate dataloader
        if isinstance(X_proc, DataFrame):
            dataloader = DataLoader(
                SparkPredictDataset(X_proc),
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X_proc),
                batch_size=self.batch_size,
                shuffle=False
            )

        res_explain: List[NDArray[np.float32]] = []
        masks_dict: Dict[str, List[NDArray[np.float32]]] = {}
        
        for _, data in enumerate(dataloader):
            data = data.to(self.device).float()
            M_explain, masks = self.network.forward_masks(data)
            res_explain.append(M_explain.cpu().detach().numpy())
            
            # Initialize mask lists in dictionary if not already done
            if not masks_dict:
                masks_dict = {key: [] for key in masks.keys()}
            
            # Append each mask to its corresponding list
            for key, mask in masks.items():
                masks_dict[key].append(mask.cpu().detach().numpy())
        
        res_explain_array = np.vstack(res_explain)
        
        # Convert lists of masks to numpy arrays
        final_masks = {key: np.vstack(mask_list) for key, mask_list in masks_dict.items()}
        
        if normalize:
            res_explain_array = (res_explain_array - res_explain_array.min()) / (
                res_explain_array.max() - res_explain_array.min()
            )
        
        return res_explain_array, final_masks

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
        compute_importance=True
    ):
        # Convert input data
        X_train = self._prepare_input(X_train)
        if eval_set is not None:
            new_eval_set = []
            for X_val, y_val in eval_set:
                X_val = self._prepare_input(X_val)
                new_eval_set.append((X_val, y_val))
            eval_set = new_eval_set

        # Store fit parameters
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        
        return super().fit(
            X_train,
            y_train,
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
            compute_importance=compute_importance
        )


class TabNetRegressor(TabModel):
    def __post_init__(self):
        super(TabNetRegressor, self).__post_init__()
        self._task = 'regression'
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = 'mse'

    def _prepare_input(self, X):
        """Convert input data to appropriate format"""
        if isinstance(X, DataFrame):
            self.input_dim = len(X.columns)
            return X.toPandas()
        return X

    def prepare_target(self, y):
        return y

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights
    ):
        if len(y_train.shape) != 2:
            msg = "Targets should be 2D : (n_samples, n_regression) " + \
                  f"but y_train.shape={y_train.shape} given.\n" + \
                  "Use reshape(-1, 1) for single regression."
            raise ValueError(msg)
        self.output_dim = y_train.shape[1]
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict_func(self, outputs):
        return outputs

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score

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
        compute_importance=True
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
            X_train,
            y_train,
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
            compute_importance=compute_importance
        )
