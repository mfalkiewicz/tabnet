import torch
import numpy as np
from scipy.special import softmax
from pytorch_tabnet.utils import SparsePredictDataset, PredictDataset, filter_weights
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.multiclass_utils import infer_multitask_output, check_output_dim
from torch.utils.data import DataLoader
import scipy
from typing import List, Optional, Union, Dict, Tuple, TYPE_CHECKING

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


class TabNetMultiTaskClassifier(TabModel):
    def __post_init__(self):
        super(TabNetMultiTaskClassifier, self).__post_init__()
        self._task = "classification"
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = "logloss"

    def prepare_target(self, y: Union[np.ndarray, DataFrame, List[str]]) -> Union[np.ndarray, List[str]]:
        """Prepare target data for training.
        
        Args:
            y: Target data, either numpy array for direct values or list of column names for Spark DataFrame
            
        Returns:
            Prepared target data
        """
        if isinstance(y, (list, str)):
            # For Spark DataFrame, return column names
            return y if isinstance(y, list) else [y]
            
        y_mapped = y.copy()
        for task_idx in range(y.shape[1]):
            task_mapper = self.target_mapper[task_idx]
            y_mapped[:, task_idx] = np.vectorize(task_mapper.get)(y[:, task_idx])
        return y_mapped

    def compute_loss(self, y_pred: List[torch.Tensor], y_true: torch.Tensor) -> torch.Tensor:
        """Compute the loss according to network output and targets.

        Args:
            y_pred: List of output tensors from network
            y_true: Target tensor

        Returns:
            Loss tensor
        """
        loss = 0
        y_true = y_true.long()
        if isinstance(self.loss_fn, list):
            # if you specify a different loss for each task
            for task_loss, task_output, task_id in zip(
                self.loss_fn, y_pred, range(len(self.loss_fn))
            ):
                loss += task_loss(task_output, y_true[:, task_id])
        else:
            # same loss function is applied to all tasks
            for task_id, task_output in enumerate(y_pred):
                loss += self.loss_fn(task_output, y_true[:, task_id])

        loss /= len(y_pred)
        return loss

    def stack_batches(self, list_y_true: List[np.ndarray], list_y_score: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Stack batches of predictions and targets.
        
        Args:
            list_y_true: List of target arrays
            list_y_score: List of prediction score arrays
            
        Returns:
            Tuple of stacked targets and scores
        """
        y_true = np.vstack(list_y_true)
        y_score = []
        for i in range(len(self.output_dim)):
            score = np.vstack([x[i] for x in list_y_score])
            score = softmax(score, axis=1)
            y_score.append(score)
        return y_true, y_score

    def update_fit_params(
        self,
        X_train: Union[np.ndarray, DataFrame],
        y_train: Union[np.ndarray, List[str]],
        eval_set: Optional[List[Tuple[Union[np.ndarray, DataFrame], Union[np.ndarray, List[str]]]]],
        weights: Optional[Union[int, Dict[int, float]]] = None
    ) -> None:
        """Update model parameters before fitting.
        
        Args:
            X_train: Training features
            y_train: Training targets or list of target column names for Spark
            eval_set: Optional validation set
            weights: Optional sample weights
        """
        if _HAVE_PYSPARK and isinstance(X_train, DataFrame):
            if not isinstance(y_train, (list, str)):
                raise ValueError("When using Spark DataFrames, y_train must be a column name (str) or list of column names")
                
            # Get unique classes for each task from Spark DataFrame
            y_train_cols = [y_train] if isinstance(y_train, str) else y_train
            train_labels = []
            for col in y_train_cols:
                classes = X_train.select(col).distinct().rdd.map(lambda x: x[0]).collect()
                train_labels.append(np.array(sorted(classes)))
                
            output_dim = [len(classes) for classes in train_labels]
            
            # Verify eval set if provided
            if eval_set:
                for X_val, y_val in eval_set:
                    if not isinstance(X_val, DataFrame):
                        raise TypeError("X_val must be DataFrame when X_train is DataFrame")
                    if not isinstance(y_val, (list, str)):
                        raise TypeError("y_val must be column name(s) when using Spark DataFrames")
                    
                    y_val_cols = [y_val] if isinstance(y_val, str) else y_val
                    for task_idx, col in enumerate(y_val_cols):
                        val_classes = X_val.select(col).distinct().rdd.map(lambda x: x[0]).collect()
                        check_output_dim(train_labels[task_idx], np.array(val_classes))
        else:
            output_dim, train_labels = infer_multitask_output(y_train)
            if eval_set:
                for _, y in eval_set:
                    for task_idx in range(y.shape[1]):
                        check_output_dim(train_labels[task_idx], y[:, task_idx])
                        
        self.output_dim = output_dim
        self.classes_ = train_labels
        self.target_mapper = [
            {class_label: index for index, class_label in enumerate(classes)}
            for classes in self.classes_
        ]
        self.preds_mapper = [
            {str(index): str(class_label) for index, class_label in enumerate(classes)}
            for classes in self.classes_
        ]
        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict(self, X: Union[np.ndarray, DataFrame]) -> List[np.ndarray]:
        """Make predictions on input data.

        Args:
            X: Input features

        Returns:
            List of predictions for each task
        """
        self.network.eval()

        if _HAVE_PYSPARK and isinstance(X, DataFrame):
            dataloader = SparkDataset(
                X,
                feature_cols=X.columns,
            ).make_loader(
                batch_size=self.batch_size,
                shuffle=False,
                num_epochs=1
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

        results = {}
        for data in dataloader:
            if isinstance(data, dict):  # Petastorm loader returns dict
                data = torch.tensor(
                    np.column_stack([data[col].numpy() for col in X.columns]),
                    device=self.device
                ).float()
            else:
                data = data.to(self.device).float()
                
            output, _ = self.network(data)
            predictions = [
                torch.argmax(torch.nn.Softmax(dim=1)(task_output), dim=1)
                .cpu()
                .detach()
                .numpy()
                .reshape(-1)
                for task_output in output
            ]

            for task_idx in range(len(self.output_dim)):
                results[task_idx] = results.get(task_idx, []) + [predictions[task_idx]]
                
        # stack all tasks individually
        results = [np.hstack(task_res) for task_res in results.values()]
        # map all tasks individually
        results = [
            np.vectorize(self.preds_mapper[task_idx].get)(task_res.astype(str))
            for task_idx, task_res in enumerate(results)
        ]
        return results

    def predict_proba(self, X: Union[np.ndarray, DataFrame]) -> List[np.ndarray]:
        """Make probability predictions.

        Args:
            X: Input features

        Returns:
            List of probability arrays for each task
        """
        self.network.eval()

        if _HAVE_PYSPARK and isinstance(X, DataFrame):
            dataloader = SparkDataset(
                X,
                feature_cols=X.columns,
            ).make_loader(
                batch_size=self.batch_size,
                shuffle=False,
                num_epochs=1
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

        results = {}
        for data in dataloader:
            if isinstance(data, dict):  # Petastorm loader returns dict
                data = torch.tensor(
                    np.column_stack([data[col].numpy() for col in X.columns]),
                    device=self.device
                ).float()
            else:
                data = data.to(self.device).float()
                
            output, _ = self.network(data)
            predictions = [
                torch.nn.Softmax(dim=1)(task_output).cpu().detach().numpy()
                for task_output in output
            ]
            for task_idx in range(len(self.output_dim)):
                results[task_idx] = results.get(task_idx, []) + [predictions[task_idx]]
                
        res = [np.vstack(task_res) for task_res in results.values()]
        return res
