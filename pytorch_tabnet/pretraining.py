import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_tabnet import tab_network
from pytorch_tabnet.utils import (
    create_explain_matrix,
    filter_weights,
    SparsePredictDataset,
    PredictDataset,
    check_input,
    create_group_matrix,
)
from torch.nn.utils import clip_grad_norm_
from pytorch_tabnet.pretraining_utils import (
    create_dataloaders,
    validate_eval_set,
)
from pytorch_tabnet.metrics import (
    UnsupMetricContainer,
    check_metrics,
    UnsupervisedLoss,
)
from pytorch_tabnet.abstract_model import TabModel
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


class TabNetPretrainer(TabModel):
    def __post_init__(self):
        super(TabNetPretrainer, self).__post_init__()
        self._task = "unsupervised"
        self._default_loss = UnsupervisedLoss
        self._default_metric = "unsup_loss_numpy"

    def prepare_target(self, y):
        return y

    def compute_loss(self, output, embedded_x, obf_vars):
        return self.loss_fn(output, embedded_x, obf_vars)

    def update_fit_params(self, weights):
        self.updated_weights = weights
        filter_weights(self.updated_weights)
        self.preds_mapper = None

    def fit(
        self,
        X_train: Union[np.ndarray, DataFrame],
        eval_set: Optional[List[Union[np.ndarray, DataFrame]]] = None,
        eval_name: Optional[List[str]] = None,
        loss_fn: Optional[callable] = None,
        pretraining_ratio: float = 0.5,
        weights: Union[int, np.ndarray] = 0,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
        num_workers: int = 0,
        drop_last: bool = True,
        callbacks: Optional[List[callable]] = None,
        pin_memory: bool = True,
        warm_start: bool = False,
    ):
        """Train a neural network for self-supervised learning.

        Parameters
        ----------
        X_train : Union[np.ndarray, DataFrame]
            Training data, either numpy array or Spark DataFrame
        eval_set : Optional[List[Union[np.ndarray, DataFrame]]]
            List of evaluation sets
        eval_name : Optional[List[str]]
            Names for evaluation sets
        loss_fn : Optional[callable]
            Custom loss function
        pretraining_ratio : float
            Ratio of features to mask for reconstruction
        weights : Union[int, np.ndarray]
            Sample weights
        max_epochs : int
            Maximum training epochs
        patience : int
            Early stopping patience
        batch_size : int
            Training batch size
        virtual_batch_size : int
            Ghost Batch Normalization size
        num_workers : int
            DataLoader workers
        drop_last : bool
            Whether to drop last incomplete batch
        callbacks : Optional[List[callable]]
            Training callbacks
        pin_memory : bool
            Whether to pin memory in DataLoader
        warm_start : bool
            Whether to warm start from previous fit
        """
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory and (self.device.type != "cpu")
        self.pretraining_ratio = pretraining_ratio
        eval_set = eval_set if eval_set else []

        if _HAVE_PYSPARK and isinstance(X_train, DataFrame):
            self.input_dim = len(X_train.columns)
        else:
            self.input_dim = X_train.shape[1]

        self._stop_training = False

        if loss_fn is None:
            self.loss_fn = self._default_loss
        else:
            self.loss_fn = loss_fn

        if _HAVE_PYSPARK and isinstance(X_train, DataFrame):
            # For Spark DataFrames, we don't need to check input format
            pass
        else:
            check_input(X_train)

        self.update_fit_params(weights)

        # Validate eval sets
        eval_names = validate_eval_set(eval_set, eval_name, X_train)
        
        # Construct data loaders
        train_dataloader, valid_dataloaders = self._construct_loaders(X_train, eval_set)

        if not hasattr(self, "network") or not warm_start:
            self._set_network()

        self._update_network_params()
        self._set_metrics(eval_names)
        self._set_optimizer()
        self._set_callbacks(callbacks)

        # Training loop
        self._callback_container.on_train_begin()

        for epoch_idx in range(self.max_epochs):
            self._callback_container.on_epoch_begin(epoch_idx)
            self._train_epoch(train_dataloader)

            for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
                self._predict_epoch(eval_name, valid_dataloader)

            self._callback_container.on_epoch_end(
                epoch_idx, logs=self.history.epoch_metrics
            )

            if self._stop_training:
                break

        self._callback_container.on_train_end()
        self.network.eval()

    def _construct_loaders(
        self,
        X_train: Union[np.ndarray, DataFrame],
        eval_set: Optional[List[Union[np.ndarray, DataFrame]]] = None
    ) -> Tuple[DataLoader, List[DataLoader]]:
        """Construct data loaders for training and evaluation.

        Parameters
        ----------
        X_train : Union[np.ndarray, DataFrame]
            Training data
        eval_set : Optional[List[Union[np.ndarray, DataFrame]]]
            List of evaluation sets

        Returns
        -------
        Tuple[DataLoader, List[DataLoader]]
            Training and evaluation data loaders
        """
        if _HAVE_PYSPARK and isinstance(X_train, DataFrame):
            # Handle Spark DataFrame
            train_dataset = SparkDataset(
                X_train,
                feature_cols=X_train.columns,
            )
            train_dataloader = train_dataset.make_loader(
                batch_size=self.batch_size,
                shuffle=True,
                num_epochs=None  # Infinite for training
            )

            valid_dataloaders = []
            if eval_set:
                for X_val in eval_set:
                    if not isinstance(X_val, DataFrame):
                        raise TypeError("Eval set must be DataFrame when X_train is DataFrame")
                    
                    val_dataset = SparkDataset(
                        X_val,
                        feature_cols=X_val.columns,
                    )
                    valid_dataloaders.append(
                        val_dataset.make_loader(
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_epochs=1
                        )
                    )

            return train_dataloader, valid_dataloaders

        else:
            # Handle numpy arrays
            return create_dataloaders(
                X_train,
                eval_set,
                self.updated_weights,
                self.batch_size,
                self.num_workers,
                self.drop_last,
                self.pin_memory,
            )

    def _train_epoch(self, train_loader):
        """Train one epoch of the network.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        """
        self.network.train()
        
        for batch_idx, data in enumerate(train_loader):
            self._callback_container.on_batch_begin(batch_idx)
            
            if isinstance(data, dict):  # Petastorm loader returns dict
                batch_data = torch.tensor(
                    np.column_stack([data[col].numpy() for col in data.keys()]),
                    device=self.device
                ).float()
            else:
                batch_data = data.to(self.device).float()

            self.optimizer.zero_grad()
            
            output, embedded_x, obf_vars = self.network(batch_data)
            loss = self.compute_loss(output, embedded_x, obf_vars)
            
            loss.backward()
            if self.clip_value:
                clip_grad_norm_(self.network.parameters(), self.clip_value)
                
            self.optimizer.step()
            
            self._callback_container.on_batch_end(batch_idx)
            
        return

    def _predict_epoch(self, name, loader):
        """Predict an epoch and update metrics.
        
        Parameters
        ----------
        name : str
            Name of the validation set
        loader : DataLoader
            Data loader for predictions
        """
        self.network.eval()
        
        list_loss = []
        for batch_idx, data in enumerate(loader):
            if isinstance(data, dict):  # Petastorm loader returns dict
                batch_data = torch.tensor(
                    np.column_stack([data[col].numpy() for col in data.keys()]),
                    device=self.device
                ).float()
            else:
                batch_data = data.to(self.device).float()
                
            output, embedded_x, obf_vars = self.network(batch_data)
            loss = self.compute_loss(output, embedded_x, obf_vars)
            list_loss.append(loss.cpu().detach().numpy())
            
        metrics_logs = {
            metric_name: metric_func(list_loss)
            for metric_name, metric_func in self._metrics.items()
        }
        
        self.network.train()
        self.history.epoch_metrics.update({name + "_" + k: v for k, v in metrics_logs.items()})
        
        return

    def _set_network(self):
        """Setup the network and explain matrix."""
        if not hasattr(self, "pretraining_ratio"):
            self.pretraining_ratio = 0.5
        torch.manual_seed(self.seed)

        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)

        self.network = tab_network.TabNetPretraining(
            self.input_dim,
            pretraining_ratio=self.pretraining_ratio,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            n_shared_decoder=self.n_shared_decoder,
            n_indep_decoder=self.n_indep_decoder,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
            group_attention_matrix=self.group_matrix.to(self.device),
        ).to(self.device)

        self.reducing_matrix = create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )

    def _update_network_params(self):
        self.network.virtual_batch_size = self.virtual_batch_size
        self.network.pretraining_ratio = self.pretraining_ratio

    def _set_metrics(self, eval_names):
        """Set attributes relative to the metrics.

        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.

        """
        metrics = [self._default_metric]

        metrics = check_metrics(metrics)
        # Set metric container for each sets
        self._metric_container_dict = {}
        for name in eval_names:
            self._metric_container_dict.update(
                {name: UnsupMetricContainer(metrics, prefix=f"{name}_")}
            )

        self._metrics = []
        self._metrics_names = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        # Early stopping metric is the last eval metric
        self.early_stopping_metric = (
            self._metrics_names[-1] if len(self._metrics_names) > 0 else None
        )

    def _set_optimizer(self):
        pass

    def _set_callbacks(self, callbacks):
        pass

    def _callback_container(self):
        pass

    def _predict_batch(self, X):
        """
        Predict one batch of data.

        Parameters
        ----------
        X : torch.Tensor
            Owned products

        Returns
        -------
        np.array
            model scores
        """
        X = X.to(self.device).float()
        return self.network(X)

    def stack_batches(self, list_output, list_embedded_x, list_obfuscation):
        output = np.vstack(list_output)
        embedded_x = np.vstack(list_embedded_x)
        obf_vars = np.vstack(list_obfuscation)
        return output, embedded_x, obf_vars

    def predict(self, X):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        predictions : np.array
            Predictions of the regression problem
        """
        self.network.eval()

        if scipy.sparse.issparse(X):
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
        embedded_res = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()
            output, embeded_x, _ = self.network(data)
            predictions = output.cpu().detach().numpy()
            results.append(predictions)
            embedded_res.append(embeded_x.cpu().detach().numpy())
        res_output = np.vstack(results)
        embedded_inputs = np.vstack(embedded_res)
        return res_output, embedded_inputs
