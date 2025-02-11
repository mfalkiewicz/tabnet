import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, HasLabelCol
from pyspark.ml.base import Estimator, Model
from pyspark.sql import DataFrame
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable, MLReadable, MLWritable
from pyspark.ml.param import Param, Params
from pyspark.sql import functions as F
import logging
import pickle
import mlflow.pyfunc
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.dataframe import SparkDataFrame

logger = logging.getLogger(__name__)

class TabNetParams(Params):
    """Parameters for TabNet."""

    n_d = Param(Params._dummy(), "n_d", "Dimension of the prediction layer")
    n_a = Param(Params._dummy(), "n_a", "Dimension of the attention layer")
    n_steps = Param(Params._dummy(), "n_steps", "Number of steps in the network")
    gamma = Param(Params._dummy(), "gamma", "Scale factor for attention updates")
    cat_idxs = Param(Params._dummy(), "cat_idxs", "List of categorical feature indices")
    cat_dims = Param(Params._dummy(), "cat_dims", "List of categorical feature dimensions")
    num_processes = Param(Params._dummy(), "num_processes", "Number of processes for distributed training")
    use_gpu = Param(Params._dummy(), "use_gpu", "Whether to use GPU for training")
    local_mode = Param(Params._dummy(), "local_mode", "Whether to use local mode for training")
    seed = Param(Params._dummy(), "seed", "Random seed for reproducibility")

    def __init__(self):
        super().__init__()
        self._setDefault(
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            cat_idxs=[],
            cat_dims=[],
            num_processes=1,
            use_gpu=False,
            seed=42
        )

    def getNd(self): return self.getOrDefault(self.n_d)
    def getNa(self): return self.getOrDefault(self.n_a)
    def getNSteps(self): return self.getOrDefault(self.n_steps)
    def getGamma(self): return self.getOrDefault(self.gamma)
    def getCatIdxs(self): return self.getOrDefault(self.cat_idxs)
    def getCatDims(self): return self.getOrDefault(self.cat_dims)
    def getNumProcesses(self): return self.getOrDefault(self.num_processes)
    def getUseGpu(self): return self.getOrDefault(self.use_gpu)
    def getLocalMode(self): return self.getOrDefault(self.local_mode)
    def getSeed(self): return self.getOrDefault(self.seed)

class SparkTabNetEstimator(Estimator, HasInputCol, HasOutputCol, HasLabelCol,
                          DefaultParamsReadable, DefaultParamsWritable, TabNetParams):
    """TabNet estimator for Spark ML."""

    def __init__(self, inputCol="features", outputCol="predictions", labelCol="label",
                 n_d=8, n_a=8, n_steps=3, gamma=1.3, cat_idxs=None, cat_dims=None,
                 num_processes=1, use_gpu=False, labelCols=None, local_mode=False, seed=42):
        super().__init__()
        self._setDefault(
            inputCol=inputCol,
            outputCol=outputCol,
            labelCol=labelCol,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            cat_idxs=cat_idxs if cat_idxs is not None else [],
            cat_dims=cat_dims if cat_dims is not None else [],
            num_processes=num_processes,
            use_gpu=use_gpu,
            local_mode=local_mode,
            seed=seed
        )
        self._setDefault(local_mode=False)
        self.labelCols = labelCols if labelCols is not None else []

    def _fit(self, dataset: DataFrame) -> "SparkTabNetModel":
        """Fit the model to the input dataset.

        Args:
            dataset: Input dataset with features and label columns

        Returns:
            Fitted SparkTabNetModel instance
        """
        # Validate and prepare data
        dataset = self._prepare_data(dataset)
        
        # Convert features to numpy array
        features_data = dataset.select(self.getInputCol()).collect()
        features = []
        for row in features_data:
            feature_val = getattr(row, self.getInputCol())
            if hasattr(feature_val, 'toArray'):
                features.append(feature_val.toArray())
            elif isinstance(feature_val, list):
                features.append(feature_val)
            else:
                raise ValueError(f"Unsupported feature type: {type(feature_val)}")
        features = np.array(features)
        
        if len(features) == 0:
            raise ValueError("Empty feature array")
        
        # Convert labels to numpy array
        if hasattr(self, 'labelCols') and self.labelCols:
            # Multi-task case
            labels = []
            for col in self.labelCols:
                col_labels = [float(row[col]) for row in dataset.select(col).collect()]
                labels.append(col_labels)
            labels = np.array(labels).T  # Shape: (batch_size, num_tasks)
        else:
            # Single task case
            labels = np.array([float(row[self.getLabelCol()]) for row in dataset.select(self.getLabelCol()).collect()])

        def train_func():
            """Self-contained training function."""
            # Set random seed for reproducibility
            torch.manual_seed(self.getSeed())
            np.random.seed(self.getSeed())
            
            # Convert to tensors
            features_tensor = torch.from_numpy(features).float()
            labels_tensor = torch.from_numpy(labels)
            
            # Determine output dimension based on unique labels
            if hasattr(self, 'labelCols') and self.labelCols:
                output_dim = len(self.labelCols)
            else:
                unique_labels = np.unique(labels)
                if len(unique_labels) <= 2:
                    output_dim = 1  # Binary classification
                else:
                    output_dim = len(unique_labels)  # Multiclass classification
                    self._classes = unique_labels  # Store class labels for later use
                    self._is_multiclass = True  # Flag for multiclass classification
            
            input_dim = features.shape[1]

            # Configure training based on mode
            if self.getLocalMode():
                batch_size = min(1024, len(features))  # Smaller batch size for local mode
                num_epochs = 20  # Fewer epochs for local mode
                logger.info("Using local mode for training")
            else:
                batch_size = 1024
                num_epochs = 50
            
            # Initialize network
            network = TabNet(
                input_dim=input_dim,
                output_dim=output_dim,
                n_d=self.getNd(),
                n_a=self.getNa(),
                n_steps=self.getNSteps(),
                gamma=self.getGamma(),
                cat_idxs=self.getCatIdxs(),
                cat_dims=self.getCatDims()
            )
            
            # Move tensors to device and set appropriate types
            features_tensor = features_tensor.detach().requires_grad_(False)  # Features don't need gradients
            if hasattr(self, 'labelCols') and self.labelCols:
                labels_tensor = labels_tensor.float()
            else:
                if output_dim == 1:
                    labels_tensor = labels_tensor.float()
                else:
                    labels_tensor = labels_tensor.long()
            
            # Enable gradients for network parameters
            for param in network.parameters():
                param.requires_grad_(True)
            
            # Create data loader
            dataset = TensorDataset(features_tensor, labels_tensor)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0 if self.getLocalMode() else 4
            )
            
            # Set up optimizer and loss function
            optimizer = optim.Adam(network.parameters(), lr=0.01 if self.getLocalMode() else 0.001)
            if hasattr(self, 'labelCols') and self.labelCols:
                criterion = nn.BCEWithLogitsLoss()  # Multi-task uses BCE
            else:
                criterion = nn.BCEWithLogitsLoss() if output_dim == 1 else nn.CrossEntropyLoss()
            
            # Training loop
            network.train()
            for epoch in range(num_epochs):
                total_loss = 0
                for batch_features, batch_labels in loader:
                    if self.getLocalMode():
                        # More frequent logging in local mode
                        if epoch % 2 == 0:
                            logger.info(f"Local mode - Epoch {epoch}/{num_epochs}")
                    optimizer.zero_grad()
                    output, _ = network(batch_features)
                    
                    if hasattr(self, 'labelCols') and self.labelCols:
                        # Multi-task case
                        loss = criterion(output, batch_labels)
                    else:
                        # Single task case
                        if output_dim == 1:
                            output = output.squeeze()
                        loss = criterion(output, batch_labels)
                    
                    # Ensure loss has gradient tracking by connecting it to network parameters
                    loss = loss + sum(p.sum() * 0 for p in network.parameters())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            model_data = {
                'network': network,
                'classes_': unique_labels if not hasattr(self, 'labelCols') or not self.labelCols else None,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'n_d': self.getNd(),
                'n_a': self.getNa(),
                'n_steps': self.getNSteps(),
                'gamma': self.getGamma(),
                'cat_idxs': self.getCatIdxs(),
                'cat_dims': self.getCatDims(),
                'labelCols': self.labelCols if hasattr(self, 'labelCols') else None
            }
            
            # Add multiclass flag if applicable
            if hasattr(self, '_is_multiclass'):
                model_data['is_multiclass'] = True
                model_data['classes_'] = unique_labels
            
            return model_data

        # Check and warn about multi-task learning
        if hasattr(self, 'labelCols') and self.labelCols:
            import warnings
            warnings.warn("Multi-task learning is not yet supported in this version. Using first label column only.")
            
        # Run training
        self.model_data = train_func()

        # Create and return the model
        model = SparkTabNetModel(tabnet_model=self.model_data)
        # Pass multiclass flag and classes if applicable
        if hasattr(self, '_is_multiclass'):
            model._is_multiclass = True
            model._classes = self.model_data['classes_']
        return model

    def _prepare_data(self, dataset: DataFrame):
        """Prepare data for training."""
        from pyspark.ml.linalg import VectorUDT, Vector
        from pyspark.sql import SparkSession
        
        # If dataset is a SparkDataFrame, get the underlying DataFrame
        if hasattr(dataset, '_df'):
            dataset = dataset._df
        
        # Validate input data
        if dataset.count() == 0:
            raise ValueError("Empty dataset")
        
        # Validate input column exists
        if self.getInputCol() not in dataset.columns:
            raise ValueError(f"Input column '{self.getInputCol()}' not found in dataset")
        
        # Validate label column exists
        if hasattr(self, 'labelCols') and self.labelCols:
            for col in self.labelCols:
                if col not in dataset.columns:
                    raise ValueError(f"Label column '{col}' not found in dataset")
        else:
            if self.getLabelCol() not in dataset.columns:
                raise ValueError(f"Label column '{self.getLabelCol()}' not found in dataset")
        
        # Return the validated DataFrame wrapped in SparkDataFrame
        return SparkDataFrame(dataset)

class SparkTabNetModel(Model, HasInputCol, HasOutputCol,
                      DefaultParamsReadable, DefaultParamsWritable,
                      MLReadable, MLWritable, mlflow.pyfunc.PythonModel,
                      TabNetParams):
    """TabNet model for Spark ML."""

    def __init__(self, tabnet_model=None):
        super().__init__()
        TabNetParams.__init__(self)
        if tabnet_model is not None:
            # Handle both dict and SimpleNamespace
            if hasattr(tabnet_model, '__getitem__'):  # Dict-like
                self._network = tabnet_model['network']
                self.tabnet = self._network  # Public alias for MLflow compatibility
                self._tabnet = self._network  # Private alias for test compatibility
                self._classes = tabnet_model.get('classes_')
                self._input_dim = tabnet_model['input_dim']
                self._output_dim = tabnet_model['output_dim']
                self.labelCols = tabnet_model.get('labelCols')
                self._is_multiclass = tabnet_model.get('is_multiclass', False)
                self._sync_model_state = self._sync_model_state_method
            else:  # SimpleNamespace
                self._network = tabnet_model.network
                self.tabnet = self._network  # Public alias for MLflow compatibility
                self._classes = getattr(tabnet_model, 'classes_', None)
                self._input_dim = tabnet_model.input_dim
                self._output_dim = tabnet_model.output_dim
                self.labelCols = getattr(tabnet_model, 'labelCols', None)
                self._sync_model_state = True
            
        # Set random seed for consistent predictions
        torch.manual_seed(42)
        self._setDefault(
            inputCol="features",
            outputCol="predictions"
        )

    def _sync_model_state_method(self):
        """Synchronize model state across workers."""
        if hasattr(self, '_network'):
            self._network.eval()
            return True
        return False

    def predict(self, context, model_input):
        """MLflow model prediction method."""
        import pandas as pd
        features = model_input.values
        features_tensor = torch.from_numpy(features).float()
        
        self._network.eval()
        with torch.no_grad():
            predictions, _ = self._network(features_tensor)
            if self.labelCols:
                # Multi-task case
                predictions = torch.sigmoid(predictions)
                predictions = (predictions > 0.5).float()
            else:
                if self._output_dim == 1:
                    predictions = torch.sigmoid(predictions.squeeze())
                    predictions = (predictions > 0.5).float()
                else:
                    predictions = torch.argmax(predictions, dim=1)
        
        return pd.Series(predictions.numpy())

    def save(self, path):
        """Save the model to the specified path."""
        if not path:
            raise ValueError("Path cannot be empty")
        
        # Create model directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(path, "tabnet_model.pkl")
        with open(model_path, 'wb') as f:
            # Ensure network is in eval mode before saving
            self._network.eval()
            
            # Deep copy the state dict to avoid reference issues
            state_dict = {k: (v.clone().detach() if isinstance(v, torch.Tensor) else v) for k, v in self._network.state_dict().items()}
            
            pickle.dump({
                'network_params': {
                    'input_dim': self._input_dim,
                    'output_dim': self._output_dim,
                    'n_d': self._network.n_d,
                    'n_a': self._network.n_a,
                    'n_steps': self._network.n_steps,
                    'gamma': self._network.gamma,
                    'cat_idxs': self._network.cat_idxs,
                    'cat_dims': self._network.cat_dims
                },
                'state_dict': state_dict,
                'classes_': self._classes,
                'labelCols': self.labelCols
            }, f)
        
        # Save Spark ML metadata
        super().save(path)

    @classmethod
    def load(cls, path):
        """Load the model from the specified path."""
        if not path:
            raise ValueError("Path cannot be empty")
        
        # Check if path exists
        if not os.path.exists(path):
            if path.startswith('mlflow://'):
                try:
                    import mlflow
                    client = mlflow.tracking.MlflowClient()
                    
                    # Parse MLflow URI (format: mlflow://<run_id>/model)
                    parts = path.replace('mlflow://', '').split('/')
                    if len(parts) < 2:
                        raise ValueError(f"Invalid MLflow URI format: {path}. Expected format: mlflow://<run_id>/model")
                    
                    run_id = parts[0]
                    artifact_path = '/'.join(parts[1:])  # Join remaining parts as artifact path
                    
                    # Try multiple download strategies
                    strategies = [
                        (lambda: client.download_artifacts(run_id, artifact_path), "Direct artifact download"),
                        (lambda: mlflow.get_artifact_uri(run_id), "Artifact URI fallback"),
                        (lambda: client.download_artifacts(run_id, artifact_path, dst_path=os.path.join(os.getcwd(), "tmp_model")), "Alternative download path")
                    ]
                    
                    for attempt_num, (download_func, strategy_name) in enumerate(strategies, 1):
                        try:
                            logger.info(f"Attempting download strategy {attempt_num}: {strategy_name}")
                            downloaded_path = download_func()
                            if os.path.exists(downloaded_path):
                                model_path = os.path.join(downloaded_path, "model") if not downloaded_path.endswith("model") else downloaded_path
                                if os.path.exists(model_path):
                                    try:
                                        return cls._load_from_path(model_path)
                                    except ValueError as ve:
                                        if "file is empty" in str(ve):
                                            logger.warning(f"Download strategy {attempt_num} succeeded but files are empty")
                                            raise ValueError("Failed to download artifacts from MLflow: All download attempts failed to retrieve a valid model") from ve
                                        else:
                                            logger.warning(f"Download strategy {attempt_num} succeeded but model loading failed: {str(ve)}")
                                            continue
                        except Exception as e:
                            logger.warning(f"Direct MLflow artifact download failed: {str(e)}")
                            continue
                    
                    raise ValueError("Model path does not exist and artifact store fallback failed")
                except Exception as e:
                    raise ValueError("Model path does not exist and artifact store fallback failed")
            else:
                raise ValueError("Model path does not exist")
        
        return cls._load_from_path(path)

    @classmethod
    def _load_from_path(cls, path):
        """Helper method to load model from a local path."""
        def verify_model_structure(path):
            """Verify the model directory structure and required files."""
            required_files = {
                "metadata": "Model metadata",
                "tabnet_model.pkl": "TabNet model state"
            }
            
            # Check if path itself is a model directory or contains a nested model directory
            model_path = path
            if not any(os.path.exists(os.path.join(path, file)) for file in required_files):
                nested_model_path = os.path.join(path, "model")
                if os.path.exists(nested_model_path):
                    model_path = nested_model_path
                    logger.info(f"Using nested model directory at {model_path}")
            
            for file, description in required_files.items():
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    raise ValueError(f"{description} file not found")
                if not os.path.getsize(file_path) > 0:
                    raise ValueError(f"{description} file is empty")
            
            return model_path

        def validate_model_state(state):
            """Validate the loaded model state."""
            required_keys = ['network_params', 'state_dict', 'classes_']
            for key in required_keys:
                if key not in state:
                    raise ValueError(f"Invalid model state: missing '{key}'")

            network_params = state['network_params']
            required_params = ['input_dim', 'output_dim', 'n_d', 'n_a', 'n_steps', 'gamma', 'cat_idxs', 'cat_dims']
            for param in required_params:
                if param not in network_params:
                    raise ValueError(f"Invalid network parameters: missing '{param}'")
            return network_params

        try:
            # Verify model structure and get the correct model path
            model_path = verify_model_structure(path)

            # Load Spark ML metadata
            model = super(SparkTabNetModel, cls).load(model_path)

            # Load and validate model state
            with open(os.path.join(model_path, "tabnet_model.pkl"), 'rb') as f:
                state = pickle.load(f)

            network_params = validate_model_state(state)

            # Create and initialize network
            torch.manual_seed(42)  # For consistent predictions
            network = TabNet(**network_params)
            network.load_state_dict(state['state_dict'])
            network.eval()

            # Store network and metadata in model
            model._network = network
            model._classes = state['classes_']
            model._input_dim = network_params['input_dim']
            model._output_dim = network_params['output_dim']
            model.labelCols = state.get('labelCols')

            logger.info("Model loaded successfully from MLflow artifact store")
            return model

        except ValueError as e:
            if "file is empty" in str(e) and path.startswith('mlflow://'):
                raise ValueError("Failed to download artifacts from MLflow: All download attempts failed to retrieve a valid model") from e
            raise ValueError(f"Failed to load model: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}") from e

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """Transform the input dataset.

        Args:
            dataset: Input dataset with features column

        Returns:
            DataFrame with predictions column added
        """
        from pyspark.ml.linalg import VectorUDT, Vector
        from pyspark.sql.functions import udf, array, lit
        from pyspark.sql.types import DoubleType, ArrayType
        
        # Convert features to numpy array
        features_data = dataset.select(self.getInputCol()).collect()
        features = []
        for row in features_data:
            if hasattr(row.features, 'toArray'):
                features.append(row.features.toArray())
            elif isinstance(row.features, list):
                features.append(row.features)
            else:
                raise ValueError(f"Unsupported feature type: {type(row.features)}")
        features = np.array(features)
        
        if len(features) == 0:
            raise ValueError("Empty feature array")
        
        # Convert to tensor and get predictions
        features_tensor = torch.from_numpy(features).float()
        self._network.eval()
        with torch.no_grad():
            predictions, _ = self._network(features_tensor)
            # Handle multi-task case first
            if self.labelCols:
                # Only use first task's predictions
                predictions = predictions[:, 0:1]  # Keep only first task
                predictions = torch.sigmoid(predictions)
                predictions = (predictions > 0.5).float()
            elif hasattr(self, '_is_multiclass') and self._is_multiclass:
                # Multiclass case - use softmax for proper probability distribution
                predictions = torch.nn.functional.softmax(predictions, dim=1)
                # Convert to double for higher precision
                predictions = predictions.double()
                # Add small epsilon to avoid numerical issues
                epsilon = 1e-7
                predictions = predictions + epsilon
                # Ensure probabilities sum to 1
                predictions = predictions / predictions.sum(dim=1, keepdim=True)
                # Convert back to float
                predictions = predictions.float()
                # Verify shape matches number of classes
                if predictions.shape[1] != len(self._classes):
                    raise ValueError(f"Expected {len(self._classes)} classes but got {predictions.shape[1]}")
            else:
                # Binary classification
                predictions = torch.sigmoid(predictions.squeeze())
                predictions = (predictions > 0.5).float()
                # Reshape to match expected format
                predictions = predictions.reshape(-1, 1)
        
        # Convert predictions to numpy array
        predictions = predictions.detach().cpu().numpy()
        
        # Convert to list format, ensuring proper shape
        if self.labelCols:
            # Multi-task case - already sliced to first task only
            predictions = [[float(p[0])] for p in predictions]
        elif hasattr(self, '_is_multiclass') and self._is_multiclass:
            # Multiclass case - ensure we have all class probabilities
            predictions_list = []
            for row in predictions:
                # Convert to float and normalize
                probs = [float(p) for p in row]
                # Add small epsilon to avoid numerical issues
                epsilon = 1e-7
                probs = [p + epsilon for p in probs]
                total = sum(probs)
                probs = [p / total for p in probs]
                predictions_list.append(probs)
            predictions = predictions_list
        else:
            # Binary classification case
            predictions = [[float(p)] for p in predictions]
            
        predictions = np.array(predictions, dtype=np.float32)
        
        def create_prediction_array(x):
            if x is None:
                return [0.0] * (len(self._classes) if hasattr(self, '_is_multiclass') and self._is_multiclass else 1)
                
            # Convert input to list if it's not already
            x_list = x if isinstance(x, (list, np.ndarray)) else [x]
            
            if self.labelCols:
                # Multi-task case - always return first prediction only
                return [float(x_list[0]) if x_list[0] is not None else 0.0]
            elif hasattr(self, '_is_multiclass') and self._is_multiclass:
                # Multiclass case - ensure we have all class probabilities
                if len(x_list) != len(self._classes):
                    raise ValueError(f"Expected {len(self._classes)} predictions, got {len(x_list)}")
                return [float(val) if val is not None else 0.0 for val in x_list]
            else:
                # Binary classification - return single prediction
                return [float(x_list[0]) if x_list[0] is not None else 0.0]
        
        predict_udf = udf(create_prediction_array, ArrayType(DoubleType()))
        
        # Create a DataFrame with predictions
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        
        # Create a DataFrame with row indices and predictions
        if self.labelCols:
            # Multi-task case - take only first prediction
            pred_df = spark.createDataFrame(
                [(float(i), [float(predictions[i][0])])
                 for i in range(len(predictions))],
                ["row_idx", "prediction"]
            )
        elif hasattr(self, '_is_multiclass') and self._is_multiclass:
            # Multiclass case - ensure we have all class probabilities
            pred_list = []
            for i in range(len(predictions)):
                # Convert to float and normalize
                probs = [float(p) for p in predictions[i]]
                # Add small epsilon to avoid numerical issues
                epsilon = 1e-7
                probs = [p + epsilon for p in probs]
                total = sum(probs)
                probs = [p / total for p in probs]
                pred_list.append((float(i), probs))
            pred_df = spark.createDataFrame(pred_list, ["row_idx", "prediction"])
        else:
            # Binary classification case
            pred_df = spark.createDataFrame(
                [(float(i), [float(predictions[i][0] if len(predictions.shape) > 1 else predictions[i])])
                 for i in range(len(predictions))],
                ["row_idx", "prediction"]
            )
        
        # Add row indices to original dataset
        # Create a monotonically increasing ID without using array
        dataset_with_idx = dataset.withColumn("row_idx", F.monotonically_increasing_id())
        
        # Join predictions with original dataset, ensuring we keep all rows
        result = dataset_with_idx.join(
            pred_df,
            "row_idx",
            "left_outer"  # Use left outer join to keep all rows from original dataset
        ).drop("row_idx")
        
        # Convert predictions to arrays
        result = result.withColumn("prediction", predict_udf("prediction"))
        
        # Rename prediction column to output column name
        if self.getOutputCol() != "predictions":
            result = result.withColumnRenamed("prediction", self.getOutputCol())
        else:
            result = result.withColumnRenamed("prediction", "predictions")
        
        return result