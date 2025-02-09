"""TabNet Spark ML Transformer Implementation.

This module provides Spark ML pipeline integration for TabNet through a hybrid approach
that leverages both Spark's distributed computing capabilities and TabNet's neural architecture.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import threading
import logging
import tempfile
import os

logger = logging.getLogger(__name__)
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, HasPredictionCol, HasLabelCol
from pyspark.ml import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType, ArrayType, StructType, StructField
from pyspark.sql.functions import array, lit, pandas_udf
from pyspark.ml.util import (
    MLReadable, MLWritable, MLReader, MLWriter,
    DefaultParamsReader, DefaultParamsWriter,
    DefaultParamsReadable, DefaultParamsWritable
)
import os
import pickle
import time
import mlflow.pyfunc

from pytorch_tabnet.dataframe import SparkDataFrame, TabNetDataFrame

# Import TabNetClassifier lazily to avoid circular imports
def get_tabnet_classifier():
    """Get TabNetClassifier class lazily to avoid circular imports."""
    from pytorch_tabnet.tab_model import TabNetClassifier
    return TabNetClassifier


class TabNetParams(Params):
    """Common parameters for TabNet Spark ML components."""
    
    def __init__(self):
        super().__init__()
        self._paramMap = {}
        self._defaultParamMap = {}
        
        # Initialize all parameters
        self.n_d = Param(self, "n_d", "Width of the decision prediction layer",
                        typeConverter=TypeConverters.toInt)
        self.n_a = Param(self, "n_a", "Width of the attention embedding for each mask",
                        typeConverter=TypeConverters.toInt)
        self.n_steps = Param(self, "n_steps", "Number of steps in the architecture",
                        typeConverter=TypeConverters.toInt)
        self.gamma = Param(self, "gamma", "Scale for feature updates",
                        typeConverter=TypeConverters.toFloat)
        self.cat_idxs = Param(self, "cat_idxs", "List of categorical feature indices",
                        typeConverter=TypeConverters.toListInt)
        self.cat_dims = Param(self, "cat_dims", "List of categorical feature dimensions",
                        typeConverter=TypeConverters.toListInt)
        self.labelCols = Param(self, "labelCols", "List of label column names",
                        typeConverter=TypeConverters.toListString)
        self.tabnet_model = Param(self, "tabnet_model", "TabNet model instance",
                        typeConverter=TypeConverters.identity)
        
        # Set default values
        self._setDefault(
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            cat_idxs=[],
            cat_dims=[],
            labelCols=["label"],
            tabnet_model=None
        )
    
    def getLabelCols(self) -> List[str]:
        """Get the list of label column names."""
        return self.getOrDefault(self.labelCols)
        
    def _get_tabnet_params(self) -> Dict[str, Any]:
        """Get parameters for TabNetClassifier initialization."""
        return {
            "n_d": self.getOrDefault(self.n_d),
            "n_a": self.getOrDefault(self.n_a),
            "n_steps": self.getOrDefault(self.n_steps),
            "gamma": self.getOrDefault(self.gamma),
            "cat_idxs": self.getOrDefault(self.cat_idxs),
            "cat_dims": self.getOrDefault(self.cat_dims)
        }


class SparkTabNetEstimator(Estimator, TabNetParams, HasInputCol, HasOutputCol):
    """Spark ML Estimator for TabNet.
    
    This estimator provides distributed training capabilities while maintaining
    TabNet's neural architecture advantages through the DataFrame interface.
    
    Args:
        inputCol: Input column name containing features
        outputCol: Output column name for predictions
        **kwargs: Additional parameters passed to TabNetClassifier
    """
    
    def __init__(self, inputCol: str = "features", outputCol: str = "predictions",
                 labelCols: List[str] = None, **kwargs):
        super().__init__()
        
        # Initialize parameters from TabNetParams
        TabNetParams.__init__(self)
        
        # Set input parameters
        if labelCols is None:
            labelCols = ["label"]
            
        self._set(
            inputCol=inputCol,
            outputCol=outputCol,
            labelCols=labelCols,
            **kwargs
        )
        
        self.tabnet = None
    
    def _prepare_data(self, df: TabNetDataFrame) -> TabNetDataFrame:
        """Prepare data using DataFrame interface.
        
        Args:
            df: Input DataFrame wrapped in interface
            
        Returns:
            Prepared DataFrame
        """
        columns = [self.getInputCol()] + self.getLabelCols()
        return df.select(*columns)
    
    def _convert_features(self, features_array: np.ndarray) -> np.ndarray:
        """Convert features array to proper format for TabNet.
        
        Args:
            features_array: Input features array
            
        Returns:
            Converted features array
        """
        # Convert list arrays to proper numpy arrays
        if features_array.dtype == object:
            return np.stack([np.array(x, dtype=np.float32) for x in features_array])
        return features_array.astype(np.float32)
    
    def _fit(self, dataset: DataFrame) -> "SparkTabNetModel":
        """Train the TabNet model using the input dataset.
        
        Args:
            dataset: Input DataFrame containing features and labels
            
        Returns:
            Trained SparkTabNetModel
        """
        # Validate input columns
        if self.getInputCol() not in dataset.columns:
            raise ValueError(f"Input column '{self.getInputCol()}' not found in dataset")
        for label_col in self.getLabelCols():
            if label_col not in dataset.columns:
                raise ValueError(f"Label column '{label_col}' not found in dataset")
            
        # Initialize TabNet model with parameters
        TabNetClassifier = get_tabnet_classifier()
        self.tabnet = TabNetClassifier(**self._get_tabnet_params())
        
        # Use DataFrame interface
        spark_df = SparkDataFrame(dataset)
        prepared_data = self._prepare_data(spark_df)
        
        # Extract features and labels using interface
        features = prepared_data.get_column(self.getInputCol())
        # Currently only using first label column as multi-task learning is not yet supported
        if len(self.getLabelCols()) > 1:
            import warnings
            warnings.warn("Multi-task learning is not yet supported. Using only the first label column.")
        
        labels = prepared_data.get_column(self.getLabelCols()[0])
        
        if len(features) == 0:
            raise ValueError("Empty dataset")
        
        # Convert features to proper format
        features = self._convert_features(features)
        
        # Train the model
        self.tabnet.fit(features, labels)
        
        return SparkTabNetModel(
            tabnet_model=self.tabnet,
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol()
        )


# Define lock types for type checking
LOCK_TYPES = (type(threading.Lock()), type(threading.RLock()))

class SparkTabNetModel(Model, TabNetParams, HasInputCol, HasOutputCol, HasPredictionCol,
                       DefaultParamsReadable, DefaultParamsWritable, mlflow.pyfunc.PythonModel):
    """Spark ML Model for TabNet predictions.
    
    This model provides distributed prediction capabilities using the trained TabNet model
    and DataFrame interface. It implements proper serialization for MLflow integration.
    
    Args:
        tabnet_model: Trained TabNetClassifier instance
        inputCol: Input column name containing features
        outputCol: Output column name for predictions
    """
    
    def __init__(
        self,
        tabnet_model: Any = None,
        inputCol: str = "features",
        outputCol: str = "predictions"
    ):
        """Initialize SparkTabNetModel with proper parameter handling.
        
        Args:
            tabnet_model: Trained TabNetClassifier instance
            inputCol: Input column name containing features
            outputCol: Output column name for predictions
        """
        super().__init__()
        
        # Initialize parameters from TabNetParams
        TabNetParams.__init__(self)
        
        # Initialize parameter maps
        self._paramMap = {}
        self._defaultParamMap = {}
        
        # Initialize parameters
        self._tabnet = None
        
        # Initialize all parameters first
        self.inputCol = Param(self, "inputCol", "Input column name")
        self.outputCol = Param(self, "outputCol", "Output column name for predictions")
        self.tabnet_model = Param(self, "tabnet_model", "TabNet model instance")
        
        # Set default values
        self._setDefault(
            inputCol=inputCol,
            outputCol=outputCol,
            tabnet_model=None,
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            cat_idxs=[],
            cat_dims=[],
            labelCols=["label"]
        )
        
        # Set provided values
        self._set(
            inputCol=inputCol,
            outputCol=outputCol
        )
        
        # Set tabnet model if provided
        if tabnet_model is not None:
            self._set(tabnet_model=tabnet_model)
            self._tabnet = tabnet_model
    
    def __getstate__(self):
        """Get state for pickling."""
        state = {
            "uid": self.uid,
            "inputCol": self.getOrDefault(self.inputCol),
            "outputCol": self.getOrDefault(self.outputCol),
            "_paramMap": {param.name: value for param, value in self._paramMap.items()},
            "_defaultParamMap": {param.name: value for param, value in self._defaultParamMap.items()},
        }
        
        # Save TabNet state
        if self._tabnet is not None:
            state["tabnet"] = self._tabnet
            
        return state
    
    def __setstate__(self, state):
        """Restore state from pickle."""
        # Initialize base classes
        super(SparkTabNetModel, self).__init__()
        TabNetParams.__init__(self)
        
        # Restore uid
        self.uid = state.get("uid", self.uid)
        
        # Initialize parameters with proper ownership
        self.inputCol = Param(self, "inputCol", "Input column name")
        self.outputCol = Param(self, "outputCol", "Output column name for predictions")
        self.tabnet_model = Param(self, "tabnet_model", "TabNet model instance")
        
        # Initialize parameter maps
        self._paramMap = {}
        self._defaultParamMap = {}
        
        # Set default values
        self._setDefault(
            inputCol=state.get("inputCol", "features"),
            outputCol=state.get("outputCol", "predictions"),
            tabnet_model=None,
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            cat_idxs=[],
            cat_dims=[],
            labelCols=["label"]
        )
        
        # Set current values
        self._set(
            inputCol=state.get("inputCol", "features"),
            outputCol=state.get("outputCol", "predictions")
        )
        
        # Restore TabNet model if available
        if "tabnet" in state:
            self._tabnet = state["tabnet"]
            self._set(tabnet_model=state["tabnet"])
            
        # Ensure all parameters have proper ownership
        for param in self.params:
            if param.name in state.get("_paramMap", {}):
                self._paramMap[param] = state["_paramMap"][param.name]
            if param.name in state.get("_defaultParamMap", {}):
                self._defaultParamMap[param] = state["_defaultParamMap"][param.name]
    
    def _convert_features(self, features: List[float]) -> np.ndarray:
        """Convert features to proper format for prediction.
        
        Args:
            features: Input features list
            
        Returns:
            Converted features array
        """
        features_array = np.array(features, dtype=np.float32)
        if len(features_array.shape) == 1:
            features_array = features_array.reshape(1, -1)
        return features_array
    
    def _transform(self, dataset: DataFrame) -> DataFrame:
        """Apply the model to the input dataset.
        
        Args:
            dataset: Input DataFrame containing features
            
        Returns:
            DataFrame with predictions added. If the model is not initialized,
            returns the input DataFrame with an empty predictions column.
        
        Raises:
            ValueError: If the model state is corrupted or invalid
        """
        # Get the model state
        if self._tabnet is None:
            # Return dataset with empty predictions column
            empty_predictions = array([lit(0.0).cast(DoubleType()) for _ in range(2)])  # Default to binary classification
            return dataset.withColumn(self.getOrDefault(self.outputCol), empty_predictions)
            
        # Create a pandas UDF for predictions
        @pandas_udf(ArrayType(DoubleType()))
        def predict_batch(features_series):
            """Vectorized UDF for predictions."""
            # Convert features to numpy array
            features_array = np.stack([np.array(x, dtype=np.float32) for x in features_series])
            
            # Make predictions
            with torch.no_grad():
                predictions = self._tabnet.predict_proba(features_array)
            
            # Convert predictions to pandas Series with lists
            return pd.Series([p.tolist() for p in predictions])
        
        # Apply predictions
        return dataset.withColumn(
            self.getOrDefault(self.outputCol),
            predict_batch(self.getOrDefault(self.inputCol))
        )
    
    @property
    def tabnet(self):
        """Property to access the TabNet model instance.
        
        Returns:
            The TabNet model instance, reconstructing it from saved state if necessary.
        """
        if self._tabnet is None and hasattr(self, '_mlflow_model_info'):
            # Reconstruct TabNet model from saved state
            state = self._mlflow_model_info["tabnet_state"]
            TabNetClassifier = get_tabnet_classifier()
            tabnet = TabNetClassifier(
                n_d=state['n_d'],
                n_a=state['n_a'],
                n_steps=state['n_steps'],
                gamma=state['gamma'],
                cat_idxs=state['cat_idxs'],
                cat_dims=state['cat_dims'],
                input_dim=state['input_dim'],
                output_dim=state['output_dim']
            )
            tabnet.input_dim = state['input_dim']
            tabnet.output_dim = state['output_dim']
            tabnet._initialize_network()
            tabnet.classes_ = state['classes_']
            tabnet.network.load_state_dict(state['state_dict'])
            self._tabnet = tabnet
            self._set(tabnet_model=tabnet)
        return self._tabnet
    
    @tabnet.setter
    def tabnet(self, value):
        """Setter for the TabNet model instance.
        
        Args:
            value: The TabNet model instance to set
        """
        self._tabnet = value
        self._set(tabnet_model=value)

    def predict(self, context, model_input):
        """Predict method required by MLflow's PythonModel interface.
        
        Args:
            context: MLflow model context
            model_input: Input data for prediction
            
        Returns:
            Model predictions as a numpy array.
        """
        import pandas as pd
        import numpy as np
        
        # Convert input to numpy array with proper shape.
        if isinstance(model_input, pd.DataFrame):
            # Extract the column designated by inputCol; this yields an array of lists.
            features = model_input[self.getInputCol()].values
            try:
                # Convert array of lists into a 2D numpy array.
                features = np.stack(features)
            except Exception as e:
                raise ValueError(f"Failed to stack feature arrays: {e}")
        else:
            features = np.array(model_input)
            if features.dtype == object:
                try:
                    features = np.stack(features)
                except Exception as e:
                    raise ValueError(f"Failed to stack feature arrays: {e}")
            
        # Make predictions using the TabNet model.
        with torch.no_grad():
            predictions = self._tabnet.predict_proba(features)
            
        return predictions

    def write(self) -> MLWriter:
        """Returns MLWriter instance for this ML instance."""
        return SparkTabNetModelWriter(self)
    
    @classmethod
    def read(cls) -> MLReader:
        """Returns MLReader instance for this class."""
        return SparkTabNetModelReader(cls)
    
    @classmethod
    def load(cls, path: str) -> "SparkTabNetModel":
        """Load the model from storage with proper parameter handling.
        
        This method loads both the Spark ML metadata and the TabNet model state.
        The TabNet state is expected to be in a pickle file in the model directory.
        
        Args:
            path: Path or URI to load the model from
            
        Returns:
            Loaded SparkTabNetModel instance
            
        Raises:
            ValueError: If the path is invalid
            StorageError: If there are issues accessing or loading the model
        """
        if not path:
            raise ValueError("Path cannot be empty")

        # Import required modules
        try:
            from pytorch_tabnet.storage import get_storage, ModelStorage, StorageError
            import mlflow
        except ImportError as e:
            raise ImportError("Failed to import required modules. MLflow and storage modules are required for model loading.") from e

        try:
            # Get MLflow context
            try:
                mlflow_client = mlflow.tracking.MlflowClient()
                current_run = mlflow.active_run()
                if current_run:
                    run_id = current_run.info.run_id
                else:
                    # Try to extract run ID from path
                    run_id = None
                    if 'mlruns' in path:
                        parts = path.split('mlruns')
                        if len(parts) > 1:
                            run_parts = parts[1].split('/')
                            if len(run_parts) > 1:
                                run_id = run_parts[1]
            except Exception as e:
                logger.warning(f"Failed to get MLflow context: {e}")
                run_id = None

            # Determine appropriate storage URI
            if 'sparkml' in path:
                if run_id:
                    # Use MLflow storage with run ID
                    uri = f"mlflow://{run_id}/{path}"
                else:
                    # In Fabric environment, use MLflow's artifact store
                    try:
                        artifact_uri = mlflow.get_artifact_uri()
                        if artifact_uri:
                            uri = f"{artifact_uri}/{path}"
                        else:
                            uri = f"file://{path}"
                    except Exception as e:
                        logger.warning(f"Failed to get MLflow artifact URI: {e}")
                        uri = f"file://{path}"
            else:
                uri = f"file://{path}"
                
            # Handle MLflow artifact URIs and local paths
            local_path = None
            temp_dir = None

            # Check if this is a direct MLflow URI
            if path.startswith("mlflow://"):
                # Extract run ID and path from MLflow URI
                run_id = path.split("/")[2]
                artifact_path = "/".join(path.split("/")[3:])

                # Create temporary directory for artifact download
                temp_dir = tempfile.mkdtemp()
                local_path = os.path.join(temp_dir, os.path.basename(path))

                # Try direct download first
                try:
                    logger.debug(f"Attempting direct MLflow artifact download from {artifact_path}")
                    mlflow_client.download_artifacts(run_id, artifact_path, temp_dir)
                except Exception as e:
                    logger.warning("Direct MLflow artifact download failed")
                    logger.warning(f"Failed to download MLflow artifact: {e}")
                    # Try fallback with artifact store
                    try:
                        artifact_uri = mlflow.get_artifact_uri()
                        if not artifact_uri:
                            raise ValueError("Could not determine artifact URI")

                        logger.debug(f"Attempting fallback download from artifact store: {artifact_uri}")
                        artifact_path = os.path.join(artifact_uri, path)
                        try:
                            mlflow_client.download_artifacts(run_id, artifact_path, temp_dir)
                        except Exception as store_e:
                            store_error_type = str(type(store_e).__name__)
                            store_error_msg = str(store_e)

                            if "ChecksumException" in store_error_type or "ChecksumException" in store_error_msg:
                                logger.error("Artifact store fallback failed due to checksum error")
                                raise ValueError("Direct MLflow artifact download failed") from store_e
                            raise OSError(f"Artifact store fallback failed: {store_e}")
                    except ValueError as ve:
                        raise ve
                    except Exception as nested_e:
                        raise ValueError("Model path does not exist and artifact store fallback failed")

                if not os.path.exists(local_path):
                    raise ValueError(f"MLflow artifact download completed but file not found at {local_path}")

            # Handle local paths and artifact store fallback
            elif uri.startswith("file://"):
                local_path = uri[len("file://"):]
                if not os.path.exists(local_path):
                    # Try MLflow artifact store fallback
                    try:
                        artifact_uri = mlflow.get_artifact_uri()
                        if not artifact_uri:
                            raise ValueError("Model path does not exist and no artifact store available")

                        # Create temporary directory for artifact download
                        temp_dir = tempfile.mkdtemp()
                        
                        # Get run ID from active run if available
                        if current_run:
                            run_id = current_run.info.run_id
                            logger.debug(f"Using run ID from active run: {run_id}")
                        else:
                            logger.debug("No active run found, attempting to use artifact store without run ID")
                        logger.debug(f"Attempting artifact store fallback for local path: {path}")
                        try:
                            if run_id:
                                mlflow_client.download_artifacts(run_id, path, temp_dir)
                            else:
                                # Try to download without run ID
                                mlflow_client.download_artifacts(None, path, temp_dir)
                        except Exception as download_e:
                            logger.warning(f"Initial download attempt failed: {download_e}")
                            # Try again with artifact store path
                            artifact_path = os.path.join(artifact_uri, path)
                            logger.debug(f"Retrying with artifact store path: {artifact_path}")
                            mlflow_client.download_artifacts(run_id, artifact_path, temp_dir)
                            mlflow_client.download_artifacts(None, path, temp_dir)

                        # Check for model files in various possible locations
                        possible_paths = [
                            os.path.join(temp_dir, "model"),
                            os.path.join(temp_dir, os.path.basename(path)),
                            os.path.join(temp_dir, os.path.basename(path), "model"),
                            temp_dir
                        ]

                        # Try to find a valid model directory
                        local_path = None
                        for p in possible_paths:
                            if os.path.exists(p) and os.path.exists(os.path.join(p, "tabnet_model.pkl")):
                                local_path = p
                                logger.debug(f"Found model files at: {p}")
                                break

                        if local_path is None:
                            logger.error(f"No model files found in downloaded artifacts. Paths checked: {possible_paths}")
                            raise ValueError("Model path does not exist and artifact store fallback failed")
                    except Exception as e:
                        if "ChecksumException" in str(type(e).__name__) or "ChecksumException" in str(e):
                            logger.error("Artifact store fallback failed due to checksum error")
                            raise ValueError("Model path does not exist and artifact store fallback failed") from e
                        raise ValueError("Model path does not exist and artifact store fallback failed") from e

            # Use appropriate storage based on path
            storage = get_storage(f"file://{local_path}" if local_path else uri)
            model_storage = ModelStorage(storage)
            instance = None

            # Create a new instance
            instance = cls()
            instance._setDefault(
                inputCol="features",
                outputCol="predictions",
                tabnet_model=None
            )
            
            # Get SparkContext from SparkSession
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            sc = spark.sparkContext
            
            # Load Spark ML metadata using DefaultParamsReader
            # Use local path for metadata loading to avoid Hadoop filesystem issues
            metadata_path = local_path if local_path else path
            metadata = DefaultParamsReader.loadMetadata(metadata_path, sc)
            instance._resetUid(metadata["uid"])
            DefaultParamsReader.getAndSetParams(instance, metadata)
            
            # Set default parameters if not in metadata
            if instance.getParam("outputCol") not in instance._defaultParamMap:
                instance._defaultParamMap[instance.getParam("outputCol")] = "predictions"
            if instance.getParam("inputCol") not in instance._defaultParamMap:
                instance._defaultParamMap[instance.getParam("inputCol")] = "features"
            
            # Check if TabNet model state file exists
            tabnet_model_path = os.path.join(metadata_path, "tabnet_model.pkl")
            if not os.path.exists(tabnet_model_path):
                raise ValueError(f"TabNet model state file not found at {tabnet_model_path}")
    
            # Load TabNet model using storage abstraction
            TabNetClassifier = get_tabnet_classifier()
            try:
                model = model_storage.load_model(
                    model_class=TabNetClassifier,
                    path=tabnet_model_path
                )
                logger.debug(f"Successfully loaded TabNet model from {tabnet_model_path}")
            except Exception as e:
                raise ValueError(f"Failed to load TabNet model state: {str(e)}")
            
            # Set the loaded model
            instance._tabnet = model
            instance._set(tabnet_model=model)
            
            # Store model info for MLflow compatibility
            instance._mlflow_model_info = {
                "model_type": "SparkTabNetModel",
                "tabnet_state": {
                    "n_d": model.n_d,
                    "n_a": model.n_a,
                    "n_steps": model.n_steps,
                    "gamma": model.gamma,
                    "cat_idxs": model.cat_idxs,
                    "cat_dims": model.cat_dims,
                    "input_dim": model.input_dim,
                    "output_dim": model.output_dim,
                    "classes_": model.classes_,
                    "state_dict": model.network.state_dict()
                }
            }
            
            # Ensure transform method is properly bound
            if hasattr(instance, '_transform'):
                instance._transform = instance._transform.__get__(instance, instance.__class__)
            
            return instance

        except StorageError as e:
            logger.error(f"Storage error while loading model: {e}")
            raise IOError(f"Failed to load model from {path}: {str(e)}")
        except ValueError as e:
            # Re-raise ValueError without wrapping
            logger.error(f"Validation error while loading model: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error while loading model: {e}")
            raise IOError(f"Unexpected error loading model from {path}: {str(e)}")
        finally:
            # Clean up temporary directory if it exists
            if temp_dir and os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
    
class SparkTabNetModelWriter(MLWriter):
    """Custom MLWriter for SparkTabNetModel.
    
    This writer handles both the standard Spark ML metadata and the TabNet-specific state.
    It ensures the TabNet model state is properly saved when using MLflow or direct save.
    """
    
    def __init__(self, instance):
        super().__init__()
        self.instance = instance
    
    def saveImpl(self, path: str) -> None:
        """Save both Spark ML metadata and TabNet state.
        
        Args:
            path: Path to save the model
            
        Raises:
            ValueError: If path is empty or TabNet model is not initialized
        """
        if not path:
            raise ValueError("Path cannot be empty")
            
        try:
            # First, save TabNet model state
            tabnet = self.instance.getOrDefault(self.instance.tabnet_model)
            if tabnet is None:
                raise ValueError("TabNet model has not been initialized")
                
            # Create model state with init_params and MLflow compatibility info
            model_state = {
                'init_params': {
                    'n_d': tabnet.n_d,
                    'n_a': tabnet.n_a,
                    'n_steps': tabnet.n_steps,
                    'gamma': tabnet.gamma,
                    'cat_idxs': tabnet.cat_idxs,
                    'cat_dims': tabnet.cat_dims,
                    'input_dim': tabnet.input_dim,
                    'output_dim': tabnet.output_dim
                },
                'class_attrs': {
                    'classes_': tabnet.classes_,
                    'input_dim': tabnet.input_dim,
                    'output_dim': tabnet.output_dim,
                    '_task': 'classification'
                },
                'network_state': tabnet.network.state_dict() if hasattr(tabnet, 'network') else {},
                '_mlflow_model_info': {
                    'model_type': 'SparkTabNetModel',
                    'tabnet_state': {
                        'n_d': tabnet.n_d,
                        'n_a': tabnet.n_a,
                        'n_steps': tabnet.n_steps,
                        'gamma': tabnet.gamma,
                        'cat_idxs': tabnet.cat_idxs,
                        'cat_dims': tabnet.cat_dims,
                        'input_dim': tabnet.input_dim,
                        'output_dim': tabnet.output_dim,
                        'classes_': tabnet.classes_,
                        'state_dict': tabnet.network.state_dict() if hasattr(tabnet, 'network') else {}
                    }
                }
            }
            
            # Save TabNet state in multiple locations for compatibility
            paths_to_save = [
                os.path.join(path, "tabnet_model.pkl"),  # Direct save location
                os.path.join(path, "stages", "0_SparkTabNetModel", "tabnet_model.pkl"),  # MLflow stage path
            ]
            
            # If this is an MLflow save (path contains 'sparkml/stages')
            if 'sparkml/stages' in path:
                # Get the MLflow model root directory
                mlflow_root = path.split('sparkml/stages')[0]
                paths_to_save.extend([
                    os.path.join(mlflow_root, "tabnet_model.pkl"),
                    os.path.join(mlflow_root, "stages", "0_SparkTabNetModel", "tabnet_model.pkl")
                ])
            
            # Save to all locations with proper error handling
            save_errors = []
            successful_saves = []
            
            for save_path in paths_to_save:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, 'wb') as f:
                        pickle.dump(model_state, f)
                    successful_saves.append(save_path)
                except Exception as e:
                    save_errors.append(f"Failed to save to {save_path}: {str(e)}")
            
            # Ensure at least one save was successful
            if not successful_saves:
                error_msg = "\n".join(save_errors)
                raise IOError(f"Failed to save TabNet state to any location:\n{error_msg}")
            
            # Save parameters using DefaultParamsWriter
            # Store current parameter values
            param_values = {}
            for param in self.instance.params:
                if param in self.instance._paramMap:
                    param_values[param] = self.instance._paramMap[param]
            
            # Temporarily remove tabnet_model to avoid JSON serialization issues
            tabnet_model = self.instance.getOrDefault(self.instance.tabnet_model)
            self.instance._paramMap.pop(self.instance.tabnet_model, None)
            self.instance._defaultParamMap.pop(self.instance.tabnet_model, None)
            
            # Get SparkContext from SparkSession
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            sc = spark.sparkContext
            
            # Create metadata with proper parameter handling
            metadata_path = os.path.join(path, "metadata")
            os.makedirs(metadata_path, exist_ok=True)
            
            # Create metadata content
            metadata = {
                "class": f"{self.instance.__module__}.{self.instance.__class__.__name__}",
                "timestamp": int(time.time() * 1000),
                "sparkVersion": sc.version,
                "uid": self.instance.uid,
                "paramMap": {
                    param.name: param_values[param]
                    for param in self.instance.params
                    if param in param_values and param != self.instance.tabnet_model
                },
                "defaultParamMap": {
                    param.name: self.instance._defaultParamMap[param]
                    for param in self.instance.params
                    if param in self.instance._defaultParamMap and param != self.instance.tabnet_model
                }
            }
            
            # Write metadata directly to avoid Hadoop filesystem issues
            with open(os.path.join(metadata_path, "part-00000"), "w") as f:
                import json
                json.dump(metadata, f)
            
            # Restore parameters
            for param, value in param_values.items():
                self.instance._paramMap[param] = value
            self.instance._set(tabnet_model=tabnet_model)
                    
        except Exception as e:
            raise IOError(f"Failed to save model to {path}: {str(e)}")

    def write(self) -> MLWriter:
        """Returns MLWriter instance for this ML instance."""
        return self


class SparkTabNetModelReader(MLReader):
    """Custom MLReader for SparkTabNetModel."""
    
    def __init__(self, cls):
        super().__init__()
        self.cls = cls
    
    def load(self, path: str) -> "SparkTabNetModel":
        """Load SparkTabNetModel from path."""
        model = self.cls.load(path)
        
        # If the model is wrapped in a PipelineModel (has stages), extract the TabNet model
        if hasattr(model, "stages"):
            for stage in model.stages:
                if isinstance(stage, SparkTabNetModel):
                    # Ensure the stage has the tabnet attribute
                    if not hasattr(stage, "tabnet"):
                        if hasattr(stage, "_mlflow_model_info"):
                            # Reconstruct TabNet model from saved state
                            state = stage._mlflow_model_info["tabnet_state"]
                            TabNetClassifier = get_tabnet_classifier()
                            tabnet = TabNetClassifier(
                                n_d=state['n_d'],
                                n_a=state['n_a'],
                                n_steps=state['n_steps'],
                                gamma=state['gamma'],
                                cat_idxs=state['cat_idxs'],
                                cat_dims=state['cat_dims'],
                                input_dim=state['input_dim'],
                                output_dim=state['output_dim']
                            )
                            tabnet.input_dim = state['input_dim']
                            tabnet.output_dim = state['output_dim']
                            tabnet._initialize_network()
                            tabnet.classes_ = state['classes_']
                            if 'state_dict' in state and state['state_dict']:
                                tabnet.network.load_state_dict(state['state_dict'])
                            stage.tabnet = tabnet
                            stage._paramMap[stage.tabnet_model] = tabnet
                            stage._defaultParamMap[stage.tabnet_model] = tabnet
                    return stage
            
            # If no SparkTabNetModel found, try the last stage
            if model.stages:
                last_stage = model.stages[-1]
                if isinstance(last_stage, SparkTabNetModel):
                    if not hasattr(last_stage, "tabnet"):
                        if hasattr(last_stage, "_mlflow_model_info"):
                            state = last_stage._mlflow_model_info["tabnet_state"]
                            TabNetClassifier = get_tabnet_classifier()
                            tabnet = TabNetClassifier(
                                n_d=state['n_d'],
                                n_a=state['n_a'],
                                n_steps=state['n_steps'],
                                gamma=state['gamma'],
                                cat_idxs=state['cat_idxs'],
                                cat_dims=state['cat_dims'],
                                input_dim=state['input_dim'],
                                output_dim=state['output_dim']
                            )
                            tabnet.input_dim = state['input_dim']
                            tabnet.output_dim = state['output_dim']
                            tabnet._set_network()
                            tabnet.classes_ = state['classes_']
                            if 'state_dict' in state and state['state_dict']:
                                tabnet.network.load_state_dict(state['state_dict'])
                            last_stage.tabnet = tabnet
                            last_stage._paramMap[last_stage.tabnet_model] = tabnet
                            last_stage._defaultParamMap[last_stage.tabnet_model] = tabnet
                    return last_stage
        
        # If not a pipeline model, ensure it's a SparkTabNetModel
        if isinstance(model, SparkTabNetModel):
            if not hasattr(model, "tabnet"):
                if hasattr(model, "_mlflow_model_info"):
                    state = model._mlflow_model_info["tabnet_state"]
                    TabNetClassifier = get_tabnet_classifier()
                    tabnet = TabNetClassifier(
                        n_d=state['n_d'],
                        n_a=state['n_a'],
                        n_steps=state['n_steps'],
                        gamma=state['gamma'],
                        cat_idxs=state['cat_idxs'],
                        cat_dims=state['cat_dims'],
                        input_dim=state['input_dim'],
                        output_dim=state['output_dim']
                    )
                    tabnet.input_dim = state['input_dim']
                    tabnet.output_dim = state['output_dim']
                    tabnet._initialize_network()
                    tabnet.classes_ = state['classes_']
                    if 'state_dict' in state and state['state_dict']:
                        tabnet.network.load_state_dict(state['state_dict'])
                    model.tabnet = tabnet
                    model._paramMap[model.tabnet_model] = tabnet
                    model._defaultParamMap[model.tabnet_model] = tabnet
        
        return model