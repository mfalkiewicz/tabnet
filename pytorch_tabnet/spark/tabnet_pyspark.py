"""TabNet PySpark integration using TorchDistributor.

This module provides distributed training capabilities for TabNet using PySpark's
TorchDistributor, with efficient data loading and MLflow integration.
"""

from typing import List, Dict, Any, Optional
import os
import json
import logging
import mlflow
import mlflow.mleap
import torch
import numpy as np
import time
import tempfile
import shutil
from torchinfo import summary
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.ml.param.shared import (
    HasInputCol, HasOutputCol, HasLabelCol,
    Param, Params, TypeConverters
)
from pyspark.ml.base import Estimator, Model
from pyspark.ml.util import MLReadable, MLWritable, MLWriter, MLReader
from typing import TypeVar, overload, TYPE_CHECKING
from typing_extensions import override

if TYPE_CHECKING:
    RL = TypeVar("RL", bound="TabNetModel")
from pyspark.sql import DataFrame
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, FloatType
from pyspark.ml.linalg import VectorUDT, DenseVector

from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.spark.provider import SparkDataProvider
from pytorch_tabnet.spark.factory import create_tabnet_model, validate_loaded_model


class TabNetParams(HasInputCol, HasOutputCol, HasLabelCol):
    """Shared parameters for TabNet estimator and model."""
    
    n_d = Param(Params._dummy(), "n_d", "Width of the decision prediction layer",
                typeConverter=TypeConverters.toInt)
    n_a = Param(Params._dummy(), "n_a", "Width of the attention embedding",
                typeConverter=TypeConverters.toInt)
    n_steps = Param(Params._dummy(), "n_steps", "Number of steps in the architecture",
                   typeConverter=TypeConverters.toInt)
    gamma = Param(Params._dummy(), "gamma", "Scale factor for attention updates",
                 typeConverter=TypeConverters.toFloat)
    n_independent = Param(Params._dummy(), "n_independent", 
                         "Number of independent GLU layers",
                         typeConverter=TypeConverters.toInt)
    n_shared = Param(Params._dummy(), "n_shared",
                    "Number of shared GLU layers",
                    typeConverter=TypeConverters.toInt)
    virtual_batch_size = Param(Params._dummy(), "virtual_batch_size",
                             "Size of virtual batches",
                             typeConverter=TypeConverters.toInt)
    momentum = Param(Params._dummy(), "momentum",
                    "Momentum for batch normalization",
                    typeConverter=TypeConverters.toFloat)
    mask_type = Param(Params._dummy(), "mask_type",
                     "Type of mask to use (sparsemax or entmax)",
                     typeConverter=TypeConverters.toString)
    epochs = Param(Params._dummy(), "epochs",
                  "Number of training epochs",
                  typeConverter=TypeConverters.toInt)
    learning_rate = Param(Params._dummy(), "learning_rate",
                         "Learning rate for optimizer",
                         typeConverter=TypeConverters.toFloat)
    batch_size = Param(Params._dummy(), "batch_size",
                      "Training batch size",
                      typeConverter=TypeConverters.toInt)
    
    def __init__(self):
        super().__init__()
        self._setDefault(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            virtual_batch_size=128,
            momentum=0.02,
            mask_type="sparsemax",
            epochs=10,
            learning_rate=0.001,
            batch_size=1024
        )
    
    def _validate_params(self):
        """Validate parameter values."""
        if self.getNSteps() <= 0:
            raise ValueError("n_steps must be positive")
        if self.getNIndependent() == 0 and self.getNShared() == 0:
            raise ValueError("n_independent and n_shared cannot both be zero")
        if self.getMaskType() not in ["sparsemax", "entmax"]:
            raise ValueError("mask_type must be either 'sparsemax' or 'entmax'")
    
    def _get_model_params(self) -> Dict[str, Any]:
        """Get parameters for TabNet model initialization."""
        self._validate_params()  # Validate parameters before returning
        return {
            "n_d": self.getNd(),
            "n_a": self.getNa(),
            "n_steps": self.getNSteps(),
            "gamma": self.getGamma(),
            "n_independent": self.getNIndependent(),
            "n_shared": self.getNShared(),
            "virtual_batch_size": self.getVirtualBatchSize(),
            "momentum": self.getMomentum(),
            "mask_type": self.getMaskType()
        }
    
    # Getters and setters for all parameters
    def getNd(self) -> int:
        return self.getOrDefault(self.n_d)
    
    def setNd(self, value: int) -> "TabNetParams":
        return self._set(n_d=value)
    
    def getNa(self) -> int:
        return self.getOrDefault(self.n_a)
    
    def setNa(self, value: int) -> "TabNetParams":
        return self._set(n_a=value)
    
    def getNSteps(self) -> int:
        return self.getOrDefault(self.n_steps)
    
    def setNSteps(self, value: int) -> "TabNetParams":
        if value <= 0:
            raise ValueError("n_steps must be positive")
        return self._set(n_steps=value)
    
    def getGamma(self) -> float:
        return self.getOrDefault(self.gamma)
    
    def setGamma(self, value: float) -> "TabNetParams":
        return self._set(gamma=value)
    
    def getNIndependent(self) -> int:
        return self.getOrDefault(self.n_independent)
    
    def setNIndependent(self, value: int) -> "TabNetParams":
        if value == 0 and self.getNShared() == 0:
            raise ValueError("n_independent and n_shared cannot both be zero")
        return self._set(n_independent=value)
    
    def getNShared(self) -> int:
        return self.getOrDefault(self.n_shared)
    
    def setNShared(self, value: int) -> "TabNetParams":
        if value == 0 and self.getNIndependent() == 0:
            raise ValueError("n_independent and n_shared cannot both be zero")
        return self._set(n_shared=value)
    
    def getVirtualBatchSize(self) -> int:
        return self.getOrDefault(self.virtual_batch_size)
    
    def setVirtualBatchSize(self, value: int) -> "TabNetParams":
        return self._set(virtual_batch_size=value)
    
    def getMomentum(self) -> float:
        return self.getOrDefault(self.momentum)
    
    def setMomentum(self, value: float) -> "TabNetParams":
        return self._set(momentum=value)
    
    def getMaskType(self) -> str:
        return self.getOrDefault(self.mask_type)
    
    def setMaskType(self, value: str) -> "TabNetParams":
        if value not in ["sparsemax", "entmax"]:
            raise ValueError("mask_type must be either 'sparsemax' or 'entmax'")
        return self._set(mask_type=value)
    
    def getEpochs(self) -> int:
        return self.getOrDefault(self.epochs)
    
    def setEpochs(self, value: int) -> "TabNetParams":
        if value <= 0:
            raise ValueError("epochs must be positive")
        return self._set(epochs=value)
    
    def getLearningRate(self) -> float:
        return self.getOrDefault(self.learning_rate)
    
    def setLearningRate(self, value: float) -> "TabNetParams":
        if value <= 0:
            raise ValueError("learning_rate must be positive")
        return self._set(learning_rate=value)
    
    def getBatchSize(self) -> int:
        return self.getOrDefault(self.batch_size)
    
    def setBatchSize(self, value: int) -> "TabNetParams":
        if value <= 0:
            raise ValueError("batch_size must be positive")
        return self._set(batch_size=value)


class TabNetEstimator(Estimator, TabNetParams, MLReadable, MLWritable):
    """PySpark Estimator for distributed TabNet training."""
    
    def __init__(
        self,
        inputCol: str = "features",
        outputCol: str = "prediction",
        labelCol: str = "label",
        **kwargs
    ):
        super().__init__()
        self._set(inputCol=inputCol, outputCol=outputCol, labelCol=labelCol)
        self.setParams(**kwargs)
        self.num_processes = kwargs.get("num_processes", 2)
        self.use_gpu = kwargs.get("use_gpu", False)
    
    def setParams(self, **kwargs) -> "TabNetEstimator":
        """Set parameters for the estimator."""
        for param, value in kwargs.items():
            if hasattr(self, f"set{param[0].upper()}{param[1:]}"):
                getattr(self, f"set{param[0].upper()}{param[1:]}")(value)
        return self
    
    def _fit(self, dataset: DataFrame) -> "TabNetModel":
        """Train TabNet model using TorchDistributor."""
        from pyspark.ml.torch.distributor import TorchDistributor
        
        # Check for empty dataset
        if dataset.rdd.isEmpty():
            raise ValueError("Cannot train on empty dataset")
        
        # Get feature dimension and number of classes
        first_row = dataset.select(self.getInputCol()).first()
        if first_row is None or first_row[0] is None:
            raise ValueError("Input features cannot be None")
        input_dim = len(first_row[0])
        n_classes = dataset.select(self.getLabelCol()).distinct().count()
        output_dim = int(n_classes)
        
        # Get feature dimension and number of classes
        first_row = dataset.select(self.getInputCol()).first()
        if first_row is None or first_row[0] is None:
            raise ValueError("Input features cannot be None")
        input_dim = len(first_row[0].toArray())  # Ensure we get the actual vector length
        n_classes = dataset.select(self.getLabelCol()).distinct().count()
        output_dim = int(n_classes)

        # Create model parameters
        model_params = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            **self._get_model_params()
        }
        
        # Convert Spark DataFrame to pandas DataFrame for training
        pandas_df = dataset.select(self.getInputCol(), self.getLabelCol()).toPandas()
        
        def train_tabnet(pandas_data):
            """Training function for TorchDistributor."""
            # Set random seeds for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)
            
            # Create TabNet model using factory function
            model = create_tabnet_model({
                "input_dim": input_dim,
                "output_dim": output_dim,
                **model_params
            })
            
            # Convert pandas data to numpy arrays
            features = np.vstack([row.toArray() for row in pandas_data[self.getInputCol()]])
            labels = pandas_data[self.getLabelCol()].values
            
            # Create model summary string
            model_summary = str(summary(
                model,
                input_size=(self.getBatchSize(), input_dim),
                verbose=0
            ))
            
            # Train model with configured parameters
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.getLearningRate(),
                eps=1e-7  # Prevent division by zero
            )
            criterion = torch.nn.CrossEntropyLoss()
            
            # Convert data to tensors
            features_tensor = torch.from_numpy(features).float()
            labels_tensor = torch.from_numpy(labels).long()
            
            # Create dataset and dataloader
            dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.getBatchSize(),
                shuffle=True
            )
            
            # Train using batched data
            for epoch in range(self.getEpochs()):
                epoch_loss = 0.0
                epoch_acc = 0.0
                n_batches = 0
                
                for batch_features, batch_labels in dataloader:
                    # Zero gradients and perform forward pass
                    optimizer.zero_grad()
                    outputs, _ = model(batch_features)
                    
                    # Calculate loss and backpropagate
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate metrics
                    with torch.no_grad():
                        predictions = outputs.argmax(dim=1)
                        accuracy = (predictions == batch_labels).float().mean()
                    
                    # Update epoch metrics
                    epoch_loss += loss.item()
                    epoch_acc += accuracy.item()
                    n_batches += 1
                
                # Calculate and log average metrics
                avg_loss = epoch_loss / max(n_batches, 1)  # Avoid division by zero
                avg_acc = epoch_acc / max(n_batches, 1)
                
                # Store metrics for later logging
                metrics = {
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "accuracy": avg_acc
                }
            
            # Return model state dict and metrics
            return {
                "state_dict": model.state_dict(),
                "metrics": metrics,
                "model_summary": model_summary
            }
        
        # Train distributed
        distributor = TorchDistributor(
            num_processes=self.num_processes,
            use_gpu=self.use_gpu,
            local_mode=True
        )
        # Run distributed training
        result = distributor.run(train_tabnet, pandas_df)
        
        # Log metrics and artifacts on driver
        active_run = mlflow.active_run()
        if active_run is None:
            with mlflow.start_run(nested=True) as run:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
                    f.write(result["model_summary"])
                    mlflow.log_artifact(f.name, "model_summary.txt")
                mlflow.log_params(model_params)
                mlflow.log_metrics(result["metrics"])
        else:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
                f.write(result["model_summary"])
                mlflow.log_artifact(f.name, "model_summary.txt")
            mlflow.log_params(model_params)
            mlflow.log_metrics(result["metrics"])
        
        # Create and return model
        return self._create_model(result["state_dict"], input_dim, output_dim)
    
    def _create_model(
        self,
        state_dict: Dict[str, torch.Tensor],
        input_dim: int,
        output_dim: int
    ) -> "TabNetModel":
        """Create TabNetModel with trained parameters."""
        model = TabNetModel(
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol(),
            input_dim=input_dim,
            output_dim=output_dim
        )
        model.setParams(**self._get_model_params())
        model._torch_model = create_tabnet_model({
            "input_dim": input_dim,
            "output_dim": output_dim,
            **self._get_model_params()
        })
        model._torch_model.load_state_dict(state_dict)
        return model


class TabNetModel(Model, TabNetParams, MLReadable, MLWritable):
    """PySpark Model for TabNet predictions."""
    
    def __init__(
        self,
        inputCol: str = "features",
        outputCol: str = "prediction",
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None
    ):
        super().__init__()
        self._set(inputCol=inputCol, outputCol=outputCol)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._torch_model = None

    def setParams(self, **kwargs) -> "TabNetModel":
        """Set parameters for the model.
        
        Args:
            **kwargs: Parameter name-value pairs to set
            
        Returns:
            self: The model instance
        """
        for param, value in kwargs.items():
            if hasattr(self, f"set{param[0].upper()}{param[1:]}"):
                getattr(self, f"set{param[0].upper()}{param[1:]}")(value)
        return self

    @classmethod
    def read(cls) -> MLReader["TabNetModel"]:
        """Returns an MLReader instance for this class."""
        from pytorch_tabnet.spark.persistence import TabNetModelReader
        return TabNetModelReader()
    
    def write(self) -> MLWriter:
        """Returns an MLWriter instance for this ML instance."""
        from pytorch_tabnet.spark.persistence import TabNetModelWriter
        return TabNetModelWriter(self)

    def save(self, path: str) -> None:
        """Save model to disk with MLflow integration."""
        # Save using MLWriter with improved persistence
        self.write().save(path)
        
        # Ensure MLmodel file exists in the root directory
        mlmodel_path = os.path.join(path, "MLmodel")
        if not os.path.exists(mlmodel_path):
            metadata = {
                "class": "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel",
                "timestamp": int(time.time() * 1000),
                "sparkVersion": "3.4.0",
                "uid": self.uid,
                "params": {
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                    "inputCol": self.getInputCol(),
                    "outputCol": self.getOutputCol(),
                    **self._get_model_params()
                }
            }
            os.makedirs(os.path.dirname(mlmodel_path), exist_ok=True)
            with open(mlmodel_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        # Handle MLflow integration if in active run
        active_run = mlflow.active_run()
        if active_run:
            # Log model artifacts
            mlflow.log_artifacts(path, "model")
            
            # Log as PyTorch model
            mlflow.pytorch.log_model(
                self._torch_model,
                "pytorch_model",
                registered_model_name="tabnet_pytorch"
            )
            
            # Log as Spark model
            mlflow.spark.log_model(
                self,
                "spark_model",
                registered_model_name="tabnet_spark"
            )
            
            # Log as pyfunc model
            class TabNetWrapper(mlflow.pyfunc.PythonModel):
                def __init__(self, torch_model):
                    self.torch_model = torch_model
                
                def predict(self, context, model_input):
                    import torch
                    import numpy as np
                    features = torch.from_numpy(
                        np.vstack([row.toArray() for row in model_input[self.getInputCol()]])
                    ).float()
                    with torch.no_grad():
                        outputs, _ = self.torch_model(features)
                        return torch.softmax(outputs, dim=1).numpy()
            
            mlflow.pyfunc.log_model(
                "pyfunc_model",
                python_model=TabNetWrapper(self._torch_model),
                registered_model_name="tabnet_pyfunc"
            )

    def save_mleap(self, path: str) -> None:
        """Save model in MLeap format for efficient serving."""
        try:
            # Create dummy data for MLeap
            dummy_df = self._create_dummy_data()
            
            # Save model in MLeap format
            mlflow.mleap.save_model(
                spark_model=self,
                path=path,
                sample_input=dummy_df
            )
        except Exception as e:
            logging.warning(f"Failed to save model in MLeap format: {str(e)}")
            logging.warning("Model will be saved without MLeap support")
            raise

    def get_pipeline(self) -> "Pipeline":
        """Get a Spark ML Pipeline containing this model."""
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import VectorAssembler
        
        assembler = VectorAssembler(
            inputCols=[self.getInputCol()],
            outputCol="assembled_features"
        )
        
        return Pipeline(stages=[assembler, self])
    
    def _create_dummy_data(self) -> DataFrame:
        """Create dummy data for MLeap serialization."""
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        schema = StructType([
            StructField(self.getInputCol(), VectorUDT()),
            StructField(self.getOutputCol(), ArrayType(FloatType()))
        ])
        dummy_data = [(
            DenseVector([0.0] * self.input_dim),
            [0.0] * self.output_dim
        )]
        return spark.createDataFrame(dummy_data, schema)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """Apply model to input dataset."""
        try:
            # Validate input column exists
            if self.getInputCol() not in dataset.columns:
                raise ValueError(f"Input column '{self.getInputCol()}' not found in dataset")
                
            # Validate torch model is initialized
            if self._torch_model is None:
                raise ValueError("Model has not been initialized or trained")
                
            # Validate feature dimensions before processing
            first_row = dataset.select(self.getInputCol()).first()
            if first_row is None:
                raise ValueError("Dataset is empty")
            feature_vector = first_row[0]
            if hasattr(feature_vector, 'size'):
                feature_dim = feature_vector.size
            elif hasattr(feature_vector, 'toArray'):
                feature_dim = len(feature_vector.toArray())
            else:
                feature_dim = len(feature_vector)
                
            if feature_dim != self.input_dim:
                raise ValueError(
                    f"Feature dimension mismatch. Model expects {self.input_dim} features, "
                    f"but got {feature_dim}. This usually happens when the feature engineering "
                    f"pipeline used during training differs from the one used for prediction. "
                    f"Ensure you're using the same pipeline stages (like VectorAssembler) "
                    f"with the same configuration for both training and prediction."
                )
        except ValueError as e:
            # Wrap ValueError in PythonException for Pandas UDF compatibility
            from pyspark.errors.exceptions.captured import PythonException
            import traceback
            raise PythonException(
                desc=str(e),
                stackTrace=traceback.format_exc()
            )
            
        from pyspark.sql.window import Window
        from pyspark.sql.functions import row_number, lit
        
        # Add deterministic row identifier
        window = Window.orderBy(lit(1))
        dataset_with_id = dataset.withColumn("row_id", row_number().over(window))
        
        # Capture required attributes in closure
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        input_dim = self.input_dim
        torch_model = self._torch_model

        def process_batch(iterator):
            for pdf in iterator:
                if pdf.empty:
                    yield pd.DataFrame({output_col: []})
                    continue
                
                # Convert features to numpy array
                features_list = []
                for feature in pdf[input_col]:
                    # Handle dict representation of PySpark Vector
                    if isinstance(feature, dict) and 'type' in feature and feature['type'] == 1:  # DenseVector
                        arr = np.array(feature['values'])
                    elif hasattr(feature, 'toArray'):
                        arr = feature.toArray()
                    else:
                        raise ValueError(f"Unsupported feature type: {type(feature)}")
                    
                    # Validate dimensions
                    if len(arr) != input_dim:
                        raise ValueError(f"Expected feature dimension {input_dim}, got {len(arr)}. Feature: {arr}")
                    
                    features_list.append(arr)
                
                # Convert to tensor and make predictions
                features_tensor = torch.from_numpy(np.vstack(features_list)).float()
                with torch.no_grad():
                    outputs, _ = torch_model(features_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                
                # Yield predictions as DataFrame
                yield pd.DataFrame({
                    output_col: [p.tolist() for p in probabilities]
                })
        
        # Apply predictions in a distributed manner
        predictions = dataset_with_id.mapInPandas(
            process_batch,
            schema=StructType([
                StructField(self.getOutputCol(), ArrayType(FloatType()))
            ])
        )
        
        # Return predictions and drop row_id
        return predictions.drop("row_id")