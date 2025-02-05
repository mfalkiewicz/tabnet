"""TabNet Spark ML Transformer Implementation.

This module provides Spark ML pipeline integration for TabNet through a hybrid approach
that leverages both Spark's distributed computing capabilities and TabNet's neural architecture.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, HasPredictionCol, HasLabelCol
from pyspark.ml import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.ml.util import MLReadable, MLWritable, MLReader, MLWriter, DefaultParamsReader, DefaultParamsWriter
import os

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.dataframe import SparkDataFrame, TabNetDataFrame


class TabNetParams(Params):
    """Common parameters for TabNet Spark ML components."""
    
    n_d = Param(Params._dummy(), "n_d", "Width of the decision prediction layer", 
                typeConverter=TypeConverters.toInt)
    n_a = Param(Params._dummy(), "n_a", "Width of the attention embedding for each mask", 
                typeConverter=TypeConverters.toInt)
    n_steps = Param(Params._dummy(), "n_steps", "Number of steps in the architecture", 
                typeConverter=TypeConverters.toInt)
    gamma = Param(Params._dummy(), "gamma", "Scale for feature updates", 
                typeConverter=TypeConverters.toFloat)
    cat_idxs = Param(Params._dummy(), "cat_idxs", "List of categorical feature indices", 
                typeConverter=TypeConverters.toListInt)
    cat_dims = Param(Params._dummy(), "cat_dims", "List of categorical feature dimensions",
                 typeConverter=TypeConverters.toListInt)
    labelCols = Param(Params._dummy(), "labelCols", "List of label column names",
                 typeConverter=TypeConverters.toListString)
    
    def __init__(self):
        super().__init__()
        self._setDefault(
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            cat_idxs=[],
            cat_dims=[],
            labelCols=["label"]
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
        if labelCols is None:
            labelCols = ["label"]
        self._set(inputCol=inputCol, outputCol=outputCol, labelCols=labelCols)
        self.tabnet = None
        self._set(**kwargs)
    
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


class SparkTabNetModel(Model, TabNetParams, HasInputCol, HasOutputCol, HasPredictionCol, MLReadable, MLWritable):
    """Spark ML Model for TabNet predictions.
    
    This model provides distributed prediction capabilities using the trained TabNet model
    and DataFrame interface.
    
    Args:
        tabnet_model: Trained TabNetClassifier instance
        inputCol: Input column name containing features
        outputCol: Output column name for predictions
    """
    
    def __init__(
        self,
        tabnet_model: TabNetClassifier = None,
        inputCol: str = "features",
        outputCol: str = "predictions"
    ):
        super().__init__()
        self.tabnet = tabnet_model
        self._set(inputCol=inputCol, outputCol=outputCol)
    
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
            DataFrame with predictions added
        """
        # Create a copy of the model's state
        model_state = {
            'n_d': self.tabnet.n_d,
            'n_a': self.tabnet.n_a,
            'n_steps': self.tabnet.n_steps,
            'gamma': self.tabnet.gamma,
            'cat_idxs': self.tabnet.cat_idxs,
            'cat_dims': self.tabnet.cat_dims,
            'input_dim': self.tabnet.input_dim,
            'output_dim': self.tabnet.output_dim,
            'classes_': self.tabnet.classes_,
            'state_dict': self.tabnet.network.state_dict()
        }
        
        @pandas_udf(ArrayType(DoubleType()))
        def predict_batch(features_series):
            """Vectorized UDF for predictions."""
            # Create a new model instance for this executor
            local_model = TabNetClassifier(
                n_d=model_state['n_d'],
                n_a=model_state['n_a'],
                n_steps=model_state['n_steps'],
                gamma=model_state['gamma'],
                cat_idxs=model_state['cat_idxs'],
                cat_dims=model_state['cat_dims'],
                input_dim=model_state['input_dim'],
                output_dim=model_state['output_dim']
            )
            # Set classes
            local_model.classes_ = model_state['classes_']
            # Load the model state
            local_model.network.load_state_dict(model_state['state_dict'])
            
            # Convert features to numpy array
            features_array = np.stack([np.array(x, dtype=np.float32) for x in features_series])
            
            # Make predictions
            with torch.no_grad():
                predictions = local_model.predict_proba(features_array)
            
            # Convert predictions to pandas Series with lists
            return pd.Series([p.tolist() for p in predictions])
        
        # Apply predictions
        return dataset.withColumn(
            self.getOutputCol(),
            predict_batch(self.getInputCol())
        )
    
    def write(self) -> MLWriter:
        """Returns MLWriter instance for this ML instance."""
        return DefaultParamsWriter(self)
    
    @classmethod
    def read(cls) -> MLReader:
        """Returns MLReader instance for this class."""
        return DefaultParamsReader(cls)
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        self.write().save(path)
        # Save TabNet model separately
        tabnet_path = os.path.join(path, "tabnet_model")
        self.tabnet.save_model(tabnet_path)
    
    @classmethod
    def load(cls, path: str) -> "SparkTabNetModel":
        """Load the model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded SparkTabNetModel instance
        """
        # Create model instance without TabNet model
        model = cls()
        # Load parameters
        reader = DefaultParamsReader(cls)
        params = reader.load(path)
        params._copyValues(model)
        # Load TabNet model
        tabnet_path = os.path.join(path, "tabnet_model")
        model.tabnet = TabNetClassifier()
        model.tabnet = model.tabnet.load_model(tabnet_path)
        return model