from typing import Optional, Union, Tuple
import numpy as np
import torch
try:
    from pyspark.sql import DataFrame as SparkDataFrame
    _HAVE_PYSPARK = True
except ImportError:
    _HAVE_PYSPARK = False


class InputValidator:
    """Handles input validation logic for TabNet models."""
    
    def validate_dimensions(self, X: Union[np.ndarray, 'SparkDataFrame'], y: Optional[Union[np.ndarray, str]] = None) -> None:
        """Validates input and target dimensions.
        
        Parameters
        ----------
        X : Union[np.ndarray, SparkDataFrame]
            Input features
        y : Optional[Union[np.ndarray, str]]
            Target values or column name for Spark
        
        Raises
        ------
        ValueError
            If dimensions are invalid
        """
        if isinstance(X, np.ndarray):
            self._validate_numpy_input_dimensions(X)
            if y is not None and not isinstance(y, str):
                self._validate_numpy_target_dimensions(X, y)
        elif _HAVE_PYSPARK and isinstance(X, SparkDataFrame):
            self._validate_spark_input_dimensions(X)
            if y is not None and not isinstance(y, str):
                raise ValueError("Target must be a column name (str) when using Spark DataFrames")
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

    def validate_types(self, X: Union[np.ndarray, 'SparkDataFrame'], y: Optional[Union[np.ndarray, str]] = None) -> None:
        """Validates input and target types.
        
        Parameters
        ----------
        X : Union[np.ndarray, SparkDataFrame]
            Input features
        y : Optional[Union[np.ndarray, str]]
            Target values or column name for Spark
            
        Raises
        ------
        TypeError
            If types are invalid
        """
        if isinstance(X, np.ndarray):
            self._validate_numpy_types(X)
            if y is not None and not isinstance(y, str):
                self._validate_numpy_target_types(y)
        elif _HAVE_PYSPARK and isinstance(X, SparkDataFrame):
            self._validate_spark_types(X)
            if y is not None and not isinstance(y, str):
                raise TypeError("Target must be a column name (str) when using Spark DataFrames")
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

    def _validate_numpy_input_dimensions(self, X: np.ndarray) -> None:
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D array, got {len(X.shape)}D array instead")

    def _validate_numpy_target_dimensions(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(y.shape) > 2:
            raise ValueError(f"Target array has too many dimensions: {len(y.shape)}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Input and target have different numbers of samples: {X.shape[0]} vs {y.shape[0]}"
            )

    def _validate_spark_input_dimensions(self, X: 'SparkDataFrame') -> None:
        if len(X.columns) == 0:
            raise ValueError("DataFrame has no columns")

    def _validate_numpy_types(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(X)}")
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"Array dtype must be numeric, got {X.dtype}")

    def _validate_numpy_target_types(self, y: np.ndarray) -> None:
        if not isinstance(y, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(y)}")
        if not np.issubdtype(y.dtype, np.number):
            raise TypeError(f"Array dtype must be numeric, got {y.dtype}")

    def _validate_spark_types(self, X: 'SparkDataFrame') -> None:
        if not _HAVE_PYSPARK:
            raise ImportError("PySpark is required for Spark DataFrame support")
        if not isinstance(X, SparkDataFrame):
            raise TypeError(f"Expected Spark DataFrame, got {type(X)}")


class ModelOutputs:
    """Standardized model outputs container."""
    
    def __init__(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        feature_importances: Optional[np.ndarray] = None,
        attention_masks: Optional[dict] = None
    ):
        """
        Parameters
        ----------
        predictions : Union[np.ndarray, torch.Tensor]
            Model predictions
        feature_importances : Optional[np.ndarray]
            Feature importance scores
        attention_masks : Optional[dict]
            Attention masks from the model
        """
        self.predictions = self._convert_to_numpy(predictions)
        self.feature_importances = feature_importances
        self.attention_masks = attention_masks

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of predictions."""
        return self.predictions.shape

    def _convert_to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Converts torch tensors to numpy arrays."""
        if isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        return data

    def to_dict(self) -> dict:
        """Converts outputs to dictionary format."""
        return {
            "predictions": self.predictions,
            "feature_importances": self.feature_importances,
            "attention_masks": self.attention_masks
        }