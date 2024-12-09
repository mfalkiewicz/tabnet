from __future__ import annotations
from typing import Union, List, Dict, Protocol, Any, Iterator, TypeVar, Tuple
import numpy as np
import torch
from numpy.typing import NDArray
from pyspark.sql import DataFrame as SparkDataFrame
import pandas as pd

# Type variables
T_co = TypeVar('T_co', covariant=True)
Shape = Tuple[int, ...]

# Basic types
Array = NDArray[Any]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ArrayLike = Union[Array, List[float]]
DataFrameLike = Union[SparkDataFrame, pd.DataFrame]
TensorLike = Union[torch.Tensor, ArrayLike]

# Complex types
BatchType = Union[TensorLike, Dict[str, TensorLike]]
TargetType = Union[ArrayLike, str]
MetricsType = Dict[str, float]

class DataLoaderProtocol(Protocol[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        ...
    
    def __next__(self) -> T_co:
        ...
    
    def __len__(self) -> int:
        ...

class ModelOutput(Protocol):
    def cpu(self) -> ModelOutput:
        ...
    def detach(self) -> ModelOutput:
        ...
    def numpy(self) -> FloatArray:
        ... 