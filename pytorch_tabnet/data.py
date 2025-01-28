from typing import Iterator, Tuple, Optional
import torch
from torch.utils.data import IterableDataset
from pytorch_tabnet.dataframe import (
    TabNetDataFrame,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame
)

class TabularDataBatch:
    """Represents a batch of tabular data (features and optional targets)"""
    
    def __init__(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        self.X = X
        self.y = y
        
    def pin_memory(self):
        """Returns a copy with tensors pinned to CUDA memory"""
        X_pin = self.X.pin_memory() if torch.cuda.is_available() else self.X
        y_pin = self.y.pin_memory() if self.y is not None and torch.cuda.is_available() else self.y
        return TabularDataBatch(X_pin, y_pin)

from abc import ABC, abstractmethod

class TabularDataProvider(IterableDataset, ABC):
    """Base class for data providers that return TabularDataBatch instances"""
    
    @abstractmethod
    def __iter__(self) -> Iterator[TabularDataBatch]:
        pass
        
    @abstractmethod
    def __len__(self):
        pass