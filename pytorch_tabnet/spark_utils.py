from typing import List, Optional, Union, Dict, Iterator, Tuple
import torch
import numpy as np
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import StructType
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from .typing import DataLoaderProtocol
from contextlib import contextmanager
import psutil

class SparkDataset(Dataset):
    """Memory-efficient Spark DataFrame Dataset for TabNet"""
    
    def __init__(
        self,
        spark_df: DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        batch_size: int = 256,
        shuffle: bool = True,
        num_epochs: Optional[int] = None
    ):
        self.spark_df = spark_df
        if feature_cols is None:
            self.feature_cols = [col for col in spark_df.columns if col != target_col]
        else:
            self.feature_cols = feature_cols
        self.target_col = target_col
        self.batch_size = self._adjust_batch_size(batch_size)
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self._len = None
        
        # Cache optimized DataFrame
        self.df = self._prepare_dataframe()
        
    def _adjust_batch_size(self, initial_size: int) -> int:
        """Dynamically adjust batch size based on available memory"""
        mem = psutil.virtual_memory()
        available_mb = mem.available // (1024 ** 2)
        adjusted_size = max(2, min(initial_size, available_mb // 10))
        return adjusted_size
        
    def _prepare_dataframe(self) -> DataFrame:
        """Optimize DataFrame for batched loading"""
        df = self.spark_df.select(self.feature_cols + ([self.target_col] if self.target_col else []))
        
        if self.shuffle:
            df = df.orderBy(F.rand())
            
        return df.repartition(max(1, self._get_len() // self.batch_size)).cache()
        
    def _get_len(self) -> int:
        """Get total number of rows in DataFrame"""
        if self._len is None:
            self._len = self.spark_df.count()
        return self._len
        
    def __len__(self) -> int:
        return (self._get_len() + self.batch_size - 1) // self.batch_size
        
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Iterate through DataFrame partitions as batches"""
        for partition in self.df.rdd.mapPartitions(lambda x: [list(x)]).toLocalIterator():
            if not partition:
                continue
                
            try:
                # Convert Spark Rows to pandas DataFrame
                pdf = pd.DataFrame([row.asDict() for row in partition])
                
                # Convert to tensors
                features = torch.from_numpy(pdf[self.feature_cols].values.astype(np.float32))
                targets = torch.from_numpy(pdf[self.target_col].values.astype(np.float32)) if self.target_col else None
                
                yield features, targets
            finally:
                # Cleanup Spark resources
                del partition
                torch.cuda.empty_cache()

@contextmanager
def spark_data_loader(
    df: DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
    batch_size: int = 256,
    shuffle: bool = True,
    num_epochs: Optional[int] = None
) -> DataLoaderProtocol:
    """Create a DataLoader for a Spark DataFrame.
    
    Args:
        df: Input Spark DataFrame
        feature_cols: List of feature column names
        target_col: Target column name for training, None for prediction
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_epochs: Number of epochs (None for infinite)
        
    Returns:
        DataLoader for the dataset
    """
    dataset = SparkDataset(df, feature_cols, target_col, batch_size, shuffle, num_epochs)
    try:
        yield DataLoader(dataset, batch_size=None, pin_memory=True, num_workers=0)
    finally:
        dataset.df.unpersist()
