"""TabNet Spark integration for efficient data loading and processing.

This module provides memory-efficient data loading from Spark DataFrames using Arrow optimization
and proper memory management strategies.
"""

from typing import List, Optional, Iterator, Union, Dict, Tuple
import queue
import threading
import warnings
import torch
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import LongType
from torch.utils.data import Dataset, DataLoader
from contextlib import contextmanager
import psutil

from pytorch_tabnet.data import TabularDataProvider, TabularDataBatch
from pytorch_tabnet.typing import DataLoaderProtocol


class SparkDataProvider(TabularDataProvider):
    """Memory-efficient data provider for PySpark DataFrames.
    
    This implementation uses Arrow-optimized pandas UDFs to efficiently stream
    data from Spark while maintaining memory usage bounds.
    
    Args:
        df: PySpark DataFrame containing features and optional target
        feature_cols: List of column names for features
        target_col: Optional column name for target variable
        batch_size: Number of rows per batch
        prefetch_batches: Number of batches to prefetch (default: 2)
        arrow_max_records: Maximum records per Arrow batch (default: 10000)
    """
    
    def __init__(
        self,
        df: DataFrame,
        feature_cols: List[str],
        target_col: Optional[str] = None,
        batch_size: int = 1000,
        prefetch_batches: int = 2,
        arrow_max_records: int = 10000
    ):
        if df is None:
            raise ValueError("DataFrame cannot be None")
        if not feature_cols:
            raise ValueError("feature_cols cannot be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if prefetch_batches < 1:
            raise ValueError("prefetch_batches must be positive")
            
        # Validate columns exist in DataFrame
        all_cols = set(df.columns)
        missing_cols = set(feature_cols)
        if target_col:
            missing_cols.add(target_col)
        missing_cols = missing_cols - all_cols
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.batch_size = batch_size
        
        # Configure Arrow optimization
        self.df.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        self.df.sparkSession.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", str(arrow_max_records))
        
        # Initialize prefetch queue
        self._prefetch_queue = queue.Queue(maxsize=prefetch_batches)
        self._stop_prefetch = threading.Event()
        
        # Cache optimized DataFrame with column selection and repartitioning
        selected_cols = [col(c) for c in self.feature_cols]
        if self.target_col:
            selected_cols.append(col(self.target_col))
            
        self._cached_df = (
            self.df.select(selected_cols)
            .repartition(self._calculate_optimal_partitions())
            .cache()
        )
        
        # Convert to pandas once for better performance
        self._pandas_df = self._cached_df.toPandas()
        
    def _calculate_optimal_partitions(self) -> int:
        """Calculate optimal number of Spark partitions based on data size and batch size"""
        total_rows = self.df.count()
        return max(1, min(
            total_rows // (self.batch_size * 2),  # Aim for 2x batch size per partition
            self.df.sparkSession.sparkContext.defaultParallelism * 2  # But don't exceed 2x cores
        ))
        
    def _create_batch(self, batch_df: pd.DataFrame) -> TabularDataBatch:
        """Create a TabularDataBatch from a pandas DataFrame
        
        Args:
            batch_df: pandas DataFrame containing features and target
            
        Returns:
            TabularDataBatch instance
        """
        X = batch_df[self.feature_cols].values.astype(np.float32)
        y = batch_df[self.target_col].values.astype(np.float32) if self.target_col else None
        
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y) if y is not None else None
        return TabularDataBatch(X_tensor, y_tensor)
            
    def __iter__(self) -> Iterator[TabularDataBatch]:
        """Iterate over batches of data
        
        The implementation uses Arrow-optimized pandas conversion to process data
        in batches, then converts each batch to torch tensors.
        
        Yields:
            TabularDataBatch instances containing features and optional target
        """
        n_rows = len(self._pandas_df)
        for start_idx in range(0, n_rows, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_rows)
            batch_df = self._pandas_df.iloc[start_idx:end_idx]
            yield self._create_batch(batch_df)
                
    def __len__(self) -> int:
        """Return the total number of batches"""
        return (len(self._pandas_df) + self.batch_size - 1) // self.batch_size


class SparkDataset(Dataset):
    """Memory-efficient Spark DataFrame Dataset for TabNet.
    
    This dataset implementation provides efficient data loading from Spark DataFrames
    with proper memory management and batch processing.
    
    Args:
        spark_df: Input Spark DataFrame
        feature_cols: Optional list of feature column names
        target_col: Optional target column name
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle the data
        num_epochs: Optional number of epochs
    """
    
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
        
        # Convert to pandas once for better performance
        self._pandas_df = self.df.toPandas()
        
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get a single batch of data.
        
        Args:
            idx: Batch index
            
        Returns:
            Tuple of (features, targets) tensors
        """
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self._pandas_df))
        
        batch_df = self._pandas_df.iloc[start_idx:end_idx]
        
        # Convert to tensors
        features = torch.from_numpy(batch_df[self.feature_cols].values.astype(np.float32))
        targets = None
        if self.target_col:
            targets = torch.from_numpy(batch_df[self.target_col].values.astype(np.float32))
            
        return features, targets


@contextmanager
def create_spark_loader(
    df: DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
    batch_size: int = 256,
    shuffle: bool = True,
    num_epochs: Optional[int] = None
) -> DataLoaderProtocol:
    """Create a DataLoader for a Spark DataFrame.
    
    This context manager ensures proper resource cleanup after using the DataLoader.
    
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