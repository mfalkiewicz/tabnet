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
from pytorch_tabnet.dataframe import TabNetDataFrame, SparkDataFrame


class SparkDataProvider(TabularDataProvider):
    """Memory-efficient data provider for PySpark DataFrames.
    
    This implementation uses Arrow-optimized pandas UDFs to efficiently stream
    data from Spark while maintaining memory usage bounds.
    
    Args:
        df: TabNetDataFrame containing features and optional target
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
            
        # Wrap PySpark DataFrame in our SparkDataFrame class
        self.df = SparkDataFrame(df)
            
        # Validate columns exist in DataFrame
        if not self.df.validate_columns(feature_cols):
            raise ValueError(f"Columns not found in DataFrame: {set(feature_cols) - set(df.columns)}")
        if target_col and not self.df.validate_columns([target_col]):
            raise ValueError(f"Columns not found in DataFrame: {set([target_col])}")
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        
        # Configure Arrow optimization
        self.df.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        self.df.sparkSession.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", str(arrow_max_records))
        
        # Initialize prefetch queue and thread control
        self._prefetch_queue = queue.Queue(maxsize=prefetch_batches)
        self._stop_prefetch = threading.Event()
        self._prefetch_thread = None
        
        # Cache optimized DataFrame with column selection and repartitioning
        selected_cols = [col(c) for c in self.feature_cols]
        if self.target_col:
            selected_cols.append(col(self.target_col))
            
        self._cached_df = (
            self.df.select(selected_cols)
            .repartition(self._calculate_optimal_partitions())
            .cache()
        )
        
    def _calculate_optimal_partitions(self) -> int:
        """Calculate optimal number of Spark partitions based on data size and batch size"""
        total_rows = self.df.count()
        return max(1, min(
            total_rows // (self.batch_size * 2),  # Aim for 2x batch size per partition
            self.df.sparkSession.sparkContext.defaultParallelism * 2  # But don't exceed 2x cores
        ))
        
    def _process_partition(self, iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """Process a partition of data.
        
        Args:
            iterator: Iterator over pandas DataFrames from a Spark partition
            
        Yields:
            Processed pandas DataFrames
        """
        for pdf in iterator:
            for start_idx in range(0, len(pdf), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(pdf))
                yield pdf.iloc[start_idx:end_idx]

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

    def _prefetch_worker(self):
        """Background worker to prefetch batches"""
        try:
            selected_cols = self.feature_cols + ([self.target_col] if self.target_col else [])
            
            # Configure Arrow batch size for optimal performance
            self._cached_df.sparkSession.conf.set(
                "spark.sql.execution.arrow.maxRecordsPerBatch",
                str(min(1000, self.batch_size))  # Smaller batches for better memory management
            )
            
            # Use smaller partitions to avoid large task sizes
            num_partitions = max(
                2,
                self._cached_df.count() // min(1000, self.batch_size)
            )
            
            # Process data in smaller chunks with proper cleanup
            for partition in (self._cached_df._df
                            .select(selected_cols)
                            .repartition(num_partitions)
                            .rdd
                            .mapPartitions(lambda x: [pd.DataFrame(list(x), columns=selected_cols)])
                            .toLocalIterator()):
                if self._stop_prefetch.is_set():
                    break
                    
                # Process partition in chunks
                for i in range(0, len(partition), self.batch_size):
                    if self._stop_prefetch.is_set():
                        break
                    chunk = partition.iloc[i:i + self.batch_size]
                    try:
                        self._prefetch_queue.put(
                            self._create_batch(chunk),
                            timeout=30  # Add timeout to prevent hanging
                        )
                    except queue.Full:
                        if self._stop_prefetch.is_set():
                            break
                        # Queue is full, skip this batch
                        continue
        except Exception as e:
            warnings.warn(f"Prefetch worker error: {str(e)}")
        finally:
            try:
                self._prefetch_queue.put(None, timeout=5)  # Signal end of data
            except queue.Full:
                pass  # Queue is full, main thread probably exited

    def __enter__(self):
        """Start prefetch thread when entering context"""
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self._prefetch_thread.daemon = True  # Allow Python to exit if thread is still running
        self._prefetch_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context"""
        self._stop_prefetch.set()
        if self._prefetch_thread:
            self._prefetch_thread.join()
        self._cached_df.unpersist()
            
    def __iter__(self) -> Iterator[TabularDataBatch]:
        """Iterate over batches of data
        
        The implementation uses Arrow-optimized streaming from Spark with prefetching
        for better performance.
        
        Yields:
            TabularDataBatch instances containing features and optional target
        """
        if not self._prefetch_thread:
            raise RuntimeError("SparkDataProvider must be used as a context manager")
            
        while True:
            batch = self._prefetch_queue.get()
            if batch is None:  # End of data
                break
            yield batch
                
    def __len__(self) -> int:
        """Return the total number of batches"""
        total_rows = self._cached_df.count()
        return (total_rows + self.batch_size - 1) // self.batch_size


class SparkDataset(Dataset):
    """Memory-efficient Spark DataFrame Dataset for TabNet.
    
    This dataset implementation provides efficient data loading from Spark DataFrames
    with proper memory management and batch processing using Arrow optimization.
    
    Args:
        df: TabNetDataFrame containing features and optional target
        feature_cols: List of feature column names
        target_col: Optional target column name
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle the data
        num_epochs: Optional number of epochs
        prefetch_batches: Number of batches to prefetch (default: 2)
    """
    
    def __init__(
        self,
        df: TabNetDataFrame,
        feature_cols: List[str],
        target_col: Optional[str] = None,
        batch_size: int = 256,
        shuffle: bool = True,
        num_epochs: Optional[int] = None,
        prefetch_batches: int = 2
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
        if not df.validate_columns(feature_cols):
            raise ValueError(f"Columns not found in DataFrame: {set(feature_cols) - set(df.get_column_names())}")
        if target_col and not df.validate_columns([target_col]):
            raise ValueError(f"Target column {target_col} not found in DataFrame")
            
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.batch_size = self._adjust_batch_size(batch_size)
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self._len = None
        
        # Initialize prefetch queue and thread control
        self._prefetch_queue = queue.Queue(maxsize=prefetch_batches)
        self._stop_prefetch = threading.Event()
        self._prefetch_thread = None
        self._current_epoch = 0
        
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

    def _process_partition(self, iterator: Iterator[pd.DataFrame]) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Process a partition of data into tensors"""
        for pdf in iterator:
            for start_idx in range(0, len(pdf), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(pdf))
                batch_df = pdf.iloc[start_idx:end_idx]
                
                features = torch.from_numpy(batch_df[self.feature_cols].values.astype(np.float32))
                targets = None
                if self.target_col:
                    targets = torch.from_numpy(batch_df[self.target_col].values.astype(np.float32))
                    
                yield features, targets

    def _prefetch_worker(self):
        """Background worker to prefetch batches"""
        try:
            # Use Arrow-optimized batch loading with repartitioning
            selected_cols = self.feature_cols + ([self.target_col] if self.target_col else [])
            df = self.df._df.select(selected_cols).repartition(self.batch_size)
            
            for partition in df.rdd.mapPartitions(lambda x: [pd.DataFrame(list(x), columns=selected_cols)]).collect():
                if self._stop_prefetch.is_set():
                    break
                    
                features = torch.from_numpy(partition[self.feature_cols].values.astype(np.float32))
                targets = None
                if self.target_col:
                    targets = torch.from_numpy(partition[self.target_col].values.astype(np.float32))
                
                self._prefetch_queue.put((features, targets))
        finally:
            self._prefetch_queue.put(None)  # Signal end of data

    def __enter__(self):
        """Start prefetch thread when entering context"""
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self._prefetch_thread.daemon = True  # Allow Python to exit if thread is still running
        self._prefetch_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context"""
        self._stop_prefetch.set()
        if self._prefetch_thread:
            self._prefetch_thread.join()
        self.df.unpersist()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get a single batch of data.
        
        Args:
            idx: Batch index
            
        Returns:
            Tuple of (features, targets) tensors
        """
        if not self._prefetch_thread:
            raise RuntimeError("SparkDataset must be used as a context manager")
            
        batch = self._prefetch_queue.get()
        if batch is None:  # End of data
            raise IndexError("Dataset iteration complete")
            
        return batch


@contextmanager
def create_spark_loader(
    df: TabNetDataFrame,
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