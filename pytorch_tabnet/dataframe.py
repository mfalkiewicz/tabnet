from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple, Any
import numpy as np

class TabNetDataFrame(ABC):
    """Abstract interface for DataFrame handling in TabNet."""
    
    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """Convert DataFrame to numpy array."""
        pass
    
    @abstractmethod
    def get_column(self, col: str) -> np.ndarray:
        """Get single column as numpy array."""
        pass
    
    @abstractmethod
    def validate_columns(self, cols: List[str]) -> bool:
        """Validate column existence."""
        pass
    
    @abstractmethod
    def get_shape(self) -> Tuple[int, int]:
        """Get DataFrame dimensions."""
        pass
    
    @classmethod
    @abstractmethod
    def from_numpy(cls, data: np.ndarray, columns: List[str]) -> 'TabNetDataFrame':
        """Create DataFrame from numpy array."""
        pass

class PandasDataFrame(TabNetDataFrame):
    """Pandas DataFrame implementation."""
    def __init__(self, df: 'pd.DataFrame'):
        self._df = df
    
    def to_numpy(self) -> np.ndarray:
        return self._df.values
    
    def get_column(self, col: str) -> np.ndarray:
        return self._df[col].values
    
    def validate_columns(self, cols: List[str]) -> bool:
        return all(col in self._df.columns for col in cols)
    
    def get_shape(self) -> Tuple[int, int]:
        return self._df.shape
    
    @classmethod
    def from_numpy(cls, data: np.ndarray, columns: List[str]) -> 'PandasDataFrame':
        import pandas as pd
        return cls(pd.DataFrame(data, columns=columns))

class PolarsDataFrame(TabNetDataFrame):
    """Polars DataFrame implementation."""
    def __init__(self, df: 'pl.DataFrame'):
        self._df = df
    
    def to_numpy(self) -> np.ndarray:
        return self._df.to_numpy()
    
    def get_column(self, col: str) -> np.ndarray:
        return self._df[col].to_numpy()
    
    def validate_columns(self, cols: List[str]) -> bool:
        return all(col in self._df.columns for col in cols)
    
    def get_shape(self) -> Tuple[int, int]:
        return self._df.shape
    
    @classmethod
    def from_numpy(cls, data: np.ndarray, columns: List[str]) -> 'PolarsDataFrame':
        import polars as pl
        # Create dictionary with column data
        data_dict = {col: data[:, i] for i, col in enumerate(columns)}
        return cls(pl.DataFrame(data_dict))

class SparkDataFrame(TabNetDataFrame):
    """Spark DataFrame implementation."""
    def __init__(self, df: 'pyspark.sql.DataFrame'):
        self._df = df
        self._spark_session = df.sparkSession
    
    def to_numpy(self) -> np.ndarray:
        return self._df.toPandas().values
    
    def get_column(self, col: str) -> np.ndarray:
        return self._df.select(col).toPandas().values.flatten()
    
    def validate_columns(self, cols: List[str]) -> bool:
        return all(col in self._df.columns for col in cols)
    
    def get_shape(self) -> Tuple[int, int]:
        return self._df.count(), len(self._df.columns)
    
    @property
    def columns(self):
        """Get DataFrame columns."""
        return self._df.columns
    
    def select(self, *cols):
        """Select columns from DataFrame."""
        return SparkDataFrame(self._df.select(*cols))
    
    @property
    def rdd(self):
        """Get RDD from DataFrame."""
        return self._df.rdd
    
    def unpersist(self):
        """Unpersist DataFrame."""
        self._df.unpersist()
    
    def repartition(self, num_partitions):
        """Repartition DataFrame."""
        return SparkDataFrame(self._df.repartition(num_partitions))
    
    def cache(self):
        """Cache DataFrame."""
        return SparkDataFrame(self._df.cache())
    
    def count(self) -> int:
        """Get number of rows in DataFrame."""
        return self._df.count()
    
    def drop(self, *cols):
        """Drop columns from DataFrame."""
        return SparkDataFrame(self._df.drop(*cols))
    
    def collect(self):
        """Collect all rows."""
        return self._df.collect()
    
    @property
    def sparkSession(self):
        """Get SparkSession."""
        return self._df.sparkSession
    
    @classmethod
    def from_numpy(cls, data: np.ndarray, columns: List[str]) -> 'SparkDataFrame':
        from pyspark.sql import SparkSession
        import pandas as pd
        # Convert to pandas first to handle schema properly
        pdf = pd.DataFrame(data, columns=columns)
        spark = SparkSession.builder.getOrCreate()
        df = spark.createDataFrame(pdf)
        return cls(df)