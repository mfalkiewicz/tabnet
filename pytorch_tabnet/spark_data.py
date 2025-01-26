from typing import List, Optional, Union
import torch
import pandas as pd
import numpy as np
import psutil
from pyspark.sql import DataFrame
from pytorch_tabnet.data import TabularDataProvider, TabularDataBatch

class SparkDataProvider(TabularDataProvider):
    def __init__(self,
                 df: DataFrame,
                 feature_cols: List[str],
                 target_col: Optional[str] = None,
                 batch_size: Union[int, str] = 256):
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Configure Arrow optimization
        self.df.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        self.df.sparkSession.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        
        # Set batch size based on available memory
        if batch_size == 'auto':
            self.batch_size = self._calculate_optimal_batch_size()
        else:
            self.batch_size = batch_size
            
        # Cache the optimized DataFrame
        self._cached_df = (self.df.select(self.feature_cols + ([self.target_col] if self.target_col else []))
                          .repartition(self._calculate_optimal_partitions())
                          .cache())
        
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory"""
        available_mem = psutil.virtual_memory().available
        row_size = len(self.feature_cols) * 4  # Assuming float32
        if self.target_col:
            row_size += 4
        
        # Use 10% of available memory
        optimal_size = int(0.1 * available_mem / row_size)
        return min(optimal_size, 10000)  # Cap at Arrow batch size
        
    def _calculate_optimal_partitions(self) -> int:
        """Calculate optimal number of Spark partitions"""
        total_rows = self.df.count()
        return max(1, total_rows // (self.batch_size * 2))
        
    def __iter__(self):
        # Use Arrow-optimized pandas conversion for each partition
        data_arrays = self._cached_df.mapInPandas(
            lambda iterator: [next(iterator).to_numpy()],
            schema=self._cached_df.schema
        ).collect()
        
        n_features = len(self.feature_cols)
        
        data_arrays = self._cached_df.mapInPandas(
            lambda iterator: [next(iterator).to_numpy()],
            schema=self._cached_df.schema
        ).collect()
        
        n_features = len(self.feature_cols)
        
        from pyspark.sql.functions import pandas_udf, PandasUDFType
        
        def process_partition(iterator):
            import pandas as pd
            import numpy as np
            import torch
            from pytorch_tabnet.data import TabularDataBatch
            
            feature_cols = self.feature_cols
            target_col = self.target_col
            batch_size = self.batch_size
            
            for pdf in iterator:
                X = pdf[feature_cols].values.astype('float32')
                y = pdf[target_col].values.astype('float32') if target_col else None
                
                n_rows = len(X)
                
                for start_idx in range(0, n_rows, batch_size):
                    end_idx = min(start_idx + batch_size, n_rows)
                    X_batch = X[start_idx:end_idx]
                    y_batch = y[start_idx:end_idx] if y is not None else None
                    
                    X_tensor = torch.from_numpy(X_batch)
                    y_tensor = torch.from_numpy(y_batch) if y_batch is not None else None
                    
                    yield TabularDataBatch(X_tensor, y_tensor)
        
        batches = self._cached_df.mapInPandas(process_partition, schema=PandasUDFType.ITER_BATCH)
        for batch in batches.flatMap(lambda x: x).collect():
            yield batch
        
    def __len__(self):
        return (self.df.count() + self.batch_size - 1) // self.batch_size