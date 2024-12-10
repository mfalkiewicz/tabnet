from typing import List, Optional, Union
import numpy as np
import torch
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader as PetastormDataLoader
from .typing import DataLoaderProtocol
from torch.utils.data import Dataset


def make_petastorm_loader(
    spark_df: DataFrame,
    feature_cols: List[str],
    target_cols: Optional[List[str]] = None,
    batch_size: int = 1024,
    shuffle: bool = True,
    num_epochs: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> DataLoaderProtocol:
    """Creates a Petastorm DataLoader from a Spark DataFrame."""
    selected_cols = feature_cols + (target_cols if target_cols else [])
    df = spark_df.select([col(c).cast("float") for c in selected_cols])

    cache_dir = cache_dir or f"/tmp/petastorm_{np.random.randint(0, 1000000)}"
    df.write.parquet(cache_dir, mode="overwrite")

    with make_batch_reader(cache_dir, num_epochs=num_epochs) as reader:
        return PetastormDataLoader(reader, batch_size=batch_size, shuffle=shuffle)


class SparkPredictDataset(Dataset):
    def __init__(self, X):
        self.X = X
        self._cached_data = None
        self._len = X.count()

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self._cached_data is None:
            # Convert to pandas only once and cache
            self._cached_data = self.X.toPandas().values
        return torch.FloatTensor(self._cached_data[idx])


class SparkTrainDataset:
    """Dataset for training with Spark DataFrames"""

    def __init__(
        self,
        spark_df: DataFrame,
        feature_cols: List[str],
        target_cols: Union[str, List[str]],
    ) -> None:
        self.spark_df = spark_df
        self.feature_cols = feature_cols
        self.target_cols = (
            [target_cols] if isinstance(target_cols, str) else target_cols
        )

    def make_loader(
        self,
        batch_size: int = 1024,
        shuffle: bool = True,
        num_epochs: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> DataLoaderProtocol:
        return make_petastorm_loader(
            self.spark_df,
            self.feature_cols,
            self.target_cols,
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            cache_dir=cache_dir,
        )
