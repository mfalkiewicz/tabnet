"""DEPRECATED: Old Spark utilities implementation.

This module is deprecated and will be removed in a future version.
Use pytorch_tabnet.spark.SparkDataset and pytorch_tabnet.spark.create_spark_loader instead.
"""

import warnings
from typing import List, Optional

from pyspark.sql import DataFrame
from pytorch_tabnet.spark.provider import SparkDataset as NewSparkDataset
from pytorch_tabnet.spark.provider import create_spark_loader as new_spark_data_loader
from pytorch_tabnet.typing import DataLoaderProtocol


class SparkDataset(NewSparkDataset):
    """DEPRECATED: Use pytorch_tabnet.spark.SparkDataset instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SparkDataset from spark_utils is deprecated and will be removed in a future version. "
            "Use pytorch_tabnet.spark.SparkDataset instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


def spark_data_loader(
    df: DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
    batch_size: int = 256,
    shuffle: bool = True,
    num_epochs: Optional[int] = None
) -> DataLoaderProtocol:
    """DEPRECATED: Use pytorch_tabnet.spark.create_spark_loader instead."""
    warnings.warn(
        "spark_data_loader is deprecated and will be removed in a future version. "
        "Use pytorch_tabnet.spark.create_spark_loader instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return new_spark_data_loader(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        batch_size=batch_size,
        shuffle=shuffle,
        num_epochs=num_epochs
    )
