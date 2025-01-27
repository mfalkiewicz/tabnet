"""DEPRECATED: Old Spark data provider implementation.

This module is deprecated and will be removed in a future version.
Use pytorch_tabnet.spark.SparkDataProvider instead.
"""

import warnings
from typing import List, Optional

from pyspark.sql import DataFrame
from pytorch_tabnet.spark.provider import SparkDataProvider as NewSparkDataProvider


class SparkDataProvider:
    """DEPRECATED: Use pytorch_tabnet.spark.SparkDataProvider instead."""

    def __init__(
        self,
        df: DataFrame,
        feature_cols: List[str],
        target_col: Optional[str] = None,
        batch_size: int = 1000,
    ):
        warnings.warn(
            "SparkDataProvider is deprecated and will be removed in a future version. "
            "Use pytorch_tabnet.spark.SparkDataProvider instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._provider = NewSparkDataProvider(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            batch_size=batch_size
        )

    def __iter__(self):
        return iter(self._provider)

    def __len__(self):
        return len(self._provider)