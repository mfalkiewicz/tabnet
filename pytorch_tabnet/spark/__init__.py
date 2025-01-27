"""TabNet Spark Integration.

This package provides efficient data loading and processing capabilities for TabNet
using Apache Spark DataFrames.
"""

from pytorch_tabnet.spark.provider import (
    SparkDataProvider,
    SparkDataset,
    create_spark_loader,
)

__all__ = [
    "SparkDataProvider",
    "SparkDataset",
    "create_spark_loader",
]