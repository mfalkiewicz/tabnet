import pytest
import torch
from pyspark.sql import SparkSession
from pytorch_tabnet.spark_utils import SparkDataset, spark_data_loader
import numpy as np

@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("tabnet-spark-test") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def sample_data(spark):
    data = [
        (float(x), float(x%3), float(x*2)) for x in range(1000)
    ]
    return spark.createDataFrame(data, ["feature1", "feature2", "target"])

def test_spark_dataset_batching(spark, sample_data):
    with spark_data_loader(
        spark,
        sample_data,
        feature_cols=["feature1", "feature2"],
        target_col="target",
        batch_size=32
    ) as loader:
        batch_count = 0
        for features, targets in loader:
            assert features.shape == (32, 2)
            assert targets.shape == (32,)
            batch_count += 1
            
        assert batch_count == (1000 + 32 - 1) // 32

def test_memory_aware_batching(spark, sample_data):
    dataset = SparkDataset(
        spark,
        sample_data,
        feature_cols=["feature1", "feature2"],
        target_col="target",
        batch_size=1024  # Will be adjusted based on available memory
    )
    assert dataset.batch_size <= 1024
    assert dataset.batch_size > 0

def test_feature_target_separation(spark, sample_data):
    with spark_data_loader(
        spark,
        sample_data,
        feature_cols=["feature1", "feature2"],
        target_col="target"
    ) as loader:
        features, targets = next(iter(loader))
        assert features.shape[1] == 2
        assert targets is not None

def test_prediction_mode(spark, sample_data):
    with spark_data_loader(
        spark,
        sample_data,
        feature_cols=["feature1", "feature2"],
        target_col=None
    ) as loader:
        features, targets = next(iter(loader))
        assert features.shape[1] == 2
        assert targets is None

def test_tensor_dtypes(spark, sample_data):
    with spark_data_loader(
        spark,
        sample_data,
        feature_cols=["feature1", "feature2"],
        target_col="target"
    ) as loader:
        features, targets = next(iter(loader))
        assert features.dtype == torch.float32
        assert targets.dtype == torch.float32
