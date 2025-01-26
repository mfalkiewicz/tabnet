import pytest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import psutil
import torch
from typing import Iterator

@pytest.fixture(scope="module")
def spark():
    spark = (SparkSession.builder
            .master("local[2]")
            .appName("tabnet-spark-test-v2")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
            .getOrCreate())
    yield spark
    spark.stop()

@pytest.fixture
def small_data(spark):
    pdf = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'y': [0, 1, 0, 1, 0]
    })
    return spark.createDataFrame(pdf)

@pytest.fixture
def large_data(spark):
    n_rows = 100000
    n_cols = 10
    pdf = pd.DataFrame(
        np.random.randn(n_rows, n_cols),
        columns=[f'feature_{i}' for i in range(n_cols-1)] + ['target']
    )
    return spark.createDataFrame(pdf)

def test_basic_functionality(small_data):
    """Test basic data provider functionality with small dataset"""
    from pytorch_tabnet.spark_data_v2 import SparkDataProviderV2
    
    provider = SparkDataProviderV2(
        small_data,
        feature_cols=['A', 'B'],
        target_col='y',
        batch_size=2
    )
    
    batches = list(provider)
    assert len(batches) == 3
    
    # Check first batch
    assert isinstance(batches[0].X, torch.Tensor)
    assert batches[0].X.shape == (2, 2)
    assert batches[0].y.shape == (2,)
    
    # Check data types
    assert batches[0].X.dtype == torch.float32
    assert batches[0].y.dtype == torch.float32

def test_memory_efficiency(large_data):
    """Test memory usage during iteration"""
    from pytorch_tabnet.spark_data_v2 import SparkDataProviderV2
    
    initial_mem = psutil.Process().memory_info().rss / 1024 / 1024
    
    provider = SparkDataProviderV2(
        large_data,
        feature_cols=[f'feature_{i}' for i in range(9)],
        target_col='target',
        batch_size=1000,
        prefetch_batches=2
    )
    
    max_mem_increase = 0
    for i, batch in enumerate(provider):
        if i >= 10:  # Check first 10 batches
            break
        current_mem = psutil.Process().memory_info().rss / 1024 / 1024
        mem_increase = current_mem - initial_mem
        max_mem_increase = max(max_mem_increase, mem_increase)
    
    # Memory increase should be bounded
    assert max_mem_increase < 500  # MB

def test_arrow_optimization(large_data):
    """Test Arrow optimization impact"""
    from pytorch_tabnet.spark_data_v2 import SparkDataProviderV2
    import time
    
    # With Arrow
    provider = SparkDataProviderV2(
        large_data,
        feature_cols=[f'feature_{i}' for i in range(9)],
        target_col='target',
        batch_size=1000
    )
    
    start = time.time()
    next(iter(provider))
    arrow_time = time.time() - start
    
    # Without Arrow
    large_data.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    provider_no_arrow = SparkDataProviderV2(
        large_data,
        feature_cols=[f'feature_{i}' for i in range(9)],
        target_col='target',
        batch_size=1000
    )
    
    start = time.time()
    next(iter(provider_no_arrow))
    no_arrow_time = time.time() - start
    
    assert arrow_time < no_arrow_time

def test_prefetch_configuration(large_data):
    """Test prefetch behavior"""
    from pytorch_tabnet.spark_data_v2 import SparkDataProviderV2
    
    provider = SparkDataProviderV2(
        large_data,
        feature_cols=[f'feature_{i}' for i in range(9)],
        target_col='target',
        batch_size=1000,
        prefetch_batches=2
    )
    
    # Verify prefetch queue size
    assert provider._prefetch_queue.maxsize == 2
    
    # Ensure prefetch doesn't block iteration
    for i, batch in enumerate(provider):
        if i >= 5:
            break
        assert isinstance(batch.X, torch.Tensor)

def test_error_handling(spark):
    """Test error handling for invalid inputs"""
    from pytorch_tabnet.spark_data_v2 import SparkDataProviderV2
    
    # Invalid DataFrame
    with pytest.raises(ValueError):
        SparkDataProviderV2(None, ['A'], 'y')
    
    # Invalid feature columns
    df = spark.createDataFrame(pd.DataFrame({'A': [1, 2], 'y': [0, 1]}))
    with pytest.raises(ValueError):
        SparkDataProviderV2(df, ['B'], 'y')  # Non-existent column
    
    # Invalid batch size
    with pytest.raises(ValueError):
        SparkDataProviderV2(df, ['A'], 'y', batch_size=0)