import pytest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import psutil
from pytorch_tabnet.spark_data import SparkDataProvider

@pytest.fixture(scope="module")
def spark():
    spark = (SparkSession.builder
            .master("local[2]")
            .appName("tabnet-spark-memory-test")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
            .getOrCreate())
    yield spark
    spark.stop()

@pytest.fixture
def large_data(spark):
    # Create a larger dataset to test memory management
    n_rows = 100000
    n_cols = 10
    pdf = pd.DataFrame(
        np.random.randn(n_rows, n_cols),
        columns=[f'feature_{i}' for i in range(n_cols-1)] + ['target']
    )
    return spark.createDataFrame(pdf)

def test_memory_efficient_loading(large_data):
    # Monitor memory usage during data loading
    initial_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    provider = SparkDataProvider(
        large_data,
        [f'feature_{i}' for i in range(9)],
        'target',
        batch_size=1000
    )
    
    # Load a few batches and check memory
    batches = []
    for i, batch in enumerate(provider):
        if i >= 5:  # Check first 5 batches
            break
        batches.append(batch)
        current_mem = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Memory increase should be reasonable (less than dataset size)
        assert current_mem - initial_mem < 500  # MB

def test_dynamic_batch_sizing(large_data):
    provider = SparkDataProvider(
        large_data,
        [f'feature_{i}' for i in range(9)],
        'target',
        batch_size='auto'  # Let the provider determine batch size
    )
    
    # Batch size should be adjusted based on available memory
    assert provider.batch_size > 0
    assert provider.batch_size <= 10000  # Should not exceed Arrow batch size
    
    # Memory usage should stay reasonable during iteration
    initial_mem = psutil.Process().memory_info().rss / 1024 / 1024
    for batch in provider:
        current_mem = psutil.Process().memory_info().rss / 1024 / 1024
        assert current_mem - initial_mem < 500  # MB

def test_arrow_optimization(large_data):
    # Verify Arrow optimization is enabled
    provider = SparkDataProvider(
        large_data,
        [f'feature_{i}' for i in range(9)],
        'target'
    )
    
    # Time batch retrieval
    import time
    start = time.time()
    next(iter(provider))
    arrow_time = time.time() - start
    
    # Disable Arrow and compare
    large_data.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    provider_no_arrow = SparkDataProvider(
        large_data,
        [f'feature_{i}' for i in range(9)],
        'target'
    )
    
    start = time.time()
    next(iter(provider_no_arrow))
    no_arrow_time = time.time() - start
    
    # Arrow should be significantly faster
    assert arrow_time < no_arrow_time