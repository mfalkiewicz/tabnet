"""Tests for TabNet Spark integration.

This module contains comprehensive tests for the Spark data provider implementation,
covering functionality, performance, and resource usage.
"""

import pytest
import numpy as np
import pandas as pd
import psutil
import time
import torch
from typing import Iterator
from pyspark.sql import SparkSession, DataFrame

from pytorch_tabnet.spark import SparkDataProvider
from pytorch_tabnet.data import TabularDataBatch

# Fixtures

@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for testing."""
    spark = (SparkSession.builder
            .master("local[2]")
            .appName("tabnet-spark-test")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
            .getOrCreate())
    yield spark
    spark.stop()

@pytest.fixture
def small_data(spark) -> DataFrame:
    """Create a small test DataFrame."""
    pdf = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'y': [0, 1, 0, 1, 0]
    })
    return spark.createDataFrame(pdf)

@pytest.fixture
def large_data(spark) -> DataFrame:
    """Create a large test DataFrame."""
    n_rows = 100000
    n_cols = 10
    pdf = pd.DataFrame(
        np.random.randn(n_rows, n_cols),
        columns=[f'feature_{i}' for i in range(n_cols-1)] + ['target']
    )
    return spark.createDataFrame(pdf)

# Basic Functionality Tests

def test_basic_iteration(small_data):
    """Test basic iteration over batches."""
    provider = SparkDataProvider(
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
    
    # Check last (partial) batch
    assert batches[2].X.shape == (1, 2)
    assert batches[2].y.shape == (1,)

def test_provider_length(small_data):
    """Test the __len__ implementation."""
    provider = SparkDataProvider(
        small_data,
        feature_cols=['A', 'B'],
        target_col='y',
        batch_size=2
    )
    assert len(provider) == 3

# Memory Management Tests

def test_memory_efficiency(large_data):
    """Test memory usage during iteration."""
    initial_mem = psutil.Process().memory_info().rss / 1024 / 1024
    
    provider = SparkDataProvider(
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

def test_prefetch_configuration(large_data):
    """Test prefetch behavior."""
    provider = SparkDataProvider(
        large_data,
        feature_cols=[f'feature_{i}' for i in range(9)],
        target_col='target',
        batch_size=1000,
        prefetch_batches=2
    )
    
    # Test that iteration works with prefetching
    batch_count = 0
    for batch in provider:
        assert isinstance(batch.X, torch.Tensor)
        assert isinstance(batch.y, torch.Tensor)
        batch_count += 1
        if batch_count >= 5:
            break
    
    # Test that we can iterate multiple times
    batch_count = 0
    for batch in provider:
        assert isinstance(batch.X, torch.Tensor)
        assert isinstance(batch.y, torch.Tensor)
        batch_count += 1
        if batch_count >= 5:
            break

# Performance Tests

def test_arrow_optimization(large_data):
    """Test Arrow optimization impact."""
    # With Arrow
    provider = SparkDataProvider(
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
    provider_no_arrow = SparkDataProvider(
        large_data,
        feature_cols=[f'feature_{i}' for i in range(9)],
        target_col='target',
        batch_size=1000
    )
    
    start = time.time()
    next(iter(provider_no_arrow))
    no_arrow_time = time.time() - start
    
    assert arrow_time < no_arrow_time

# Error Handling Tests

def test_input_validation(spark):
    """Test error handling for invalid inputs."""
    # Invalid DataFrame
    with pytest.raises(ValueError):
        SparkDataProvider(None, ['A'], 'y')
    
    # Invalid feature columns
    df = spark.createDataFrame(pd.DataFrame({'A': [1, 2], 'y': [0, 1]}))
    with pytest.raises(ValueError):
        SparkDataProvider(df, ['B'], 'y')  # Non-existent column
    
    # Invalid batch size
    with pytest.raises(ValueError):
        SparkDataProvider(df, ['A'], 'y', batch_size=0)

def test_empty_dataframe(spark):
    """Test handling of empty DataFrames."""
    from pyspark.sql.types import StructType, StructField, DoubleType
    
    # Define schema for empty DataFrame
    schema = StructType([
        StructField("A", DoubleType(), True),
        StructField("B", DoubleType(), True),
        StructField("y", DoubleType(), True)
    ])
    
    # Create empty DataFrame with schema
    df = spark.createDataFrame([], schema)
    provider = SparkDataProvider(df, ['A', 'B'], 'y')
    assert len(list(provider)) == 0

def test_null_values(spark):
    """Test handling of null values."""
    pdf = pd.DataFrame({
        'A': [1, None, 3],
        'B': [10, 20, None],
        'y': [0, 1, None]
    })
    df = spark.createDataFrame(pdf)
    
    # Create provider with null values
    provider = SparkDataProvider(df, ['A', 'B'], 'y')
    
    # Get first batch
    batch = next(iter(provider))
    
    # Verify that nulls are handled (typically converted to 0 or NaN)
    assert isinstance(batch.X, torch.Tensor)
    assert isinstance(batch.y, torch.Tensor)
    assert not torch.isnan(batch.X).all()  # Some values should be non-NaN
    assert not torch.isnan(batch.y).all()  # Some values should be non-NaN
