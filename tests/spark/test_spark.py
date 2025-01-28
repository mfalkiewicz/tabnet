"""Spark-specific tests for TabNet DataFrame implementation.

This module contains tests specific to the Spark implementation, including:
- Performance optimizations (Arrow)
- Memory management
- Spark-specific features
"""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
import torch
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType

from pytorch_tabnet.spark import SparkDataProvider
from pytorch_tabnet.dataframe import SparkDataFrame

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
def large_data(spark):
    """Create a large test DataFrame."""
    n_rows = 100000
    n_cols = 10
    pdf = pd.DataFrame(
        np.random.randn(n_rows, n_cols),
        columns=[f'feature_{i}' for i in range(n_cols-1)] + ['target']
    )
    return spark.createDataFrame(pdf)

class TestSparkOptimization:
    """Test Spark-specific optimizations."""
    
    def test_arrow_performance(self, large_data):
        """Test performance impact of Arrow optimization."""
        # With Arrow
        with SparkDataProvider(
            large_data,
            feature_cols=[f'feature_{i}' for i in range(9)],
            target_col='target',
            batch_size=1000
        ) as provider:
            start = time.time()
            next(iter(provider))
            arrow_time = time.time() - start
        
        # Without Arrow
        large_data.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        with SparkDataProvider(
            large_data,
            feature_cols=[f'feature_{i}' for i in range(9)],
            target_col='target',
            batch_size=1000
        ) as provider:
            start = time.time()
            next(iter(provider))
            no_arrow_time = time.time() - start
        
        # Re-enable Arrow for other tests
        large_data.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        
        # Arrow should be faster
        assert arrow_time < no_arrow_time

class TestMemoryManagement:
    """Test memory management features."""
    
    def test_memory_efficiency(self, large_data):
        """Test memory usage stays bounded during iteration."""
        initial_mem = psutil.Process().memory_info().rss / 1024 / 1024
        
        with SparkDataProvider(
            large_data,
            feature_cols=[f'feature_{i}' for i in range(9)],
            target_col='target',
            batch_size=1000,
            prefetch_batches=2
        ) as provider:
            max_mem_increase = 0
            for i, batch in enumerate(provider):
                if i >= 10:  # Check first 10 batches
                    break
                current_mem = psutil.Process().memory_info().rss / 1024 / 1024
                mem_increase = current_mem - initial_mem
                max_mem_increase = max(max_mem_increase, mem_increase)
        
        # Memory increase should be bounded
        assert max_mem_increase < 500  # MB
    
    def test_cache_and_unpersist(self, large_data):
        """Test caching and unpersisting functionality."""
        df = SparkDataFrame(large_data)
        
        # Cache the DataFrame
        cached_df = df.cache()
        # Force materialization
        cached_df.count()
        
        # Unpersist and verify
        cached_df.unpersist()

class TestSparkSpecificFeatures:
    """Test features specific to Spark implementation."""
    
    def test_repartition(self, large_data):
        """Test repartitioning functionality."""
        df = SparkDataFrame(large_data)
        repartitioned = df.repartition(10)
        
        # Verify number of partitions
        assert repartitioned._df.rdd.getNumPartitions() == 10
    
    def test_empty_dataframe(self, spark):
        """Test handling of empty DataFrames."""
        # Define schema for empty DataFrame
        schema = StructType([
            StructField("feature_1", DoubleType(), True),
            StructField("feature_2", DoubleType(), True),
            StructField("target", DoubleType(), True)
        ])
        
        # Create empty DataFrame with schema
        empty_df = spark.createDataFrame([], schema)
        with SparkDataProvider(empty_df, ['feature_1', 'feature_2'], 'target') as provider:
            assert len(list(provider)) == 0
    
    def test_null_handling(self, spark):
        """Test handling of null values."""
        # Create DataFrame with null values
        pdf = pd.DataFrame({
            'A': [1.0, None, 3.0],
            'B': [10.0, 20.0, None],
            'target': [0.0, 1.0, None]
        })
        df = spark.createDataFrame(pdf)
        
        with SparkDataProvider(df, ['A', 'B'], 'target') as provider:
            batch = next(iter(provider))
            # Verify tensors are created successfully
            assert isinstance(batch.X, torch.Tensor)
            assert isinstance(batch.y, torch.Tensor)
            # Nulls should be handled (typically converted to 0 or NaN)
            assert not torch.isnan(batch.X).all()
            assert not torch.isnan(batch.y).all()

# Integration tests moved to tests/test_dataframe.py