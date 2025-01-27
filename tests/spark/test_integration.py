"""Integration tests for TabNet Spark functionality."""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
import torch
from pyspark.sql import SparkSession
from pytorch_tabnet.spark import SparkDataProvider, SparkDataset, create_spark_loader


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
def small_data(spark):
    """Create a small test DataFrame."""
    pdf = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'y': [0, 1, 0, 1, 0]
    })
    return spark.createDataFrame(pdf)


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


class TestBasicFunctionality:
    """Test basic functionality of the Spark data provider."""

    def test_data_provider_iteration(self, small_data):
        """Test basic iteration functionality."""
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

    def test_data_provider_length(self, small_data):
        """Test length calculation."""
        provider = SparkDataProvider(small_data, ['A', 'B'], 'y', batch_size=3)
        assert len(provider) == 2


class TestMemoryEfficiency:
    """Test memory efficiency and management."""

    def test_memory_usage_during_iteration(self, large_data):
        """Test memory usage stays bounded during iteration."""
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

    def test_prefetch_behavior(self, large_data):
        """Test prefetch queue behavior."""
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


class TestArrowOptimization:
    """Test Arrow optimization functionality."""

    def test_arrow_functionality(self, large_data):
        """Test Arrow functionality works correctly."""
        # With Arrow enabled
        provider = SparkDataProvider(
            large_data,
            feature_cols=[f'feature_{i}' for i in range(9)],
            target_col='target',
            batch_size=1000
        )
        
        # Get first batch with Arrow
        arrow_batch = next(iter(provider))
        assert isinstance(arrow_batch.X, torch.Tensor)
        assert isinstance(arrow_batch.y, torch.Tensor)
        assert arrow_batch.X.shape[1] == 9  # Number of features
        
        # Without Arrow
        large_data.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        provider_no_arrow = SparkDataProvider(
            large_data,
            feature_cols=[f'feature_{i}' for i in range(9)],
            target_col='target',
            batch_size=1000
        )
        
        # Get first batch without Arrow
        no_arrow_batch = next(iter(provider_no_arrow))
        assert isinstance(no_arrow_batch.X, torch.Tensor)
        assert isinstance(no_arrow_batch.y, torch.Tensor)
        assert no_arrow_batch.X.shape[1] == 9  # Number of features
        
        # Verify data consistency
        assert torch.allclose(arrow_batch.X, no_arrow_batch.X, rtol=1e-5, atol=1e-5)
        assert torch.allclose(arrow_batch.y, no_arrow_batch.y, rtol=1e-5, atol=1e-5)
        
        # Reset Arrow configuration
        large_data.sparkSession.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")


class TestErrorHandling:
    """Test error handling and validation."""

    def test_input_validation(self, spark):
        """Test validation of input parameters."""
        # Invalid DataFrame
        with pytest.raises(ValueError, match="DataFrame cannot be None"):
            SparkDataProvider(None, ['A'], 'y')
        
        # Invalid feature columns
        df = spark.createDataFrame(pd.DataFrame({'A': [1, 2], 'y': [0, 1]}))
        with pytest.raises(ValueError, match="Columns not found in DataFrame: {'B'}"):
            SparkDataProvider(df, ['B'], 'y')  # Non-existent column
        
        # Invalid batch size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            SparkDataProvider(df, ['A'], 'y', batch_size=0)
        
        # Empty feature columns
        with pytest.raises(ValueError, match="feature_cols cannot be empty"):
            SparkDataProvider(df, [], 'y')
        
        # Invalid prefetch value
        with pytest.raises(ValueError, match="prefetch_batches must be positive"):
            SparkDataProvider(df, ['A'], 'y', prefetch_batches=0)
        
        # Target column not in DataFrame
        with pytest.raises(ValueError, match="Columns not found in DataFrame: {'target'}"):
            SparkDataProvider(df, ['A'], 'target')


class TestDataLoader:
    """Test DataLoader functionality."""

    def test_data_loader_creation(self, small_data):
        """Test creating and using DataLoader."""
        with create_spark_loader(
            small_data,
            feature_cols=['A', 'B'],
            target_col='y',
            batch_size=2
        ) as loader:
            # Check loader yields correct batches
            batch_count = 0
            for features, targets in loader:
                assert isinstance(features, torch.Tensor)
                assert isinstance(targets, torch.Tensor)
                batch_count += 1
            
            # With 5 rows and batch_size=2, expect 3 batches
            assert batch_count == 3