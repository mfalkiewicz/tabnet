"""Tests for TabNet DataFrame implementations and integrations.

This module implements a three-layer testing strategy:
1. Interface compliance tests - verify each implementation follows the TabNetDataFrame contract
2. Implementation-specific tests - test features unique to each backend
3. Integration tests - verify each backend works with training code
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from pyspark.sql import SparkSession
import torch
import psutil
from typing import Type, Dict, Any

from pytorch_tabnet.dataframe import (
    TabNetDataFrame,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame
)

# Common Test Data

@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Create sample data that can be used across different implementations."""
    n_samples = 100
    n_features = 5
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)
    
    # Column names
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    return {
        'X': X,
        'y': y,
        'feature_cols': feature_cols,
        'n_samples': n_samples,
        'n_features': n_features
    }

@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for testing."""
    spark = (SparkSession.builder
            .master("local[2]")
            .appName("tabnet-test")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
            .getOrCreate())
    yield spark
    spark.stop()

@pytest.fixture
def pandas_df(sample_data) -> PandasDataFrame:
    """Create a PandasDataFrame with sample data."""
    df = pd.DataFrame(
        sample_data['X'],
        columns=sample_data['feature_cols']
    )
    df['target'] = sample_data['y']
    return PandasDataFrame(df)

@pytest.fixture
def polars_df(sample_data) -> PolarsDataFrame:
    """Create a PolarsDataFrame with sample data."""
    df = pl.DataFrame(
        {
            **{col: sample_data['X'][:, i] for i, col in enumerate(sample_data['feature_cols'])},
            'target': sample_data['y']
        }
    )
    return PolarsDataFrame(df)

@pytest.fixture
def spark_df(spark, sample_data) -> SparkDataFrame:
    """Create a SparkDataFrame with sample data."""
    pandas_df = pd.DataFrame(
        sample_data['X'],
        columns=sample_data['feature_cols']
    )
    pandas_df['target'] = sample_data['y']
    df = spark.createDataFrame(pandas_df)
    return SparkDataFrame(df)

# Interface Compliance Tests

class TestTabNetDataFrameInterface:
    """Verify each implementation follows the TabNetDataFrame contract."""
    
    @pytest.mark.parametrize("df_fixture", ["pandas_df", "polars_df", "spark_df"])
    def test_to_numpy(self, df_fixture, sample_data, request):
        """Test conversion to numpy array."""
        df = request.getfixturevalue(df_fixture)
        data = df.to_numpy()
        
        assert isinstance(data, np.ndarray)
        assert data.shape[0] == sample_data['n_samples']
        assert data.shape[1] == sample_data['n_features'] + 1  # features + target
    
    @pytest.mark.parametrize("df_fixture", ["pandas_df", "polars_df", "spark_df"])
    def test_get_column(self, df_fixture, sample_data, request):
        """Test getting a single column."""
        df = request.getfixturevalue(df_fixture)
        col = sample_data['feature_cols'][0]
        
        data = df.get_column(col)
        assert isinstance(data, np.ndarray)
        assert data.shape[0] == sample_data['n_samples']
        assert len(data.shape) == 1  # Should be 1D array
    
    @pytest.mark.parametrize("df_fixture", ["pandas_df", "polars_df", "spark_df"])
    def test_validate_columns(self, df_fixture, sample_data, request):
        """Test column validation."""
        df = request.getfixturevalue(df_fixture)
        
        # Test valid columns
        assert df.validate_columns(sample_data['feature_cols'])
        
        # Test invalid columns
        assert not df.validate_columns(['nonexistent_column'])
    
    @pytest.mark.parametrize("df_fixture", ["pandas_df", "polars_df", "spark_df"])
    def test_get_shape(self, df_fixture, sample_data, request):
        """Test getting DataFrame dimensions."""
        df = request.getfixturevalue(df_fixture)
        shape = df.get_shape()
        
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        assert shape[0] == sample_data['n_samples']
        assert shape[1] == sample_data['n_features'] + 1  # features + target
    
    @pytest.mark.parametrize("impl", [PandasDataFrame, PolarsDataFrame, SparkDataFrame])
    def test_from_numpy(self, sample_data, impl):
        """Test creating DataFrame from numpy array."""
        data = sample_data['X']
        cols = sample_data['feature_cols']
        
        df = impl.from_numpy(data, cols)
        assert isinstance(df, impl)
        
        # Verify data was correctly transferred
        np_data = df.to_numpy()
        assert np.allclose(data, np_data)

# Implementation-Specific Tests will be added in subsequent commits
# Integration Tests will be added in subsequent commits