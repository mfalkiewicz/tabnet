"""Shared test fixtures for Spark integration tests."""

import pytest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing.
    
    This fixture is shared across all tests and cleaned up after the session.
    """
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
    """Create a small test DataFrame.
    
    This fixture provides a small DataFrame suitable for basic functionality tests.
    """
    pdf = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'y': [0, 1, 0, 1, 0]
    })
    return spark.createDataFrame(pdf)


@pytest.fixture
def large_data(spark):
    """Create a large test DataFrame.
    
    This fixture provides a larger DataFrame suitable for performance and memory tests.
    """
    n_rows = 100000
    n_cols = 10
    pdf = pd.DataFrame(
        np.random.randn(n_rows, n_cols),
        columns=[f'feature_{i}' for i in range(n_cols-1)] + ['target']
    )
    return spark.createDataFrame(pdf)