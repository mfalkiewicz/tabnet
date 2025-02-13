"""Shared test fixtures for Spark integration tests."""

import pytest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing.
    
    This fixture is shared across all tests and cleaned up after the session.
    """
    spark = (SparkSession.builder
            .master("local[2]")
            .appName("tabnet-spark-test")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memory", "2g")
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate())
    
    yield spark
    
    # Clean up
    spark.catalog.clearCache()
    spark.stop()


@pytest.fixture
def small_data(spark):
    """Create a small test DataFrame.
    
    This fixture provides a small DataFrame suitable for basic functionality tests.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create features and labels
    n_samples = 10
    n_features = 5
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 2, size=n_samples)
    
    # Convert to Spark vectors
    data = [(Vectors.dense(x), float(y)) for x, y in zip(features, labels)]
    
    # Create schema
    schema = StructType([
        StructField("features", VectorUDT()),
        StructField("label", DoubleType())
    ])
    
    return spark.createDataFrame(data, schema)


@pytest.fixture
def large_data(spark):
    """Create a large test DataFrame.
    
    This fixture provides a larger DataFrame suitable for performance and memory tests.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000  # Reduced from 100000 to avoid memory issues
    n_features = 10
    
    # Create features
    features = np.random.randn(n_samples, n_features)
    # Create binary labels
    labels = np.random.randint(0, 2, size=n_samples)
    
    # Convert to Spark vectors
    data = [(Vectors.dense(x), float(y)) for x, y in zip(features, labels)]
    
    # Create schema
    schema = StructType([
        StructField("features", VectorUDT()),
        StructField("label", DoubleType())
    ])
    
    return spark.createDataFrame(data, schema)


@pytest.fixture
def multiclass_data(spark):
    """Create a test DataFrame for multiclass classification.
    
    This fixture provides a DataFrame with multiple classes for testing
    multiclass classification scenarios.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 100  # Small dataset for quick testing
    n_features = 10
    n_classes = 3
    
    # Create features
    features = np.random.randn(n_samples, n_features)
    # Create multiclass labels
    labels = np.random.randint(0, n_classes, size=n_samples)
    
    # Convert to Spark vectors
    data = [(Vectors.dense(x), float(y)) for x, y in zip(features, labels)]
    
    # Create schema
    schema = StructType([
        StructField("features", VectorUDT()),
        StructField("label", DoubleType())
    ])
    
    return spark.createDataFrame(data, schema)