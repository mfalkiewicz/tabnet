import pytest
from typing import Dict, Union, Tuple, Generator
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import tempfile
import os
import sys
from pytorch_tabnet.typing import DataFrameLike, FloatArray


@pytest.fixture(scope="session", autouse=True)
def configure_spark_python():
    """Configure PySpark Python executable paths."""
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    """Create a SparkSession fixture to be used across tests"""
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("tabnet_tests")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.sql.execution.arrow.enabled", "true")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config(
            "spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={tempfile.gettempdir()}"
        )
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def classification_data(
    spark: SparkSession,
) -> Dict[str, Union[Tuple[FloatArray, FloatArray], DataFrameLike]]:
    """Create simple classification dataset"""
    np.random.seed(42)
    n_samples = 100  # Reduced for faster testing
    n_features = 10

    X = np.random.randn(n_samples, n_features).astype(np.float64)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    pdf = pd.DataFrame(X, columns=feature_names)
    pdf["target"] = y

    # Convert all columns to float for Spark compatibility
    for col in pdf.columns:
        pdf[col] = pdf[col].astype(float)

    try:
        sdf = spark.createDataFrame(pdf)
    except Exception as e:
        pytest.skip(f"Failed to create Spark DataFrame: {str(e)}")

    return {"numpy": (X, y), "pandas": pdf, "spark": sdf}


def test_python_version_consistency(spark: SparkSession):
    """Verify Python versions match between driver and workers."""
    # Check driver Python version
    driver_version = sys.version_info[:2]
    
    # Check worker Python version using UDF
    def get_python_version():
        import sys
        return '.'.join(map(str, sys.version_info[:2]))
    
    worker_version = spark.sql("SELECT 1").select(
        spark.udf.register("get_python_version", get_python_version)()
    ).collect()[0][0]
    
    assert f"{driver_version[0]}.{driver_version[1]}" == worker_version


def test_spark_python_configuration():
    """Verify PySpark Python configuration."""
    assert 'PYSPARK_PYTHON' in os.environ
    assert 'PYSPARK_DRIVER_PYTHON' in os.environ
    assert os.path.exists(os.environ['PYSPARK_PYTHON'])
    assert os.path.exists(os.environ['PYSPARK_DRIVER_PYTHON'])
