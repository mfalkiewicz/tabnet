import pytest
from typing import Dict, Union, Tuple, Generator
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import tempfile
from pytorch_tabnet.typing import DataFrameLike, FloatArray

@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    """Create a SparkSession fixture to be used across tests"""
    spark = (SparkSession.builder
            .master("local[*]")
            .appName("tabnet_tests")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.default.parallelism", "2")
            .config("spark.sql.execution.arrow.enabled", "true")
            .config("spark.driver.extraJavaOptions", 
                   f"-Djava.io.tmpdir={tempfile.gettempdir()}")
            .getOrCreate())
    yield spark
    spark.stop()

@pytest.fixture
def classification_data(spark: SparkSession) -> Dict[str, Union[Tuple[FloatArray, FloatArray], DataFrameLike]]:
    """Create simple classification dataset"""
    np.random.seed(42)
    n_samples = 100  # Reduced for faster testing
    n_features = 10
    
    X = np.random.randn(n_samples, n_features).astype(np.float64)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    pdf = pd.DataFrame(X, columns=feature_names)
    pdf['target'] = y
    
    # Convert all columns to float for Spark compatibility
    for col in pdf.columns:
        pdf[col] = pdf[col].astype(float)
    
    try:
        sdf = spark.createDataFrame(pdf)
    except Exception as e:
        pytest.skip(f"Failed to create Spark DataFrame: {str(e)}")
    
    return {
        'numpy': (X, y),
        'pandas': pdf,
        'spark': sdf
    } 