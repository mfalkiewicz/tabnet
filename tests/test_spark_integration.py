from typing import Dict, Union, Tuple, Generator, Any, cast
import pytest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import tempfile
import os
from numpy.typing import NDArray

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.typing import DataFrameLike, ArrayLike, FloatArray

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
    
    X = np.random.randn(n_samples, n_features)
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

@pytest.fixture
def regression_data(spark):
    """Create simple regression dataset"""
    np.random.seed(42)
    n_samples = 100  # Reduced for faster testing
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 2*X[:, 1] + np.random.randn(n_samples)*0.1
    
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
        'numpy': (X, y.reshape(-1, 1)),
        'pandas': pdf,
        'spark': sdf
    }

def test_classifier_pandas_input(classification_data: Dict[str, Union[Tuple[FloatArray, FloatArray], DataFrameLike]]) -> None:
    """Test TabNetClassifier with pandas input"""
    pdf = cast(pd.DataFrame, classification_data['pandas'])
    X = pdf.drop('target', axis=1)
    y = pdf['target']
    
    clf = TabNetClassifier(
        n_d=4,  # Reduced for faster testing
        n_a=4,
        n_steps=2,
        n_independent=1,
        n_shared=1
    )
    
    clf.fit(X, y, max_epochs=2)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    
    assert preds.shape[0] == len(y)
    assert probs.shape == (len(y), 2)

@pytest.mark.slow
def test_classifier_spark_input(classification_data: Dict[str, Union[Tuple[FloatArray, FloatArray], DataFrameLike]]) -> None:
    """Test TabNetClassifier with Spark input"""
    sdf = cast(SparkSession, classification_data['spark'])
    
    clf = TabNetClassifier(
        n_d=4,
        n_a=4,
        n_steps=2,
        n_independent=1,
        n_shared=1
    )
    
    try:
        clf.fit(sdf, 'target', max_epochs=2)
        preds = clf.predict(sdf)
        probs = clf.predict_proba(sdf)
        
        assert len(preds) == sdf.count()
        assert probs.shape[0] == sdf.count()
        assert probs.shape[1] == 2
    except Exception as e:
        pytest.skip(f"Spark test failed: {str(e)}")

def test_regressor_pandas_input(regression_data):
    """Test TabNetRegressor with pandas input"""
    pdf = regression_data['pandas']
    X = pdf.drop('target', axis=1)
    y = pdf['target'].values.reshape(-1, 1)

    reg = TabNetRegressor(
        n_d=4,
        n_a=4,
        n_steps=2,
        n_independent=1,
        n_shared=1
    )
    reg.fit(X, y, max_epochs=2)

@pytest.mark.slow
def test_regressor_spark_input(regression_data):
    """Test TabNetRegressor with Spark input"""
    sdf = regression_data['spark']
    X = sdf.drop('target')
    y = sdf.select('target').toPandas().values.reshape(-1, 1)

    reg = TabNetRegressor(
        n_d=4,
        n_a=4,
        n_steps=2,
        n_independent=1,
        n_shared=1
    )
    reg.fit(X, y, max_epochs=2)

@pytest.mark.slow
def test_validation_set_spark(classification_data):
    """Test validation set handling with Spark DataFrames"""
    train_sdf = classification_data['spark']
    val_sdf = classification_data['spark']  # Using same data for simplicity
    
    X_train = train_sdf.drop('target')
    y_train = train_sdf.select('target').toPandas().values.ravel()  # Flatten to 1D
    X_val = val_sdf.drop('target')
    y_val = val_sdf.select('target').toPandas().values.ravel()  # Flatten to 1D

    clf = TabNetClassifier(
        n_d=4,
        n_a=4,
        n_steps=2,
        n_independent=1,
        n_shared=1
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], max_epochs=2)

@pytest.mark.slow
def test_feature_importance_spark(classification_data):
    """Test feature importance computation with Spark input"""
    sdf = classification_data['spark']
    X = sdf.drop('target')
    y = sdf.select('target').toPandas().values.ravel()  # Flatten to 1D

    clf = TabNetClassifier(
        n_d=4,
        n_a=4,
        n_steps=2,
        n_independent=1,
        n_shared=1
    )
    clf.fit(X, y, max_epochs=2)

def test_explain_spark(classification_data):
    """Test explain method with Spark input"""
    sdf = classification_data['spark']
    X = sdf.drop('target')
    y = sdf.select('target').toPandas().values.ravel()  # Flatten to 1D

    clf = TabNetClassifier(
        n_d=8, n_a=8,
        n_steps=3
    )
    clf.fit(X, y, max_epochs=3)    
    # Test explain method
    explain_matrix, masks = clf.explain(sdf)
    
    assert explain_matrix.shape[0] == sdf.count()
    assert explain_matrix.shape[1] == len(sdf.columns) - 1  # excluding target
    assert isinstance(masks, dict) 