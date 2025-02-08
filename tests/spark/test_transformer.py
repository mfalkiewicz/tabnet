"""Tests for TabNet Spark ML transformer implementation."""

import pytest
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType
import os

from pytorch_tabnet.spark.transformer import (
    SparkTabNetEstimator,
    SparkTabNetModel,
    get_tabnet_classifier
)
from pytorch_tabnet.dataframe import SparkDataFrame


@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for testing."""
    return (SparkSession.builder
            .master("local[2]")
            .appName("tabnet-transformer-test")
            .getOrCreate())


@pytest.fixture(scope="module")
def test_data(spark):
    """Create test data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate random features and binary labels
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.random.randint(0, 2, size=n_samples)
    
    # Create DataFrame with balanced classes
    data = [(features[i].tolist(), float(labels[i])) for i in range(n_samples)]
    schema = StructType([
        StructField("features", ArrayType(DoubleType())),
        StructField("label", DoubleType())
    ])
    
    return spark.createDataFrame(data, schema)


def test_spark_tabnet_estimator_initialization():
    """Test proper initialization of SparkTabNetEstimator with various parameters."""
    estimator = SparkTabNetEstimator(
        inputCol="features",
        outputCol="predictions",
        n_d=8,
        n_steps=3
    )
    assert estimator.getInputCol() == "features"
    assert estimator.getOutputCol() == "predictions"
    assert estimator.getOrDefault(estimator.n_d) == 8
    assert estimator.getOrDefault(estimator.n_steps) == 3


def test_dataframe_interface_integration(test_data):
    """Test DataFrame interface integration."""
    # Create wrapped DataFrame
    wrapped_df = SparkDataFrame(test_data)
    
    # Verify interface methods
    assert isinstance(wrapped_df.to_numpy(), np.ndarray)
    assert isinstance(wrapped_df.get_column("features"), np.ndarray)
    assert wrapped_df.validate_columns(["features", "label"])
    assert len(wrapped_df.get_shape()) == 2


def test_prepare_spark_data(test_data):
    """Test data preparation method handles various input formats correctly."""
    estimator = SparkTabNetEstimator(inputCol="features", outputCol="predictions")
    
    # Create wrapped DataFrame
    wrapped_df = SparkDataFrame(test_data)
    
    # Verify DataFrame schema and interface methods
    assert "features" in wrapped_df.columns
    assert "label" in wrapped_df.columns
    assert wrapped_df.count() > 0
    
    # Test data preparation
    prepared_data = estimator._prepare_data(wrapped_df)
    assert isinstance(prepared_data, SparkDataFrame)
    assert prepared_data.validate_columns(["features", "label"])


def test_spark_tabnet_model_transform(test_data):
    """Test model transformation produces correct output schema and values."""
    # Train model
    estimator = SparkTabNetEstimator(
        inputCol="features",
        outputCol="predictions",
        n_d=8,
        n_steps=3
    )
    model = estimator.fit(test_data)
    
    # Transform data using DataFrame interface
    wrapped_df = SparkDataFrame(test_data)
    result = model.transform(test_data)
    
    # Verify predictions
    assert "predictions" in result.columns
    assert result.count() == test_data.count()
    
    # Check prediction values are valid
    predictions = result.select("predictions").collect()
    for row in predictions:
        assert isinstance(row.predictions, list)
        assert all(isinstance(x, float) for x in row.predictions)


def test_pipeline_integration(test_data):
    """Test SparkTabNet works correctly in a Spark ML pipeline."""
    # Create pipeline
    pipeline = Pipeline(stages=[
        SparkTabNetEstimator(inputCol="features", outputCol="predictions")
    ])
    
    # Fit pipeline
    model = pipeline.fit(test_data)
    
    # Transform data using DataFrame interface
    wrapped_df = SparkDataFrame(test_data)
    predictions = model.transform(test_data)
    
    # Verify results
    assert "predictions" in predictions.columns
    assert predictions.count() == test_data.count()


def test_data_type_compatibility(spark):
    """Test handling of different data types and schemas."""
    # Create test cases with different data types
    test_cases = [
        # Numeric features with balanced classes
        spark.createDataFrame(
            [(np.random.randn(5).tolist(), float(i % 2)) for i in range(10)],
            ["features", "label"]
        ),
        # Integer features with balanced classes
        spark.createDataFrame(
            [(np.random.randint(0, 10, 5).tolist(), float(i % 2)) for i in range(10)],
            ["features", "label"]
        ),
        # Mixed numeric types with balanced classes
        spark.createDataFrame(
            [([float(x) for x in np.random.randn(5)], float(i % 2)) for i in range(10)],
            ["features", "label"]
        )
    ]
    
    estimator = SparkTabNetEstimator(inputCol="features", outputCol="predictions")
    
    for test_df in test_cases:
        # Test with DataFrame interface
        wrapped_df = SparkDataFrame(test_df)
        model = estimator.fit(test_df)
        result = model.transform(test_df)
        assert result.count() == test_df.count()


def test_large_scale_performance(spark):
    """Test performance with large datasets."""
    # Create large dataset with balanced classes
    n_samples = 10000
    n_features = 20
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.array([i % 2 for i in range(n_samples)])  # Ensure balanced classes
    
    data = [(features[i].tolist(), float(labels[i])) for i in range(n_samples)]
    schema = StructType([
        StructField("features", ArrayType(DoubleType())),
        StructField("label", DoubleType())
    ])
    large_df = spark.createDataFrame(data, schema)
    
    # Train and transform using DataFrame interface
    wrapped_df = SparkDataFrame(large_df)
    estimator = SparkTabNetEstimator(inputCol="features", outputCol="predictions")
    model = estimator.fit(large_df)
    result = model.transform(large_df)
    
    # Verify results
    assert result.count() == n_samples


def test_model_persistence(tmp_path, test_data):
    """Test model save and load functionality."""
    # Train model
    estimator = SparkTabNetEstimator(inputCol="features", outputCol="predictions")
    model = estimator.fit(test_data)
    
    # Save model
    model_path = str(tmp_path / "tabnet_model")
    model.save(model_path)
    
    # Load model
    loaded_model = SparkTabNetModel.load(model_path)
    
    # Compare predictions using DataFrame interface
    wrapped_df = SparkDataFrame(test_data)
    original_preds = model.transform(test_data).select("predictions").collect()
    loaded_preds = loaded_model.transform(test_data).select("predictions").collect()
    
    for orig, loaded in zip(original_preds, loaded_preds):
        np.testing.assert_array_almost_equal(orig.predictions, loaded.predictions)


def test_custom_label_column(spark):
    """Test that estimator works with custom label columns and handles multi-task warning."""
    # Create test data with multiple label columns
    n_samples = 100
    n_features = 5
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    labels1 = np.random.randint(0, 2, size=n_samples)
    labels2 = np.random.randint(0, 3, size=n_samples)  # Second task with 3 classes
    
    data = [(features[i].tolist(), float(labels1[i]), float(labels2[i])) for i in range(n_samples)]
    schema = StructType([
        StructField("features", ArrayType(DoubleType())),
        StructField("task1_label", DoubleType()),
        StructField("task2_label", DoubleType())
    ])
    df = spark.createDataFrame(data, schema)
    
    # Train model with multiple label columns
    estimator = SparkTabNetEstimator(
        inputCol="features",
        outputCol="predictions",
        labelCols=["task1_label", "task2_label"]
    )
    
    # Verify warning about multi-task not being supported
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = estimator.fit(df)
        assert len(w) > 0
        assert any("Multi-task learning is not yet supported" in str(warning.message) for warning in w)
    
    # Transform data
    result = model.transform(df)
    
    # Verify predictions (should be based on first label column only)
    assert "predictions" in result.columns
    assert result.count() == df.count()
    predictions = result.select("predictions").collect()
    for row in predictions:
        assert isinstance(row.predictions, list)
        assert all(isinstance(x, float) for x in row.predictions)
        # Should only have predictions for first task
        assert len(row.predictions) == 2  # Binary classification has 2 probabilities


def test_pickle_serialization(test_data):
    """Test that the entire SparkTabNetModel can be serialized using pickle."""
    import pickle
    from io import BytesIO
    
    # Train model
    estimator = SparkTabNetEstimator(inputCol="features", outputCol="predictions")
    model = estimator.fit(test_data)
    
    # Test pickling the entire model
    bio = BytesIO()
    pickle.dump(model, bio)
    bio.seek(0)
    loaded_model = pickle.load(bio)
    
    # Compare predictions
    original_preds = model.transform(test_data).select("predictions").collect()
    loaded_preds = loaded_model.transform(test_data).select("predictions").collect()
    
    for orig, loaded in zip(original_preds, loaded_preds):
        np.testing.assert_array_almost_equal(orig.predictions, loaded.predictions)


def test_model_serialization_edge_cases(spark, tmp_path):
    """Test edge cases in model serialization."""
    # Create minimal test data
    # Create data with two classes for classification
    df = spark.createDataFrame([
        (np.random.randn(5).tolist(), float(0)),
        (np.random.randn(5).tolist(), float(1))
    ], ["features", "label"])
    
    # Train model
    estimator = SparkTabNetEstimator(inputCol="features", outputCol="predictions")
    model = estimator.fit(df)
    
    # Test saving to empty path
    with pytest.raises(ValueError, match="Path cannot be empty"):
        model.save("")
    
    # Test loading from empty path
    with pytest.raises(ValueError, match="Path cannot be empty"):
        SparkTabNetModel.load("")
    
    # Test loading from non-existent path
    with pytest.raises(ValueError, match="Model path does not exist"):
        SparkTabNetModel.load(str(tmp_path / "nonexistent"))
    
    # Save model without TabNet state
    model_path = str(tmp_path / "incomplete_model")
    model.save(model_path)
    os.remove(os.path.join(model_path, "tabnet_model.pkl"))
    
    # Test loading model with missing TabNet state
    with pytest.raises(ValueError, match="TabNet model state file not found"):
        SparkTabNetModel.load(model_path)


def test_mlflow_integration(spark, tmp_path, test_data):
    """Test MLflow integration for model serialization."""
    import mlflow
    import mlflow.spark

    # Train model
    estimator = SparkTabNetEstimator(inputCol="features", outputCol="predictions")
    model = estimator.fit(test_data)

    # Log model using MLflow
    with mlflow.start_run():
        model_path = str(tmp_path / "mlflow_model")
        mlflow.spark.save_model(model, model_path)

        # Load model using MLflow
        loaded_model = mlflow.spark.load_model(model_path)

        print("\nLoaded model type:", type(loaded_model))
        print("Loaded model attributes:", dir(loaded_model))
        if hasattr(loaded_model, "stages"):
            print("Pipeline stages:", [type(stage) for stage in loaded_model.stages])
            for stage in loaded_model.stages:
                print(f"Stage type: {type(stage)}")
                print(f"Stage attributes: {dir(stage)}")
                if hasattr(stage, "_mlflow_model_info"):
                    print("Stage MLflow info:", stage._mlflow_model_info)

        # Verify predictions
        original_preds = model.transform(test_data).select("predictions").collect()
        loaded_preds = loaded_model.transform(test_data).select("predictions").collect()

        for orig, loaded in zip(original_preds, loaded_preds):
            np.testing.assert_array_almost_equal(orig.predictions, loaded.predictions)

        # Instead of asserting a top-level 'tabnet', extract the SparkTabNetModel stage.
        if hasattr(loaded_model, "stages"):
            spark_tabnet_stage = None
            for stage in loaded_model.stages:
                if hasattr(stage, "tabnet"):
                    spark_tabnet_stage = stage
                    break
            assert spark_tabnet_stage is not None, "SparkTabNetModel stage not found in loaded pipeline"
            assert spark_tabnet_stage.tabnet is not None, "SparkTabNetModel stage missing tabnet attribute"
        else:
            # For non-pipeline models, assert directly.
            assert hasattr(loaded_model, 'tabnet') and loaded_model.tabnet is not None, "Loaded model tabnet attribute missing"


def test_edge_cases(spark, test_data):
    """Test handling of edge cases and invalid inputs."""
    # Empty DataFrame
    empty_df = spark.createDataFrame(
        [], 
        StructType([
            StructField("features", ArrayType(DoubleType())),
            StructField("label", DoubleType())
        ])
    )
    
    estimator = SparkTabNetEstimator(inputCol="features", outputCol="predictions")
    
    # Should raise error for empty DataFrame
    with pytest.raises(ValueError):
        estimator.fit(empty_df)
    
    # Invalid column name
    with pytest.raises(ValueError):
        SparkTabNetEstimator(inputCol="nonexistent", outputCol="predictions").fit(test_data)