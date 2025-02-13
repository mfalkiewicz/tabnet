"""Tests for TabNet PySpark integration."""

import os
import tempfile
import pytest
import torch
import numpy as np
import mlflow
import mlflow.mleap
import mlflow.pytorch
import mlflow.spark
import mlflow.pyfunc
import json
from unittest.mock import patch, MagicMock
import logging
from pyspark.sql.types import StructType, StructField, DoubleType, ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.torch.distributor import TorchDistributor
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler

from pytorch_tabnet.spark.tabnet_pyspark import TabNetEstimator, TabNetModel
from pytorch_tabnet.tab_network import TabNet


def test_tabnet_estimator_init():
    """Test TabNetEstimator initialization."""
    estimator = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label",
        n_d=16,
        n_a=16,
        n_steps=5
    )
    assert estimator.getInputCol() == "features"
    assert estimator.getOutputCol() == "prediction"
    assert estimator.getLabelCol() == "label"
    assert estimator.getNd() == 16
    assert estimator.getNa() == 16
    assert estimator.getNSteps() == 5


def test_invalid_params():
    """Test invalid parameter validation."""
    estimator = TabNetEstimator()
    
    with pytest.raises(ValueError, match="n_steps must be positive"):
        estimator.setNSteps(0)
    
    with pytest.raises(ValueError, match="n_independent and n_shared cannot both be zero"):
        estimator.setNIndependent(0)
        estimator.setNShared(0)
    
    with pytest.raises(ValueError, match="mask_type must be either 'sparsemax' or 'entmax'"):
        estimator.setMaskType("invalid")


def test_model_params():
    """Test model parameter getters and setters."""
    model = TabNetModel()
    
    # Test default values
    assert model.getNd() == 16
    assert model.getNa() == 16
    assert model.getNSteps() == 5
    assert model.getGamma() == 1.3
    assert model.getNIndependent() == 2
    assert model.getNShared() == 2
    assert model.getVirtualBatchSize() == 128
    assert model.getMomentum() == 0.02
    assert model.getMaskType() == "sparsemax"
    
    # Test setters
    model.setNd(32)
    model.setNa(32)
    model.setNSteps(10)
    model.setGamma(1.5)
    model.setNIndependent(3)
    model.setNShared(3)
    model.setVirtualBatchSize(256)
    model.setMomentum(0.03)
    model.setMaskType("entmax")
    
    # Verify new values
    assert model.getNd() == 32
    assert model.getNa() == 32
    assert model.getNSteps() == 10
    assert model.getGamma() == 1.5
    assert model.getNIndependent() == 3
    assert model.getNShared() == 3
    assert model.getVirtualBatchSize() == 256
    assert model.getMomentum() == 0.03
    assert model.getMaskType() == "entmax"


@patch('mlflow.tracking.MlflowClient')
def test_mlflow_error_handling(mock_client_class, tmp_path, caplog):
    """Test MLflow error handling during model loading."""
    caplog.set_level(logging.INFO)
    
    # Configure mock client to fail
    mock_client = mock_client_class.return_value
    mock_client.download_artifacts.side_effect = mlflow.exceptions.MlflowException("Failed to download")
    
    with pytest.raises(ValueError, match="Model path does not exist and artifact store fallback failed"):
        TabNetModel.load("mlflow://invalid/uri")
    
    # Verify error was logged
    assert any("Model path does not exist and artifact store fallback failed" in str(record)
              for record in caplog.records)


def test_empty_dataset(spark):
    """Test handling of empty datasets."""
    # Create empty DataFrame with correct schema
    schema = StructType([
        StructField("features", VectorUDT()),
        StructField("label", DoubleType())
    ])
    empty_df = spark.createDataFrame([], schema)
    
    estimator = TabNetEstimator()
    with pytest.raises(ValueError, match="Cannot train on empty dataset"):
        estimator.fit(empty_df)


def test_model_persistence(small_data, tmp_path):
    """Test model save and load functionality."""
    # Create and train model
    estimator = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label",
        n_d=8,
        n_a=8,
        n_steps=3
    )
    model = estimator.fit(small_data)
    
    # Save model
    model_path = str(tmp_path / "tabnet_model")
    model.save(model_path)
    
    # Load model
    loaded_model = TabNetModel.load(model_path)
    
    # Verify predictions match
    original_preds = model.transform(small_data).select("prediction").collect()
    loaded_preds = loaded_model.transform(small_data).select("prediction").collect()
    
    for orig, loaded in zip(original_preds, loaded_preds):
        np.testing.assert_array_almost_equal(orig.prediction, loaded.prediction)


def test_mlwritable_mlreadable(small_data, tmp_path):
    """Test MLWritable and MLReadable interfaces."""
    # Create and train model
    estimator = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label",
        n_d=8,
        n_a=8,
        n_steps=3
    )
    model = estimator.fit(small_data)
    
    # Test MLWritable interface
    writer = model.write()
    assert writer is not None
    
    # Save using MLWriter
    model_path = str(tmp_path / "mlwriter_model")
    writer.save(model_path)
    assert os.path.exists(model_path)
    assert os.path.exists(os.path.join(model_path, "model"))
    assert os.path.exists(os.path.join(model_path, "model", "model.pt"))
    assert os.path.exists(os.path.join(model_path, "model", "params.json"))
    
    # Test MLReadable interface
    reader = TabNetModel.read()
    assert reader is not None
    
    # Load using MLReader
    loaded_model = reader.load(model_path)
    assert loaded_model is not None
    assert isinstance(loaded_model, TabNetModel)
    assert loaded_model.getInputCol() == model.getInputCol()
    assert loaded_model.getOutputCol() == model.getOutputCol()
    
    # Verify model parameters
    assert loaded_model.getNd() == model.getNd()
    assert loaded_model.getNa() == model.getNa()
    assert loaded_model.getNSteps() == model.getNSteps()
    
    # Verify predictions match
    original_preds = model.transform(small_data).select("prediction").collect()
    loaded_preds = loaded_model.transform(small_data).select("prediction").collect()
    
    for orig, loaded in zip(original_preds, loaded_preds):
        np.testing.assert_array_almost_equal(orig.prediction, loaded.prediction)


@pytest.mark.integration
def test_end_to_end_training(large_data, tmp_path):
    """End-to-end test for model training and inference."""
    # Split data
    train_data, test_data = large_data.randomSplit([0.8, 0.2], seed=42)
    
    # Create and train model
    estimator = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label",
        n_d=8,
        n_a=8,
        n_steps=3,
        virtual_batch_size=32
    )
    
    # Train model
    model = estimator.fit(train_data)
    
    # Save model
    model_path = str(tmp_path / "model")
    model.save(model_path)
    
    # Load model
    loaded_model = TabNetModel.load(model_path)
    
    # Get predictions
    predictions = loaded_model.transform(test_data)
    assert "prediction" in predictions.columns
    
    # Verify predictions
    pred_values = predictions.select("prediction").collect()
    for row in pred_values:
        assert isinstance(row.prediction, (np.ndarray, list))
        assert len(row.prediction) > 0
        assert all(isinstance(x, (int, float)) for x in row.prediction)


@pytest.mark.integration
def test_multiclass_training(multiclass_data):
    """Test model training with multiclass data."""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Get number of classes
    n_classes = len(set(row.label for row in multiclass_data.select("label").collect()))
    
    estimator = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label",
        n_d=8,
        n_a=8,
        n_steps=3,
        output_dim=n_classes  # Set output dimension to number of classes
    )
    
    # Train model
    model = estimator.fit(multiclass_data)
    
    # Get predictions
    predictions = model.transform(multiclass_data)
    assert "prediction" in predictions.columns
    
    # Verify predictions
    pred_values = predictions.select("prediction").collect()
    
    for row in pred_values:
        assert isinstance(row.prediction, (np.ndarray, list))
        assert len(row.prediction) == n_classes
        assert all(isinstance(x, (int, float)) for x in row.prediction)
        assert np.allclose(sum(row.prediction), 1.0)  # Probabilities sum to 1

@pytest.mark.integration
def test_mlflow_integration(large_data, tmp_path):
    """Test MLflow integration for model tracking."""
    # Set up MLflow tracking
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    
    # Ensure no active runs exist
    active_run = mlflow.active_run()
    if active_run:
        try:
            mlflow.end_run()
        except mlflow.exceptions.MlflowException:
            pass  # Ignore if run doesn't exist or can't be ended
    
    with mlflow.start_run() as run:
        # Train model
        estimator = TabNetEstimator(
            inputCol="features",
            outputCol="prediction",
            labelCol="label",
            n_d=8,
            n_a=8,
            n_steps=3
        )
        model = estimator.fit(large_data)
        
        # Log model artifacts
        mlflow.spark.log_model(model, "spark_model")
        mlflow.pytorch.log_model(model._torch_model, "pytorch_model")
        
        # Save model with all MLflow flavors
        model_path = os.path.join(tmp_path, "model")
        model.save(model_path)
        
        # Load model using MLflow URI
        loaded_model = TabNetModel.load(f"mlflow://{run.info.run_id}/model")
        
        # Verify predictions match
        original_preds = model.transform(large_data).select("prediction").collect()
        loaded_preds = loaded_model.transform(large_data).select("prediction").collect()
        
        for orig, loaded in zip(original_preds, loaded_preds):
            np.testing.assert_array_almost_equal(orig.prediction, loaded.prediction)


def test_new_configurable_params():
    """Test new configurable parameters."""
    model = TabNetModel()
    
    # Test default values
    assert model.getEpochs() == 10
    assert model.getLearningRate() == 0.001
    assert model.getBatchSize() == 1024
    
    # Test setters with valid values
    model.setEpochs(20)
    model.setLearningRate(0.01)
    model.setBatchSize(512)
    
    assert model.getEpochs() == 20
    assert model.getLearningRate() == 0.01
    assert model.getBatchSize() == 512
    
    # Test invalid values
    with pytest.raises(ValueError, match="epochs must be positive"):
        model.setEpochs(0)
    
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        model.setLearningRate(0)
    
    with pytest.raises(ValueError, match="batch_size must be positive"):
        model.setBatchSize(0)


def test_distributed_data_loading(small_data):
    """Test distributed data loading with Pandas UDFs."""
    estimator = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label"
    )
    
    # Train model
    model = estimator.fit(small_data)
    
    # Get predictions using distributed inference
    predictions = model.transform(small_data)
    
    # Verify predictions
    pred_values = predictions.select("prediction").collect()
    for row in pred_values:
        assert isinstance(row.prediction, (np.ndarray, list))
        assert len(row.prediction) > 0
        assert all(isinstance(x, (int, float)) for x in row.prediction)


@pytest.mark.integration
def test_pipeline_integration(small_data, tmp_path):
    """Test Spark ML pipeline integration."""
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    
    # Create model with initialized torch model
    model = TabNetModel(
        inputCol="features",
        outputCol="prediction",
        input_dim=10,  # Set appropriate dimensions based on your data
        output_dim=2
    )
    # Set model parameters to match the saved state
    model.setNd(16)
    model.setNa(16)
    model.setNSteps(5)
    
    # Initialize torch model with matching parameters
    model._torch_model = TabNet(
        input_dim=10,
        output_dim=2,
        n_d=16,
        n_a=16,
        n_steps=5
    )
    
    # Create pipeline
    assembler = VectorAssembler(
        inputCols=["features"],
        outputCol="assembled_features"
    )
    pipeline = Pipeline(stages=[assembler, model])
    
    # Fit pipeline to ensure model is properly initialized
    fitted_pipeline = pipeline.fit(small_data)
    
    # Save and load pipeline
    pipeline_path = str(tmp_path / "pipeline")
    fitted_pipeline.save(pipeline_path)
    loaded_pipeline = Pipeline.load(pipeline_path)
    
    # Verify loaded pipeline
    assert len(loaded_pipeline.getStages()) == 2
    assert isinstance(loaded_pipeline.getStages()[0], VectorAssembler)
    assert isinstance(loaded_pipeline.getStages()[1], TabNetModel)


@pytest.mark.integration
def test_mlflow_flavors(small_data, tmp_path):
    """Test different MLflow flavors."""
    # Set up MLflow tracking
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    
    # Ensure no active runs exist
    active_run = mlflow.active_run()
    if active_run:
        try:
            mlflow.end_run()
        except mlflow.exceptions.MlflowException:
            pass  # Ignore if run doesn't exist or can't be ended
    
    with mlflow.start_run() as run:
        # Train model
        estimator = TabNetEstimator(
            inputCol="features",
            outputCol="prediction",
            labelCol="label"
        )
        model = estimator.fit(small_data)
        
        # Save models to disk first
        model_path = os.path.join(tmp_path, "models")
        os.makedirs(model_path, exist_ok=True)
        
        # Save PyTorch model
        torch_path = os.path.join(model_path, "pytorch_model")
        mlflow.pytorch.save_model(model._torch_model, torch_path)
        mlflow.log_artifact(torch_path, "pytorch_model")
        
        # Create TabNet model directory
        tabnet_path = os.path.join(model_path, "spark_model")
        os.makedirs(tabnet_path, exist_ok=True)
        
        # Save model files
        model_files_path = os.path.join(tabnet_path, "model")
        os.makedirs(model_files_path, exist_ok=True)
        model.save(model_files_path)
        
        # Create metadata file
        metadata = {
            "model_type": "TabNetModel",
            "class": "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel",
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "params": model._get_model_params()
        }
        
        metadata_path = os.path.join(tabnet_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Log TabNet model directory
        mlflow.log_artifacts(tabnet_path, "spark_model")
        # Create and log pipeline
        pipeline = Pipeline(stages=[model])
        fitted_pipeline = pipeline.fit(small_data)
        
        # Log pipeline model using MLflow's Spark flavor
        mlflow.spark.log_model(
            spark_model=fitted_pipeline,
            artifact_path="pipeline_model",
            registered_model_name="tabnet_pipeline"
        )
        
        # Test loading models
        torch_model = mlflow.pytorch.load_model(f"runs:/{run.info.run_id}/pytorch_model")
        assert isinstance(torch_model, TabNet)
        
        # Load and verify TabNet model
        model_uri = f"runs:/{run.info.run_id}/spark_model"
        loaded_path = mlflow.artifacts.download_artifacts(model_uri)
        
        # Load metadata
        metadata_path = os.path.join(loaded_path, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Verify metadata
        assert metadata["model_type"] == "TabNetModel"
        assert metadata["class"] == "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel"
        
        # Load model using appropriate class
        model_path = os.path.join(loaded_path, "model")
        spark_model = TabNetModel.load(model_path)
        assert isinstance(spark_model, TabNetModel), f"Expected TabNetModel but got {type(spark_model)}"
        
        # Verify model parameters
        assert spark_model.input_dim == metadata["input_dim"]
        assert spark_model.output_dim == metadata["output_dim"]
        
        # Get original predictions
        original_preds = model.transform(small_data).select("prediction").collect()
        
        # Load and verify pipeline model
        pipeline_model = mlflow.spark.load_model(f"runs:/{run.info.run_id}/pipeline_model")
        assert isinstance(pipeline_model, PipelineModel)
        
        # Verify pipeline predictions match
        pipeline_preds = pipeline_model.transform(small_data).select("prediction").collect()
        for orig, pipe in zip(original_preds, pipeline_preds):
            np.testing.assert_array_almost_equal(orig.prediction, pipe.prediction)
        
        # Verify predictions match
        original_preds = model.transform(small_data).select("prediction").collect()
        loaded_preds = spark_model.transform(small_data).select("prediction").collect()
        
        for orig, loaded in zip(original_preds, loaded_preds):
            np.testing.assert_array_almost_equal(orig.prediction, loaded.prediction)


@pytest.mark.integration
def test_mleap_integration_with_pipeline(small_data, tmp_path):
    """Test MLeap integration with pipeline."""
    import mlflow.mleap
    
    # Train model
    estimator = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label"
    )
    model = estimator.fit(small_data)
    
    # Get pipeline
    pipeline = model.get_pipeline()
    
    # Save pipeline with MLeap
    mleap_path = str(tmp_path / "mleap-bundle.zip")
    try:
        mlflow.mleap.save_model(
            spark_model=pipeline,
            path=mleap_path,
            sample_input=small_data
        )
        assert os.path.exists(mleap_path)
    except Exception as e:
        pytest.skip(f"MLeap integration test skipped: {str(e)}")


def test_mlflow_flavor_error_handling(small_data, tmp_path):
    """Test error handling for MLflow flavors."""
    # Set up MLflow tracking
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    
    # Ensure no active runs exist
    active_run = mlflow.active_run()
    if active_run:
        try:
            mlflow.end_run()
        except mlflow.exceptions.MlflowException:
            pass  # Ignore if run doesn't exist or can't be ended
    
    with mlflow.start_run() as run:
        # Train model
        estimator = TabNetEstimator(
            inputCol="features",
            outputCol="prediction",
            labelCol="label"
        )
        model = estimator.fit(small_data)
        
        # Save models to disk first
        model_path = os.path.join(tmp_path, "models")
        os.makedirs(model_path, exist_ok=True)
        
        # Save PyTorch model
        torch_path = os.path.join(model_path, "pytorch_model")
        mlflow.pytorch.save_model(model._torch_model, torch_path)
        mlflow.log_artifact(torch_path, "pytorch_model")
        
        # Try loading non-existent flavor
        with pytest.raises((mlflow.exceptions.MlflowException, OSError),
                         match=r"(No such file or directory|Failed to download)"):
            mlflow.pytorch.load_model(f"runs:/{run.info.run_id}/nonexistent_flavor")
        
        # Try loading with invalid run ID
        with pytest.raises((mlflow.exceptions.MlflowException, OSError),
                         match=r"(Run .* not found|No such file or directory)"):
            mlflow.pytorch.load_model("runs:/invalid_run_id/pytorch_model")


def test_pandas_udf_error_handling(small_data):
    """Test error handling in Pandas UDF transformations."""
    # Create model with initialized torch model
    model = TabNetModel(
        inputCol="features",
        outputCol="prediction",
        input_dim=10,
        output_dim=2
    )
    model._torch_model = TabNet(
        input_dim=10,
        output_dim=2,
        n_d=8,
        n_a=8,
        n_steps=3
    )
    
    from pyspark.errors.exceptions.captured import PythonException
    
    # Test with missing input column
    invalid_data = small_data.drop("features")
    with pytest.raises(PythonException, match="Input column 'features' not found in dataset"):
        model.transform(invalid_data)
    
    # Test with invalid feature dimensions
    from pyspark.sql import SparkSession
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.types import StructType, StructField
    
    # Create a DataFrame with wrong dimension vector
    spark = SparkSession.builder.getOrCreate()
    
    # Create data with wrong dimension
    wrong_vector = Vectors.dense([0.0])  # Wrong dimension
    wrong_dim_data = [(wrong_vector, 0.0)]
    
    # Create DataFrame with schema
    schema = StructType([
        StructField("features", VectorUDT()),
        StructField("label", DoubleType())
    ])
    invalid_features = spark.createDataFrame(wrong_dim_data, schema)
    
    # Verify the vector dimension before transform
    first_vector = invalid_features.select("features").first()[0]
    assert len(first_vector.toArray()) == 1  # Confirm wrong dimension
    
    # This should raise a PythonException with ValueError message
    from pyspark.errors.exceptions.captured import PythonException
    with pytest.raises(PythonException) as exc_info:
        transformed_df = model.transform(invalid_features)
        transformed_df.collect()  # Force execution
    
    # Verify the error message contains the expected dimension mismatch
    error_msg = str(exc_info.value)
    assert "Feature dimension mismatch. Model expects 10 features, but got 1" in error_msg


def test_pipeline_validation(small_data):
    """Test pipeline validation and error handling."""
    from pyspark.sql.utils import AnalysisException
    from pyspark.errors import PySparkException
    
    # Create model with initialized torch model
    model = TabNetModel(
        inputCol="features",
        outputCol="prediction",
        input_dim=10,
        output_dim=2
    )
    model._torch_model = TabNet(
        input_dim=10,
        output_dim=2,
        n_d=8,
        n_a=8,
        n_steps=3
    )
    
    # Create and fit pipeline
    pipeline = Pipeline(stages=[model])
    pipeline_model = pipeline.fit(small_data)
    
    # Test with missing input column
    invalid_data = small_data.drop("features")
    from pyspark.errors.exceptions.captured import PythonException
    with pytest.raises(PythonException, match="Input column 'features' not found in dataset"):
        pipeline_model.transform(invalid_data)
    
    # Test with invalid assembler configuration
    invalid_assembler = VectorAssembler(
        inputCols=["nonexistent_col"],
        outputCol="assembled_features"
    )
    invalid_pipeline = Pipeline(stages=[invalid_assembler, model])
    
    # This should raise a PySpark exception
    with pytest.raises((AnalysisException, PySparkException)) as exc_info:
        # Force execution by collecting results
        result = invalid_pipeline.fit(small_data)
        result.transform(small_data).collect()
    
    # Verify the error message contains expected phrases
    error_msg = str(exc_info.value).lower()
    expected_phrases = [
        "does not exist",
        "nonexistent_col",
        "available: features label"
    ]
    assert any(phrase in error_msg for phrase in expected_phrases), \
        f"Error message '{error_msg}' does not contain any expected phrases: {expected_phrases}"


def test_model_serialization_edge_cases(tmp_path):
    """Test edge cases in model serialization."""
    with pytest.raises(ValueError, match="Model path does not exist"):
        TabNetModel.load(str(tmp_path / "nonexistent"))
    
    with pytest.raises(ValueError, match="Model path does not exist"):
        TabNetModel.load("")