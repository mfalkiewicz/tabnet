"""Tests for TabNet persistence improvements."""

import os
import json
import pytest
import mlflow
import numpy as np
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler

from pytorch_tabnet.spark.tabnet_pyspark import TabNetEstimator, TabNetModel


def test_mlmodel_metadata_format(small_data, tmp_path):
    """Test that metadata is saved in MLmodel format."""
    # Train model
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
    
    # Verify MLmodel file exists and has correct format
    mlmodel_path = os.path.join(model_path, "MLmodel")
    assert os.path.exists(mlmodel_path), "MLmodel file not found"
    
    print(f"\nMLmodel path: {mlmodel_path}")
    print(f"MLmodel exists: {os.path.exists(mlmodel_path)}")
    
    # Read and print raw contents
    with open(mlmodel_path, "r") as f:
        raw_content = f.read()
    print(f"\nMLmodel file contents:\n{raw_content}\n")
    
    try:
        # Try to parse as JSON
        with open(mlmodel_path, "r") as f:
            metadata = json.load(f)
        print("Successfully parsed as JSON")
    except json.JSONDecodeError:
        print("Failed to parse as JSON, might be YAML")
        import yaml
        with open(mlmodel_path, "r") as f:
            metadata = yaml.safe_load(f)
        print("Successfully parsed as YAML")
    
    # Verify required metadata fields
    assert "class" in metadata, "Missing 'class' in metadata"
    assert metadata["class"] == "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel"
    assert "timestamp" in metadata, "Missing 'timestamp' in metadata"
    assert "sparkVersion" in metadata, "Missing 'sparkVersion' in metadata"
    assert "params" in metadata, "Missing 'params' in metadata"
    
    # Verify model parameters are saved
    params = metadata["params"]
    assert params["input_dim"] == model.input_dim
    assert params["output_dim"] == model.output_dim
    assert params["n_d"] == model.getNd()
    assert params["n_a"] == model.getNa()
    assert params["n_steps"] == model.getNSteps()


def test_pipeline_stage_persistence(small_data, tmp_path):
    """Test that TabNet model is correctly persisted within a pipeline."""
    # Create preprocessing stages
    assembler = VectorAssembler(
        inputCols=["features"],
        outputCol="assembled_features"
    )
    scaler = StandardScaler(
        inputCol="assembled_features",
        outputCol="scaled_features",
        withStd=True,
        withMean=True
    )
    
    # Create and train TabNet model
    tabnet = TabNetEstimator(
        inputCol="scaled_features",
        outputCol="prediction",
        labelCol="label",
        n_d=8,
        n_a=8,
        n_steps=3
    )
    
    # Create and fit pipeline
    pipeline = Pipeline(stages=[assembler, scaler, tabnet])
    pipeline_model = pipeline.fit(small_data)
    
    # Save pipeline
    pipeline_path = str(tmp_path / "pipeline")
    pipeline_model.save(pipeline_path)
    
    # Verify each stage's metadata is saved correctly
    for i, stage in enumerate(pipeline_model.stages):
        stage_path = os.path.join(pipeline_path, f"stages/{i}")
        mlmodel_path = os.path.join(stage_path, "MLmodel")
        
        if isinstance(stage, TabNetModel):
            # Verify TabNet stage has correct metadata
            assert os.path.exists(mlmodel_path), f"MLmodel file not found for stage {i}"
            
            with open(mlmodel_path, "r") as f:
                metadata = json.load(f)
            
            assert metadata["class"] == "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel"
            assert "params" in metadata
            
            # Verify model files are saved
            model_dir = os.path.join(stage_path, "model")
            assert os.path.exists(os.path.join(model_dir, "model.pt")), "PyTorch model file not found"
            assert os.path.exists(os.path.join(model_dir, "params.json")), "Parameters file not found"


@pytest.mark.integration
def test_mlflow_pipeline_persistence(small_data, tmp_path):
    """Test MLflow integration with pipeline persistence."""
    # Set up MLflow tracking
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    
    with mlflow.start_run() as run:
        # Create pipeline with preprocessing
        assembler = VectorAssembler(
            inputCols=["features"],
            outputCol="assembled_features"
        )
        scaler = StandardScaler(
            inputCol="assembled_features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        tabnet = TabNetEstimator(
            inputCol="scaled_features",
            outputCol="prediction",
            labelCol="label",
            n_d=8,
            n_a=8,
            n_steps=3
        )
        
        # Create and fit pipeline
        pipeline = Pipeline(stages=[assembler, scaler, tabnet])
        pipeline_model = pipeline.fit(small_data)
        
        # Log pipeline model
        mlflow.spark.log_model(
            spark_model=pipeline_model,
            artifact_path="pipeline",
            registered_model_name="tabnet_pipeline"
        )
        
        # Get predictions from original pipeline
        original_preds = pipeline_model.transform(small_data).select("prediction").collect()
        
        # Load pipeline from MLflow
        loaded_pipeline = mlflow.spark.load_model(f"runs:/{run.info.run_id}/pipeline")
        
        # Verify loaded pipeline structure
        assert isinstance(loaded_pipeline, PipelineModel)
        assert len(loaded_pipeline.stages) == 3
        assert isinstance(loaded_pipeline.stages[-1], TabNetModel)
        
        # Verify predictions match
        loaded_preds = loaded_pipeline.transform(small_data).select("prediction").collect()
        for orig, loaded in zip(original_preds, loaded_preds):
            np.testing.assert_array_almost_equal(orig.prediction, loaded.prediction)
        
        # Verify TabNet stage metadata
        artifact_path = mlflow.artifacts.download_artifacts(f"runs:/{run.info.run_id}/pipeline")
        
        # Debug logging
        print("\nDEBUG: Pipeline model stages:")
        for i, stage in enumerate(pipeline_model.stages):
            print(f"Stage {i}: {stage.__class__.__name__} (uid: {stage.uid})")
        
        # List MLflow artifact directory
        print("\nDEBUG: MLflow artifact directory:")
        os.system(f"ls -R {artifact_path}")
        
        # Get TabNet stage path
        stage_uid = pipeline_model.stages[-1].uid
        if stage_uid.startswith("TabNetModel_"):
            stage_uid = stage_uid[len("TabNetModel_"):]  # Remove TabNetModel_ prefix
        stage_name = f"2_TabNetModel_{stage_uid}"
        tabnet_stage_path = os.path.join(artifact_path, "sparkml", "stages", stage_name)
        mlmodel_path = os.path.join(tabnet_stage_path, "MLmodel")
        
        print(f"\nDEBUG: Looking for MLmodel at: {mlmodel_path}")
        
        assert os.path.exists(mlmodel_path), "TabNet MLmodel file not found in MLflow artifacts"
        
        with open(mlmodel_path, "r") as f:
            metadata = json.load(f)
        
        assert metadata["class"] == "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel"
        assert "params" in metadata
        
        # Verify model files are present
        model_dir = os.path.join(tabnet_stage_path, "model")
        assert os.path.exists(os.path.join(model_dir, "model.pt")), "PyTorch model file not found"
        assert os.path.exists(os.path.join(model_dir, "params.json")), "Parameters file not found"


def test_factory_function_usage(small_data, tmp_path):
    """Test that the model factory function is used during loading."""
    # Train model
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
    
    # Verify model parameters match
    assert loaded_model.getNd() == model.getNd()
    assert loaded_model.getNa() == model.getNa()
    assert loaded_model.getNSteps() == model.getNSteps()
    assert loaded_model.input_dim == model.input_dim
    assert loaded_model.output_dim == model.output_dim
    
    # Verify predictions match
    original_preds = model.transform(small_data).select("prediction").collect()
    loaded_preds = loaded_model.transform(small_data).select("prediction").collect()
    
    for orig, loaded in zip(original_preds, loaded_preds):
        np.testing.assert_array_almost_equal(orig.prediction, loaded.prediction)