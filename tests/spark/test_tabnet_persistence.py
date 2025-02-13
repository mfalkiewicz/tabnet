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

    # Verify MLflow flavor definitions
    assert "flavors" in metadata, "Missing 'flavors' in metadata"
    flavors = metadata["flavors"]
    
    # Verify spark flavor
    assert "spark" in flavors, "Missing spark flavor"
    spark_flavor = flavors["spark"]
    assert "model_data" in spark_flavor, "Missing model_data in spark flavor"
    assert spark_flavor["model_data"] == "sparkml"
    assert "spark_version" in spark_flavor, "Missing spark_version in spark flavor"
    
    # Verify python_function flavor
    assert "python_function" in flavors, "Missing python_function flavor"
    pyfunc_flavor = flavors["python_function"]
    assert "loader_module" in pyfunc_flavor, "Missing loader_module in python_function flavor"
    assert pyfunc_flavor["loader_module"] == "mlflow.spark"
    assert "model_path" in pyfunc_flavor, "Missing model_path in python_function flavor"
    assert pyfunc_flavor["model_path"] == "sparkml"

    # Verify model artifacts are in correct location
    sparkml_path = os.path.join(model_path, "sparkml")
    assert os.path.exists(sparkml_path), "sparkml directory not found"
    assert os.path.exists(os.path.join(sparkml_path, "model.pt")), "model.pt not found in sparkml"
    assert os.path.exists(os.path.join(sparkml_path, "params.json")), "params.json not found in sparkml"
    assert os.path.exists(os.path.join(sparkml_path, "metadata")), "metadata not found in sparkml"

    # Verify environment files
    assert os.path.exists(os.path.join(model_path, "requirements.txt")), "requirements.txt not found"
    assert os.path.exists(os.path.join(model_path, "python_env.yaml")), "python_env.yaml not found"


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
        if isinstance(stage, TabNetModel):
            # Find stage directory with correct prefix
            stage_prefix = f"{i}_"
            stage_dir = None
            stages_path = os.path.join(pipeline_path, "stages")
            if os.path.exists(stages_path):
                for dirname in os.listdir(stages_path):
                    if dirname.startswith(stage_prefix):
                        stage_dir = dirname
                        break
            
            # Verify TabNet stage has correct metadata
            assert stage_dir is not None, f"Stage directory with prefix {stage_prefix} not found"
            stage_path = os.path.join(stages_path, stage_dir)
            mlmodel_path = os.path.join(stage_path, "MLmodel")
            assert os.path.exists(mlmodel_path), f"MLmodel file not found for stage {i}"
            
            with open(mlmodel_path, "r") as f:
                metadata = json.load(f)
            
            # Verify flavor definitions in pipeline stage
            assert "flavors" in metadata, "Missing 'flavors' in stage metadata"
            flavors = metadata["flavors"]
            assert "spark" in flavors, "Missing spark flavor in stage"
            assert "python_function" in flavors, "Missing python_function flavor in stage"
            
            assert metadata["class"] == "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel"
            assert "params" in metadata
            
            # Verify model files are saved in sparkml structure
            sparkml_path = os.path.join(stage_path, "sparkml")
            assert os.path.exists(sparkml_path), "sparkml directory not found"
            assert os.path.exists(os.path.join(sparkml_path, "model.pt")), "PyTorch model file not found"
            assert os.path.exists(os.path.join(sparkml_path, "params.json")), "Parameters file not found"
            assert os.path.exists(os.path.join(sparkml_path, "metadata")), "Metadata file not found"


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
        
        # Find TabNet stage directory with correct prefix
        stage_prefix = "2_"  # TabNet is the third stage (index 2)
        stage_dir = None
        stages_path = os.path.join(artifact_path, "sparkml", "stages")
        if os.path.exists(stages_path):
            for dirname in os.listdir(stages_path):
                if dirname.startswith(stage_prefix):
                    stage_dir = dirname
                    break
        
        assert stage_dir is not None, f"Stage directory with prefix {stage_prefix} not found"
        tabnet_stage_path = os.path.join(stages_path, stage_dir)
        mlmodel_path = os.path.join(tabnet_stage_path, "MLmodel")
        
        print(f"\nDEBUG: Looking for MLmodel at: {mlmodel_path}")
        
        assert os.path.exists(mlmodel_path), "TabNet MLmodel file not found in MLflow artifacts"
        
        with open(mlmodel_path, "r") as f:
            metadata = json.load(f)
        
        # Verify MLflow flavor definitions in logged model
        assert "flavors" in metadata, "Missing 'flavors' in logged model metadata"
        flavors = metadata["flavors"]
        assert "spark" in flavors, "Missing spark flavor in logged model"
        assert "python_function" in flavors, "Missing python_function flavor in logged model"
        
        assert metadata["class"] == "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel"
        assert "params" in metadata
        
        # Verify model files are present in sparkml structure
        sparkml_path = os.path.join(tabnet_stage_path, "sparkml")
        assert os.path.exists(sparkml_path), "sparkml directory not found"
        assert os.path.exists(os.path.join(sparkml_path, "model.pt")), "PyTorch model file not found"
        assert os.path.exists(os.path.join(sparkml_path, "params.json")), "Parameters file not found"
        assert os.path.exists(os.path.join(sparkml_path, "metadata")), "Metadata file not found"

        # Verify environment files
        assert os.path.exists(os.path.join(artifact_path, "requirements.txt")), "requirements.txt not found"
        assert os.path.exists(os.path.join(artifact_path, "python_env.yaml")), "python_env.yaml not found"


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


def test_standalone_mlflow_logging(small_data, tmp_path):
    """Test MLflow logging of standalone TabNet model."""
    # Set up MLflow tracking
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    
    with mlflow.start_run() as run:
        # Create and train TabNet model
        tabnet = TabNetEstimator(
            inputCol="features",
            outputCol="prediction",
            labelCol="label",
            n_d=8,
            n_a=8,
            n_steps=3
        )
        model = tabnet.fit(small_data)
        
        # Log standalone model
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="standalone_model",
            registered_model_name="tabnet_standalone"
        )
        
        # Get predictions from original model
        original_preds = model.transform(small_data).select("prediction").collect()
        
        # Load model from MLflow
        loaded_model = mlflow.spark.load_model(f"runs:/{run.info.run_id}/standalone_model")
        
        # Verify loaded model type (MLflow wraps models in a pipeline)
        assert isinstance(loaded_model, PipelineModel)
        assert len(loaded_model.stages) == 1
        assert isinstance(loaded_model.stages[0], TabNetModel)
        
        # Verify predictions match
        loaded_preds = loaded_model.transform(small_data).select("prediction").collect()
        for orig, loaded in zip(original_preds, loaded_preds):
            np.testing.assert_array_almost_equal(orig.prediction, loaded.prediction)
        
        # Download and verify artifacts
        artifact_path = mlflow.artifacts.download_artifacts(f"runs:/{run.info.run_id}/standalone_model")
        
        # Verify MLmodel file
        mlmodel_path = os.path.join(artifact_path, "MLmodel")
        assert os.path.exists(mlmodel_path), "MLmodel file not found"
        
        import yaml
        with open(mlmodel_path, "r") as f:
            metadata = yaml.safe_load(f)
        
        # Verify flavor definitions
        assert "flavors" in metadata
        assert "spark" in metadata["flavors"]
        assert "python_function" in metadata["flavors"]
        assert metadata["flavors"]["spark"]["model_data"] == "sparkml"
        
        # Debug: Print directory structure
        print("\nArtifact directory structure:")
        for root, dirs, files in os.walk(artifact_path):
            level = root.replace(artifact_path, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{subindent}{f}")

        # Verify model files
        sparkml_path = os.path.join(artifact_path, "sparkml")
        assert os.path.exists(sparkml_path), "sparkml directory not found"
        
        # Debug: Print sparkml directory contents
        print("\nSparkml directory contents:")
        if os.path.exists(sparkml_path):
            for f in os.listdir(sparkml_path):
                print(f"    {f}")
        
        # Find stage directory with correct prefix
        stage_prefix = "0_"
        stage_dir = None
        stages_path = os.path.join(sparkml_path, "stages")
        if os.path.exists(stages_path):
            for dirname in os.listdir(stages_path):
                if dirname.startswith(stage_prefix):
                    stage_dir = dirname
                    break
        
        assert stage_dir is not None, f"Stage directory with prefix {stage_prefix} not found"
        stage_path = os.path.join(stages_path, stage_dir)
        assert os.path.exists(stage_path), f"Stage directory not found at {stage_path}"
        
        # Verify model files are directly in sparkml directory
        sparkml_dir = os.path.join(stage_path, "sparkml")
        assert os.path.exists(sparkml_dir), "sparkml directory not found"
        assert os.path.exists(os.path.join(sparkml_dir, "model.pt")), "model.pt not found"
        assert os.path.exists(os.path.join(sparkml_dir, "params.json")), "params.json not found"
        assert os.path.exists(os.path.join(sparkml_dir, "metadata")), "metadata not found"
        
        # Verify MLmodel file exists at stage root
        assert os.path.exists(os.path.join(stage_path, "MLmodel")), "MLmodel not found at stage root"
        
        # Verify environment files
        assert os.path.exists(os.path.join(artifact_path, "requirements.txt"))
        assert os.path.exists(os.path.join(artifact_path, "python_env.yaml"))


def test_model_artifacts_persistence(small_data, tmp_path):
    """Test that model artifacts are saved in both standard and sparkml paths."""
    # Set up MLflow tracking
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    
    with mlflow.start_run() as run:
        # Create and train TabNet model
        tabnet = TabNetEstimator(
            inputCol="features",
            outputCol="prediction",
            labelCol="label",
            n_d=8,
            n_a=8,
            n_steps=3
        )
        model = tabnet.fit(small_data)
        
        # Create pipeline with single TabNet stage
        pipeline = Pipeline(stages=[model])
        pipeline_model = pipeline.fit(small_data)
        
        # Log pipeline model
        mlflow.spark.log_model(
            spark_model=pipeline_model,
            artifact_path="pipeline",
            registered_model_name="tabnet_pipeline"
        )
        
        # Download artifacts
        artifact_path = mlflow.artifacts.download_artifacts(f"runs:/{run.info.run_id}/pipeline")
        
        # Get TabNet stage path
        stage_uid = model.uid
        if stage_uid.startswith("TabNetModel_"):
            stage_uid = stage_uid[len("TabNetModel_"):]
        stage_name = f"0_TabNetModel_{stage_uid}"
        
        # List contents for debugging
        print("\nArtifact directory structure:")
        for root, dirs, files in os.walk(artifact_path):
            level = root.replace(artifact_path, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{subindent}{f}")

        # Find stage directory with correct prefix
        stage_prefix = "0_"
        stage_dir = None
        stages_path = os.path.join(artifact_path, "sparkml", "stages")
        if os.path.exists(stages_path):
            for dirname in os.listdir(stages_path):
                if dirname.startswith(stage_prefix):
                    stage_dir = dirname
                    break
        
        assert stage_dir is not None, f"Stage directory with prefix {stage_prefix} not found"
        stage_path = os.path.join(stages_path, stage_dir)
        sparkml_dir = os.path.join(stage_path, "sparkml")
        
        # Verify sparkml directory structure
        assert os.path.exists(sparkml_dir), f"Model directory not found at {sparkml_dir}"
        assert os.path.exists(os.path.join(sparkml_dir, "model.pt")), \
            "Model file not found in sparkml directory"
        assert os.path.exists(os.path.join(sparkml_dir, "params.json")), \
            "Parameters file not found in sparkml directory"
        assert os.path.exists(os.path.join(sparkml_dir, "metadata")), \
            "Metadata file not found in sparkml directory"
        
        # Verify MLmodel file exists at stage root with proper flavor definitions
        mlmodel_path = os.path.join(stage_path, "MLmodel")
        assert os.path.exists(mlmodel_path), "MLmodel file not found at stage root"
        
        with open(mlmodel_path, "r") as f:
            metadata = json.load(f)
            
            # Verify class and parameters
            assert metadata["class"] == "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel", \
                "Incorrect model class"
            assert "params" in metadata, "Missing parameters in metadata"
            
            # Verify flavor definitions
            assert "flavors" in metadata, "Missing flavors in metadata"
            flavors = metadata["flavors"]
            
            assert "spark" in flavors, "Missing spark flavor"
            assert flavors["spark"]["model_data"] == "sparkml", \
                "Incorrect model_data path in spark flavor"
            
            assert "python_function" in flavors, "Missing python_function flavor"
            assert flavors["python_function"]["loader_module"] == "mlflow.spark", \
                "Incorrect loader_module in python_function flavor"
            assert flavors["python_function"]["model_path"] == "sparkml", \
                "Incorrect model_path in python_function flavor"