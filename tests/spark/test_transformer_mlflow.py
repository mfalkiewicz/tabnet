"""Tests for MLflow integration with TabNet Spark transformer."""

import os
import tempfile
import pytest
import numpy as np
import pickle
from unittest.mock import patch, MagicMock
import logging
import mlflow
from types import SimpleNamespace

from pytorch_tabnet.spark.transformer import SparkTabNetModel
from tests.utils import DummyTabNet, create_test_data

# Apply patches
patch('pytorch_tabnet.tab_network.TabNet', DummyTabNet).start()
patch('pytorch_tabnet.spark.transformer.TabNet', DummyTabNet).start()


@pytest.fixture
def mock_tabnet_model():
    """Create a mock TabNet model."""
    network = DummyTabNet(
        input_dim=10,
        output_dim=2,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[]
    )
    return {
        'network': network,
        'classes_': np.array([0, 1]),
        'input_dim': 10,
        'output_dim': 2,
        'n_d': 8,
        'n_a': 8,
        'n_steps': 3,
        'gamma': 1.3,
        'cat_idxs': [],
        'cat_dims': []
    }


@pytest.fixture
def mock_network():
    """Create a mock network."""
    return DummyTabNet(
        input_dim=10,
        output_dim=2,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[]
    )


@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for testing."""
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
            .master("local[1]")
            .appName("test")
            .getOrCreate())
    yield spark
    spark.stop()

@pytest.fixture
def mock_mlflow_client():
    """Create a mock MLflow client."""
    return MagicMock()


def test_load_local_file(mock_tabnet_model, mock_network):
    """Test loading model from local file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model")
        os.makedirs(model_path)

        # Create a SparkTabNetModel instance
        spark_model = SparkTabNetModel(tabnet_model=mock_tabnet_model)
        
        # Save the model
        spark_model.save(model_path)
        
        # Load the model
        loaded_model = SparkTabNetModel.load(model_path)
        
        # Verify the loaded model
        assert loaded_model._network is not None
        assert loaded_model._input_dim == mock_tabnet_model['input_dim']
        assert loaded_model._output_dim == mock_tabnet_model['output_dim']
        assert np.array_equal(loaded_model._classes, mock_tabnet_model['classes_'])

@patch('mlflow.tracking.MlflowClient')
def test_load_mlflow_artifact(mock_client_class, mock_tabnet_model, mock_network, tmp_path, caplog, spark):
    """Test loading model from MLflow artifact store with enhanced error handling."""
    caplog.set_level(logging.INFO)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Save a mock model to the temporary directory
    spark_model = SparkTabNetModel(tabnet_model=mock_tabnet_model)
    spark_model._setDefault(inputCol="features", outputCol="predictions")  # Set defaults for Spark params
    spark_model.save(str(model_dir))

    # Set up mock client with multiple download attempts
    mock_client = mock_client_class.return_value
    local_path = os.path.join(str(model_dir), "model")
    mock_client.download_artifacts.side_effect = [
        mlflow.exceptions.MlflowException("Run 'test' not found"),  # First attempt fails
        local_path,  # Second attempt succeeds
        local_path   # Third attempt (not reached)
    ]

    with patch('mlflow.get_artifact_uri') as mock_get_uri, \
         patch('os.path.exists') as mock_exists, \
         patch('os.path.getsize') as mock_getsize, \
         patch('pyspark.ml.util.DefaultParamsReader.loadMetadata') as mock_load_metadata, \
         patch('builtins.open') as mock_open:
        mock_get_uri.return_value = local_path
        
        # Mock metadata loading with defaultParamMap
        mock_load_metadata.return_value = {
            "class": "pytorch_tabnet.spark.transformer.SparkTabNetModel",
            "timestamp": 1234567890,
            "sparkVersion": "3.0.0",
            "uid": "test",
            "paramMap": {"inputCol": "features", "outputCol": "predictions"},
            "defaultParamMap": {
                "inputCol": "features",
                "outputCol": "predictions",
                "n_d": 8,
                "n_a": 8,
                "n_steps": 3,
                "gamma": 1.3,
                "cat_idxs": [],
                "cat_dims": [],
                "num_processes": 1,
                "use_gpu": False
            }
        }
        
        # Mock file existence and size checks
        def exists_check(path):
            if isinstance(path, str):
                if path == local_path or path.endswith('metadata') or path.endswith('tabnet_model.pkl'):
                    return True
                if path.startswith('file:'):  # Handle Spark's file:// paths
                    clean_path = path.replace('file:', '')
                    return exists_check(clean_path)
            return False
        mock_exists.side_effect = exists_check
        
        # Mock file size checks
        mock_getsize.return_value = 1000  # Non-empty files

        # Mock file content for tabnet_model.pkl
        mock_data = pickle.dumps({
            'network_params': {
                'input_dim': mock_tabnet_model['input_dim'],
                'output_dim': mock_tabnet_model['output_dim'],
                'n_d': mock_tabnet_model['n_d'],
                'n_a': mock_tabnet_model['n_a'],
                'n_steps': mock_tabnet_model['n_steps'],
                'gamma': mock_tabnet_model['gamma'],
                'cat_idxs': mock_tabnet_model['cat_idxs'],
                'cat_dims': mock_tabnet_model['cat_dims']
            },
            'state_dict': mock_network.state_dict(),
            'classes_': mock_tabnet_model['classes_']
        })
        
        # Set up mock file object for binary reading
        from io import BytesIO
        mock_file = BytesIO(mock_data)
        mock_file_context = MagicMock()
        mock_file_context.__enter__.return_value = mock_file
        mock_file_context.__exit__.return_value = None
        mock_open.return_value = mock_file_context

        # Load model using MLflow URI
        loaded_model = SparkTabNetModel.load("mlflow://test/model")
        
        # Verify the loaded model
        assert loaded_model._network is not None
        assert loaded_model._input_dim == mock_tabnet_model['input_dim']
        assert loaded_model._output_dim == mock_tabnet_model['output_dim']
        assert np.array_equal(loaded_model._classes, mock_tabnet_model['classes_'])
        
        # Verify logging messages
        assert any("Direct MLflow artifact download failed" in record.message
                  for record in caplog.records)

@patch('mlflow.tracking.MlflowClient')
def test_load_mlflow_artifact_empty_files(mock_client_class, mock_tabnet_model, tmp_path, caplog):
    """Test handling of empty model files."""
    caplog.set_level(logging.INFO)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Save a mock model
    spark_model = SparkTabNetModel(tabnet_model=mock_tabnet_model)
    spark_model.save(str(model_dir))

    mock_client = mock_client_class.return_value
    local_path = os.path.join(str(model_dir), "model")
    mock_client.download_artifacts.return_value = local_path

    with patch('os.path.exists') as mock_exists, \
         patch('os.path.getsize') as mock_getsize:
        mock_exists.return_value = True
        mock_getsize.return_value = 0  # Empty files

        with pytest.raises(ValueError, match="Failed to download artifacts from MLflow: All download attempts failed to retrieve a valid model"):
            SparkTabNetModel.load("mlflow://test/model")

@patch('mlflow.tracking.MlflowClient')
def test_load_mlflow_artifact_all_strategies_fail(mock_client_class, caplog):
    """Test behavior when all download strategies fail."""
    caplog.set_level(logging.INFO)
    
    mock_client = mock_client_class.return_value
    # Mock all download attempts to fail
    mock_client.download_artifacts.side_effect = [
        mlflow.exceptions.MlflowException("Download failed"),  # First strategy
        mlflow.exceptions.MlflowException("Download failed"),  # Third strategy
    ]
    
    with patch('mlflow.get_artifact_uri') as mock_get_uri, \
         patch('os.path.exists') as mock_exists:
        # Mock second strategy to fail
        mock_get_uri.side_effect = mlflow.exceptions.MlflowException("Failed to get artifact URI")
        mock_exists.return_value = False
        
        with pytest.raises(ValueError, match="Model path does not exist and artifact store fallback failed"):
            SparkTabNetModel.load("mlflow://test/model")
        
        # Verify all strategies were attempted
        assert sum(1 for record in caplog.records if "Attempting download strategy" in record.message) >= 2


def test_load_nonexistent_path():
    """Test loading model from nonexistent path."""
    with pytest.raises(ValueError, match="Model path does not exist"):
        SparkTabNetModel.load("/nonexistent/path")


@patch('mlflow.tracking.MlflowClient')
def test_load_mlflow_direct_download_failure(mock_client_class, caplog):
    """Test direct MLflow download failure with fallback."""
    caplog.set_level(logging.DEBUG)

    # Set up mock client to fail both direct download and fallback
    mock_client = mock_client_class.return_value
    mock_client.download_artifacts.side_effect = mlflow.exceptions.MlflowException("Failed to download")

    with patch('mlflow.get_artifact_uri') as mock_get_uri:
        # Configure mock for fallback attempt to fail
        mock_get_uri.side_effect = mlflow.exceptions.MlflowException("Failed to get artifact URI")

        with pytest.raises(ValueError, match="Model path does not exist and artifact store fallback failed"):
            SparkTabNetModel.load("mlflow://invalid/uri")

        # Verify both attempts were made
        assert mock_client.download_artifacts.called
        assert mock_get_uri.called

    # Verify warning was logged
    assert any("Direct MLflow artifact download failed" in record.message
              and record.levelname == "WARNING" for record in caplog.records)

@patch('mlflow.tracking.MlflowClient')
def test_load_mlflow_fallback_failure(mock_client_class, caplog):
    """Test MLflow fallback path failure."""
    caplog.set_level(logging.DEBUG)

    # Set up mock client to fail first attempt
    mock_client = mock_client_class.return_value
    mock_client.download_artifacts.side_effect = mlflow.exceptions.MlflowException("Failed to download")

    with patch('mlflow.get_artifact_uri') as mock_get_uri:
        # Configure mock for fallback attempt
        mock_get_uri.side_effect = mlflow.exceptions.MlflowException("Failed to get artifact URI")

        with pytest.raises(ValueError, match="Model path does not exist and artifact store fallback failed"):
            SparkTabNetModel.load("mlflow://invalid/uri")

        # Verify that both attempts were made
        assert mock_client.download_artifacts.called
        assert mock_get_uri.called


@patch('mlflow.tracking.MlflowClient')
def test_load_artifact_store_fallback(mock_client_class, mock_tabnet_model, mock_network, tmp_path, caplog):
    """Test loading model with artifact store fallback."""
    caplog.set_level(logging.DEBUG)

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Save mock model state
    spark_model = SparkTabNetModel(tabnet_model=mock_tabnet_model)
    spark_model.save(str(model_dir))

    # Set up mock client
    mock_client = mock_client_class.return_value
    local_path = os.path.join(str(model_dir), "model")
    mock_client.download_artifacts.side_effect = [
        mlflow.exceptions.MlflowException("Run 'test' not found"),  # First call fails
        local_path  # Second call succeeds
    ]

    with patch('mlflow.get_artifact_uri') as mock_get_uri, \
         patch('os.path.exists') as mock_exists, \
         patch('os.path.getsize') as mock_getsize, \
         patch('pyspark.ml.util.DefaultParamsReader.loadMetadata') as mock_load_metadata, \
         patch('builtins.open') as mock_open:
        mock_get_uri.return_value = local_path
        
        # Mock metadata loading with defaultParamMap
        mock_load_metadata.return_value = {
            "class": "pytorch_tabnet.spark.transformer.SparkTabNetModel",
            "timestamp": 1234567890,
            "sparkVersion": "3.0.0",
            "uid": "test",
            "paramMap": {"inputCol": "features", "outputCol": "predictions"},
            "defaultParamMap": {
                "inputCol": "features",
                "outputCol": "predictions",
                "n_d": 8,
                "n_a": 8,
                "n_steps": 3,
                "gamma": 1.3,
                "cat_idxs": [],
                "cat_dims": [],
                "num_processes": 1,
                "use_gpu": False
            }
        }
        
        # Mock file existence checks
        def exists_check(path):
            if isinstance(path, str):
                # Handle both the direct path and nested model directory cases
                if path == local_path or path == os.path.join(local_path, "model"):
                    return True
                if path.endswith('metadata') or path.endswith('model/metadata'):
                    return True
                if path.endswith('tabnet_model.pkl') or path.endswith('model/tabnet_model.pkl'):
                    return True
                # Handle file:// protocol paths
                if path.startswith('file://'):
                    clean_path = path.replace('file://', '')
                    return exists_check(clean_path)
            return False
        mock_exists.side_effect = exists_check
        
        # Mock file size checks
        mock_getsize.return_value = 1000  # Non-empty files
        
        # Mock file content for tabnet_model.pkl
        mock_data = pickle.dumps({
            'network_params': {
                'input_dim': mock_tabnet_model['input_dim'],
                'output_dim': mock_tabnet_model['output_dim'],
                'n_d': mock_tabnet_model['n_d'],
                'n_a': mock_tabnet_model['n_a'],
                'n_steps': mock_tabnet_model['n_steps'],
                'gamma': mock_tabnet_model['gamma'],
                'cat_idxs': mock_tabnet_model['cat_idxs'],
                'cat_dims': mock_tabnet_model['cat_dims']
            },
            'state_dict': mock_network.state_dict(),
            'classes_': mock_tabnet_model['classes_']
        })
        
        # Set up mock file object for binary reading
        from io import BytesIO
        mock_file = BytesIO(mock_data)
        mock_file_context = MagicMock()
        mock_file_context.__enter__.return_value = mock_file
        mock_file_context.__exit__.return_value = None
        mock_open.return_value = mock_file_context

        # Load model using MLflow URI
        loaded_model = SparkTabNetModel.load("mlflow://test/model")

        # Verify the loaded model
        assert loaded_model._network is not None
        assert loaded_model._input_dim == mock_tabnet_model['input_dim']
        assert loaded_model._output_dim == mock_tabnet_model['output_dim']
        assert np.array_equal(loaded_model._classes, mock_tabnet_model['classes_'])

        # Verify warning was logged
        assert any("Direct MLflow artifact download failed" in record.message
                  and record.levelname == "WARNING" for record in caplog.records)


@pytest.mark.integration
def test_end_to_end_mlflow_dfs_integration(spark, tmp_path, caplog):
    """End-to-end integration test for MLflow model loading with DFS emulation.
    
    This test verifies the complete flow of:
    1. Training a model on synthetic data
    2. Saving it as an MLflow model
    3. Loading the model using DFS emulation
    4. Executing predictions
    """
    caplog.set_level(logging.INFO)
    
    # Create synthetic test data
    df = create_test_data(spark, n_samples=100, n_features=10)
    
    # Create and configure model
    model_config = {
        'input_dim': 10,
        'output_dim': 2,
        'n_d': 8,
        'n_a': 8,
        'n_steps': 3,
        'gamma': 1.3,
        'cat_idxs': [],
        'cat_dims': []
    }
    
    network = DummyTabNet(**model_config)
    tabnet_model = {
        'network': network,
        'classes_': np.array([0, 1]),
        **model_config
    }
    
    # Initialize SparkTabNetModel
    spark_model = SparkTabNetModel(tabnet_model=tabnet_model)
    spark_model._setDefault(inputCol="features", outputCol="predictions")
    
    # Create DFS-like temporary directory structure
    dfs_root = tmp_path / "dfs"
    model_path = dfs_root / "models" / "tabnet"
    os.makedirs(model_path, exist_ok=True)
    
    # Configure MLflow tracking
    mlflow.set_tracking_uri(mlflow.get_tracking_uri())
    
    # Save model using MLflow
    with mlflow.start_run() as run:
        # Save model to DFS-like path
        spark_model.save(str(model_path))
        mlflow.log_artifacts(str(model_path), "model")
        run_id = run.info.run_id
        
        # Log the run ID and artifact path for debugging
        logging.info(f"MLflow run ID: {run_id}")
        logging.info(f"MLflow artifact path: {mlflow.get_artifact_uri('model')}")
    
    # Load model using MLflow with DFS path
    mlflow_uri = f"mlflow://{run_id}/model"
    logging.info(f"Attempting to load model from URI: {mlflow_uri}")
    loaded_model = SparkTabNetModel.load(mlflow_uri)
    
    # Verify model structure
    assert loaded_model._network is not None
    assert loaded_model._input_dim == model_config['input_dim']
    assert loaded_model._output_dim == model_config['output_dim']
    assert np.array_equal(loaded_model._classes, tabnet_model['classes_'])
    
    # Execute predictions
    predictions = loaded_model.transform(df)
    assert "predictions" in predictions.columns
    
    # Verify predictions shape and values
    pred_count = predictions.select("predictions").count()
    assert pred_count == 100  # Matches input sample count
    
    # Log successful test completion
    assert any("Model loaded successfully from MLflow artifact store" in record.message
              for record in caplog.records)
