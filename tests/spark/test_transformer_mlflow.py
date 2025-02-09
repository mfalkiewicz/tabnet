"""Tests for MLflow integration in SparkTabNetModel."""

import os
import tempfile
import pytest
import mlflow
import numpy as np
import logging
import pickle
from unittest.mock import patch, MagicMock

logger = logging.getLogger(__name__)

from pytorch_tabnet.spark.transformer import SparkTabNetModel
from pytorch_tabnet.tab_model import TabNetClassifier

@pytest.fixture
def mock_mlflow_client():
    """Create a mock MLflow client."""
    with patch('mlflow.tracking.MlflowClient') as mock_client:
        yield mock_client()

@pytest.fixture
def mock_tabnet_model():
    """Create a mock TabNet model."""
    model = TabNetClassifier()
    model.n_d = 8
    model.n_a = 8
    model.n_steps = 3
    model.gamma = 1.3
    model.cat_idxs = []
    model.cat_dims = []
    model.input_dim = 10
    model.output_dim = 2
    model.classes_ = np.array([0, 1])
    return model

@pytest.fixture
def mock_network():
    """Create a mock network with state dict."""
    network = MagicMock()
    network.state_dict.return_value = {}
    return network

@pytest.fixture
def mock_tabnet_model(mock_network):
    """Create a mock TabNet model."""
    model = TabNetClassifier()
    model.n_d = 8
    model.n_a = 8
    model.n_steps = 3
    model.gamma = 1.3
    model.cat_idxs = []
    model.cat_dims = []
    model.input_dim = 10
    model.output_dim = 2
    model.classes_ = np.array([0, 1])
    model.network = mock_network
    return model

def test_load_local_file(mock_tabnet_model, mock_network):
    """Test loading model from local file."""
    # Save model to temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model")
        os.makedirs(model_path)
        
        # Create a SparkTabNetModel instance
        spark_model = SparkTabNetModel(tabnet_model=mock_tabnet_model)
        
        # Save the model
        spark_model.write().save(model_path)
        
        # Mock network for loaded model
        with patch('pytorch_tabnet.tab_model.TabNetClassifier._initialize_network') as mock_init:
            mock_init.return_value = mock_network
            
            # Load the model
            loaded_model = SparkTabNetModel.load(model_path)
            
            # Verify model attributes
            assert loaded_model.tabnet.n_d == mock_tabnet_model.n_d
            assert loaded_model.tabnet.n_a == mock_tabnet_model.n_a
            assert loaded_model.tabnet.n_steps == mock_tabnet_model.n_steps
            assert loaded_model.tabnet.gamma == mock_tabnet_model.gamma
            assert loaded_model.tabnet.input_dim == mock_tabnet_model.input_dim
            assert loaded_model.tabnet.output_dim == mock_tabnet_model.output_dim
            assert np.array_equal(loaded_model.tabnet.classes_, mock_tabnet_model.classes_)
def test_load_mlflow_artifact(mock_mlflow_client, mock_tabnet_model, mock_network, tmp_path):
    """Test loading model from MLflow artifact store."""
    # Create a temporary directory for the model
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    
    # Save a mock model to the temporary directory
    spark_model = SparkTabNetModel(tabnet_model=mock_tabnet_model)
    spark_model.write().save(str(model_dir))
    
    # Create TabNet model state file
    with open(os.path.join(model_dir, "tabnet_model.pkl"), "wb") as f:
        pickle.dump({
            "init_params": {
                "n_d": mock_tabnet_model.n_d,
                "n_a": mock_tabnet_model.n_a,
                "n_steps": mock_tabnet_model.n_steps,
                "gamma": mock_tabnet_model.gamma,
                "cat_idxs": mock_tabnet_model.cat_idxs,
                "cat_dims": mock_tabnet_model.cat_dims,
                "input_dim": mock_tabnet_model.input_dim,
                "output_dim": mock_tabnet_model.output_dim
            },
            "class_attrs": {
                "classes_": mock_tabnet_model.classes_,
                "input_dim": mock_tabnet_model.input_dim,
                "output_dim": mock_tabnet_model.output_dim,
                "_task": "classification"
            },
            "network_state": {}
        }, f)
    
    # Mock MLflow artifact download to copy from temp directory
    def mock_download_artifacts(run_id, artifact_path, dst_path):
        import shutil
        os.makedirs(dst_path, exist_ok=True)
        shutil.copytree(model_dir, os.path.join(dst_path, "model"), dirs_exist_ok=True)
        logger.debug(f"Mock downloaded artifacts to {dst_path}")
        
    mock_mlflow_client.download_artifacts.side_effect = mock_download_artifacts
    
    # Mock MLflow run context and artifact URI
    with patch('mlflow.active_run') as mock_run, \
         patch('mlflow.get_artifact_uri') as mock_get_uri, \
         patch('pytorch_tabnet.tab_model.TabNetClassifier._initialize_network') as mock_init:
        
        mock_run.return_value = MagicMock(info=MagicMock(run_id='test_run_id'))
        mock_get_uri.return_value = "mlflow-artifacts://test"
        mock_init.return_value = mock_network
        
        # Test loading from MLflow URI
        model_uri = "mlflow://test_run_id/model"
        loaded_model = SparkTabNetModel.load(model_uri)
        loaded_model = SparkTabNetModel.load(model_uri)
        
        # Verify model attributes
        assert loaded_model.tabnet.n_d == mock_tabnet_model.n_d
        assert loaded_model.tabnet.n_a == mock_tabnet_model.n_a
        assert loaded_model.tabnet.n_steps == mock_tabnet_model.n_steps
        assert loaded_model.tabnet.gamma == mock_tabnet_model.gamma
        assert loaded_model.tabnet.input_dim == mock_tabnet_model.input_dim
        assert loaded_model.tabnet.output_dim == mock_tabnet_model.output_dim
        assert np.array_equal(loaded_model.tabnet.classes_, mock_tabnet_model.classes_)
        
        # Verify MLflow client was called
        assert mock_mlflow_client.download_artifacts.call_count >= 1

def test_load_nonexistent_path():
    """Test loading model from nonexistent path."""
    with pytest.raises(ValueError, match="Model path does not exist"):
        SparkTabNetModel.load("/nonexistent/path")
def test_load_invalid_mlflow_uri(mock_mlflow_client, caplog):
    """Test loading model from invalid MLflow URI."""
    # Set up logging capture
    caplog.set_level(logging.DEBUG)
    
    # Mock both direct and fallback paths to fail
    mock_mlflow_client.download_artifacts.side_effect = Exception("Failed to download")
    
    with patch('mlflow.get_artifact_uri') as mock_get_uri:
        # First attempt: direct MLflow URI
        with pytest.raises(ValueError, match="Model path does not exist and artifact store fallback failed"):
            SparkTabNetModel.load("mlflow://invalid/uri")
            
        # Verify warning was logged
        assert any("Direct MLflow artifact download failed" in record.message and record.levelname == "WARNING" for record in caplog.records)

        # Reset log capture
        caplog.clear()

        # Second attempt: fallback path
        mock_get_uri.return_value = "mlflow-artifacts://test"
        with pytest.raises(ValueError, match="Model path does not exist and artifact store fallback failed"):
            SparkTabNetModel.load("/nonexistent/path")
        
        # Verify both paths were attempted
        assert mock_mlflow_client.download_artifacts.call_count >= 2
        
        # Verify temporary directory cleanup was attempted
        assert "Cleaned up temporary directory" in caplog.text
        assert mock_mlflow_client.download_artifacts.call_count >= 2

def test_load_artifact_store_fallback(mock_mlflow_client, mock_tabnet_model, mock_network, caplog):
    """Test loading model with artifact store fallback."""
    # Set up logging capture
    caplog.set_level(logging.DEBUG)
    
    # Mock MLflow artifact store
    with patch('mlflow.get_artifact_uri') as mock_get_uri, \
         patch('pytorch_tabnet.tab_model.TabNetClassifier._initialize_network') as mock_init:
        mock_get_uri.return_value = "mlflow-artifacts://test"
        mock_init.return_value = mock_network
        
        # Create a temporary directory for the model
        model_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(model_dir, "model"), exist_ok=True)

        # Save mock model state
        spark_model = SparkTabNetModel(tabnet_model=mock_tabnet_model)
        spark_model.write().save(os.path.join(model_dir, "model"))

        # Create TabNet model state file
        with open(os.path.join(model_dir, "model", "tabnet_model.pkl"), "wb") as f:
            pickle.dump({
                "init_params": {
                    "n_d": mock_tabnet_model.n_d,
                    "n_a": mock_tabnet_model.n_a,
                    "n_steps": mock_tabnet_model.n_steps,
                    "gamma": mock_tabnet_model.gamma,
                    "cat_idxs": mock_tabnet_model.cat_idxs,
                    "cat_dims": mock_tabnet_model.cat_dims,
                    "input_dim": mock_tabnet_model.input_dim,
                    "output_dim": mock_tabnet_model.output_dim
                },
                "class_attrs": {
                    "classes_": mock_tabnet_model.classes_,
                    "input_dim": mock_tabnet_model.input_dim,
                    "output_dim": mock_tabnet_model.output_dim,
                    "_task": "classification"
                },
                "network_state": {}
            }, f)

        # Create metadata directory
        metadata_path = os.path.join(model_dir, "model", "metadata")
        os.makedirs(metadata_path, exist_ok=True)
        with open(os.path.join(metadata_path, "part-00000"), "w") as f:
            f.write('{"class":"org.apache.spark.ml.PipelineModel","timestamp":1234567890,"sparkVersion":"3.5.0","uid":"pipeline_123","paramMap":{},"defaultParamMap":{}}')

        # Reset mock to ensure clean state
        mock_mlflow_client.download_artifacts.reset_mock()

        # Define mock download function with closure over model_dir
        def mock_download(run_id, artifact_path, dst_path):
            logger.debug(f"Mock downloading artifacts to {dst_path}")
            import shutil
            os.makedirs(dst_path, exist_ok=True)
            # Copy the model directory to the destination
            model_src = os.path.join(model_dir, "model")
            # Copy directly to the model subdirectory
            model_dst = os.path.join(dst_path, "model")
            os.makedirs(model_dst, exist_ok=True)
            # Copy contents of model directory
            for item in os.listdir(model_src):
                src_item = os.path.join(model_src, item)
                dst_item = os.path.join(model_dst, item)
                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_item, dst_item)
            # Copy tabnet_model.pkl to ensure it's in the right place
            shutil.copy2(
                os.path.join(model_src, "tabnet_model.pkl"),
                os.path.join(model_dst, "tabnet_model.pkl")
            )
            # Log directory structure
            logger.debug(f"Files in destination root: {os.listdir(dst_path)}")
            logger.debug(f"Files in model directory: {os.listdir(model_dst)}")
            logger.debug(f"Model file exists: {os.path.exists(os.path.join(model_dst, 'tabnet_model.pkl'))}")
            return True

        # Mock active run
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"

        # Create a custom side effect handler that preserves run_id
        class SideEffectHandler:
            def __init__(self):
                self.call_count = 0

            def __call__(self, run_id, artifact_path, dst_path):
                self.call_count += 1
                logger.debug(f"Download attempt {self.call_count} with run_id: {run_id}, path: {artifact_path}")
                
                if self.call_count == 1:
                    raise Exception("Direct path failed")
                
                # For the second attempt, ensure we're using the correct path
                if self.call_count == 2:
                    # Copy the model to the destination
                    result = mock_download(run_id, artifact_path, dst_path)
                    
                    # Log directory structure before returning
                    logger.debug(f"Files in destination root: {os.listdir(dst_path)}")
                    if os.path.exists(os.path.join(dst_path, "model")):
                        model_dir = os.path.join(dst_path, "model")
                        logger.debug(f"Files in model directory: {os.listdir(model_dir)}")
                        if os.path.exists(os.path.join(model_dir, "tabnet_model.pkl")):
                            logger.debug("Found tabnet_model.pkl in model directory")
                            return result
                        else:
                            logger.error("tabnet_model.pkl not found in model directory")
                            raise ValueError("Model file not found after download")
                    else:
                        logger.error("Model directory not found in downloaded artifacts")
                        raise ValueError("Model directory not found")
                
                return False

        # Set up mock with side effect handler
        mock_mlflow_client.download_artifacts.side_effect = SideEffectHandler()

        # Mock get_artifact_uri to return a valid URI
        mock_get_uri.return_value = "mlflow-artifacts://test"

        # Test loading with fallback
        with patch('mlflow.active_run', return_value=mock_run):
            loaded_model = SparkTabNetModel.load("/nonexistent/path")
        
        # Verify model attributes
        assert loaded_model.tabnet.n_d == mock_tabnet_model.n_d
        assert loaded_model.tabnet.n_a == mock_tabnet_model.n_a
        assert loaded_model.tabnet.n_steps == mock_tabnet_model.n_steps
        assert loaded_model.tabnet.gamma == mock_tabnet_model.gamma
        assert loaded_model.tabnet.input_dim == mock_tabnet_model.input_dim
        assert loaded_model.tabnet.output_dim == mock_tabnet_model.output_dim
        assert np.array_equal(loaded_model.tabnet.classes_, mock_tabnet_model.classes_)
        
        # Verify MLflow client was called multiple times
        assert mock_mlflow_client.download_artifacts.call_count >= 2
        
        # Verify network initialization
        mock_init.assert_called_once()
        
        # Verify logging behavior
        assert "Initial download attempt failed: Direct path failed" in caplog.text
        assert "Retrying with artifact store path: /nonexistent/path" in caplog.text
        assert "Found model files at:" in caplog.text
        assert "Successfully loaded TabNet model" in caplog.text
        assert "Cleaned up temporary directory" in caplog.text