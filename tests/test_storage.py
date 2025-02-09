"""Tests for storage backend implementations."""

import os
import pytest
import tempfile
import shutil
import threading
import time
from pathlib import Path
import pickle
import mlflow
import numpy as np
from unittest.mock import patch, MagicMock

from pytorch_tabnet.storage import (
    get_storage,
    LocalStorage,
    MLflowStorage,
    ModelStorage,
    StorageError,
    StorageWriteError,
    StorageLockError
)
from pytorch_tabnet.tab_model import TabNetClassifier

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)

@pytest.fixture
def local_storage(temp_dir):
    """Create a LocalStorage instance."""
    return LocalStorage(temp_dir)

@pytest.fixture
def mlflow_storage(temp_dir):
    """Create an MLflowStorage instance with a new MLflow run."""
    mlflow.set_tracking_uri(f"file://{temp_dir}/mlruns")
    with mlflow.start_run() as run:
        storage = MLflowStorage(f"mlflow://{run.info.run_id}")
        yield storage

def test_local_storage_basic_operations(local_storage, temp_dir):
    """Test basic operations with LocalStorage."""
    test_path = "test.txt"
    test_data = b"test data"
    
    # Test write
    local_storage.write_bytes(test_path, test_data)
    assert os.path.exists(os.path.join(temp_dir, test_path))
    
    # Test exists
    assert local_storage.exists(test_path)
    assert not local_storage.exists("nonexistent.txt")
    
    # Test read
    read_data = local_storage.read_bytes(test_path)
    assert read_data == test_data
    
    # Test delete
    local_storage._delete(test_path)
    assert not local_storage.exists(test_path)

def test_local_storage_locking(local_storage):
    """Test file locking mechanism."""
    test_path = "test.txt"
    test_data = b"test data"
    
    def write_with_lock():
        with local_storage.lock(test_path):
            local_storage.write_bytes(test_path, test_data)
            time.sleep(0.1)  # Simulate work
    
    # Start multiple threads trying to write simultaneously
    threads = [
        threading.Thread(target=write_with_lock)
        for _ in range(3)
    ]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    # Verify final state
    assert local_storage.exists(test_path)
    assert local_storage.read_bytes(test_path) == test_data
    assert not local_storage.exists(f"{test_path}.lock")

def test_local_storage_lock_timeout(local_storage):
    """Test lock timeout behavior."""
    test_path = "test.txt"
    
    # Create a lock file and hold it
    with local_storage.lock(test_path, timeout=1):
        # Try to acquire another lock, should timeout
        with pytest.raises(StorageLockError):
            with local_storage.lock(test_path, timeout=1):
                pass

def test_mlflow_storage_basic_operations(mlflow_storage):
    """Test basic operations with MLflowStorage."""
    test_path = "test.txt"
    test_data = b"test data"
    
    # Test write
    mlflow_storage.write_bytes(test_path, test_data)
    
    # Test exists
    assert mlflow_storage.exists(test_path)
    assert not mlflow_storage.exists("nonexistent.txt")
    
    # Test read
    read_data = mlflow_storage.read_bytes(test_path)
    assert read_data == test_data

def test_model_storage_save_load(temp_dir):
    """Test saving and loading a model through ModelStorage."""
    # Create a simple model
    model = TabNetClassifier(
        n_d=8,
        n_a=8,
        n_steps=3
    )
    
    # Create some dummy data and fit the model
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    # Save and load the model
    storage = LocalStorage(temp_dir)
    model_storage = ModelStorage(storage)
    
    save_path = "model.pkl"
    model_storage.save_model(model, save_path)
    
    loaded_model = model_storage.load_model(TabNetClassifier, save_path)
    
    # Verify loaded model has same parameters
    assert loaded_model.n_d == model.n_d
    assert loaded_model.n_a == model.n_a
    assert loaded_model.n_steps == model.n_steps
    
    # Verify predictions match
    np.testing.assert_array_almost_equal(
        model.predict_proba(X),
        loaded_model.predict_proba(X)
    )

def test_get_storage_factory():
    """Test storage factory function."""
    # Test local storage
    storage = get_storage("file:///tmp/test")
    assert isinstance(storage, LocalStorage)
    
    # Test MLflow storage
    storage = get_storage("mlflow://run_id/path")
    assert isinstance(storage, MLflowStorage)
    
    # Test invalid scheme
    with pytest.raises(ValueError):
        get_storage("invalid://path")

def test_concurrent_model_operations(temp_dir):
    """Test concurrent model operations with locking."""
    storage = LocalStorage(temp_dir)
    model_storage = ModelStorage(storage)
    
    # Create a simple model
    model = TabNetClassifier(
        n_d=8,
        n_a=8,
        n_steps=3
    )
    
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    save_path = "model.pkl"
    
    def save_and_load():
        # Save model
        model_storage.save_model(model, save_path)
        # Load model
        loaded_model = model_storage.load_model(TabNetClassifier, save_path)
        # Verify basic property
        assert loaded_model.n_d == model.n_d
    
    # Run concurrent operations
    threads = [
        threading.Thread(target=save_and_load)
        for _ in range(3)
    ]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()

@pytest.mark.integration
def test_mlflow_integration_end_to_end(temp_dir):
    """End-to-end test of MLflow integration."""
    mlflow.set_tracking_uri(f"file://{temp_dir}/mlruns")
    
    with mlflow.start_run() as run:
        # Create and train model
        model = TabNetClassifier(
            n_d=8,
            n_a=8,
            n_steps=3
        )
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        
        # Save model using MLflow storage
        storage = MLflowStorage(f"mlflow://{run.info.run_id}")
        model_storage = ModelStorage(storage)
        
        save_path = "model.pkl"
        model_storage.save_model(model, save_path)
        
        # Load model using MLflow storage
        loaded_model = model_storage.load_model(TabNetClassifier, save_path)
        
        # Verify predictions match
        np.testing.assert_array_almost_equal(
            model.predict_proba(X),
            loaded_model.predict_proba(X)
        )

def test_error_handling(temp_dir):
    """Test error handling in storage operations."""
    storage = LocalStorage(temp_dir)
    
    # Test read nonexistent file
    with pytest.raises(StorageError):
        storage.read_bytes("nonexistent.txt")
    
    # Test write to invalid path (using a root path that's definitely not writable)
    invalid_path = "/root/test.txt"
    with pytest.raises(StorageError):
        storage.write_bytes(invalid_path, b"test")
    
    # Test lock acquisition failure
    def simulate_lock_failure():
        with patch.object(storage, 'write_bytes', side_effect=StorageWriteError("Simulated error")):
            with pytest.raises(StorageLockError):
                with storage.lock("test.txt", timeout=1):
                    pass

    simulate_lock_failure()