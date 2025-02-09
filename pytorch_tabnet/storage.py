"""Storage backend abstraction for TabNet model serialization.

This module provides a unified interface for accessing model files across different
storage backends (local filesystem, HDFS, S3, Azure Blob Storage, etc.) with proper
error handling and retries.
"""

import os
import tempfile
import shutil
import logging
import threading
from pathlib import Path
from typing import Optional, Union, BinaryIO, Dict, Any
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
import time
import json
import pickle
from urllib.parse import urlparse

import mlflow
from mlflow.tracking.artifact_utils import get_artifact_uri
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass

class StorageConnectionError(StorageError):
    """Raised when connection to storage backend fails."""
    pass

class StorageReadError(StorageError):
    """Raised when reading from storage fails."""
    pass

class StorageWriteError(StorageError):
    """Raised when writing to storage fails."""
    pass

class StorageLockError(StorageError):
    """Raised when acquiring a lock fails."""
    pass

class BaseStorage(ABC):
    """Abstract base class for storage implementations."""
    
    def __init__(self, base_path: str, **kwargs):
        self.base_path = base_path
        self._lock = threading.Lock()
        
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists in storage."""
        pass
    
    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Read bytes from storage."""
        pass
    
    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to storage."""
        pass
    
    @contextmanager
    def lock(self, path: str, timeout: int = 60):
        """Acquire a lock for the given path."""
        lock_path = f"{path}.lock"
        start_time = time.time()
        acquired = False
        
        try:
            while True:
                with self._lock:
                    if not self.exists(lock_path):
                        try:
                            self.write_bytes(lock_path, str(os.getpid()).encode())
                            acquired = True
                            break
                        except StorageWriteError:
                            pass
                
                if time.time() - start_time > timeout:
                    raise StorageLockError(f"Failed to acquire lock for {path}")
                
                time.sleep(1)
            
            yield
            
        finally:
            if acquired:
                try:
                    with self._lock:
                        if self.exists(lock_path):
                            self._delete(lock_path)
                except Exception as e:
                    logger.warning(f"Failed to release lock {lock_path}: {e}")
                
    @abstractmethod
    def _delete(self, path: str) -> None:
        """Delete a path from storage."""
        pass

class LocalStorage(BaseStorage):
    """Local filesystem storage implementation."""
    
    def exists(self, path: str) -> bool:
        return os.path.exists(os.path.join(self.base_path, path))
    
    def read_bytes(self, path: str) -> bytes:
        try:
            with open(os.path.join(self.base_path, path), 'rb') as f:
                return f.read()
        except IOError as e:
            raise StorageReadError(f"Failed to read {path}: {e}")
    
    def write_bytes(self, path: str, data: bytes) -> None:
        full_path = os.path.join(self.base_path, path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'wb') as f:
                f.write(data)
        except (IOError, OSError) as e:
            raise StorageWriteError(f"Failed to write {path}: {e}")
            
    def _delete(self, path: str) -> None:
        try:
            os.remove(os.path.join(self.base_path, path))
        except OSError as e:
            logger.warning(f"Failed to delete {path}: {e}")

class MLflowStorage(BaseStorage):
    """MLflow artifact storage implementation."""
    
    def __init__(self, base_path: str, **kwargs):
        super().__init__(base_path)
        self.client = mlflow.tracking.MlflowClient()
        self.temp_dir = tempfile.mkdtemp()
        
    def _get_artifact_uri(self, path: str) -> str:
        """Get the full artifact URI for a path."""
        # Convert mlflow:// to runs:/ for proper artifact handling
        if self.base_path.startswith("mlflow://"):
            run_id = self.base_path.split("mlflow://")[1].split("/")[0]
            return f"runs:/{run_id}/{path}"
        return os.path.join(self.base_path, path)
    
    def exists(self, path: str) -> bool:
        try:
            artifact_uri = self._get_artifact_uri(path)
            # Use MLflow's artifact utilities instead of direct repository access
            client = mlflow.tracking.MlflowClient()
            run_id = self.base_path.split("mlflow://")[1].split("/")[0]
            
            # List artifacts in the run
            artifacts = client.list_artifacts(run_id)
            return any(artifact.path == path for artifact in artifacts)
        except Exception as e:
            logger.warning(f"Failed to check existence of {path}: {e}")
            return False
    
    def read_bytes(self, path: str) -> bytes:
        try:
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=self._get_artifact_uri(path),
                dst_path=self.temp_dir
            )
            with open(local_path, 'rb') as f:
                return f.read()
        except Exception as e:
            raise StorageReadError(f"Failed to read {path} from MLflow: {e}")
    
    def write_bytes(self, path: str, data: bytes) -> None:
        temp_path = os.path.join(self.temp_dir, path)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        try:
            # Write to temp file
            with open(temp_path, 'wb') as f:
                f.write(data)
            
            # Get run ID
            run_id = self.base_path.split("mlflow://")[1].split("/")[0]
            
            # Log to MLflow using client
            self.client.log_artifact(run_id, temp_path, artifact_path=os.path.dirname(path))
            
            # Wait for artifact to be available
            max_retries = 3
            for attempt in range(max_retries):
                if self.exists(path):
                    return
                time.sleep(1)
            raise StorageWriteError(f"Failed to verify {path} was written to MLflow")
            
        except Exception as e:
            raise StorageWriteError(f"Failed to write {path} to MLflow: {e}")
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
                
    def _delete(self, path: str) -> None:
        # MLflow doesn't support artifact deletion
        # This is a no-op
        pass
    
    def __del__(self):
        """Cleanup temporary directory on deletion."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir {self.temp_dir}: {e}")

def get_storage(uri: str, **kwargs) -> BaseStorage:
    """Factory function to create appropriate storage backend.
    
    Args:
        uri: Storage URI (e.g., file:///path, mlflow://run_id/path)
        **kwargs: Additional arguments passed to storage implementation
        
    Returns:
        Storage implementation instance
    """
    parsed = urlparse(uri)
    
    if parsed.scheme in ('', 'file'):
        return LocalStorage(parsed.path or parsed.netloc, **kwargs)
    elif parsed.scheme == 'mlflow':
        if not parsed.netloc:
            raise ValueError("MLflow URI must include run ID")
        return MLflowStorage(uri, **kwargs)
    else:
        raise ValueError(f"Unsupported storage scheme: {parsed.scheme}")

class ModelStorage:
    """High-level interface for model storage operations."""
    
    def __init__(self, storage: BaseStorage):
        self.storage = storage
        
    def save_model(self, model: Any, path: str) -> None:
        """Save model to storage with proper locking and error handling."""
        # Only use locking for local storage
        context = self.storage.lock(path) if isinstance(self.storage, LocalStorage) else nullcontext()
        
        with context:
            try:
                # Create a dictionary of everything needed
                save_dict = {
                    "init_params": model.get_params(),
                    "class_attrs": {
                        "preds_mapper": model.preds_mapper,
                        "classes_": model.classes_,
                        "_class_map": model._class_map,
                        "feature_importances_": getattr(model, "feature_importances_", None),
                        "_task": model._task,
                        "input_dim": model.input_dim,
                        "output_dim": model.output_dim,
                        "version": "1.0.0"
                    },
                    "network_state": model.network.state_dict(),
                }
                
                # Serialize with retries
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        serialized = pickle.dumps(save_dict)
                        self.storage.write_bytes(path, serialized)
                        break
                    except (StorageWriteError, pickle.PickleError) as e:
                        if attempt == max_retries - 1:
                            raise StorageError(f"Failed to save model after {max_retries} attempts: {e}")
                        time.sleep(1)
                        
            except Exception as e:
                raise StorageError(f"Failed to save model: {e}")
    
    def load_model(self, model_class: type, path: str) -> Any:
        """Load model from storage with proper error handling."""
        # Only use locking for local storage
        context = self.storage.lock(path) if isinstance(self.storage, LocalStorage) else nullcontext()
        
        with context:
            try:
                # Read with retries
                max_retries = 3
                serialized = None
                
                for attempt in range(max_retries):
                    try:
                        serialized = self.storage.read_bytes(path)
                        break
                    except StorageReadError as e:
                        if attempt == max_retries - 1:
                            raise StorageError(f"Failed to read model after {max_retries} attempts: {e}")
                        time.sleep(1)
                
                if serialized is None:
                    raise StorageError(f"Failed to read model from {path}")
                
                # Deserialize
                try:
                    loaded_dict = pickle.loads(serialized)
                except pickle.UnpicklingError as e:
                    raise StorageError(f"Failed to deserialize model: {e}")
                
                # Create new model instance
                model = model_class(**loaded_dict["init_params"])
                
                # Restore attributes
                for k, v in loaded_dict["class_attrs"].items():
                    setattr(model, k, v)
                
                # Initialize network
                model._initialize_network()
                
                # Load state dict
                model.network.load_state_dict(loaded_dict["network_state"])
                
                return model
                
            except Exception as e:
                raise StorageError(f"Failed to load model: {e}")