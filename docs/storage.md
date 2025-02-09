# TabNet Storage System Documentation

## Overview

The TabNet storage system provides a unified interface for model serialization across different storage backends (local filesystem, MLflow artifacts, etc.). It handles distributed filesystem access with proper error handling, retries, and concurrency control.

## Key Components

### Storage Backends

1. **LocalStorage**
   - Direct filesystem access for local development and testing
   - Supports file locking for concurrent access
   - Uses standard Python file I/O operations

2. **MLflowStorage**
   - Integration with MLflow artifact storage
   - Handles distributed filesystem access
   - Manages temporary files for MLflow artifact operations

### High-Level Interface

The `ModelStorage` class provides a high-level interface for model operations:
- Save/load models with proper error handling
- Automatic retries for transient failures
- Concurrent access protection via file locking

## Usage Examples

### Local Development

```python
from pytorch_tabnet.storage import get_storage, ModelStorage
from pytorch_tabnet.tab_model import TabNetClassifier

# Initialize storage for local development
storage = get_storage("file:///path/to/models")
model_storage = ModelStorage(storage)

# Save model
model = TabNetClassifier(...)
model_storage.save_model(model, "model.pkl")

# Load model
loaded_model = model_storage.load_model(TabNetClassifier, "model.pkl")
```

### MLflow Integration

```python
import mlflow

# Start MLflow run
with mlflow.start_run() as run:
    # Initialize MLflow storage
    storage = get_storage(f"mlflow://{run.info.run_id}")
    model_storage = ModelStorage(storage)
    
    # Save model as MLflow artifact
    model_storage.save_model(model, "model.pkl")
```

## Testing with MinIO (Local S3-Compatible Storage)

1. Start MinIO server:
```bash
docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
```

2. Configure environment:
```bash
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
```

3. Run tests:
```bash
pytest tests/test_storage.py -v
```

## Error Handling

The storage system defines several exception types:
- `StorageError`: Base exception for storage operations
- `StorageConnectionError`: Connection issues
- `StorageReadError`: Read operation failures
- `StorageWriteError`: Write operation failures
- `StorageLockError`: Lock acquisition failures

Example error handling:
```python
from pytorch_tabnet.storage import StorageError

try:
    model = model_storage.load_model(TabNetClassifier, "model.pkl")
except StorageError as e:
    logger.error(f"Failed to load model: {e}")
    # Handle error appropriately
```

## Concurrency Control

The storage system uses file-based locking to handle concurrent access:

```python
# Automatic locking during model operations
with storage.lock("model.pkl"):
    # Perform atomic operations
    model_storage.save_model(model, "model.pkl")
```

Lock timeout can be configured:
```python
# Set 30-second timeout for lock acquisition
with storage.lock("model.pkl", timeout=30):
    # ... operations ...
```

## Production Deployment

For production environments:

1. Configure proper permissions:
```bash
# Example: Azure Storage
export AZURE_STORAGE_CONNECTION_STRING="..."
# or AWS S3
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

2. Use appropriate storage URI:
```python
# MLflow with Azure backend
storage = get_storage("mlflow://run_id/path")
```

3. Configure logging:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Best Practices

1. **Error Handling**
   - Always wrap storage operations in try/except blocks
   - Log errors with appropriate context
   - Implement retries for transient failures

2. **Resource Management**
   - Use context managers for locks
   - Clean up temporary files
   - Release resources properly

3. **Testing**
   - Run tests with different storage backends
   - Test concurrent access patterns
   - Verify error handling behavior

4. **Monitoring**
   - Log storage operations
   - Track timing metrics
   - Monitor lock contention

## Implementation Details

### File Locking

The storage system uses file-based locking:
```python
lock_path = f"{path}.lock"
with storage._lock:  # Thread-safe check
    if not storage.exists(lock_path):
        storage.write_bytes(lock_path, str(os.getpid()).encode())
```

### Retries

Operations implement exponential backoff:
```python
max_retries = 3
for attempt in range(max_retries):
    try:
        return storage.read_bytes(path)
    except StorageReadError:
        if attempt == max_retries - 1:
            raise
        time.sleep(2 ** attempt)
```

### Cleanup

Temporary resources are managed:
```python
def __del__(self):
    """Cleanup temporary directory on deletion."""
    try:
        shutil.rmtree(self.temp_dir)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp dir: {e}")