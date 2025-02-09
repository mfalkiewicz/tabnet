# Fabric Integration Guide

This guide explains how to use TabNet models with Azure Fabric, including model saving, loading, and common troubleshooting steps.

## Model Loading in Fabric

TabNet models in Fabric environments are loaded through MLflow's artifact store. The model loading process has been enhanced to handle both local files and MLflow artifacts seamlessly.

### Loading Process

When loading a model in Fabric:

1. The system first attempts to load the model from the specified local path
2. If the local file doesn't exist, it attempts to retrieve the model from MLflow's artifact store
3. The model is automatically downloaded to a temporary location if needed
4. The model state is reconstructed from the downloaded artifacts

### Example Usage

```python
import mlflow
from synapse.ml.predict import MLFlowTransformer

# Load model using MLflow transformer
model = MLFlowTransformer(
    inputCols=["feature1", "feature2"],
    outputCol="predictions",
    modelName="my_tabnet_model",
    modelVersion=1
)
```

### Configuration

No special configuration is needed for Fabric environments. The model loading process automatically detects and handles MLflow artifact URIs.

## Troubleshooting

### Common Issues

1. **Model Path Does Not Exist**
   - **Symptom:** ValueError with message "Model path does not exist"
   - **Solution:** Ensure the model is properly registered in MLflow and the artifact store is accessible
   - **Check:**
     - Verify MLflow tracking URI is correctly set
     - Confirm model registration status
     - Check artifact store permissions

2. **MLflow Artifact Store Access**
   - **Symptom:** IOError when accessing MLflow artifact store
   - **Solution:** Verify MLflow configuration and connectivity
   - **Check:**
     - MLflow tracking URI configuration
     - Network connectivity to artifact store
     - Authentication credentials

### Best Practices

1. **Model Registration**
   - Always register models through MLflow's model registry
   - Use meaningful model names and versions
   - Include relevant tags and descriptions

2. **Artifact Management**
   - Monitor artifact storage usage
   - Clean up unused artifacts periodically
   - Maintain proper versioning of artifacts

3. **Error Handling**
   - Implement proper error handling in your applications
   - Log relevant information for troubleshooting
   - Consider implementing retries for transient failures

## Technical Details

### Model Loading Implementation

The TabNet model loading process in Fabric has been enhanced to support multiple storage scenarios:

1. **Direct Local File Access**
   - Attempts to load model directly from local filesystem
   - Uses standard file I/O operations

2. **MLflow Artifact Store**
   - Supports loading from MLflow artifact store URIs
   - Automatically handles artifact downloading
   - Manages temporary storage for downloaded artifacts

3. **Fallback Mechanism**
   - If local file not found, attempts MLflow artifact store
   - Provides graceful degradation path
   - Maintains backward compatibility

### URI Handling

The system supports multiple URI formats:

1. **Local File URIs**
   ```
   file:///path/to/model
   ```

2. **MLflow URIs**
   ```
   mlflow://<run_id>/path/to/model
   ```

3. **Artifact Store URIs**
   ```
   mlflow-artifacts://<artifact-root>/path/to/model
   ```

## Migration Guide

### From Previous Versions

If you're upgrading from a previous version, no changes are required in your code. The new model loading mechanism is backward compatible and will automatically handle both local and MLflow artifact store scenarios.

### Best Practices for New Projects

For new projects:

1. Always use MLflow model registry for model management
2. Implement proper error handling for model loading
3. Use the provided MLFlowTransformer interface
4. Monitor and log model loading operations

## Support and Resources

- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Azure Fabric Documentation](https://docs.microsoft.com/en-us/azure/fabric/)
- [TabNet Documentation](https://github.com/microsoft/tabnet)

For issues and support:
- File issues on the GitHub repository
- Contact Azure Fabric support for platform-specific issues
- Consult MLflow documentation for MLflow-related questions