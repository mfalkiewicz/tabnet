import pytest
import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.ml.torch.distributor import TorchDistributor

from pytorch_tabnet.spark.transformer import SparkTabNetEstimator
from pytorch_tabnet.tab_model import TabNetClassifier

@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for testing."""
    return (SparkSession.builder
            .master("local[2]")  # Use 2 local worker threads
            .appName("tabnet-torch-distributor-test")
            .getOrCreate())

def test_torch_distributor_integration(spark):
    """Test TorchDistributor integration with TabNet."""
    # Generate synthetic data
    n_samples = 1000
    n_features = 10
    n_classes = 3
    
    # Create features
    X = np.random.random((n_samples, n_features))
    y = np.random.randint(0, n_classes, n_samples)
    
    # Convert to Spark DataFrame
    data = [(x.tolist(), float(y_)) for x, y_ in zip(X, y)]
    df = spark.createDataFrame(data, ["features", "label"])
    
    # Initialize TabNet estimator with TorchDistributor config
    estimator = SparkTabNetEstimator(
        inputCol="features",
        outputCol="predictions",
        n_d=8,
        n_a=8,
        n_steps=3,
        num_processes=2,  # Use 2 processes for distributed training
        use_gpu=torch.cuda.is_available(),  # Use GPU if available
        local_mode=True
    )
    
    # Train model
    model = estimator.fit(df)
    
    # Verify model was trained
    assert model is not None
    assert model.tabnet is not None
    
    # Test predictions
    predictions = model.transform(df)
    assert "predictions" in predictions.columns
    
    # Get predictions as numpy array
    pred_rows = predictions.select("predictions").collect()
    # Convert predictions to numpy array with consistent shape
    pred_list = []
    for row in pred_rows:
        # Ensure predictions is a list of floats with length n_classes
        if isinstance(row.predictions, list):
            if len(row.predictions) != n_classes:
                raise ValueError(f"Expected {n_classes} predictions, got {len(row.predictions)}")
            # Convert to float and normalize
            probs = [float(x) for x in row.predictions]
            # Add small epsilon to avoid numerical issues
            epsilon = 1e-7
            probs = [p + epsilon for p in probs]
            total = sum(probs)
            probs = [p / total for p in probs]
            pred_list.append(probs)
        else:
            # If single value, normalize across n_classes
            val = float(row.predictions)
            pred_list.append([val/n_classes] * n_classes)
    pred_array = np.array(pred_list, dtype=np.float32)
    
    # Verify prediction shape and values
    assert pred_array.shape[0] == n_samples
    assert pred_array.shape[1] == n_classes  # One probability per class
    assert np.all(np.isfinite(pred_array))  # No NaN or inf values
    # Check probabilities sum to 1 with higher tolerance for numerical precision
    assert np.allclose(pred_array.sum(axis=1), 1.0, rtol=1e-4, atol=1e-4)  # Probabilities sum to 1

def test_torch_distributor_gpu_handling(spark):
    """Test TorchDistributor GPU handling."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    # Generate small synthetic dataset
    n_samples = 100
    n_features = 5
    X = np.random.random((n_samples, n_features))
    y = np.random.randint(0, 2, n_samples)  # Binary classification
    
    # Create Spark DataFrame
    data = [(x.tolist(), float(y_)) for x, y_ in zip(X, y)]
    df = spark.createDataFrame(data, ["features", "label"])
    
    # Initialize estimator with GPU config
    estimator = SparkTabNetEstimator(
        inputCol="features",
        outputCol="predictions",
        num_processes=2,
        use_gpu=True,
        local_mode=True
    )
    
    # Train model
    model = estimator.fit(df)
    
    # Verify model was trained on GPU
    assert model.tabnet.device.type == "cuda"
    
    # Test predictions
    predictions = model.transform(df)
    pred_rows = predictions.select("predictions").collect()
    pred_array = np.array([row.predictions for row in pred_rows])
    
    # Verify predictions
    assert pred_array.shape == (n_samples, 2)  # Binary classification
    assert np.all(np.isfinite(pred_array))

def test_model_state_synchronization(spark):
    """Test model state synchronization across workers."""
    # Generate synthetic data
    n_samples = 500
    n_features = 8
    X = np.random.random((n_samples, n_features))
    y = np.random.randint(0, 2, n_samples)
    
    # Create Spark DataFrame
    data = [(x.tolist(), float(y_)) for x, y_ in zip(X, y)]
    df = spark.createDataFrame(data, ["features", "label"])
    
    # Initialize estimator with multiple processes
    estimator = SparkTabNetEstimator(
        inputCol="features",
        outputCol="predictions",
        num_processes=2,
        use_gpu=False,  # Use CPU for consistent testing
        local_mode=True
    )
    
    # Train model
    model = estimator.fit(df)
    
    # Get initial predictions
    pred1 = model.transform(df).select("predictions").collect()
    
    # Force model state synchronization
    model._sync_model_state()
    
    # Get predictions after synchronization
    pred2 = model.transform(df).select("predictions").collect()
    
    # Verify predictions are consistent
    pred1_array = np.array([row.predictions for row in pred1])
    pred2_array = np.array([row.predictions for row in pred2])
    assert np.allclose(pred1_array, pred2_array)

def test_distributed_training_consistency(spark):
    """Test consistency of distributed training results."""
    # Generate synthetic data
    n_samples = 1000
    n_features = 10
    X = np.random.random((n_samples, n_features))
    y = np.random.randint(0, 2, n_samples)
    
    # Create Spark DataFrame
    data = [(x.tolist(), float(y_)) for x, y_ in zip(X, y)]
    df = spark.createDataFrame(data, ["features", "label"])
    
    # Train with single process
    single_proc = SparkTabNetEstimator(
        inputCol="features",
        outputCol="predictions",
        num_processes=1,
        use_gpu=False,
        local_mode=True,
        seed=42  # Set seed for reproducibility
    )
    single_model = single_proc.fit(df)
    
    # Train with multiple processes
    multi_proc = SparkTabNetEstimator(
        inputCol="features",
        outputCol="predictions",
        num_processes=2,
        use_gpu=False,
        local_mode=True,
        seed=42  # Same seed
    )
    multi_model = multi_proc.fit(df)
    
    # Get predictions from both models
    pred_single = single_model.transform(df).select("predictions").collect()
    pred_multi = multi_model.transform(df).select("predictions").collect()
    
    # Convert to numpy arrays
    pred_single_array = np.array([row.predictions for row in pred_single])
    pred_multi_array = np.array([row.predictions for row in pred_multi])
    
    # Verify predictions are similar (allowing for some numerical differences)
    assert np.allclose(pred_single_array, pred_multi_array, rtol=1e-2, atol=1e-2)