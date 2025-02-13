"""Tests for TorchDistributor integration with TabNet."""

import pytest
import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.ml.torch.distributor import TorchDistributor
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import StructType, StructField, DoubleType

from pytorch_tabnet.spark.tabnet_pyspark import TabNetEstimator, TabNetModel


def test_torch_distributor_integration(large_data):
    """Test TorchDistributor integration with TabNet."""
    # Initialize TabNet estimator
    estimator = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label",
        n_d=8,
        n_a=8,
        n_steps=3,
        num_processes=2,  # Use 2 processes for distributed training
        use_gpu=torch.cuda.is_available()  # Use GPU if available
    )
    
    # Train model
    model = estimator.fit(large_data)
    
    # Verify model was trained
    assert model is not None
    assert model._torch_model is not None
    
    # Test predictions
    predictions = model.transform(large_data)
    assert "prediction" in predictions.columns
    
    # Get predictions as numpy array
    pred_rows = predictions.select("prediction").collect()
    pred_list = []
    for row in pred_rows:
        pred_list.append(row.prediction)
    pred_array = np.array(pred_list)
    
    # Verify prediction shape and values
    assert pred_array.shape[0] == large_data.count()
    assert np.all(np.isfinite(pred_array))  # No NaN or inf values


def test_torch_distributor_gpu_handling(small_data):
    """Test TorchDistributor GPU handling."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Initialize estimator with GPU config
    estimator = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label",
        num_processes=2,
        use_gpu=True
    )
    
    # Train model
    model = estimator.fit(small_data)
    
    # Test predictions
    predictions = model.transform(small_data)
    pred_rows = predictions.select("prediction").collect()
    pred_array = np.array([row.prediction for row in pred_rows])
    
    # Verify predictions
    assert pred_array.shape[0] == small_data.count()
    assert np.all(np.isfinite(pred_array))


def test_distributed_training_consistency(small_data):
    """Test consistency of distributed training results."""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Train with single process
    single_proc = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label",
        num_processes=1,
        use_gpu=False,
        n_d=8,
        n_a=8,
        n_steps=3,
        virtual_batch_size=32
    )
    single_model = single_proc.fit(small_data)
    
    # Train with multiple processes
    multi_proc = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label",
        num_processes=2,
        use_gpu=False,
        n_d=8,
        n_a=8,
        n_steps=3,
        virtual_batch_size=32
    )
    multi_model = multi_proc.fit(small_data)
    
    # Get predictions from both models
    pred_single = single_model.transform(small_data).select("prediction").collect()
    pred_multi = multi_model.transform(small_data).select("prediction").collect()
    
    # Convert to numpy arrays
    pred_single_array = np.array([row.prediction for row in pred_single])
    pred_multi_array = np.array([row.prediction for row in pred_multi])
    
    # Verify predictions are similar (allowing for some numerical differences)
    assert np.allclose(pred_single_array, pred_multi_array, rtol=1e-2, atol=1e-2)


def test_error_handling(spark):
    """Test error handling in distributed training."""
    # Create schema for invalid data
    schema = StructType([
        StructField("features", VectorUDT()),
        StructField("label", DoubleType())
    ])
    
    # Create DataFrame with missing features
    data = [(None, 0.0), (None, 1.0)]
    df = spark.createDataFrame(data, schema)
    
    estimator = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label"
    )
    
    # Verify training fails gracefully
    with pytest.raises(ValueError, match="Input features cannot be None"):
        estimator.fit(df)


def test_model_persistence(small_data, tmp_path):
    """Test model persistence in distributed setting."""
    # Train model
    estimator = TabNetEstimator(
        inputCol="features",
        outputCol="prediction",
        labelCol="label",
        num_processes=2
    )
    model = estimator.fit(small_data)
    
    # Save and load model
    model_path = str(tmp_path / "tabnet_model")
    model.save(model_path)
    loaded_model = TabNetModel.load(model_path)
    
    # Compare predictions
    orig_preds = model.transform(small_data).select("prediction").collect()
    loaded_preds = loaded_model.transform(small_data).select("prediction").collect()
    
    # Verify predictions match
    orig_array = np.array([row.prediction for row in orig_preds])
    loaded_array = np.array([row.prediction for row in loaded_preds])
    assert np.array_equal(orig_array, loaded_array)