"""Tests for TabNet model factory functions."""

import pytest
import torch

from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.spark.factory import create_tabnet_model, validate_loaded_model


def test_create_tabnet_model():
    """Test creating TabNet model from parameters."""
    params = {
        "input_dim": 10,
        "output_dim": 2,
        "n_d": 8,
        "n_a": 8,
        "n_steps": 3,
        "gamma": 1.3,
        "n_independent": 2,
        "n_shared": 2,
        "virtual_batch_size": 128,
        "momentum": 0.02,
        "mask_type": "sparsemax"
    }
    
    model = create_tabnet_model(params)
    assert isinstance(model, TabNet)
    assert model.input_dim == params["input_dim"]
    assert model.output_dim == params["output_dim"]
    assert model.n_d == params["n_d"]
    assert model.n_a == params["n_a"]
    assert model.n_steps == params["n_steps"]


def test_missing_parameters():
    """Test error handling for missing parameters."""
    # Missing required parameter
    params = {
        "input_dim": 10,
        "output_dim": 2,
        # Missing n_d
        "n_a": 8,
        "n_steps": 3,
        "gamma": 1.3,
        "n_independent": 2,
        "n_shared": 2,
        "virtual_batch_size": 128,
        "momentum": 0.02,
        "mask_type": "sparsemax"
    }
    
    with pytest.raises(ValueError, match="Missing required parameters"):
        create_tabnet_model(params)


def test_invalid_parameters():
    """Test error handling for invalid parameters."""
    base_params = {
        "input_dim": 10,
        "output_dim": 2,
        "n_d": 8,
        "n_a": 8,
        "n_steps": 3,
        "gamma": 1.3,
        "n_independent": 2,
        "n_shared": 2,
        "virtual_batch_size": 128,
        "momentum": 0.02,
        "mask_type": "sparsemax"
    }
    
    # Test invalid n_steps
    invalid_params = base_params.copy()
    invalid_params["n_steps"] = 0
    with pytest.raises(ValueError, match="n_steps must be positive"):
        create_tabnet_model(invalid_params)
    
    # Test invalid n_independent and n_shared
    invalid_params = base_params.copy()
    invalid_params["n_independent"] = 0
    invalid_params["n_shared"] = 0
    with pytest.raises(ValueError, match="n_independent and n_shared cannot both be zero"):
        create_tabnet_model(invalid_params)
    
    # Test invalid mask_type
    invalid_params = base_params.copy()
    invalid_params["mask_type"] = "invalid"
    with pytest.raises(ValueError, match="mask_type must be either 'sparsemax' or 'entmax'"):
        create_tabnet_model(invalid_params)


def test_validate_loaded_model():
    """Test validation of loaded model parameters."""
    # Create model with known parameters
    params = {
        "input_dim": 10,
        "output_dim": 2,
        "n_d": 8,
        "n_a": 8,
        "n_steps": 3,
        "gamma": 1.3,
        "n_independent": 2,
        "n_shared": 2,
        "virtual_batch_size": 128,
        "momentum": 0.02,
        "mask_type": "sparsemax"
    }
    model = create_tabnet_model(params)
    
    # Test successful validation
    validate_loaded_model(model, params)
    
    # Test mismatched input dimension
    wrong_params = params.copy()
    wrong_params["input_dim"] = 20
    with pytest.raises(ValueError, match="Model input dimension mismatch"):
        validate_loaded_model(model, wrong_params)
    
    # Test mismatched output dimension
    wrong_params = params.copy()
    wrong_params["output_dim"] = 3
    with pytest.raises(ValueError, match="Model output dimension mismatch"):
        validate_loaded_model(model, wrong_params)
    
    # Test mismatched architecture parameters
    wrong_params = params.copy()
    wrong_params["n_d"] = 16
    with pytest.raises(ValueError, match="Model n_d mismatch"):
        validate_loaded_model(model, wrong_params)


def test_model_state_dict():
    """Test model state dict compatibility after creation."""
    params = {
        "input_dim": 10,
        "output_dim": 2,
        "n_d": 8,
        "n_a": 8,
        "n_steps": 3,
        "gamma": 1.3,
        "n_independent": 2,
        "n_shared": 2,
        "virtual_batch_size": 128,
        "momentum": 0.02,
        "mask_type": "sparsemax"
    }
    
    # Create two models with same parameters
    model1 = create_tabnet_model(params)
    model2 = create_tabnet_model(params)
    
    # Get state dict from first model
    state_dict = model1.state_dict()
    
    # Load state dict into second model
    model2.load_state_dict(state_dict)
    
    # Verify models have same parameters
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        assert name1 == name2
        assert torch.equal(param1, param2)


def test_model_forward_pass():
    """Test model can perform forward pass after creation."""
    params = {
        "input_dim": 10,
        "output_dim": 2,
        "n_d": 8,
        "n_a": 8,
        "n_steps": 3,
        "gamma": 1.3,
        "n_independent": 2,
        "n_shared": 2,
        "virtual_batch_size": 128,
        "momentum": 0.02,
        "mask_type": "sparsemax"
    }
    
    model = create_tabnet_model(params)
    
    # Create dummy input
    batch_size = 32
    x = torch.randn(batch_size, params["input_dim"])
    
    # Perform forward pass
    with torch.no_grad():
        outputs, _ = model(x)
    
    # Verify output shape
    assert outputs.shape == (batch_size, params["output_dim"])