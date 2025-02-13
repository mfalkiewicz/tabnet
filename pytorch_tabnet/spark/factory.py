"""Factory functions for creating TabNet models."""

from typing import Dict, Any
import logging

from pytorch_tabnet.tab_network import TabNet


def create_tabnet_model(params: Dict[str, Any]) -> TabNet:
    """Create a TabNet model instance from parameters.
    
    This factory function encapsulates the logic for creating a TabNet model
    with the correct architecture and parameters. It's used during model loading
    to ensure consistent model reconstruction.
    
    Args:
        params: Dictionary containing model parameters including:
            - input_dim: Input feature dimension
            - output_dim: Output dimension (number of classes)
            - n_d: Width of the decision prediction layer
            - n_a: Width of the attention embedding
            - n_steps: Number of steps in the architecture
            - gamma: Scale factor for attention updates
            - n_independent: Number of independent GLU layers
            - n_shared: Number of shared GLU layers
            - virtual_batch_size: Size of virtual batches
            - momentum: Momentum for batch normalization
            - mask_type: Type of mask to use (sparsemax or entmax)
            
    Returns:
        Initialized TabNet model
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    required_params = {
        "input_dim", "output_dim", "n_d", "n_a", "n_steps",
        "gamma", "n_independent", "n_shared", "virtual_batch_size",
        "momentum", "mask_type"
    }
    
    # Validate required parameters
    missing_params = required_params - set(params.keys())
    if missing_params:
        msg = f"Missing required parameters: {missing_params}"
        logging.error(msg)
        raise ValueError(msg)
    
    # Validate parameter values
    if params["n_steps"] <= 0:
        raise ValueError("n_steps must be positive")
    if params["n_independent"] == 0 and params["n_shared"] == 0:
        raise ValueError("n_independent and n_shared cannot both be zero")
    if params["mask_type"] not in ["sparsemax", "entmax"]:
        raise ValueError("mask_type must be either 'sparsemax' or 'entmax'")
    
    try:
        # Create and return TabNet model
        return TabNet(
            input_dim=params["input_dim"],
            output_dim=params["output_dim"],
            n_d=params["n_d"],
            n_a=params["n_a"],
            n_steps=params["n_steps"],
            gamma=params["gamma"],
            n_independent=params["n_independent"],
            n_shared=params["n_shared"],
            virtual_batch_size=params["virtual_batch_size"],
            momentum=params["momentum"],
            mask_type=params["mask_type"]
        )
    except Exception as e:
        msg = f"Error creating TabNet model: {str(e)}"
        logging.error(msg)
        raise ValueError(msg)


def validate_loaded_model(model: TabNet, params: Dict[str, Any]) -> None:
    """Validate that a loaded model matches the expected parameters.
    
    Args:
        model: Loaded TabNet model instance
        params: Dictionary of expected parameters
        
    Raises:
        ValueError: If model parameters don't match expected values
    """
    # Verify key attributes match saved parameters
    if model.input_dim != params["input_dim"]:
        raise ValueError(
            f"Model input dimension mismatch: expected {params['input_dim']}, "
            f"got {model.input_dim}"
        )
    if model.output_dim != params["output_dim"]:
        raise ValueError(
            f"Model output dimension mismatch: expected {params['output_dim']}, "
            f"got {model.output_dim}"
        )
    
    # Verify architecture parameters
    if model.n_d != params["n_d"]:
        raise ValueError(
            f"Model n_d mismatch: expected {params['n_d']}, got {model.n_d}"
        )
    if model.n_a != params["n_a"]:
        raise ValueError(
            f"Model n_a mismatch: expected {params['n_a']}, got {model.n_a}"
        )
    if model.n_steps != params["n_steps"]:
        raise ValueError(
            f"Model n_steps mismatch: expected {params['n_steps']}, "
            f"got {model.n_steps}"
        )