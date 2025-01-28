from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class ModelState:
    """Tracks the state of a TabNet model."""
    initialized: bool = False
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    current_epoch: int = 0
    device: Optional[torch.device] = None
    training: bool = False
    
    def validate(self):
        """Validates that the model is in a proper state for operations."""
        if not self.initialized:
            raise RuntimeError("Model not initialized")
        if self.input_dim is None:
            raise ValueError("Input dimension not set")
        if self.output_dim is None:
            raise ValueError("Output dimension not set")
        if self.device is None:
            raise ValueError("Device not set")

    def update(self, **kwargs: Dict[str, Any]) -> None:
        """Updates state attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid state attribute: {key}")

    def to_dict(self) -> Dict[str, Any]:
        """Converts state to dictionary for serialization."""
        return {
            "initialized": self.initialized,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "current_epoch": self.current_epoch,
            "device": str(self.device) if self.device else None,
            "training": self.training
        }

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> 'ModelState':
        """Creates ModelState from dictionary."""
        if state_dict.get("device"):
            state_dict["device"] = torch.device(state_dict["device"])
        return cls(**state_dict)