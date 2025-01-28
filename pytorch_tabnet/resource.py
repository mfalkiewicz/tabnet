from typing import List, Optional, Any
import gc
import psutil
import torch
from abc import ABC, abstractmethod


class Resource(ABC):
    """Abstract base class for resources that need cleanup."""
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup the resource."""
        pass


class TensorResource(Resource):
    """Resource class for PyTorch tensors."""
    
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def cleanup(self) -> None:
        """Cleanup the tensor by moving it to CPU and freeing CUDA memory."""
        if self.tensor is not None and self.tensor.is_cuda:
            self.tensor = self.tensor.cpu()
            torch.cuda.empty_cache()


class ResourceManager:
    """Manages resources and their cleanup."""
    
    def __init__(self):
        self._resources: List[Resource] = []

    def register(self, resource: Resource) -> None:
        """Register a resource for management.
        
        Parameters
        ----------
        resource : Resource
            Resource to manage
        """
        self._resources.append(resource)

    def cleanup(self) -> None:
        """Cleanup all registered resources."""
        # Clean resources in reverse order (LIFO)
        for resource in reversed(self._resources):
            resource.cleanup()
        self._resources.clear()
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MemoryMonitor:
    """Monitors system and GPU memory usage."""
    
    def __init__(self):
        self._process = psutil.Process()

    @property
    def system_memory_usage(self) -> float:
        """Get current system memory usage in MB."""
        return self._process.memory_info().rss / (1024 * 1024)

    @property
    def gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in MB if available."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return None


class MemoryManager:
    """Manages memory usage and cleanup."""
    
    def __init__(self, max_memory_mb: Optional[float] = None):
        """
        Parameters
        ----------
        max_memory_mb : Optional[float]
            Maximum allowed memory usage in MB
        """
        self.max_memory = max_memory_mb
        self._monitor = MemoryMonitor()
        self._resource_manager = ResourceManager()

    def register_resource(self, resource: Any) -> None:
        """Register a resource for memory management.
        
        Parameters
        ----------
        resource : Any
            Resource to manage. If not a Resource instance,
            it will be wrapped in appropriate Resource class.
        """
        if isinstance(resource, Resource):
            self._resource_manager.register(resource)
        elif isinstance(resource, torch.Tensor):
            self._resource_manager.register(TensorResource(resource))

    def check_memory(self) -> None:
        """Check memory usage and cleanup if necessary."""
        if self.max_memory is None:
            return

        system_usage = self._monitor.system_memory_usage
        gpu_usage = self._monitor.gpu_memory_usage

        if system_usage > self.max_memory or (gpu_usage and gpu_usage > self.max_memory):
            self._cleanup_memory()

    def _cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        self._resource_manager.cleanup()


class ResourceContext:
    """Context manager for automatic resource cleanup."""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager or MemoryManager()

    def __enter__(self):
        return self.memory_manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.memory_manager._cleanup_memory()