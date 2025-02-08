"""TabNet implementation in PyTorch with Spark ML integration."""

import warnings

# Filter Pydantic deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

# Version of the pytorch-tabnet package
__version__ = "4.1.0"
