try:
    from ._version import get_versions

    __version__ = get_versions()["version"]
    del get_versions
except ImportError:
    __version__ = "0.0.0"

from .tab_model import TabNetRegressor, TabNetClassifier
from .tab_network import TabNet

__all__ = ["TabNetRegressor", "TabNetClassifier", "TabNet", "__version__"]
