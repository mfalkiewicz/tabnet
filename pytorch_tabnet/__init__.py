try:
    from ._version import get_versions

    __version__ = get_versions()["version"]
    del get_versions
except ImportError:
    __version__ = "5.0.0"

from .tab_model import TabNetRegressor, TabNetClassifier
from .tab_network import TabNet
from . import spark

__all__ = ["TabNetRegressor", "TabNetClassifier", "TabNet", "spark", "__version__"]

from . import _version
__version__ = _version.get_versions()['version']
