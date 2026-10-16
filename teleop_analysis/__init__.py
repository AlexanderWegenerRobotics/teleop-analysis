from .catalog import Catalog, open_folder
from .loaders import register
from .loaders.base import Loader

__all__ = ["Catalog", "open_folder", "register", "Loader"]
__version__ = "0.1.0"
