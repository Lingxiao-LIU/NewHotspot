from .hotspot import Hotspot

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "hotspot"
__version__ = importlib_metadata.version(package_name)
