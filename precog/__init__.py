import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__ or __package__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback version for development

version_split = __version__.split(".")
__spec_version__ = (1000 * int(version_split[0])) + (10 * int(version_split[1])) + (1 * int(version_split[2]))
