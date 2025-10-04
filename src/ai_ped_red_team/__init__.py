"""Top-level package for ai_ped_red_team."""

from importlib import metadata

try:
    __version__ = metadata.version("ai-ped-red-team")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]
