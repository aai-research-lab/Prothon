"""Prothon: efficient comparison of protein conformational ensembles."""

from __future__ import annotations

MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 13)

try:
    from ._version import __version__
except ImportError:
    __version__ = "2.1.0.dev0"

from .core.prothon_core import Prothon
from .utils import load_trajectories

__all__ = ["Prothon", "load_trajectories", "__version__"]
