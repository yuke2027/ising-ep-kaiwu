"""Reusable PyTorch Ising EP core.

This package intentionally contains no Kaiwu dependency. Official SDK access is
kept at the ``examples.mnist_kaiwu`` boundary.
"""

from .model import IsingEPModel
from .mapping import QuboProblem

__all__ = ["IsingEPModel", "QuboProblem"]
