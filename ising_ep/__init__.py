"""Thin compatibility package for the PyTorch Ising EP core.

Implementations live in ``src.ising_ep``. Official Kaiwu 1.4.1 calls live in
``examples.mnist_kaiwu``.
"""

from .mapping import build_qubo, decode_aux, energy
from .model import IsingEPModel

__all__ = ["IsingEPModel", "build_qubo", "decode_aux", "energy"]
