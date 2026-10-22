"""Reusable PyTorch Ising equilibrium-propagation core.

The energy and the QUBO mapping live on :class:`IsingEPModel`. The only module
that touches the licensed Kaiwu SDK is :mod:`ising_ep.kaiwu_solver`, which is
imported lazily so the rest of the package works without the enterprise wheel.
"""

from .kaiwu_solver import AnnealSettings, KaiwuSolver, init_license
from .model import IsingEPModel

__all__ = [
    "AnnealSettings",
    "IsingEPModel",
    "KaiwuSolver",
    "init_license",
]
