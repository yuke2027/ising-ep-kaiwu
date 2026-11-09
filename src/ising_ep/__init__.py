"""Reusable PyTorch Ising equilibrium-propagation core.

The energy model and the QUBO mapping are pure Torch. The only module that
touches the licensed Kaiwu SDK is :mod:`ising_ep.kaiwu_solver`, which is
imported lazily so the rest of the package works without the enterprise wheel.
"""

from .kaiwu_solver import AnnealSettings, KaiwuSolver, init_license
from .mapping import build_qubo, qubo_energy
from .model import IsingEPModel

__all__ = [
    "AnnealSettings",
    "IsingEPModel",
    "KaiwuSolver",
    "build_qubo",
    "init_license",
    "qubo_energy",
]
