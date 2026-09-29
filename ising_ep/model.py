"""Backward-compatible import path for the PyTorch Ising EP model.

The implementation lives in ``src/ising_ep/model.py``; this module remains as a
thin legacy entry point for historical scripts and tests.
"""

from src.ising_ep.model import IsingEPModel, ep_gradients

__all__ = ["IsingEPModel", "ep_gradients"]
