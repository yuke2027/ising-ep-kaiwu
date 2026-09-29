"""Shared pytest helpers for the Ising EP refactor."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def to_numpy(value: Any) -> np.ndarray:
    """Convert NumPy arrays or tensors to a detached NumPy array."""
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def set_parameter(parameter: Any, value: Any) -> None:
    """Set a NumPy array or torch parameter without changing its identity."""
    array = np.asarray(value, dtype=np.float64)
    if hasattr(parameter, "data") and hasattr(parameter, "copy_"):
        with torch.no_grad():
            parameter.copy_(torch.as_tensor(array, dtype=parameter.dtype))
        return
    parameter[...] = array
