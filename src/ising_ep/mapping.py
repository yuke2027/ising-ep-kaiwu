"""Torch energy, QUBO mapping and auxiliary-spin decoding utilities.

The QUBO convention is ``q.T @ Q @ q + offset`` where ``q = (s + 1) / 2`` and
``Q`` is upper triangular. Physical variables are ordered as hidden spins first,
followed by output spins.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class QuboProblem:
    """QUBO matrix and metadata needed to recover physical states.

    Attributes:
        Q: Upper-triangular ``(n_hidden + n_output, n_hidden + n_output)`` matrix.
        constant: Scalar offset added to ``q.T @ Q @ q``.
        n_hidden: Number of hidden spins.
        n_output: Number of output spins.
        variable_order: Fixed physical variable order, ``("hidden", "output")``.
    """

    Q: torch.Tensor
    constant: float
    n_hidden: int
    n_output: int
    variable_order: tuple[str, str] = ("hidden", "output")


def _tensor(value: Any, name: str, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} cannot be converted to a float64 tensor") from exc
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def energy(
    x: Any,
    h: Any,
    o: Any,
    W: Any,
    V: Any,
    b_h: Any,
    b_o: Any,
) -> float:
    """Return the scalar EP energy ``-xWh - hVo - b_h·h - b_o·o``."""
    x_t = _tensor(x, "x").reshape(-1)
    h_t = _tensor(h, "hidden").reshape(-1)
    o_t = _tensor(o, "output").reshape(-1)
    W_t = _tensor(W, "W")
    V_t = _tensor(V, "V")
    bh_t = _tensor(b_h, "b_h").reshape(-1)
    bo_t = _tensor(b_o, "b_o").reshape(-1)
    value = -x_t @ W_t @ h_t - h_t @ V_t @ o_t - bh_t @ h_t - bo_t @ o_t
    return float(value.item())


def cost_c(o: Any, y: Any) -> float:
    """Return the EP output cost ``n_output - y·o``."""
    output = _tensor(o, "output").reshape(-1)
    target = _tensor(y, "target").reshape(-1)
    return float((output.numel() - target @ output).item())


def free_energy(
    x: Any,
    h: Any,
    o: Any,
    y: Any,
    W: Any,
    V: Any,
    b_h: Any,
    b_o: Any,
    beta: float,
) -> float:
    """Return ``energy + beta * cost`` for one physical state."""
    beta_value = float(beta)
    if not math.isfinite(beta_value) or beta_value < 0:
        raise ValueError("beta must be a non-negative finite value")
    return energy(x, h, o, W, V, b_h, b_o) + beta_value * cost_c(o, y)


def hidden_field(x: Any, W: Any, b_h: Any) -> torch.Tensor:
    """Return the hidden local field ``W.T @ x + b_h``."""
    x_t = _tensor(x, "x").reshape(-1)
    W_t = _tensor(W, "W")
    b_t = _tensor(b_h, "b_h").reshape(-1)
    return W_t.T @ x_t + b_t


def output_field(b_o: Any, y: Any, beta: float) -> torch.Tensor:
    """Return the output local field ``b_o + beta * y``."""
    beta_value = float(beta)
    if not math.isfinite(beta_value) or beta_value < 0:
        raise ValueError("beta must be a non-negative finite value")
    return _tensor(b_o, "b_o").reshape(-1) + beta_value * _tensor(y, "target").reshape(-1)


def clip_solver_problem(
    V: Any,
    H_h: Any,
    H_o: Any,
    bias_lim: float,
    couple_lim: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip fields and couplings for a solver copy without changing parameters."""
    return (
        torch.clamp(_tensor(V, "V"), -float(couple_lim), float(couple_lim)),
        torch.clamp(_tensor(H_h, "H_h"), -float(bias_lim), float(bias_lim)),
        torch.clamp(_tensor(H_o, "H_o"), -float(bias_lim), float(bias_lim)),
    )


def coupling_and_fields(V: Any, H_h: Any, H_o: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the signed Ising coupling matrix and concatenated local fields."""
    V_t = _tensor(V, "V")
    h_t = _tensor(H_h, "H_h").reshape(-1)
    o_t = _tensor(H_o, "H_o").reshape(-1)
    n_h, n_o = h_t.numel(), o_t.numel()
    if V_t.shape != (n_h, n_o):
        raise ValueError(f"V must have shape ({n_h}, {n_o}), got {tuple(V_t.shape)}")
    J = torch.zeros((n_h + n_o, n_h + n_o), dtype=V_t.dtype)
    J[:n_h, n_h:] = V_t
    J[n_h:, :n_h] = V_t.T
    return J, torch.cat([h_t, o_t])


def build_qubo(V: Any, H_h: Any, H_o: Any, beta: float) -> QuboProblem:
    """Build the upper-triangular QUBO for ``E + beta * C``.

    Variables are ordered hidden spins first, then output spins. Binary
    variables use ``q = (s + 1) / 2`` and ``Q`` is upper triangular.

    Args:
        V: Hidden-output coupling matrix with shape ``(n_hidden, n_output)``.
        H_h: Hidden local field with shape ``(n_hidden,)``.
        H_o: Output local field including ``beta * target`` with shape
            ``(n_output,)``.
        beta: Non-negative nudging coefficient.

    Returns:
        A :class:`QuboProblem` whose matrix represents the EP energy up to the
        recorded constant offset.

    Raises:
        ValueError: If ``beta`` is negative or ``V`` does not match the fields.
    """
    beta_value = float(beta)
    if not math.isfinite(beta_value) or beta_value < 0:
        raise ValueError("beta must be a non-negative finite value")
    V_t = _tensor(V, "V").detach().clone()
    h_t = _tensor(H_h, "H_h").detach().reshape(-1)
    o_t = _tensor(H_o, "H_o").detach().reshape(-1)
    n_h, n_o = h_t.numel(), o_t.numel()
    if V_t.shape != (n_h, n_o):
        raise ValueError(f"V must have shape ({n_h}, {n_o}), got {tuple(V_t.shape)}")

    n = n_h + n_o
    Q = torch.zeros((n, n), dtype=V_t.dtype)
    Q[:n_h, :n_h] = torch.diag(2.0 * V_t.sum(dim=1) - 2.0 * h_t)
    Q[n_h:, n_h:] = torch.diag(2.0 * V_t.sum(dim=0) - 2.0 * o_t)
    Q[:n_h, n_h:] = -4.0 * V_t
    constant = float((beta_value * n_o + h_t.sum() + o_t.sum() - V_t.sum()).item())
    return QuboProblem(Q=Q, constant=constant, n_hidden=n_h, n_output=n_o)


def spins_to_bits(spins: Any) -> torch.Tensor:
    """Convert ±1 spins to 0/1 QUBO variables."""
    return (_tensor(spins, "spins").reshape(-1) + 1.0) * 0.5


def qubo_energy(Q: Any, constant: float, spins: Any) -> float:
    """Evaluate ``q.T @ Q @ q + constant`` for one physical state.

    Args:
        Q: Upper-triangular QUBO matrix. Columns follow hidden spins, then
            output spins.
        constant: Scalar offset stored with ``Q``.
        spins: Physical spins in ``{+1, -1}``. Bits are ``q = (s + 1) / 2``.

    Returns:
        The scalar QUBO value.

    Raises:
        ValueError: If ``Q`` is not square or does not match ``spins``.
    """
    matrix = _tensor(Q, "Q")
    bits = spins_to_bits(spins)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != bits.numel():
        raise ValueError("Q must be square and match the spin length")
    return float((bits @ matrix @ bits + float(constant)).item())


def decode_aux(solver_spins: Any) -> torch.Tensor:
    """Decode physical spins from either auxiliary-spin gauge.

    The last entry is the auxiliary spin. Physical spins are
    ``s_solver[:-1] * s_aux``, so both auxiliary gauges restore the same
    ``{+1, -1}`` state.

    Args:
        solver_spins: Solver spins in ``{+1, -1}``, physical variables first.

    Returns:
        A ``float64`` physical vector of length ``len(solver_spins) - 1``.

    Raises:
        ValueError: If the auxiliary spin is missing or a value is not ±1.
    """
    state = _tensor(solver_spins, "solver_spins").reshape(-1)
    if state.numel() < 2:
        raise ValueError("solver spin vector must include an auxiliary spin")
    if not torch.all((state == 1.0) | (state == -1.0)):
        raise ValueError("solver spins must be exactly ±1")
    return state[:-1] * state[-1]


def decode_aux_alt(solver_spins: Any) -> torch.Tensor:
    """Decode physical spins with the alternate auxiliary convention.

    This is equivalent to :func:`decode_aux`: the last spin is the auxiliary
    gauge, and both gauges map back to the same ``{+1, -1}`` physical state.

    Args:
        solver_spins: Solver spins in ``{+1, -1}``, physical variables first.

    Returns:
        A ``float64`` physical vector of length ``len(solver_spins) - 1``.

    Raises:
        ValueError: If the auxiliary spin is missing or a value is not ±1.
    """
    state = _tensor(solver_spins, "solver_spins").reshape(-1)
    if state.numel() < 2:
        raise ValueError("solver spin vector must include an auxiliary spin")
    if not torch.all((state == 1.0) | (state == -1.0)):
        raise ValueError("solver spins must be exactly ±1")
    physical = state[:-1]
    return physical if state[-1] > 0 else -physical


def split_hidden_output(spins: Any, n_hidden: int, n_output: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a physical vector into hidden and output blocks.

    Args:
        spins: Physical spins in ``{+1, -1}``, ordered hidden then output.
        n_hidden: Number of leading hidden spins.
        n_output: Number of trailing output spins.

    Returns:
        Hidden and output ``float64`` vectors.

    Raises:
        ValueError: If ``spins`` does not have length ``n_hidden + n_output``.
    """
    state = _tensor(spins, "spins").reshape(-1)
    expected = int(n_hidden) + int(n_output)
    if state.numel() != expected:
        raise ValueError(f"expected {expected} physical spins, got {state.numel()}")
    return state[: int(n_hidden)].clone(), state[int(n_hidden) :].clone()


def iter_physical_states(n: int) -> Iterator[torch.Tensor]:
    """Yield every ±1 assignment of length ``n`` in lexicographic (-1, +1) order."""
    if int(n) < 1:
        raise ValueError("n must be positive")
    for bits in itertools.product((-1.0, 1.0), repeat=int(n)):
        yield torch.tensor(bits, dtype=torch.float64)


def energy_equivalence_report(
    ep_energies: Any,
    qubo_energies: Any,
    *,
    atol: float = 1e-8,
) -> dict[str, float]:
    """Verify that two energy tables differ only by a constant offset."""
    ep = np.asarray([_to_numpy(value) for value in ep_energies], dtype=np.float64).reshape(-1)
    qubo = np.asarray([_to_numpy(value) for value in qubo_energies], dtype=np.float64).reshape(-1)
    if ep.shape != qubo.shape:
        raise ValueError("energy tables must have the same shape")
    delta = ep - qubo
    offset = float(delta[0]) if delta.size else 0.0
    max_abs = float(np.max(np.abs(delta - offset))) if delta.size else 0.0
    if max_abs > float(atol):
        raise ValueError(f"EP and QUBO energies differ by more than a constant: {max_abs}")
    return {"constant": offset, "max_abs_residual": max_abs}


def minimizer_indices(energies: Any, *, atol: float = 1e-8) -> tuple[int, ...]:
    """Return all indices whose energy is within ``atol`` of the minimum."""
    values = np.asarray([_to_numpy(value) for value in energies], dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("energies must not be empty")
    lowest = float(np.min(values))
    return tuple(int(i) for i, value in enumerate(values.tolist()) if abs(value - lowest) <= float(atol))
