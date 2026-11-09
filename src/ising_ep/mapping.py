"""Torch mapping between the EP energy and the QUBO form Kaiwu consumes.

Conventions used throughout this module:

* Physical spins are ``s in {-1, +1}``, ordered layer by layer: the first
  hidden layer first, the output layer last.
* Binary QUBO variables are ``q = (s + 1) / 2``, so ``q in {0, 1}``.
* A QUBO is scored as ``q @ Q @ q + constant`` with ``Q`` upper triangular.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def _float64(value: Any) -> torch.Tensor:
    """Convert a value to a float64 tensor detached from autograd.

    Args:
        value: Tensor or array-like holding model parameters, fields or spins.

    Returns:
        A detached ``float64`` tensor view of ``value``.
    """
    return torch.as_tensor(value).detach().to(dtype=torch.float64)


def build_qubo(
    couplings: Sequence[Any],
    fields: Sequence[Any],
    beta: float,
) -> tuple[torch.Tensor, float]:
    """Rewrite a stacked ``F = E + beta * C`` as an upper-triangular QUBO.

    With spin layers ``s_0, ..., s_{K-1}`` (the last one is the output), the
    local fields ``H_l`` and the spin-spin couplings ``W_l`` between layer
    ``l - 1`` and ``l``, the nudged EP objective collapses to::

        F = -sum_{l>=1} s_{l-1} W_l s_l - sum_l H_l . s_l + beta * n_output

    The clamped input never appears here: it is already inside ``H_0``.
    Substituting ``s = 2q - 1`` and using ``q**2 == q`` for binary variables
    moves all linear terms onto the diagonal of ``Q``:

    * each coupling contributes ``-4 * W`` to the block between its two layers,
      plus ``+2 * W.sum(axis)`` onto each of the two diagonal blocks;
    * each local field contributes ``-2 * H`` onto its diagonal;
    * the leftovers ``-sum W + sum H + beta * n_output`` become the scalar
      offset, which shifts the objective but not its minimizers.

    A single hidden layer is ``couplings = [V]`` and ``fields = [H_h, H_o]``.

    Args:
        couplings: One weight matrix per adjacent pair of spin layers, of shape
            ``(width_l, width_{l+1})``. Empty for a single spin layer.
        fields: Local field of every spin layer, output last. The nudge
            ``beta * target`` is already part of the last field.
        beta: Nudging coefficient. It enters ``Q`` only through the last field;
            here it contributes the ``beta * n_output`` offset.

    Returns:
        A tuple ``(Q, constant)`` where ``Q`` is an upper-triangular
        ``float64`` matrix and ``constant`` is the scalar offset such that
        ``q @ Q @ q + constant == F`` for ``q = (s + 1) / 2``.

    Raises:
        ValueError: If ``fields`` is empty or ``couplings`` does not hold one
            matrix per adjacent pair of layers.
    """
    local = [_float64(field).reshape(-1) for field in fields]
    if not local:
        raise ValueError("build_qubo needs at least one local field")
    if len(couplings) != len(local) - 1:
        raise ValueError(
            f"expected {len(local) - 1} couplings for {len(local)} fields, got {len(couplings)}"
        )

    widths = [int(field.shape[0]) for field in local]
    starts = [sum(widths[:index]) for index in range(len(widths))]
    total = sum(widths)
    Q = torch.zeros((total, total), dtype=torch.float64)
    offset = torch.zeros((), dtype=torch.float64) + float(beta) * widths[-1]

    for index, coupling in enumerate(couplings):
        weight = _float64(coupling)
        left, right = starts[index], starts[index + 1]
        left_width, right_width = widths[index], widths[index + 1]
        if tuple(weight.shape) != (left_width, right_width):
            raise ValueError(
                f"coupling {index} must have shape {(left_width, right_width)}, "
                f"got {tuple(weight.shape)}"
            )
        # Diagonals carry the linear coefficients because q**2 == q.
        Q[left : left + left_width, left : left + left_width].diagonal().add_(
            2.0 * weight.sum(dim=1)
        )
        Q[right : right + right_width, right : right + right_width].diagonal().add_(
            2.0 * weight.sum(dim=0)
        )
        # Upper-triangular cross block: (2q_j - 1)(2q_k - 1) expands to 4 q_j q_k.
        Q[left : left + left_width, right : right + right_width] = -4.0 * weight
        offset = offset - weight.sum()

    for start, width, field in zip(starts, widths, local):
        Q[start : start + width, start : start + width].diagonal().add_(-2.0 * field)
        offset = offset + field.sum()
    return Q, float(offset.item())


def qubo_energy(Q: Any, constant: float, spins: Any) -> float:
    """Score a physical spin state under a QUBO built by :func:`build_qubo`.

    Args:
        Q: Upper-triangular QUBO matrix of shape ``(n, n)``.
        constant: Scalar offset returned alongside ``Q``.
        spins: Physical spins in ``{-1, +1}`` of length ``n``, first layer first.

    Returns:
        The scalar value ``q @ Q @ q + constant`` with ``q = (spins + 1) / 2``.
    """
    matrix = _float64(Q)
    bits = (_float64(spins).reshape(-1) + 1.0) * 0.5
    return float((bits @ matrix @ bits + float(constant)).item())
