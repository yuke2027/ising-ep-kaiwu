"""Tests for the EP-to-QUBO mapping on the model.

The EP energy is recomputed inline from its definition rather than reusing a
helper from the package, so the mapping is checked against the formula itself.
"""

# Test functions document intent in their summary; parametrised values do not
# benefit from Args sections.
# pylint: disable=missing-param-doc,missing-return-doc

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from ising_ep.mapping import build_qubo, qubo_energy
from ising_ep.model import IsingEPModel

from tests.conftest import iter_physical_states, to_numpy


def _reference_objective(
    x: np.ndarray,
    parts: list[np.ndarray],
    target: np.ndarray,
    params: dict[str, np.ndarray],
    beta: float,
) -> float:
    """Evaluate ``F = E + beta * C`` straight from the stacked EP definition."""
    previous = x
    energy = 0.0
    for index, current in enumerate(parts):
        energy -= previous @ params[f"weights.{index}"] @ current
        energy -= params[f"biases.{index}"] @ current
        previous = current
    output = parts[-1]
    cost = output.size - target @ output
    return float(energy + beta * cost)


def _model_with_fields(
    coupling: np.ndarray,
    hidden_field: np.ndarray,
    output_field: np.ndarray,
) -> IsingEPModel:
    """Return a model whose fields equal the arguments at a zero input.

    The input coupling is zero. Callers pass a zero target, so the hidden
    field is the first bias and the output field is the last bias, and
    ``beta`` only changes the QUBO offset.
    """
    n_hidden, n_output = np.asarray(coupling).shape
    model = IsingEPModel(
        n_in=1,
        hidden_sizes=int(n_hidden),
        n_classes=int(n_output),
        spins_per_class=1,
        seed=0,
    )
    with torch.no_grad():
        model.weights[0].zero_()
        model.weights[1].copy_(torch.as_tensor(coupling, dtype=torch.float64))
        model.biases[0].copy_(torch.as_tensor(hidden_field, dtype=torch.float64))
        model.biases[1].copy_(torch.as_tensor(output_field, dtype=torch.float64))
    return model


def _qubo(model: IsingEPModel, x, target, beta: float):
    """Build the QUBO of one sample the way the training code does."""
    with torch.no_grad():
        return build_qubo(list(model.weights[1:]), model.fields(x, target, beta), beta)


def test_qubo_matches_explicit_closed_form() -> None:
    """The QUBO matrix and offset match the hand-expanded closed form."""
    coupling = np.array([[1.0, 2.0], [3.0, 4.0]])
    hidden_field = np.array([0.5, -1.0])
    output_field = np.array([2.0, -0.5])
    beta = 0.3
    model = _model_with_fields(coupling, hidden_field, output_field)

    matrix, constant = _qubo(model, torch.zeros(1), torch.zeros(model.n_output), beta)

    # Hidden diag: 2*V.sum(axis=1) - 2*H_h = [2*3-1, 2*7+2] = [5, 16]
    # Output diag: 2*V.sum(axis=0) - 2*H_o = [2*4-4, 2*6+1] = [4, 13]
    # Cross block: -4*V
    # Offset: beta*n_out + H_h.sum() + H_o.sum() - V.sum() = 0.6 - 0.5 + 1.5 - 10
    expected = np.array(
        [
            [5.0, 0.0, -4.0, -8.0],
            [0.0, 16.0, -12.0, -16.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 13.0],
        ]
    )
    np.testing.assert_allclose(to_numpy(matrix), expected)
    assert constant == pytest.approx(-8.4)


def test_qubo_returns_matrix_and_offset_without_a_wrapper_type() -> None:
    """The mapping hands back a plain tensor and float, not a problem object."""
    model = _model_with_fields(np.ones((2, 3)), np.zeros(2), np.zeros(3))

    matrix, constant = _qubo(model, torch.zeros(1), torch.zeros(model.n_output), 0.0)

    assert matrix.shape == (5, 5)
    assert isinstance(constant, float)


@pytest.mark.parametrize("beta", [0.0, 0.7])
def test_ep_and_qubo_energies_differ_only_by_a_constant(beta: float) -> None:
    """Over all 64 states the two energies agree up to one shared offset."""
    model = IsingEPModel(n_in=3, hidden_sizes=2, n_classes=2, spins_per_class=2, seed=11)
    params = {name: to_numpy(value) for name, value in model.parameter_snapshot().items()}
    x = np.array([0.1, 0.7, -0.2])
    target = to_numpy(model.target_spins(1)).reshape(-1)

    matrix, constant = _qubo(model, x, target, beta)

    ep_values = []
    qubo_values = []
    width = model.n_hidden + model.n_output
    for spins in iter_physical_states(width):
        parts = [to_numpy(block) for block in model.split_state(spins)]
        ep_values.append(_reference_objective(x, parts, target, params, beta))
        qubo_values.append(qubo_energy(matrix, constant, spins))

    ep_array = np.asarray(ep_values)
    qubo_array = np.asarray(qubo_values)
    assert ep_array.size == 64
    # qubo folds every leftover term into `constant`, so the residual is 0.
    np.testing.assert_allclose(ep_array, qubo_array, atol=1e-10)
    assert set(np.flatnonzero(ep_array <= ep_array.min() + 1e-10)) == set(
        np.flatnonzero(qubo_array <= qubo_array.min() + 1e-10)
    )


def test_two_hidden_layers_match_the_qubo_on_every_state() -> None:
    """A second hidden layer stays inside the same QUBO identity."""
    model = IsingEPModel(n_in=2, hidden_sizes=(2, 2), n_classes=2, spins_per_class=1, seed=3)
    params = {name: to_numpy(value) for name, value in model.parameter_snapshot().items()}
    x = np.array([0.4, -0.3])
    target = to_numpy(model.target_spins(0)).reshape(-1)
    beta = 0.5
    matrix, constant = _qubo(model, x, target, beta)

    ep_values = []
    qubo_values = []
    for spins in iter_physical_states(sum(model.spin_widths)):
        parts = [to_numpy(block) for block in model.split_state(spins)]
        ep_values.append(_reference_objective(x, parts, target, params, beta))
        qubo_values.append(qubo_energy(matrix, constant, spins))

    np.testing.assert_allclose(np.asarray(ep_values), np.asarray(qubo_values), atol=1e-10)


def test_split_state_splits_on_every_layer_boundary() -> None:
    """Blocks follow the hidden widths and then the output."""
    model = IsingEPModel(n_in=1, hidden_sizes=(1, 1), n_classes=1, spins_per_class=1, seed=0)
    parts = model.split_state(np.array([1.0, -1.0, -1.0]))
    first, second, output = parts[0], parts[1], parts[2]

    np.testing.assert_array_equal(to_numpy(first), [1.0])
    np.testing.assert_array_equal(to_numpy(second), [-1.0])
    np.testing.assert_array_equal(to_numpy(output), [-1.0])


def test_iter_physical_states_enumerates_every_assignment() -> None:
    """The helper yields all 2**n spin assignments without repeats."""
    states = [tuple(to_numpy(s).tolist()) for s in iter_physical_states(3)]

    assert len(states) == 8
    assert len(set(states)) == 8
    assert set(states) == set(itertools.product((-1.0, 1.0), repeat=3))


def test_build_qubo_rejects_a_mismatched_coupling_count() -> None:
    """One coupling is required per adjacent pair of spin layers."""
    with pytest.raises(ValueError, match="couplings"):
        build_qubo([], [np.zeros(2), np.zeros(2)], 0.0)
    with pytest.raises(ValueError, match="at least one"):
        build_qubo([], [], 0.0)


def test_build_qubo_rejects_a_misshapen_coupling() -> None:
    """A coupling must match the widths of the two layers it joins."""
    with pytest.raises(ValueError, match="shape"):
        build_qubo([np.zeros((3, 2))], [np.zeros(2), np.zeros(2)], 0.0)
