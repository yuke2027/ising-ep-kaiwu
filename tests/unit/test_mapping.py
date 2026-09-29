"""Characterization tests for energy/QUBO mapping and state decoding."""

from __future__ import annotations

import numpy as np
import pytest

from src.ising_ep.mapping import (
    build_qubo,
    decode_aux,
    decode_aux_alt,
    energy_equivalence_report,
    free_energy,
    hidden_field,
    iter_physical_states,
    minimizer_indices,
    output_field,
    qubo_energy,
    split_hidden_output,
)
from src.ising_ep.model import IsingEPModel

from tests.conftest import to_numpy


def test_build_qubo_matches_explicit_closed_form() -> None:
    """The QUBO matrix matches the closed form for a two-by-two coupling."""
    v = np.array([[1.0, 2.0], [3.0, 4.0]])
    hidden = np.array([0.5, -1.0])
    output = np.array([2.0, -0.5])
    beta = 0.3

    problem = build_qubo(v, hidden, output, beta)
    q = to_numpy(problem.Q)

    expected = np.array(
        [
            [5.0, 0.0, -4.0, -8.0],
            [0.0, 16.0, -12.0, -16.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 13.0],
        ]
    )
    np.testing.assert_allclose(q, expected)
    assert problem.constant == pytest_approx(-8.4)
    assert problem.n_hidden == 2
    assert problem.n_output == 2


def test_ep_and_qubo_energies_differ_only_by_constant_on_tiny_model() -> None:
    """Enumerated EP and QUBO energies differ only by a constant."""
    model = IsingEPModel(n_in=3, n_hidden=2, n_classes=2, spins_per_class=2, seed=11)
    params = {name: to_numpy(value) for name, value in model.numpy_params().items()}
    x = np.array([0.1, 0.7, -0.2])
    y = to_numpy(model.target_spins([1]))[0]
    beta = 0.7
    problem = build_qubo(
        params["V"],
        hidden_field(x, params["W"], params["b_h"]),
        output_field(params["b_o"], y, beta),
        beta,
    )

    ep_energies = []
    qubo_energies = []
    for spins in iter_physical_states(model.n_hidden + model.n_output):
        hidden, output = split_hidden_output(spins, model.n_hidden, model.n_output)
        ep_energies.append(
            free_energy(x, hidden, output, y, params["W"], params["V"], params["b_h"], params["b_o"], beta)
        )
        qubo_energies.append(qubo_energy(problem.Q, problem.constant, spins))

    report = energy_equivalence_report(np.asarray(ep_energies), np.asarray(qubo_energies), atol=1e-10)
    assert report["max_abs_residual"] <= 1e-10
    assert minimizer_indices(np.asarray(ep_energies)) == minimizer_indices(np.asarray(qubo_energies))


def test_auxiliary_spin_decoding_restores_physical_state_for_both_gauges() -> None:
    """Both auxiliary gauges decode to the original physical spins."""
    physical = np.array([1.0, -1.0, -1.0, 1.0])
    for gauge in (-1.0, 1.0):
        solver_state = np.concatenate([physical * gauge, [gauge]])
        np.testing.assert_array_equal(decode_aux(solver_state), physical)
        np.testing.assert_array_equal(decode_aux_alt(solver_state), physical)


def test_split_hidden_output_validates_physical_length() -> None:
    """Splitting checks that the physical vector has the expected length."""
    hidden, output = split_hidden_output(np.array([1.0, -1.0, -1.0]), 1, 2)
    np.testing.assert_array_equal(hidden, [1.0])
    np.testing.assert_array_equal(output, [-1.0, -1.0])

    try:
        split_hidden_output(np.ones(3), 2, 2)
    except ValueError as exc:
        assert "physical spins" in str(exc)
    else:
        raise AssertionError("invalid physical length was accepted")


def pytest_approx(value: float):
    """Return a pytest approximate scalar."""
    return pytest.approx(value)
