"""Optional integration checks against the installed Kaiwu 1.4.1 SDK."""

from __future__ import annotations

import numpy as np
import pytest
import torch

kaiwu = pytest.importorskip("kaiwu", reason="Kaiwu SDK is not installed")

from examples.mnist_kaiwu.solve import build_kaiwu_problem
from src.ising_ep.mapping import (
    build_qubo,
    decode_aux,
    decode_aux_alt,
    iter_physical_states,
    minimizer_indices,
    qubo_energy,
    split_hidden_output,
)
from src.ising_ep.model import IsingEPModel

pytestmark = pytest.mark.integration


def _problem():
    qubo = build_qubo(
        torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=torch.float64),
        torch.tensor([0.2, -0.1], dtype=torch.float64),
        torch.tensor([-0.3, 0.15], dtype=torch.float64),
        0.25,
    )
    return build_kaiwu_problem(qubo)


def test_qubo_model_and_ising_model_use_documented_types() -> None:
    """Official conversion returns Kaiwu QUBO and Ising model types."""
    problem = _problem()

    assert isinstance(problem.qubo_model, kaiwu.core.QuboModel)
    assert isinstance(problem.ising_model, kaiwu.core.IsingModel)
    assert problem.ising_matrix.shape[0] in {4, 5}
    assert len(problem.physical_variables) == 4


def test_conversion_preserves_objective_up_to_qubo_constant() -> None:
    """Both auxiliary gauges keep the Ising objective at the QUBO value."""
    problem = _problem()
    matrix = problem.ising_matrix
    aux = problem.auxiliary_variable

    for physical in iter_physical_states(4):
        physical_array = physical.numpy()
        if aux is None:
            solver_states = [physical_array]
        else:
            solver_states = [
                np.concatenate([physical_array * aux_value, [aux_value]])
                for aux_value in (-1.0, 1.0)
            ]
        for solver_state in solver_states:
            ising_energy = -float(solver_state @ matrix @ solver_state) + problem.ising_bias
            qubo_without_external_offset = qubo_energy(problem.qubo.Q, 0.0, physical)
            assert ising_energy == pytest.approx(qubo_without_external_offset, abs=1e-8)


def test_get_sol_dict_returns_physical_bits_and_drops_aux() -> None:
    """``get_sol_dict`` returns physical bits and omits the auxiliary spin."""
    problem = _problem()
    solution = np.ones(len(problem.variables))
    decoded = kaiwu.core.get_sol_dict(solution, problem.ising_model.get_variables())

    assert set(decoded) == set(problem.physical_variables)
    assert all(float(value) in (0.0, 1.0) for value in decoded.values())
    assert problem.auxiliary_variable is not None
    assert problem.auxiliary_variable not in decoded
    aux_index = problem.ising_model.get_variables()[problem.auxiliary_variable]
    assert float(solution[int(aux_index)]) in (-1.0, 1.0)


def _solver_state(problem, physical: np.ndarray, gauge: float) -> np.ndarray:
    variables = problem.ising_model.get_variables()
    state = np.empty(problem.ising_matrix.shape[0], dtype=np.float64)
    for offset, name in enumerate(problem.physical_variables):
        state[int(variables[name])] = float(physical[offset]) * gauge
    state[int(variables[problem.auxiliary_variable])] = gauge
    return state


def _ising_objective(state: np.ndarray, problem) -> float:
    return -float(state @ problem.ising_matrix @ state) + float(problem.ising_bias)


def test_kaiwu_ising_minimizers_match_ep_free_energy() -> None:
    """Decoded Kaiwu Ising argmins match EP free-energy argmins for both gauges."""
    model = IsingEPModel(n_in=3, n_hidden=2, n_classes=2, spins_per_class=1, seed=11)
    x = np.array([0.1, 0.7, -0.2])
    target = model.target_spins([1])[0]
    beta = 0.7
    with torch.no_grad():
        h_field, o_field = model.fields(x, target, beta)
    problem = build_kaiwu_problem(build_qubo(model.V, h_field, o_field, beta))
    assert problem.auxiliary_variable is not None

    physical_states = list(iter_physical_states(model.n_hidden + model.n_output))
    ep_energies = []
    ising_records = []
    variables = problem.ising_model.get_variables()
    for spins in physical_states:
        hidden, output = split_hidden_output(spins, model.n_hidden, model.n_output)
        ep_energies.append(float(model.free_energy(x, hidden, output, target, beta).detach()))
        physical = spins.detach().cpu().numpy()
        for gauge in (-1.0, 1.0):
            state = _solver_state(problem, physical, gauge)
            physical_solver = np.array(
                [state[int(variables[name])] for name in problem.physical_variables],
                dtype=np.float64,
            )
            aux = float(state[int(variables[problem.auxiliary_variable])])
            packed = np.concatenate([physical_solver, [aux]])
            np.testing.assert_array_equal(decode_aux(packed).numpy(), physical)
            np.testing.assert_array_equal(decode_aux_alt(packed).numpy(), physical)
            ising_records.append(
                (_ising_objective(state, problem), tuple(int(value) for value in physical.tolist()))
            )

    ep_states = {
        tuple(int(value) for value in physical_states[index].tolist())
        for index in minimizer_indices(ep_energies)
    }
    lowest = min(energy for energy, _state in ising_records)
    ising_states = {state for energy, state in ising_records if abs(energy - lowest) <= 1e-8}
    assert ising_states == ep_states
