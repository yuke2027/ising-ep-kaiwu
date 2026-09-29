"""Unit tests for the direct Kaiwu boundary using a controlled fake SDK."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from examples.mnist_kaiwu import solve
from examples.mnist_kaiwu.solve import PairedSolverSettings
from src.ising_ep.mapping import build_qubo


class FakeIsingModel:
    """Minimal Ising model exposing the Kaiwu getters used by solve.py."""

    def __init__(self, variables: dict[str, int], matrix: np.ndarray, bias: float = 0.0) -> None:
        self._variables = variables
        self._matrix = matrix
        self._bias = bias

    def get_matrix(self) -> np.ndarray:
        """Return the stored Ising matrix."""
        return self._matrix

    def get_variables(self) -> dict[str, int]:
        """Return the stored variable index map."""
        return self._variables

    def get_bias(self) -> float:
        """Return the stored Ising bias."""
        return self._bias


class FakeCore:
    """Minimal ``kaiwu.core`` stand-in for conversion and solution lookup."""

    def __init__(self, variables: dict[str, int]) -> None:
        self.variables = variables

    def qubo_matrix_to_qubo_model(self, matrix: np.ndarray) -> dict[str, np.ndarray]:
        """Store the QUBO matrix without calling Kaiwu."""
        return {"matrix": np.asarray(matrix, dtype=np.float64)}

    def qubo_model_to_ising_model(self, qubo_model: dict[str, np.ndarray]) -> FakeIsingModel:
        """Return an identity Ising model with the configured variables."""
        size = len(self.variables)
        return FakeIsingModel(self.variables, np.eye(size), 0.25)

    def get_sol_dict(self, solution: np.ndarray, vars_dict: dict[str, int]) -> dict[str, float]:
        """Map each variable name to the value at its index."""
        return {name: float(solution[index]) for name, index in vars_dict.items()}


class FakeOptimizer:
    """Records annealing calls and returns a configured candidate array."""

    calls: list[dict[str, Any]] = []
    rows = np.ones((1, 3))

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def single_process_solve(self, **kwargs: Any) -> np.ndarray:
        """Record the call and return the configured rows."""
        self.calls.append({"optimizer": self.kwargs, "solve": kwargs})
        return self.rows


def fake_kaiwu(variables: dict[str, int]) -> SimpleNamespace:
    """Build a fake ``kaiwu`` module with core and classical namespaces."""
    return SimpleNamespace(
        core=FakeCore(variables),
        classical=SimpleNamespace(SimulatedAnnealingOptimizer=FakeOptimizer),
    )


def settings() -> PairedSolverSettings:
    """Return a tiny valid paired annealing configuration."""
    return PairedSolverSettings(
        initial_temperature=1.0,
        alpha=0.9,
        cutoff_temperature=0.01,
        iterations_per_t=2,
        candidate_count=2,
        process_num=1,
    )


def test_kaiwu_problem_builds_and_orders_numeric_variables(monkeypatch) -> None:
    """Physical variables are ordered by index and the auxiliary spin is separate."""
    variables = {f"b[{index}]": index for index in range(12)}
    variables["__spin__"] = 12
    monkeypatch.setattr(solve, "_import_kaiwu", lambda: fake_kaiwu(variables))
    problem = solve.build_kaiwu_problem(
        build_qubo(torch.zeros((6, 6)), torch.zeros(6), torch.zeros(6), 0.0)
    )

    assert problem.ising_matrix.shape == (13, 13)
    assert problem.ising_bias == pytest.approx(0.25)
    assert problem.physical_variables == tuple(f"b[{index}]" for index in range(12))
    assert problem.auxiliary_variable == "__spin__"


def test_solve_normalizes_candidates_and_selects_by_original_energy(monkeypatch) -> None:
    """Candidate ranking uses the original EP energy, not solver order."""
    variables = {"b[0]": 0, "b[1]": 1, "__spin__": 2}
    FakeOptimizer.calls = []
    FakeOptimizer.rows = np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]])
    monkeypatch.setattr(solve, "_import_kaiwu", lambda: fake_kaiwu(variables))
    problem = solve.build_kaiwu_problem(
        build_qubo(torch.ones((1, 1)), torch.ones(1), torch.ones(1), 0.0)
    )

    candidate, audit = solve.solve_kaiwu_problem(
        problem,
        phase="free",
        pair_seed=17,
        settings=settings(),
        energy_fn=lambda physical: float(-physical[0] + physical[1]),
        run_id="unit",
        sample_index=3,
    )

    assert candidate.source_index == 1
    assert candidate.physical.tolist() == [1.0, -1.0]
    assert candidate.solver_state.tolist() == [-1.0, 1.0, -1.0]
    assert audit.candidate_count == 2
    assert audit.requested_candidate_count == 2
    assert audit.optimizer_kwargs["rand_seed"] == 17
    assert FakeOptimizer.calls[-1]["solve"]["rand_seed"] == 17
    assert "init_solution" not in FakeOptimizer.calls[-1]["solve"]


def test_solve_passes_complete_solver_state_for_warm_start(monkeypatch) -> None:
    """A nudged solve passes the full free solver state as ``init_solution``."""
    variables = {"b[0]": 0, "b[1]": 1, "__spin__": 2}
    FakeOptimizer.calls = []
    FakeOptimizer.rows = np.array([[1.0, -1.0, -1.0]])
    monkeypatch.setattr(solve, "_import_kaiwu", lambda: fake_kaiwu(variables))
    problem = solve.build_kaiwu_problem(
        build_qubo(torch.ones((1, 1)), torch.ones(1), torch.ones(1), 0.0)
    )
    warm_start = np.array([-1.0, 1.0, -1.0])

    solve.solve_kaiwu_problem(
        problem,
        phase="nudged",
        pair_seed=19,
        settings=settings(),
        energy_fn=lambda physical: float(physical.sum()),
        initial_solver_state=warm_start,
    )

    np.testing.assert_array_equal(FakeOptimizer.calls[-1]["solve"]["init_solution"], warm_start)
    assert FakeOptimizer.calls[-1]["solve"]["rand_seed"] == 19


def test_solve_decodes_binary_candidates_with_auxiliary_first(monkeypatch) -> None:
    """Binary solver values decode through the auxiliary spin to physical spins."""
    variables = {"__spin__": 0, "b[0]": 1, "b[1]": 2}
    FakeOptimizer.calls = []
    FakeOptimizer.rows = np.array([[0.0, 1.0, 0.0]])
    monkeypatch.setattr(solve, "_import_kaiwu", lambda: fake_kaiwu(variables))
    problem = solve.build_kaiwu_problem(
        build_qubo(torch.ones((1, 1)), torch.ones(1), torch.ones(1), 0.0)
    )

    candidate, _ = solve.solve_kaiwu_problem(
        problem,
        phase="evaluation",
        pair_seed=23,
        settings=settings(),
        energy_fn=lambda physical: float(physical.sum()),
    )

    assert problem.physical_variables == ("b[0]", "b[1]")
    assert problem.auxiliary_variable == "__spin__"
    assert candidate.solver_state.tolist() == [-1.0, 1.0, -1.0]
    assert candidate.physical.tolist() == [-1.0, 1.0]


def test_solve_rejects_bad_return_shape(monkeypatch) -> None:
    """A solver return whose width does not match the variables is rejected."""
    variables = {"b[0]": 0, "b[1]": 1, "__spin__": 2}
    FakeOptimizer.calls = []
    FakeOptimizer.rows = np.zeros((1, 1, 3))
    monkeypatch.setattr(solve, "_import_kaiwu", lambda: fake_kaiwu(variables))
    problem = solve.build_kaiwu_problem(
        build_qubo(torch.ones((1, 1)), torch.ones(1), torch.ones(1), 0.0)
    )

    with pytest.raises(Exception, match="2-D candidate"):
        solve.solve_kaiwu_problem(
            problem,
            phase="free",
            pair_seed=1,
            settings=settings(),
            energy_fn=lambda physical: 0.0,
        )
