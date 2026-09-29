"""Optional official Kaiwu solver checks; skipped without SDK/license."""

from __future__ import annotations

import numpy as np
import pytest
import torch

kaiwu = pytest.importorskip("kaiwu", reason="Kaiwu SDK is not installed")

from examples.mnist_kaiwu.environment import initialize_license
from examples.mnist_kaiwu.failures import AlignmentError
from examples.mnist_kaiwu.solve import (
    PairedSolverSettings,
    build_kaiwu_problem,
    solve_kaiwu_problem,
)
from src.ising_ep.mapping import build_qubo

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _license() -> None:
    try:
        licensed = initialize_license()
    except AlignmentError as exc:
        pytest.skip(f"Kaiwu license unavailable: {exc.message}")
    if not licensed:
        pytest.skip("Kaiwu license unavailable")


def _settings() -> PairedSolverSettings:
    return PairedSolverSettings(
        initial_temperature=1.0,
        alpha=0.9,
        cutoff_temperature=0.01,
        iterations_per_t=2,
        candidate_count=2,
        process_num=1,
    )


def _problem():
    return build_kaiwu_problem(
        build_qubo(
            torch.tensor([[0.4]], dtype=torch.float64),
            torch.tensor([0.1], dtype=torch.float64),
            torch.tensor([-0.2], dtype=torch.float64),
            0.0,
        )
    )


def test_single_process_solve_returns_exact_spins() -> None:
    """Official annealing returns ±1 spins and does not use a local backend."""
    candidate, audit = solve_kaiwu_problem(
        _problem(),
        phase="evaluation",
        pair_seed=11,
        settings=_settings(),
        energy_fn=lambda physical: float(physical.sum()),
    )

    assert candidate.physical.shape == (2,)
    assert set(candidate.physical.tolist()) <= {-1.0, 1.0}
    assert set(candidate.solver_state.tolist()) <= {-1.0, 1.0}
    assert audit.requested_candidate_count == 2
    assert audit.used_local_backend is False


def test_fixed_seed_and_warm_start_are_accepted() -> None:
    """A fixed seed repeats, and the nudged phase accepts ``init_solution``."""
    problem = _problem()
    def energy(physical):
        return float(-physical[0] + physical[1])

    first, _ = solve_kaiwu_problem(
        problem,
        phase="free",
        pair_seed=19,
        settings=_settings(),
        energy_fn=energy,
    )
    second, _ = solve_kaiwu_problem(
        problem,
        phase="free",
        pair_seed=19,
        settings=_settings(),
        energy_fn=energy,
    )
    np.testing.assert_array_equal(first.solver_state, second.solver_state)
    np.testing.assert_array_equal(first.physical.numpy(), second.physical.numpy())

    nudged, audit = solve_kaiwu_problem(
        problem,
        phase="nudged",
        pair_seed=19,
        settings=_settings(),
        energy_fn=energy,
        initial_solver_state=first.solver_state,
    )
    assert nudged.solver_state.shape == first.solver_state.shape
    assert "init_solution" in audit.solve_parameter_names
