"""Direct Kaiwu 1.4.1 QUBO/Ising conversion and simulated-annealing solves."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Callable, Literal, Mapping
import uuid

import numpy as np
import torch

from src.ising_ep.mapping import QuboProblem, decode_aux

from .artifacts import canonical_sha256
from .failures import (
    INVALID_CANDIDATE,
    SDK_IMPORT_FAILED,
    SDK_INSTALL_BLOCKED,
    AlignmentError,
)

ENERGY_TOLERANCE = 1e-8
Phase = Literal["free", "nudged", "evaluation"]
EnergyFn = Callable[[torch.Tensor], float]


@dataclass(frozen=True)
class PairedSolverSettings:
    """Paired free/nudged simulated-annealing settings."""

    initial_temperature: float
    alpha: float
    cutoff_temperature: float
    iterations_per_t: int
    candidate_count: int
    process_num: int = 1

    def validate(self) -> None:
        """Validate positive numeric settings and the fixed process count."""
        if self.candidate_count < 1:
            raise AlignmentError(INVALID_CANDIDATE, "candidate_count must be positive")
        if self.process_num != 1:
            raise AlignmentError(
                INVALID_CANDIDATE,
                "paired EP solves are restricted to process_num=1",
                evidence={"process_num": self.process_num},
            )
        for name in (
            "initial_temperature",
            "alpha",
            "cutoff_temperature",
            "iterations_per_t",
        ):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise AlignmentError(INVALID_CANDIDATE, f"{name} must be a positive finite value")


@dataclass(frozen=True)
class KaiwuProblem:
    """Official Kaiwu model objects and matrix for one QUBO problem."""

    qubo: QuboProblem
    qubo_model: Any
    ising_model: Any
    ising_matrix: np.ndarray
    ising_bias: float
    variables: tuple[str, ...]
    physical_variables: tuple[str, ...]
    auxiliary_variable: str | None


@dataclass(frozen=True)
class DecodedCandidate:
    """One solver candidate and its recovered physical state."""

    physical: torch.Tensor
    solver_state: np.ndarray
    original_energy: float
    source_index: int
    content_sha256: str

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe candidate summary."""
        return {
            "physical": [int(value) for value in self.physical.detach().cpu().tolist()],
            "solver_state": [int(value) for value in self.solver_state.tolist()],
            "original_energy": self.original_energy,
            "source_index": self.source_index,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class SolveAudit:
    """Non-secret audit record for one official solve call."""

    call_id: str
    run_id: str
    sample_index: int
    phase: str
    request_digest: str
    candidate_count: int
    selected_candidate_digest: str
    selected_solver_state_sha256: str
    elapsed_s: float
    optimizer_kwargs: dict[str, Any]
    solve_parameter_names: tuple[str, ...]
    requested_candidate_count: int
    variables: tuple[str, ...]
    auxiliary_variable: str | None
    used_local_backend: bool = False

    def to_json(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible audit payload."""
        return {
            "call_id": self.call_id,
            "run_id": self.run_id,
            "sample_index": self.sample_index,
            "phase": self.phase,
            "request_digest": self.request_digest,
            "candidate_count": self.candidate_count,
            "selected_candidate_digest": self.selected_candidate_digest,
            "selected_solver_state_sha256": self.selected_solver_state_sha256,
            "elapsed_s": self.elapsed_s,
            "optimizer_kwargs": dict(self.optimizer_kwargs),
            "solve_parameter_names": list(self.solve_parameter_names),
            "requested_candidate_count": self.requested_candidate_count,
            "variables": list(self.variables),
            "auxiliary_variable": self.auxiliary_variable,
            "used_local_backend": self.used_local_backend,
        }


def _import_kaiwu() -> Any:
    try:
        import kaiwu as kw
    except ModuleNotFoundError as exc:
        raise AlignmentError(
            SDK_INSTALL_BLOCKED,
            "Kaiwu SDK is not installed in the project virtual environment",
            evidence={"import": "kaiwu"},
        ) from exc
    except ImportError as exc:
        raise AlignmentError(
            SDK_IMPORT_FAILED,
            "Kaiwu SDK import failed",
            evidence={"import": "kaiwu", "error": type(exc).__name__},
        ) from exc
    return kw


def _matrix_float64(matrix: Any, name: str) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise AlignmentError(INVALID_CANDIDATE, f"{name} must be a square matrix")
    if not np.isfinite(array).all():
        raise AlignmentError(INVALID_CANDIDATE, f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _variable_names(raw_variables: Any) -> tuple[str, ...]:
    if isinstance(raw_variables, Mapping):
        items = list(raw_variables.items())
        if all(isinstance(value, (int, np.integer)) for _, value in items):
            items.sort(key=lambda item: int(item[1]))
            return tuple(str(name) for name, _ in items)
        return tuple(str(name) for name, _ in items)
    return tuple(str(name) for name in raw_variables)


def _physical_variable_index(name: str) -> int | None:
    digits = "".join(character for character in name if character.isdigit())
    if not digits:
        return None
    return int(digits)


def _problem_layout(
    variables: tuple[str, ...],
    n_hidden: int,
    n_output: int,
) -> tuple[tuple[str, ...], str | None]:
    expected = int(n_hidden) + int(n_output)
    auxiliary = [name for name in variables if _physical_variable_index(name) is None]
    if len(auxiliary) > 1:
        raise AlignmentError(INVALID_CANDIDATE, "Kaiwu model has multiple auxiliary variables")
    physical_names = [name for name in variables if _physical_variable_index(name) is not None]
    indexed = sorted(
        ((_physical_variable_index(name), name) for name in physical_names),
        key=lambda item: (item[0], item[1]),
    )
    indices = [index for index, _ in indexed]
    if len(indexed) != expected or indices != list(range(expected)):
        raise AlignmentError(
            INVALID_CANDIDATE,
            "Kaiwu physical variables do not form a 0..N-1 ordered set",
            evidence={"variables": list(variables), "expected": expected},
        )
    return tuple(name for _, name in indexed), (auxiliary[0] if auxiliary else None)


def build_kaiwu_problem(problem: QuboProblem) -> KaiwuProblem:
    """Convert a Torch QUBO into official Kaiwu QUBO and Ising models.

    Paired solves of the returned problem share one ``pair_seed``. The nudged
    phase warm-starts from the free phase through ``init_solution``. Candidates
    are ranked by the original EP energy rather than the Kaiwu objective.

    Args:
        problem: Upper-triangular QUBO whose variables are hidden, then output.

    Returns:
        The Kaiwu models, Ising matrix, bias and variable order.

    Raises:
        AlignmentError: If Kaiwu is unavailable or the conversion API is missing.
    """
    kw = _import_kaiwu()
    matrix = _matrix_float64(problem.Q, "QUBO")
    try:
        qubo_model = kw.core.qubo_matrix_to_qubo_model(matrix)
        ising_model = kw.core.qubo_model_to_ising_model(qubo_model)
    except AttributeError as exc:
        raise AlignmentError(
            SDK_IMPORT_FAILED,
            "installed Kaiwu does not expose the documented QUBO/Ising conversion API",
            evidence={"error": type(exc).__name__},
        ) from exc
    ising_matrix = _matrix_float64(ising_model.get_matrix(), "Kaiwu Ising matrix")
    variables = _variable_names(ising_model.get_variables())
    physical, auxiliary = _problem_layout(variables, problem.n_hidden, problem.n_output)
    bias = float(ising_model.get_bias())
    if not math.isfinite(bias):
        raise AlignmentError(INVALID_CANDIDATE, "Kaiwu Ising bias is not finite")
    return KaiwuProblem(
        qubo=problem,
        qubo_model=qubo_model,
        ising_model=ising_model,
        ising_matrix=ising_matrix,
        ising_bias=bias,
        variables=variables,
        physical_variables=physical,
        auxiliary_variable=auxiliary,
    )


def _spin_value(value: Any, name: str) -> float:
    scalar = float(np.asarray(value).reshape(-1)[0])
    if scalar in (-1.0, 1.0):
        return scalar
    if scalar in (0.0, 1.0):
        return 2.0 * scalar - 1.0
    raise AlignmentError(
        INVALID_CANDIDATE,
        "Kaiwu solution value is neither a ±1 spin nor a 0/1 bit",
        evidence={"variable": name, "value": scalar},
    )


def _normalize_raw_return(raw: Any, expected_variables: int) -> np.ndarray:
    array = np.asarray(raw, dtype=np.float64)
    if array.size == 0:
        raise AlignmentError(INVALID_CANDIDATE, "official solver returned no candidates")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise AlignmentError(
            INVALID_CANDIDATE,
            "official solver return must be one solution or a 2-D candidate array",
            evidence={"shape": list(array.shape)},
        )
    if array.shape[1] != expected_variables:
        raise AlignmentError(
            INVALID_CANDIDATE,
            "official solver candidate length does not match Kaiwu variables",
            evidence={"expected": expected_variables, "actual": int(array.shape[1])},
        )
    if not np.isfinite(array).all():
        raise AlignmentError(INVALID_CANDIDATE, "official solver returned non-finite values")
    normalized = np.where(array == 0.0, -1.0, array)
    if not np.all((normalized == 1.0) | (normalized == -1.0)):
        raise AlignmentError(INVALID_CANDIDATE, "official solver candidates must be ±1 or 0/1")
    return normalized


def _decode_solution(
    solution: np.ndarray,
    problem: KaiwuProblem,
) -> tuple[torch.Tensor, np.ndarray]:
    kw = _import_kaiwu()
    solution_array = np.asarray(solution, dtype=np.float64).reshape(-1)
    if solution_array.shape[0] != len(problem.variables):
        raise AlignmentError(INVALID_CANDIDATE, "solution length does not match Kaiwu variables")
    variables_map = problem.ising_model.get_variables()
    try:
        sol_dict = kw.core.get_sol_dict(
            solution_array,
            variables_map,
        )
    except AttributeError as exc:
        raise AlignmentError(
            SDK_IMPORT_FAILED,
            "installed Kaiwu does not expose kaiwu.core.get_sol_dict",
            evidence={"error": type(exc).__name__},
        ) from exc
    if not isinstance(sol_dict, Mapping):
        raise AlignmentError(INVALID_CANDIDATE, "Kaiwu get_sol_dict did not return a mapping")
    by_name: dict[str, float] = {}
    for name in problem.variables:
        if name in sol_dict:
            by_name[name] = _spin_value(sol_dict[name], name)
            continue
        index = variables_map.get(name) if isinstance(variables_map, Mapping) else None
        if index is None:
            index = problem.variables.index(name)
        if int(index) >= solution_array.shape[0]:
            raise AlignmentError(
                INVALID_CANDIDATE,
                "Kaiwu variable is missing from both sol_dict and variables map",
                evidence={"variable": name},
            )
        by_name[name] = _spin_value(solution_array[int(index)], name)
    physical_solver = np.asarray(
        [by_name[name] for name in problem.physical_variables], dtype=np.float64
    )
    if problem.auxiliary_variable is None:
        physical = physical_solver
    else:
        physical = decode_aux(
            np.concatenate([physical_solver, [by_name[problem.auxiliary_variable]]])
        ).detach().cpu().numpy()
    solver_state = np.asarray(
        [by_name[name] for name in problem.variables], dtype=np.float64
    )
    return torch.as_tensor(physical, dtype=torch.float64), solver_state


def _lex_less(left: np.ndarray, right: np.ndarray) -> bool:
    for x_value, y_value in zip(left.tolist(), right.tolist(), strict=False):
        if x_value < y_value:
            return True
        if x_value > y_value:
            return False
    return len(left) < len(right)


def _candidate_better(left: DecodedCandidate, right: DecodedCandidate) -> bool:
    if left.original_energy < right.original_energy - ENERGY_TOLERANCE:
        return True
    if right.original_energy < left.original_energy - ENERGY_TOLERANCE:
        return False
    left_physical = left.physical.detach().cpu().numpy()
    right_physical = right.physical.detach().cpu().numpy()
    if not np.array_equal(left_physical, right_physical):
        return _lex_less(left_physical, right_physical)
    if not np.array_equal(left.solver_state, right.solver_state):
        return _lex_less(left.solver_state, right.solver_state)
    return left.content_sha256 < right.content_sha256


def _content_sha256(physical: torch.Tensor, solver_state: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(physical.detach().cpu().numpy(), dtype=np.float64).tobytes())
    digest.update(b"|")
    digest.update(np.ascontiguousarray(solver_state, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _request_digest(
    problem: KaiwuProblem,
    phase: str,
    pair_seed: int,
    settings: PairedSolverSettings,
    initial_solver_state: np.ndarray | None,
) -> str:
    payload = {
        "phase": phase,
        "pair_seed": int(pair_seed),
        "settings": {
            "initial_temperature": settings.initial_temperature,
            "alpha": settings.alpha,
            "cutoff_temperature": settings.cutoff_temperature,
            "iterations_per_t": settings.iterations_per_t,
            "candidate_count": settings.candidate_count,
            "process_num": settings.process_num,
        },
        "ising_sha256": hashlib.sha256(problem.ising_matrix.tobytes()).hexdigest(),
        "initial_solver_sha256": (
            hashlib.sha256(initial_solver_state.tobytes()).hexdigest()
            if initial_solver_state is not None
            else None
        ),
        "variables": list(problem.variables),
        "ising_bias": problem.ising_bias,
        "qubo_constant": problem.qubo.constant,
    }
    return canonical_sha256(payload)


def solve_kaiwu_problem(
    problem: KaiwuProblem,
    *,
    phase: Phase,
    pair_seed: int,
    settings: PairedSolverSettings,
    energy_fn: EnergyFn,
    initial_solver_state: np.ndarray | None = None,
    run_id: str = "",
    sample_index: int = -1,
    call_id: str | None = None,
) -> tuple[DecodedCandidate, SolveAudit]:
    """Solve one official Kaiwu Ising problem and rank candidates by EP energy.

    Free and nudged phases share ``pair_seed`` and the same annealing settings.
    A nudged call passes the free phase's complete solver state as
    ``init_solution``. Returned candidates are ranked by ``energy_fn``, the
    original EP energy.

    Args:
        problem: Result of :func:`build_kaiwu_problem`.
        phase: ``free``, ``nudged`` or ``evaluation``.
        pair_seed: Seed shared by a paired free/nudged call.
        settings: Annealing settings used for both phases.
        energy_fn: Original EP energy used to rank returned candidates.
        initial_solver_state: Complete warm-start solver state for nudged calls.
        run_id: Run identifier written to the audit record.
        sample_index: Dataset index written to the audit record.
        call_id: Optional deterministic call identifier.

    Returns:
        The selected candidate and its audit record.

    Raises:
        AlignmentError: If the phase, warm-start or official return is invalid.
    """
    if phase not in {"free", "nudged", "evaluation"}:
        raise AlignmentError(INVALID_CANDIDATE, f"invalid phase {phase!r}")
    settings.validate()
    if initial_solver_state is not None:
        initial_solver_state = np.asarray(initial_solver_state, dtype=np.float64).reshape(-1)
        if initial_solver_state.shape[0] != len(problem.variables):
            raise AlignmentError(INVALID_CANDIDATE, "warm-start length does not match Kaiwu variables")
        if not np.all((initial_solver_state == 1.0) | (initial_solver_state == -1.0)):
            raise AlignmentError(INVALID_CANDIDATE, "warm-start solver state must be ±1")

    kw = _import_kaiwu()
    try:
        optimizer_class = kw.classical.SimulatedAnnealingOptimizer
    except AttributeError as exc:
        raise AlignmentError(
            SDK_IMPORT_FAILED,
            "installed Kaiwu does not expose classical.SimulatedAnnealingOptimizer",
            evidence={"error": type(exc).__name__},
        ) from exc
    optimizer_kwargs = {
        "initial_temperature": settings.initial_temperature,
        "alpha": settings.alpha,
        "cutoff_temperature": settings.cutoff_temperature,
        "iterations_per_t": settings.iterations_per_t,
        "size_limit": settings.candidate_count,
        "rand_seed": int(pair_seed),
        "process_num": settings.process_num,
    }
    optimizer = optimizer_class(**optimizer_kwargs)
    solve_kwargs: dict[str, Any] = {
        "ising_matrix": problem.ising_matrix,
        "rand_seed": int(pair_seed),
    }
    if initial_solver_state is not None:
        solve_kwargs["init_solution"] = initial_solver_state

    started = time.perf_counter()
    raw = optimizer.single_process_solve(**solve_kwargs)
    elapsed = time.perf_counter() - started
    rows = _normalize_raw_return(raw, len(problem.variables))

    best: DecodedCandidate | None = None
    for index, row in enumerate(rows):
        physical, solver_state = _decode_solution(row, problem)
        with torch.no_grad():
            energy_value = energy_fn(physical)
        if isinstance(energy_value, torch.Tensor):
            energy = float(energy_value.detach().cpu().item())
        else:
            energy = float(energy_value)
        if not math.isfinite(energy):
            raise AlignmentError(
                INVALID_CANDIDATE,
                "original EP energy is not finite",
                evidence={"source_index": index},
            )
        candidate = DecodedCandidate(
            physical=physical,
            solver_state=solver_state,
            original_energy=energy,
            source_index=index,
            content_sha256=_content_sha256(physical, solver_state),
        )
        if best is None or _candidate_better(candidate, best):
            best = candidate
    if best is None:
        raise AlignmentError(INVALID_CANDIDATE, "no official candidate could be decoded")

    audit = SolveAudit(
        call_id=call_id or uuid.uuid4().hex,
        run_id=run_id,
        sample_index=sample_index,
        phase=phase,
        request_digest=_request_digest(problem, phase, pair_seed, settings, initial_solver_state),
        candidate_count=int(rows.shape[0]),
        selected_candidate_digest=best.content_sha256,
        selected_solver_state_sha256=hashlib.sha256(best.solver_state.tobytes()).hexdigest(),
        elapsed_s=elapsed,
        optimizer_kwargs=optimizer_kwargs,
        solve_parameter_names=tuple(solve_kwargs.keys()),
        requested_candidate_count=settings.candidate_count,
        variables=problem.variables,
        auxiliary_variable=problem.auxiliary_variable,
    )
    return best, audit
