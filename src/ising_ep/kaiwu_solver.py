"""QUBO in, ranked spin candidates out. The only module that imports ``kaiwu``.

Kaiwu 1.3.1 converts a QUBO matrix through ``kaiwu.qubo.qubo_matrix_to_qubo_model``
and ``kaiwu.conversion.qubo_model_to_ising_model``. The annealer is
``SimulatedAnnealingOptimizer.single_process_solve``. Ranking and the
negative-tail gauge fix are done here: 1.3.1 has no ``get_sorted_solutions``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Name Kaiwu gives the gauge spin appended by ``qubo_model_to_ising_model``.
AUXILIARY_VARIABLE = "__spin__"


@dataclass(frozen=True)
class AnnealSettings:
    """Simulated-annealing schedule shared by paired free and nudged solves.

    Paired equilibrium-propagation phases must anneal identically, so one
    instance is reused for both phases of a sample; only the seed and the warm
    start differ between them.

    Attributes:
        initial_temperature: Starting temperature of the anneal.
        alpha: Geometric cooling factor applied after each temperature step.
        cutoff_temperature: Temperature at which annealing stops.
        iterations_per_t: Sweeps performed at each temperature.
        candidate_count: Number of candidate solutions to return per solve,
            passed to Kaiwu as ``size_limit``.
    """

    initial_temperature: float = 1.25
    alpha: float = 0.95
    cutoff_temperature: float = 0.001
    iterations_per_t: int = 10
    candidate_count: int = 8


def _import_kaiwu() -> Any:
    """Import the licensed Kaiwu package.

    The import is deferred so unit tests can inject a fake module and so the
    rest of the package remains importable without the enterprise wheel.

    Returns:
        The imported ``kaiwu`` module.
    """
    import kaiwu as kw

    return kw


def init_license(user_id: str, sdk_code: str) -> None:
    """Activate the Kaiwu license for this process.

    Args:
        user_id: Value of ``KAIWU_USER_ID`` from the quantum cloud platform.
        sdk_code: Value of ``KAIWU_SDK_CODE`` from the quantum cloud platform.
    """
    _import_kaiwu().license.init(user_id=user_id, sdk_code=sdk_code)


def _as_float64_matrix(value: Any) -> np.ndarray:
    """Convert a CPU Torch tensor or array-like into a contiguous float64 matrix.

    Args:
        value: QUBO matrix as a Torch tensor or any array-like.

    Returns:
        A C-contiguous ``float64`` NumPy array suitable for the Kaiwu API.
    """
    return np.ascontiguousarray(value, dtype=np.float64)


def _rank_and_fix_gauge(ising_matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Sort candidates by Ising energy and pin the auxiliary spin at ``+1``.

    The 1.3.1 objective is ``H(s) = -s^T J s``. Multiplying a row by its last
    entry is the negative-tail flip: both gauges become the same physical
    state, with the trailing auxiliary spin equal to ``+1``.

    Args:
        ising_matrix: Coupling matrix including the auxiliary spin.
        rows: Candidate spin rows in solver order.

    Returns:
        ``float64`` rows, best first, each trailing entry ``+1``.
    """
    candidates = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    coupling = np.asarray(ising_matrix, dtype=np.float64)
    hamilton = -np.einsum("ij,ij->i", candidates @ coupling, candidates)
    ranked = candidates[np.argsort(hamilton)]
    return ranked * ranked[:, [-1]]


class KaiwuSolver:
    """Solves QUBO problems on the official Kaiwu classical annealer.

    One instance holds the annealing schedule, which is what makes paired
    free/nudged solves comparable: callers vary only the seed and the warm
    start.

    Attributes:
        settings: Annealing schedule applied to every :meth:`solve` call.
    """

    def __init__(self, settings: AnnealSettings | None = None) -> None:
        """Create a solver bound to one annealing schedule.

        Args:
            settings: Annealing schedule. Defaults to :class:`AnnealSettings`.
        """
        self.settings = settings or AnnealSettings()

    def solve(
        self,
        Q: Any,
        *,
        seed: int,
        init_solution: np.ndarray | None = None,
    ) -> np.ndarray:
        """Anneal one QUBO matrix and return candidate spin vectors.

        Args:
            Q: Upper-triangular QUBO matrix of shape ``(n, n)``, as a Torch
                tensor or array-like.
            seed: Random seed for the anneal. Paired free/nudged solves of the
                same sample pass the same value so the two phases stay
                comparable.
            init_solution: Optional warm start in solver order, length
                ``n + 1``. The nudged phase passes the free phase's selected
                candidate row here.

        Returns:
            A ``float64`` array of shape ``(candidate_count, n + 1)``, sorted
            so row 0 has the lowest Ising objective. Entries are ``-1.0`` or
            ``+1.0``, and the trailing auxiliary spin is fixed at ``+1.0``.
            The leading ``n`` entries of a row are the physical spins; the
            whole row is a valid ``init_solution``.

        Raises:
            RuntimeError: If the Ising model Kaiwu returns does not place the
                auxiliary spin last, which would invalidate the gauge fix.
        """
        kw = _import_kaiwu()
        matrix = _as_float64_matrix(Q)

        qubo_model = kw.qubo.qubo_matrix_to_qubo_model(matrix)
        ising_model = kw.conversion.qubo_model_to_ising_model(qubo_model)
        self._require_trailing_auxiliary(ising_model.get_variables(), matrix.shape[0])
        ising_matrix = _as_float64_matrix(ising_model.get_matrix())

        rows = self._anneal(kw, ising_matrix, seed, init_solution)
        return _rank_and_fix_gauge(ising_matrix, rows)

    def _anneal(
        self,
        kw: Any,
        ising_matrix: np.ndarray,
        seed: int,
        init_solution: np.ndarray | None,
    ) -> np.ndarray:
        """Anneal once, retrying when Kaiwu returns an empty pool.

        Args:
            kw: Imported ``kaiwu`` module.
            ising_matrix: Ising coupling matrix, including the auxiliary spin.
            seed: Random seed forwarded to the optimizer and the solve call.
            init_solution: Warm start in solver order, or ``None`` to start cold.

        Returns:
            Candidate rows whose width matches ``ising_matrix``.

        Raises:
            RuntimeError: If a cold start still returns no candidates.
        """
        width = int(ising_matrix.shape[0])
        attempts: list[np.ndarray | None] = [init_solution]
        if init_solution is not None:
            attempts.append(-np.asarray(init_solution, dtype=np.float64).reshape(-1))
            attempts.append(None)
        for attempt in attempts:
            optimizer = kw.classical.SimulatedAnnealingOptimizer(
                initial_temperature=self.settings.initial_temperature,
                alpha=self.settings.alpha,
                cutoff_temperature=self.settings.cutoff_temperature,
                iterations_per_t=self.settings.iterations_per_t,
                size_limit=self.settings.candidate_count,
                rand_seed=int(seed),
                # Paired EP solves must stay reproducible, so never fan out.
                process_num=1,
            )
            start = None if attempt is None else np.ascontiguousarray(attempt, dtype=np.float64).reshape(-1)
            try:
                raw = optimizer.single_process_solve(
                    ising_matrix=ising_matrix,
                    init_solution=start,
                    rand_seed=int(seed),
                )
            except ValueError:
                # Kaiwu 1.3.1 can raise from an empty unique-solution pool
                # instead of returning an empty array, especially on a warm start.
                if start is None:
                    break
                continue
            rows = np.atleast_2d(np.asarray(raw, dtype=np.float64))
            if rows.ndim == 2 and rows.shape[-1] == width and rows.shape[0] > 0:
                return rows
        raise RuntimeError("Kaiwu annealer returned no candidates")

    @staticmethod
    def _require_trailing_auxiliary(variables: dict, n_qubo: int) -> None:
        """Verify the auxiliary spin occupies the final solver position.

        Args:
            variables: Name-to-index map from ``ising_model.get_variables()``.
            n_qubo: Number of QUBO variables before the Ising rewrite.

        Raises:
            RuntimeError: If the map is not ``{b[i]: i, __spin__: n_qubo}``.
        """
        if len(variables) != n_qubo + 1 or variables.get(AUXILIARY_VARIABLE) != n_qubo:
            raise RuntimeError(
                "Kaiwu Ising model must expose exactly one trailing auxiliary "
                f"spin {AUXILIARY_VARIABLE!r} at index {n_qubo}; got {variables!r}"
            )
