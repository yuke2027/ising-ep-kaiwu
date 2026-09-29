"""Two-phase EP training driven directly by official Kaiwu solves."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.ising_ep.mapping import build_qubo, split_hidden_output
from src.ising_ep.model import IsingEPModel

from .artifacts import append_jsonl
from .failures import PAIRING_VIOLATION, AlignmentError
from .metrics import pair_dynamics
from .solve import (
    PairedSolverSettings,
    SolveAudit,
    build_kaiwu_problem,
    solve_kaiwu_problem,
)


@dataclass(frozen=True)
class PhasePair:
    """Selected free and nudged states for one training sample."""

    h_free: torch.Tensor
    o_free: torch.Tensor
    h_nudged: torch.Tensor
    o_nudged: torch.Tensor
    free_audit: SolveAudit
    nudged_audit: SolveAudit
    free_solver_state: np.ndarray
    nudged_solver_state: np.ndarray


def _phase_energy(model: IsingEPModel, x: Any, target: Any, beta: float):
    """Return an original-energy function for one decoded physical state."""

    def energy_fn(physical: torch.Tensor) -> float:
        hidden, output = split_hidden_output(physical, model.n_hidden, model.n_output)
        value = model.free_energy(x, hidden, output, target, beta)
        return float(value.detach().cpu().item())

    return energy_fn


def solve_phases(
    model: IsingEPModel,
    x: Any,
    target: Any,
    beta: float,
    pair_seed: int,
    settings: PairedSolverSettings,
    *,
    run_id: str,
    sample_index: int,
) -> PhasePair:
    """Solve paired free and warm-started nudged phases for one sample."""
    beta_value = float(beta)
    if not math.isfinite(beta_value) or beta_value <= 0:
        raise AlignmentError(PAIRING_VIOLATION, "nudged phase requires a strictly positive beta")
    settings.validate()
    with torch.no_grad():
        hidden_field, free_output_field = model.fields(x, target, 0.0)
        _, nudged_output_field = model.fields(x, target, beta_value)
        free_problem = build_kaiwu_problem(
            build_qubo(model.V, hidden_field, free_output_field, 0.0)
        )
        nudged_problem = build_kaiwu_problem(
            build_qubo(model.V, hidden_field, nudged_output_field, beta_value)
        )

    free_candidate, free_audit = solve_kaiwu_problem(
        free_problem,
        phase="free",
        pair_seed=pair_seed,
        settings=settings,
        energy_fn=_phase_energy(model, x, target, 0.0),
        run_id=run_id,
        sample_index=sample_index,
    )
    nudged_candidate, nudged_audit = solve_kaiwu_problem(
        nudged_problem,
        phase="nudged",
        pair_seed=pair_seed,
        settings=settings,
        energy_fn=_phase_energy(model, x, target, beta_value),
        initial_solver_state=free_candidate.solver_state,
        run_id=run_id,
        sample_index=sample_index,
    )
    if free_audit.optimizer_kwargs != nudged_audit.optimizer_kwargs:
        raise AlignmentError(PAIRING_VIOLATION, "free/nudged optimizer kwargs drifted")

    h_free, o_free = split_hidden_output(
        free_candidate.physical, model.n_hidden, model.n_output
    )
    h_nudged, o_nudged = split_hidden_output(
        nudged_candidate.physical, model.n_hidden, model.n_output
    )
    return PhasePair(
        h_free=h_free.detach(),
        o_free=o_free.detach(),
        h_nudged=h_nudged.detach(),
        o_nudged=o_nudged.detach(),
        free_audit=free_audit,
        nudged_audit=nudged_audit,
        free_solver_state=free_candidate.solver_state.copy(),
        nudged_solver_state=nudged_candidate.solver_state.copy(),
    )


def _assert_parameters_unchanged(
    model: IsingEPModel,
    snapshot: dict[str, torch.Tensor],
) -> None:
    for name, before in snapshot.items():
        current = model.state_dict()[name]
        if not torch.equal(current.detach(), before):
            raise AlignmentError(
                PAIRING_VIOLATION,
                "parameters changed during solver calls before batch update",
                evidence={"parameter": name},
            )


def train_epoch(
    model: IsingEPModel,
    optimizer: torch.optim.Optimizer,
    images: Any,
    labels: Any,
    order: Any,
    *,
    beta: float,
    batch_size: int,
    pair_seed_fn: Callable[[int], int],
    settings: PairedSolverSettings,
    run_id: str,
    seed: int,
    epoch: int,
    ledger_path: str | Path,
) -> dict[str, float]:
    """Train one epoch from a shared parameter snapshot per batch.

    One parameter snapshot is used for the whole batch. After every sample in
    that batch has been solved, one optimizer step is taken on the mean EP loss
    ``(E_nudged - E_free) / beta``.

    Args:
        model: EP network whose parameters receive the optimizer step.
        optimizer: Torch optimizer, normally ``torch.optim.SGD``.
        images: Training images aligned with ``labels``.
        labels: Integer class labels.
        order: Sample indices visited in this epoch.
        beta: Positive nudging coefficient.
        batch_size: Number of samples sharing one parameter snapshot.
        pair_seed_fn: Maps a sample index to the paired free/nudged seed.
        settings: Annealing settings shared by both phases.
        run_id: Run identifier written into the ledger.
        seed: Training seed written into the ledger.
        epoch: Epoch index written into the ledger.
        ledger_path: JSONL path for free and nudged solve audits.

    Returns:
        Batch-averaged flip statistics and the number of optimizer steps.

    Raises:
        AlignmentError: If the split, ``beta`` or an EP gradient is invalid.
    """
    indices = np.asarray(order, dtype=np.int64).reshape(-1)
    if indices.size < 1:
        raise AlignmentError(PAIRING_VIOLATION, "training split is empty")
    if int(batch_size) < 1:
        raise AlignmentError(PAIRING_VIOLATION, "batch_size must be positive")
    beta_value = float(beta)
    if not math.isfinite(beta_value) or beta_value <= 0:
        raise AlignmentError(PAIRING_VIOLATION, "beta must be positive")

    hidden_flip = 0.0
    output_flip = 0.0
    zero_update = 0.0
    hidden_zero = 0.0
    updates = 0
    images_array = np.asarray(images, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)

    start = 0
    while start < indices.size:
        batch = indices[start : start + int(batch_size)]
        start += int(batch.size)
        snapshot = model.parameter_snapshot()
        pairs: list[PhasePair] = []
        losses: list[torch.Tensor] = []
        xs: list[torch.Tensor] = []
        for sample_index in batch.tolist():
            x = torch.as_tensor(
                images_array[int(sample_index)].reshape(-1),
                dtype=model.dtype,
                device=model.device,
            )
            target = model.target_spins([int(labels_array[int(sample_index)])])[0]
            pair = solve_phases(
                model,
                x,
                target,
                beta_value,
                int(pair_seed_fn(int(sample_index))),
                settings,
                run_id=run_id,
                sample_index=int(sample_index),
            )
            pairs.append(pair)
            xs.append(x)
            losses.append(
                (
                    model.energy(x, pair.h_nudged, pair.o_nudged)
                    - model.energy(x, pair.h_free, pair.o_free)
                )
                / beta_value
            )
            dynamics = pair_dynamics(
                pair.h_free,
                pair.o_free,
                pair.h_nudged,
                pair.o_nudged,
            )
            hidden_flip += dynamics["hidden_flip"]
            output_flip += dynamics["output_flip"]
            zero_update += dynamics["zero_update"]
            hidden_zero += dynamics["hidden_zero_update"]
            append_jsonl(
                ledger_path,
                {**pair.free_audit.to_json(), "seed": seed, "epoch": epoch, "split": "train"},
            )
            append_jsonl(
                ledger_path,
                {**pair.nudged_audit.to_json(), "seed": seed, "epoch": epoch, "split": "train"},
            )

        _assert_parameters_unchanged(model, snapshot)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise AlignmentError(PAIRING_VIOLATION, "EP gradient is not finite")
        optimizer.step()
        updates += 1

    inverse = 1.0 / float(indices.size)
    return {
        "hidden_flip": hidden_flip * inverse,
        "output_flip": output_flip * inverse,
        "zero_update": zero_update * inverse,
        "hidden_zero_update": hidden_zero * inverse,
        "skip_nudge": 0.0,
        "n_samples": float(indices.size),
        "batch_updates": float(updates),
    }
