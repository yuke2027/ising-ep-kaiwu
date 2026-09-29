"""Free-phase-only evaluation on a frozen checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.ising_ep.mapping import build_qubo, split_hidden_output
from src.ising_ep.model import IsingEPModel

from .artifacts import append_jsonl, atomic_write_json
from .failures import CHECKPOINT_LEAKAGE, AlignmentError
from .metrics import classification_metrics
from .solve import PairedSolverSettings, build_kaiwu_problem, solve_kaiwu_problem


def _evaluation_energy(model: IsingEPModel, x: Any):
    """Return an unclamped free-phase energy function for evaluation."""

    def energy_fn(physical: torch.Tensor) -> float:
        hidden, output = split_hidden_output(physical, model.n_hidden, model.n_output)
        value = model.energy(x, hidden, output)
        return float(value.detach().cpu().item())

    return energy_fn


def evaluate_free_only(
    model: IsingEPModel,
    images: Any,
    labels: Any,
    settings: PairedSolverSettings,
    *,
    seed: int,
    run_id: str,
    seed_dir: Path,
    ledger_path: Path,
    eval_seed: int,
) -> dict[str, Any]:
    """Evaluate one free solve per test sample after checkpoint freeze."""
    if not (Path(seed_dir) / "checkpoint-decision.json").is_file():
        raise AlignmentError(CHECKPOINT_LEAKAGE, "test evaluation started before checkpoint freeze")
    settings.validate()
    images_array = np.asarray(images, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions: list[int] = []
    ties: list[bool] = []
    records: list[dict[str, Any]] = []
    test_dir = Path(seed_dir) / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    for index, (image, label) in enumerate(zip(images_array, labels_array, strict=True)):
        x = torch.as_tensor(image.reshape(-1), dtype=model.dtype, device=model.device)
        with torch.no_grad():
            dummy_target = torch.ones(model.n_output, dtype=model.dtype, device=model.device)
            hidden_field, output_field = model.fields(x, dummy_target, 0.0)
            problem = build_kaiwu_problem(
                build_qubo(model.V, hidden_field, output_field, 0.0)
            )
        selected, audit = solve_kaiwu_problem(
            problem,
            phase="evaluation",
            pair_seed=int(eval_seed + index),
            settings=settings,
            energy_fn=_evaluation_energy(model, x),
            run_id=run_id,
            sample_index=index,
        )
        if audit.phase != "evaluation":
            raise AlignmentError(CHECKPOINT_LEAKAGE, "test phase must be evaluation")
        hidden, output = split_hidden_output(
            selected.physical, model.n_hidden, model.n_output
        )
        prediction, tied = model.predict(hidden, output)
        predictions.append(int(prediction))
        ties.append(bool(tied))
        records.append(
            {
                "index": index,
                "label": int(label),
                "prediction": int(prediction),
                "tied": bool(tied),
                "call_id": audit.call_id,
            }
        )
        append_jsonl(
            ledger_path,
            {**audit.to_json(), "seed": seed, "epoch": -1, "split": "test"},
        )

    metrics = classification_metrics(
        labels_array,
        predictions,
        ties,
        n_classes=model.n_classes,
    )
    metrics["official_calls"] = int(images_array.shape[0])
    metrics["nudged_calls"] = 0
    atomic_write_json(test_dir / "predictions.json", {"records": records})
    atomic_write_json(test_dir / "metrics.json", metrics)
    return metrics
