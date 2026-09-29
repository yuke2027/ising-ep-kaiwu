"""Aggregate valid official seeds; reference runs never enter these stats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .failures import INSUFFICIENT_VALID_SEEDS, AlignmentError
from .metrics import population_stats

REFERENCE_HISTORY = (
    {"name": "mnist100_s0", "accuracy": 0.35, "kind": "reference"},
    {"name": "paper_sa", "accuracy": 0.15, "kind": "reference", "skip": 1.0, "zero": 1.0, "learned": False},
    {"name": "gate_b", "accuracy": 0.10, "kind": "reference"},
    {"name": "bp_mlp", "accuracy": 0.91, "kind": "reference"},
)


def valid_seed_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return records that are valid, evaluated and did not use a local backend."""
    valid = []
    for item in records:
        if item.get("used_local_backend"):
            continue
        if not item.get("valid"):
            continue
        if item.get("state") != "TEST_EVALUATED":
            continue
        valid.append(dict(item))
    return valid


def aggregate_official_seeds(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics over at least three valid official seeds."""
    valid = valid_seed_records(records)
    if len(valid) < 3:
        raise AlignmentError(
            INSUFFICIENT_VALID_SEEDS,
            "primary aggregation requires at least 3 valid official seeds",
            evidence={"n_valid": len(valid)},
        )
    accuracies = [float(item["accuracy"]) for item in valid]
    ties = [float(item["tie_rate"]) for item in valid]
    per_class_means = []
    for class_index in range(10):
        column = []
        for item in valid:
            rows = item.get("per_class") or []
            match = next((row for row in rows if int(row["class"]) == class_index), None)
            if match is None:
                raise AlignmentError(
                    INSUFFICIENT_VALID_SEEDS,
                    f"missing per-class accuracy for class {class_index}",
                )
            column.append(float(match["accuracy"]))
        per_class_means.append({"class": class_index, **population_stats(column)})
    return {
        "n_valid": len(valid),
        "seeds": [int(item["seed"]) for item in valid],
        "accuracy": population_stats(accuracies),
        "tie_rate": population_stats(ties),
        "per_class": per_class_means,
        "reference_history": [dict(item) for item in REFERENCE_HISTORY],
        "used_local_backend": False,
    }


def load_seed_statuses(output_dir: Path) -> list[dict[str, Any]]:
    """Read seed status files from a completed run directory."""
    root = Path(output_dir) / "seeds"
    records = []
    if not root.is_dir():
        return records
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        status_path = child / "status.json"
        if not status_path.is_file():
            continue
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        payload.setdefault("seed", int(child.name) if child.name.isdigit() else child.name)
        records.append(payload)
    return records
