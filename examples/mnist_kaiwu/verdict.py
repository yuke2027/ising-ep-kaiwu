"""Strict Primary_Run verdict; any missing necessary condition fails."""

from __future__ import annotations

from typing import Any, Mapping

ACCURACY_TARGET = 0.40


def verdict(
    *,
    latest_smoke_passed: bool,
    method_gate_passed: bool,
    attested: bool,
    aggregation: Mapping[str, Any] | None,
    seed_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the conjunctive Primary_Run verdict."""
    valid = [item for item in seed_records if item.get("valid") and not item.get("used_local_backend")]
    learning_ok = bool(valid) and all(
        bool((item.get("learning_evidence") or {}).get("passed")) for item in valid
    )
    n_valid = len(valid)
    if aggregation is not None:
        n_valid = int(aggregation.get("n_valid") or n_valid)
        mean_accuracy = float((aggregation.get("accuracy") or {}).get("mean") or 0.0)
    else:
        mean_accuracy = 0.0
    conditions = {
        "latest_smoke_passed": bool(latest_smoke_passed),
        "method_gate_passed": bool(method_gate_passed),
        "attested": bool(attested),
        "valid_official_seeds_ge_3": n_valid >= 3,
        "mean_accuracy_ge_target": aggregation is not None and mean_accuracy >= ACCURACY_TARGET,
        "learning_evidence": learning_ok,
        "no_local_backend": all(not item.get("used_local_backend") for item in seed_records),
    }
    success = all(conditions.values())
    return {
        "verdict": "SUCCESS" if success else "FAILED",
        "conditions": conditions,
        "accuracy_target": ACCURACY_TARGET,
        "n_valid": n_valid,
        "mean_accuracy": mean_accuracy,
        "partial_success": False,
    }
