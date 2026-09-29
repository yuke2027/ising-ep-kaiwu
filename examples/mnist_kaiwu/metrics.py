"""Classification, dynamics and learning-evidence metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .failures import ARTIFACT_INTEGRITY_ERROR, AlignmentError

PARAM_CHANGE_THRESHOLD = 1e-3
HIDDEN_UPDATE_THRESHOLD = 0.05
RELATIVE_EPS = 1e-8


def _array(value: Any, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def class_block_scores(output: Any, *, n_classes: int = 10, spins_per_class: int = 4) -> np.ndarray:
    """Return one summed score per output class block."""
    values = _array(output).reshape(-1)
    if values.size != n_classes * spins_per_class:
        raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "output length is not n_classes × spins_per_class")
    return values.reshape(n_classes, spins_per_class).sum(axis=1)


def classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    ties: Sequence[bool],
    *,
    n_classes: int = 10,
) -> dict[str, Any]:
    """Compute accuracy, tie rate, confusion matrix and per-class metrics."""
    y = _array(labels, np.int64).reshape(-1)
    predictions_array = _array(predictions, np.int64).reshape(-1)
    ties_array = _array(ties, bool).reshape(-1)
    if y.size == 0 or y.size != predictions_array.size or y.size != ties_array.size:
        raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "classification arrays must be non-empty and aligned")
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    per_class_correct = np.zeros(n_classes, dtype=np.int64)
    per_class_total = np.zeros(n_classes, dtype=np.int64)
    correct = 0
    for label, prediction in zip(y.tolist(), predictions_array.tolist(), strict=True):
        if label < 0 or label >= n_classes or prediction < 0 or prediction >= n_classes:
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "label/prediction outside class range")
        confusion[label, prediction] += 1
        per_class_total[label] += 1
        if label == prediction:
            correct += 1
            per_class_correct[label] += 1
    total = int(y.size)
    per_class = []
    for class_index in range(n_classes):
        denominator = int(per_class_total[class_index])
        accuracy = float(per_class_correct[class_index] / denominator) if denominator else 0.0
        per_class.append(
            {
                "class": class_index,
                "correct": int(per_class_correct[class_index]),
                "total": denominator,
                "accuracy": accuracy,
            }
        )
    return {
        "n": total,
        "correct": int(correct),
        "accuracy": float(correct / total),
        "tie_count": int(ties_array.sum()),
        "tie_rate": float(ties_array.mean()),
        "confusion": confusion.tolist(),
        "per_class": per_class,
    }


def pair_dynamics(h_free: Any, o_free: Any, h_nudged: Any, o_nudged: Any) -> dict[str, float]:
    """Measure hidden/output flips and zero-update indicators for one pair."""
    h0 = _array(h_free).reshape(-1)
    o0 = _array(o_free).reshape(-1)
    hb = _array(h_nudged).reshape(-1)
    ob = _array(o_nudged).reshape(-1)
    if h0.shape != hb.shape or o0.shape != ob.shape or h0.size == 0 or o0.size == 0:
        raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "paired states must be non-empty and same shape")
    hidden_flip = float(np.mean(h0 != hb))
    output_flip = float(np.mean(o0 != ob))
    hidden_zero = float(np.array_equal(h0, hb))
    zero = float(np.array_equal(h0, hb) and np.array_equal(o0, ob))
    return {
        "hidden_flip": hidden_flip,
        "output_flip": output_flip,
        "zero_update": zero,
        "hidden_zero_update": hidden_zero,
        "skip_nudge": 0.0,
    }


def l2_distance(left: Any, right: Any) -> float:
    """Return the Euclidean distance between two flattened arrays."""
    left_array = _array(left).reshape(-1)
    right_array = _array(right).reshape(-1)
    return float(np.linalg.norm(left_array - right_array))


def relative_change(initial: Any, final: Any, *, epsilon: float = RELATIVE_EPS) -> dict[str, float]:
    """Return absolute and denominator-normalized parameter changes."""
    absolute = l2_distance(initial, final)
    denominator = max(float(np.linalg.norm(_array(initial))), float(epsilon))
    return {"absolute": absolute, "relative": absolute / denominator, "epsilon": float(epsilon)}


def learning_evidence(
    *,
    w_change: float,
    v_change: float,
    hidden_nonzero_rates: Sequence[float],
) -> dict[str, Any]:
    """Evaluate the frozen learning-evidence thresholds."""
    rates = [float(value) for value in hidden_nonzero_rates]
    passed = (
        w_change > PARAM_CHANGE_THRESHOLD
        and v_change > PARAM_CHANGE_THRESHOLD
        and any(rate >= HIDDEN_UPDATE_THRESHOLD for rate in rates)
    )
    return {
        "w_change": float(w_change),
        "v_change": float(v_change),
        "param_threshold": PARAM_CHANGE_THRESHOLD,
        "hidden_nonzero_rates": rates,
        "hidden_update_threshold": HIDDEN_UPDATE_THRESHOLD,
        "passed": bool(passed),
    }


def population_stats(values: Sequence[float]) -> dict[str, float]:
    """Return mean/std/min/max for at least three scalar values."""
    array = _array(list(values)).reshape(-1)
    if array.size < 3:
        raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "scalar aggregation requires at least 3 values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }
