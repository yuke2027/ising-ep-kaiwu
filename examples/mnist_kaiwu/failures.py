"""Stable experiment failure codes and structured error records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


SDK_INSTALL_BLOCKED = "SDK_INSTALL_BLOCKED"
SDK_IMPORT_FAILED = "SDK_IMPORT_FAILED"
SDK_LICENSE_INVALID = "SDK_LICENSE_INVALID"
SDK_SMOKE_FAILED = "SDK_SMOKE_FAILED"
BACKEND_ATTESTATION_FAILED = "BACKEND_ATTESTATION_FAILED"
INVALID_CANDIDATE = "INVALID_CANDIDATE"
PAIRING_VIOLATION = "PAIRING_VIOLATION"
METHOD_GATE_FAILED = "METHOD_GATE_FAILED"
DATASET_DIGEST_MISMATCH = "DATASET_DIGEST_MISMATCH"
CALL_COUNT_MISMATCH = "CALL_COUNT_MISMATCH"
CHECKPOINT_LEAKAGE = "CHECKPOINT_LEAKAGE"
ARTIFACT_INTEGRITY_ERROR = "ARTIFACT_INTEGRITY_ERROR"
INSUFFICIENT_VALID_SEEDS = "INSUFFICIENT_VALID_SEEDS"

FAILURE_CODES = (
    SDK_INSTALL_BLOCKED,
    SDK_IMPORT_FAILED,
    SDK_LICENSE_INVALID,
    SDK_SMOKE_FAILED,
    BACKEND_ATTESTATION_FAILED,
    INVALID_CANDIDATE,
    PAIRING_VIOLATION,
    METHOD_GATE_FAILED,
    DATASET_DIGEST_MISMATCH,
    CALL_COUNT_MISMATCH,
    CHECKPOINT_LEAKAGE,
    ARTIFACT_INTEGRITY_ERROR,
    INSUFFICIENT_VALID_SEEDS,
)


class AlignmentError(Exception):
    """Typed failure used by official-path gates; never a signal to fall back."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        evidence: Mapping[str, Any] | None = None,
        evidence_id: str | None = None,
    ) -> None:
        """Create a typed failure with optional non-secret evidence."""
        if code not in FAILURE_CODES:
            raise ValueError(f"unknown failure code: {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.evidence = dict(evidence or {})
        self.evidence_id = evidence_id

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without credentials."""
        return {
            "failure_code": self.code,
            "message": self.message,
            "evidence_id": self.evidence_id,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class FailureRecord:
    """Serializable failure record written to run artifacts."""

    failure_code: str
    message: str
    blocked_stage: str
    evidence_id: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    official_call_count: int = 0

    def to_json(self) -> dict[str, Any]:
        """Return the canonical JSON payload for this failure."""
        return {
            "failure_code": self.failure_code,
            "message": self.message,
            "blocked_stage": self.blocked_stage,
            "evidence_id": self.evidence_id,
            "evidence": dict(self.evidence),
            "official_call_count": self.official_call_count,
            "used_local_backend": False,
        }


__all__ = [
    "ARTIFACT_INTEGRITY_ERROR",
    "AlignmentError",
    "BACKEND_ATTESTATION_FAILED",
    "CALL_COUNT_MISMATCH",
    "CHECKPOINT_LEAKAGE",
    "DATASET_DIGEST_MISMATCH",
    "FAILURE_CODES",
    "FailureRecord",
    "INSUFFICIENT_VALID_SEEDS",
    "INVALID_CANDIDATE",
    "METHOD_GATE_FAILED",
    "PAIRING_VIOLATION",
    "SDK_IMPORT_FAILED",
    "SDK_INSTALL_BLOCKED",
    "SDK_LICENSE_INVALID",
    "SDK_SMOKE_FAILED",
]
