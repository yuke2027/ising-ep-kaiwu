"""Atomic artifacts, canonical serialization and environment metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping
import uuid

import numpy as np

from .failures import ARTIFACT_INTEGRITY_ERROR, AlignmentError

SCHEMA_VERSION = "1.0"


def canonical_json(value: Any) -> str:
    """Serialize supported values as deterministic compact JSON."""
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of a canonical JSON payload."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    """Return the SHA-256 of exact UTF-8 text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    """Return the current UTC timestamp in canonical seconds form."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dotenv(path: Path | None = None) -> None:
    """Load ``KEY=value`` pairs from a local .env file without overriding env."""
    env_path = path or Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def credential_fingerprint() -> dict[str, Any]:
    """Return credential presence and a non-secret fingerprint."""
    load_dotenv()
    user_id = os.environ.get("KAIWU_USER_ID", "").strip()
    sdk_code = os.environ.get("KAIWU_SDK_CODE", "").strip()
    present = bool(user_id and sdk_code)
    material = f"present={int(present)};uid_len={len(user_id)};code_len={len(sdk_code)}"
    return {
        "license_available": present,
        "license_fingerprint": text_sha256(material)[:12] if present else None,
        "credential_keys_present": {
            "KAIWU_USER_ID": bool(user_id),
            "KAIWU_SDK_CODE": bool(sdk_code),
        },
    }


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON does not permit NaN or infinity")
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def assert_no_secrets(payload: Any) -> None:
    """Reject artifacts that embed credential values."""
    load_dotenv()
    text = payload if isinstance(payload, str) else canonical_json(payload)
    for key in ("KAIWU_USER_ID", "KAIWU_SDK_CODE"):
        value = os.environ.get(key, "").strip()
        if value and value in text:
            raise AlignmentError(
                ARTIFACT_INTEGRITY_ERROR,
                "artifact contains a credential value",
                evidence={"field": key},
            )


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write bytes to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text after a secret check."""
    assert_no_secrets(text)
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write canonical JSON after a secret check."""
    assert_no_secrets(payload)
    atomic_write_text(path, canonical_json(payload) + "\n")


def _artifact_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "NPZ artifacts must contain numeric arrays")
    return np.ascontiguousarray(array)


def atomic_write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """Atomically write numeric NumPy/Tensor arrays to an NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez(tmp, **{key: _artifact_array(value) for key, value in arrays.items()})
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one canonical JSON line after a secret check."""
    assert_no_secrets(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(dict(payload)) + "\n")


@dataclass(frozen=True)
class ArtifactRecord:
    """Indexed artifact metadata."""

    relative_path: str
    media_type: str
    size: int
    sha256: str
    producer_stage: str
    requirement_ids: tuple[str, ...]


class ArtifactIndex:
    """Content-addressed index for a run directory."""

    def __init__(self, root: Path) -> None:
        """Create an empty index for one run directory."""
        self.root = Path(root)
        self.records: dict[str, ArtifactRecord] = {}

    def add(
        self,
        path: Path,
        *,
        producer_stage: str,
        requirement_ids: tuple[str, ...] = (),
        media_type: str = "application/json",
    ) -> ArtifactRecord:
        """Add an existing file below the run root."""
        path = Path(path)
        rel = path.relative_to(self.root).as_posix()
        record = ArtifactRecord(
            relative_path=rel,
            media_type=media_type,
            size=int(path.stat().st_size),
            sha256=sha256_file(path),
            producer_stage=producer_stage,
            requirement_ids=requirement_ids,
        )
        self.records[rel] = record
        return record

    def to_json(self) -> dict[str, Any]:
        """Return the index payload."""
        return {
            "created_at_utc": utc_now(),
            "root": str(self.root),
            "records": {
                key: {
                    "relative_path": rec.relative_path,
                    "media_type": rec.media_type,
                    "size": rec.size,
                    "sha256": rec.sha256,
                    "producer_stage": rec.producer_stage,
                    "requirement_ids": list(rec.requirement_ids),
                }
                for key, rec in sorted(self.records.items())
            },
            "index_sha256": "",
        }

    def write(self) -> Path:
        """Write ``artifact-index.json`` under the run root."""
        payload = self.to_json()
        payload["index_sha256"] = canonical_sha256(
            {key: value for key, value in payload.items() if key != "index_sha256"}
        )
        dest = self.root / "artifact-index.json"
        atomic_write_json(dest, payload)
        return dest

    def hydrate(self) -> None:
        """Load an existing artifact index, preserving earlier records."""
        path = self.root / "artifact-index.json"
        if not path.is_file():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rel, record in (payload.get("records") or {}).items():
            self.records[rel] = ArtifactRecord(
                relative_path=str(record.get("relative_path") or rel),
                media_type=str(record.get("media_type") or "application/json"),
                size=int(record.get("size") or 0),
                sha256=str(record.get("sha256") or ""),
                producer_stage=str(record.get("producer_stage") or "unknown"),
                requirement_ids=tuple(record.get("requirement_ids") or ()),
            )


def environment_lock() -> dict[str, Any]:
    """Capture non-secret runtime environment metadata."""
    try:
        kaiwu_version = version("kaiwu")
    except PackageNotFoundError:
        kaiwu_version = ""
    credentials = credential_fingerprint()
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "kaiwu_distribution": kaiwu_version,
        "torch_distribution": _distribution_version("torch"),
        "numpy_distribution": _distribution_version("numpy"),
        "license_available": credentials["license_available"],
        "license_fingerprint": credentials["license_fingerprint"],
        "credential_keys_present": credentials["credential_keys_present"],
        "captured_at_utc": utc_now(),
        "lock_sha256": text_sha256(canonical_json({"python": sys.version.split()[0]})),
    }


def _distribution_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return ""
