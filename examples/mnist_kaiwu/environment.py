"""Project environment, license and official Kaiwu smoke checks."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import sys
from typing import Any

import torch

from src.ising_ep.mapping import build_qubo

from .artifacts import (
    atomic_write_json,
    environment_lock,
    load_dotenv,
    utc_now,
)
from .failures import (
    SDK_IMPORT_FAILED,
    SDK_INSTALL_BLOCKED,
    SDK_LICENSE_INVALID,
    AlignmentError,
    FailureRecord,
)
from .solve import PairedSolverSettings, build_kaiwu_problem, solve_kaiwu_problem

REQUIRED_PYTHON = (3, 10)
REQUIRED_KAIWU = "1.4.1"


def require_python310() -> None:
    """Fail unless the running interpreter is Python 3.10."""
    if sys.version_info[:2] != REQUIRED_PYTHON:
        raise AlignmentError(
            SDK_INSTALL_BLOCKED,
            "this project requires Python 3.10",
            evidence={"actual": sys.version.split()[0]},
        )


def import_kaiwu() -> Any:
    """Import the official Kaiwu package inside the project environment."""
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
            evidence={"error": type(exc).__name__},
        ) from exc
    return kw


def kaiwu_distribution_version() -> str:
    """Return the installed ``kaiwu`` distribution version."""
    try:
        return importlib.metadata.version("kaiwu")
    except importlib.metadata.PackageNotFoundError as exc:
        raise AlignmentError(
            SDK_INSTALL_BLOCKED,
            "Kaiwu distribution is not installed",
            evidence={"package": "kaiwu"},
        ) from exc


def require_kaiwu_141() -> str:
    """Require the installed Kaiwu distribution to be exactly 1.4.1."""
    version = kaiwu_distribution_version()
    if version != REQUIRED_KAIWU:
        raise AlignmentError(
            SDK_INSTALL_BLOCKED,
            "unsupported Kaiwu distribution version",
            evidence={"expected": REQUIRED_KAIWU, "actual": version},
        )
    return version


def initialize_license() -> bool:
    """Initialize Kaiwu from environment credentials or validate an issued license."""
    kw = import_kaiwu()
    load_dotenv()
    import os

    user_id = os.environ.get("KAIWU_USER_ID", "").strip()
    sdk_code = os.environ.get("KAIWU_SDK_CODE", "").strip()
    if user_id and sdk_code:
        try:
            kw.license.init(user_id=user_id, sdk_code=sdk_code)
        except Exception as exc:  # noqa: BLE001 - SDK defines its own failures
            raise AlignmentError(
                SDK_LICENSE_INVALID,
                "official Kaiwu license initialization failed",
                evidence={"error": type(exc).__name__},
            ) from exc
        return True
    package_dir = Path(getattr(kw, "__file__", "")).resolve().parent
    license_file = package_dir / "license.lic"
    return license_file.is_file() and license_file.stat().st_size > 0


def _write_failure(output_dir: Path, exc: AlignmentError, stage: str) -> dict[str, Any]:
    record = FailureRecord(
        failure_code=exc.code,
        message=exc.message,
        blocked_stage=stage,
        evidence=exc.evidence,
        evidence_id=exc.evidence_id,
    )
    atomic_write_json(output_dir / "failure.json", record.to_json())
    status = {"state": "FAILED", "failure": record.to_json(), "used_local_backend": False}
    atomic_write_json(output_dir / "status.json", status)
    return status


def run_environment_check(output_dir: str | Path) -> dict[str, Any]:
    """Run version, license and one tiny official solve; write evidence."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "environment.json", environment_lock())
    try:
        require_python310()
        version = require_kaiwu_141()
        licensed = initialize_license()
        if not licensed:
            raise AlignmentError(
                SDK_LICENSE_INVALID,
                "official Kaiwu authorization is not available",
            )
        problem = build_qubo(
            torch.tensor([[0.25]], dtype=torch.float64),
            torch.tensor([0.1], dtype=torch.float64),
            torch.tensor([-0.2], dtype=torch.float64),
            0.0,
        )
        kaiwu_problem = build_kaiwu_problem(problem)
        settings = PairedSolverSettings(
            initial_temperature=1.0,
            alpha=0.9,
            cutoff_temperature=0.01,
            iterations_per_t=2,
            candidate_count=2,
            process_num=1,
        )

        def energy_fn(physical: torch.Tensor) -> float:
            hidden, output = physical[:-1], physical[-1:]
            return float(
                (
                    -hidden @ torch.tensor([[0.25]], dtype=torch.float64) @ output
                    - 0.1 * hidden.sum()
                    + 0.2 * output.sum()
                ).item()
            )

        candidate, audit = solve_kaiwu_problem(
            kaiwu_problem,
            phase="evaluation",
            pair_seed=1,
            settings=settings,
            energy_fn=energy_fn,
            run_id="environment-smoke",
            sample_index=-1,
        )
        smoke = {
            "status": "PASSED",
            "kaiwu_distribution": version,
            "candidate": candidate.summary(),
            "audit": audit.to_json(),
            "used_local_backend": False,
            "checked_at_utc": utc_now(),
        }
        atomic_write_json(output / "evidence" / "latest-smoke.json", smoke)
        atomic_write_json(
            output / "evidence" / "attestation.json",
            {
                "provider": "kaiwu_official",
                "version": version,
                "license_available": True,
                "official_solve": True,
                "attested_at_utc": utc_now(),
            },
        )
        status = {
            "state": "ATTESTED",
            "kaiwu_distribution": version,
            "license_available": True,
            "latest_smoke_status": "PASSED",
            "used_local_backend": False,
            "output_dir": str(output),
        }
        atomic_write_json(output / "status.json", status)
        return status
    except AlignmentError as exc:
        return _write_failure(output, exc, "environment")


def build_parser() -> argparse.ArgumentParser:
    """Build the environment-check command-line parser."""
    parser = argparse.ArgumentParser(description="Check Python 3.10, Kaiwu 1.4.1 and license smoke.")
    parser.add_argument("--output", required=True, help="Evidence output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the environment gate and return a process status code."""
    args = build_parser().parse_args(argv)
    status = run_environment_check(args.output)
    print(status["state"], args.output)
    return 0 if status["state"] == "ATTESTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
