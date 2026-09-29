"""Thin legacy entry point for freezing direct Kaiwu environment evidence."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.mnist_kaiwu.environment import run_environment_check


def main() -> int:
    """Run the official environment check under ``evidence/environment-check``."""
    output = ROOT / "evidence" / "environment-check"
    status = run_environment_check(output)
    print(status["state"], output)
    return 0 if status["state"] == "ATTESTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
