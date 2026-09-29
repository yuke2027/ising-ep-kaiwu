"""Thin legacy entry point for rebuilding Primary_Run reports."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.mnist_kaiwu.reporting import main as reporting_main


def main(argv: list[str] | None = None) -> int:
    """Delegate to ``examples.mnist_kaiwu.reporting``."""
    return reporting_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
