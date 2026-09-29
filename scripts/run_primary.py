"""Thin legacy entry point for the PyTorch/Kaiwu experiment runner."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.mnist_kaiwu.experiment import main as experiment_main


def _strip_legacy_evidence_arg(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        argv = sys.argv[1:]
    cleaned: list[str] = []
    skip_next = False
    for index, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--evidence":
            skip_next = index + 1 < len(argv)
            continue
        if arg.startswith("--evidence="):
            continue
        cleaned.append(arg)
    return cleaned


def main(argv: list[str] | None = None) -> int:
    """Delegate to ``examples.mnist_kaiwu.experiment`` without adapter gates."""
    return experiment_main(_strip_legacy_evidence_arg(argv))


if __name__ == "__main__":
    raise SystemExit(main())
