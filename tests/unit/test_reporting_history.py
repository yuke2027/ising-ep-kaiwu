"""Read-only reporting compatibility checks for historical runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.mnist_kaiwu.reporting import inspect_run

RUN_DIR = Path(__file__).resolve().parents[2] / "runs" / "alignment-h64-e5"


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="historical run directory is not present")
def test_inspect_run_reads_historical_tree_without_training() -> None:
    """Report inspection reads a historical run tree without training."""
    inspection = inspect_run(RUN_DIR)

    assert len(inspection["seed_records"]) == 3
    assert inspection["registration"]["model"]["n_hidden"] == 64
    assert inspection["files"]["registration"] is True
    assert inspection["files"]["manifest"] is True
    assert inspection["ledger_ok"] is True
    assert all(not item.get("used_local_backend") for item in inspection["seed_records"])
