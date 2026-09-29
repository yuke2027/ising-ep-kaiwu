"""Read-only checks for the preserved historical Primary_Run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from examples.mnist_kaiwu.aggregation import aggregate_official_seeds, load_seed_statuses
from examples.mnist_kaiwu.data import manifest_from_json

RUN_DIR = Path(__file__).resolve().parents[2] / "runs" / "alignment-h64-e5"


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="historical run directory is not present")
def test_historical_registration_and_manifest_are_readable() -> None:
    """Historical registration and manifest files still parse."""
    registration = json.loads((RUN_DIR / "registration.json").read_text(encoding="utf-8"))
    manifest = manifest_from_json(json.loads((RUN_DIR / "dataset" / "mnist100-manifest.json").read_text(encoding="utf-8")))

    assert registration["model"] == {"n_classes": 10, "n_hidden": 64, "n_in": 784, "spins_per_class": 4}
    assert registration["training"] == {"batch_size": 10, "beta": 1.0, "epochs": 5, "lr": 0.01}
    assert registration["seeds"] == [0, 1, 2]
    assert len(manifest.train_indices) == 1000
    assert len(manifest.test_indices) == 100
    assert manifest.input_dtype == "float64"


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="historical run directory is not present")
def test_historical_seed_statuses_aggregate_to_recorded_baseline() -> None:
    """Historical seed statuses still aggregate to the recorded baseline."""
    statuses = load_seed_statuses(RUN_DIR)
    aggregate = aggregate_official_seeds(statuses)

    assert aggregate["n_valid"] == 3
    assert aggregate["seeds"] == [0, 1, 2]
    assert aggregate["accuracy"]["mean"] == pytest.approx(0.5433333333333333)
    assert all(not item.get("used_local_backend") for item in statuses)


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="historical run directory is not present")
def test_historical_npz_parameters_keep_expected_keys_and_shapes() -> None:
    """Historical parameter archives keep the expected keys and shapes."""
    for seed in (0, 1, 2):
        with np.load(RUN_DIR / "seeds" / str(seed) / "initial-model.npz") as archive:
            assert set(archive.files) == {"W", "V", "b_h", "b_o"}
            assert archive["W"].shape == (784, 64)
            assert archive["V"].shape == (64, 40)
            assert archive["b_h"].shape == (64,)
            assert archive["b_o"].shape == (40,)
            for key in archive.files:
                assert np.isfinite(archive[key]).all()
