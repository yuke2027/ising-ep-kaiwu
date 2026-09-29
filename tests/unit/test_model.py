"""Characterization tests for model parameters, targets, fields and prediction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ising_ep.model import IsingEPModel
from src.ising_ep.mapping import energy
from src.ising_ep.model import IsingEPModel as SourceIsingEPModel

from tests.conftest import set_parameter, to_numpy


def test_parameter_shapes_and_same_seed_reproducibility() -> None:
    """Same seeds reproduce float64 parameter shapes; different seeds do not."""
    first = IsingEPModel(n_in=5, n_hidden=3, n_classes=2, spins_per_class=2, seed=7)
    second = IsingEPModel(n_in=5, n_hidden=3, n_classes=2, spins_per_class=2, seed=7)
    different = IsingEPModel(n_in=5, n_hidden=3, n_classes=2, spins_per_class=2, seed=8)

    for key in ("W", "V", "b_h", "b_o"):
        np.testing.assert_allclose(to_numpy(first.numpy_params()[key]), to_numpy(second.numpy_params()[key]))
    assert not np.array_equal(to_numpy(first.numpy_params()["W"]), to_numpy(different.numpy_params()["W"]))
    assert to_numpy(first.numpy_params()["W"]).shape == (5, 3)
    assert to_numpy(first.numpy_params()["V"]).shape == (3, 4)
    assert to_numpy(first.numpy_params()["b_h"]).shape == (3,)
    assert to_numpy(first.numpy_params()["b_o"]).shape == (4,)


def test_target_spins_encode_four_spin_class_blocks() -> None:
    """Class labels become four-spin +1 blocks on a -1 background."""
    model = IsingEPModel(n_in=2, n_hidden=2, n_classes=10, spins_per_class=4)
    encoded = to_numpy(model.target_spins([0, 3, 9]))

    assert encoded.shape == (3, 40)
    np.testing.assert_array_equal(encoded[0, :4], 1.0)
    np.testing.assert_array_equal(encoded[1, 12:16], 1.0)
    np.testing.assert_array_equal(encoded[2, 36:40], 1.0)
    np.testing.assert_array_equal(encoded.sum(axis=1), np.full(3, -32.0))


def test_fields_match_reference_linear_formulas() -> None:
    """Local fields follow ``W.T @ x + b_h`` and ``b_o + beta * y``."""
    model = IsingEPModel(n_in=3, n_hidden=2, n_classes=2, spins_per_class=2, seed=3)
    params = {name: to_numpy(value) for name, value in model.numpy_params().items()}
    x = np.array([0.2, -0.4, 0.7])
    y = to_numpy(model.target_spins([1]))[0]

    hidden, output = model.fields(x, y, beta=0.6)

    np.testing.assert_allclose(to_numpy(hidden), params["W"].T @ x + params["b_h"])
    np.testing.assert_allclose(to_numpy(output), params["b_o"] + 0.6 * y)


def test_prediction_uses_output_block_sum_then_local_field_tie_break() -> None:
    """Class scores use output-block sums, then local fields on a tie."""
    model = IsingEPModel(n_in=2, n_hidden=2, n_classes=4, spins_per_class=2, seed=4)
    set_parameter(model.V, np.zeros((2, 8)))
    set_parameter(model.b_o, np.zeros(8))
    hidden = np.zeros(2)

    output = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    prediction, tied = model.predict_from_output(output, hidden)
    assert prediction == 0
    assert tied is True

    output = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    prediction, tied = model.predict_from_output(output, hidden)
    assert prediction == 1
    assert tied is False


def test_invalid_target_label_is_rejected() -> None:
    """Labels outside the class range raise ``ValueError``."""
    model = IsingEPModel(n_in=2, n_hidden=2, n_classes=3, spins_per_class=2)
    with pytest.raises(ValueError):
        model.target_spins([3])


def test_parameter_snapshot_is_independent_of_optimizer_updates() -> None:
    """A parameter snapshot stays unchanged after the live weights move."""
    model = IsingEPModel(n_in=2, n_hidden=2, n_classes=1, spins_per_class=2, seed=9)
    snapshot = model.parameter_snapshot()
    with torch.no_grad():
        model.W.add_(1.0)
    np.testing.assert_allclose(to_numpy(snapshot["W"]), to_numpy(model.numpy_params()["W"]) - 1.0)
    assert snapshot["W"].data_ptr() != model.W.data_ptr()


def test_legacy_model_import_is_src_implementation() -> None:
    """The compatibility package re-exports the ``src`` model class."""
    assert IsingEPModel is SourceIsingEPModel


def test_model_energy_matches_mapping_energy() -> None:
    """``IsingEPModel.energy`` matches ``mapping.energy`` within ``1e-12``."""
    model = IsingEPModel(n_in=4, n_hidden=3, n_classes=2, spins_per_class=2, seed=2)
    x = np.array([0.2, -0.5, 0.1, 0.8])
    hidden = np.array([1.0, -1.0, 1.0])
    output = np.array([1.0, -1.0, -1.0, 1.0])
    params = model.numpy_params()
    mapped = energy(x, hidden, output, params["W"], params["V"], params["b_h"], params["b_o"])
    closed = float(
        -x @ params["W"] @ hidden
        - hidden @ params["V"] @ output
        - params["b_h"] @ hidden
        - params["b_o"] @ output
    )
    observed = float(model.energy(x, hidden, output).detach().cpu())

    assert mapped == pytest.approx(closed, abs=1e-12)
    assert observed == pytest.approx(mapped, abs=1e-12)
