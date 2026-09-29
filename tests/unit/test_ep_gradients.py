"""Characterization tests for the EP free-minus-nudged gradient formula."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.ising_ep.model import IsingEPModel, ep_gradients

from tests.conftest import to_numpy


def test_ep_gradients_match_free_minus_nudged_correlations() -> None:
    """Handwritten EP gradients match free-minus-nudged outer products."""
    x = np.array([0.25, -0.5, 1.0])
    h_free = np.array([1.0, -1.0])
    o_free = np.array([1.0, -1.0, 1.0])
    h_nudged = np.array([-1.0, -1.0])
    o_nudged = np.array([-1.0, 1.0, 1.0])
    beta = 0.4

    g_w, g_v, g_bh, g_bo = ep_gradients(x, h_free, o_free, h_nudged, o_nudged, beta)

    np.testing.assert_allclose(g_w, (np.outer(x, h_free) - np.outer(x, h_nudged)) / beta)
    np.testing.assert_allclose(g_v, (np.outer(h_free, o_free) - np.outer(h_nudged, o_nudged)) / beta)
    np.testing.assert_allclose(g_bh, (h_free - h_nudged) / beta)
    np.testing.assert_allclose(g_bo, (o_free - o_nudged) / beta)


def test_ep_gradients_reject_non_positive_or_non_finite_beta() -> None:
    """Non-positive or non-finite beta is rejected."""
    x = np.ones(2)
    h = np.ones(1)
    o = np.ones(1)
    for beta in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            ep_gradients(x, h, o, h, o, beta)


def test_batch_gradient_is_average_of_per_sample_gradients() -> None:
    """A batch gradient equals the mean of the per-sample gradients."""
    samples = [
        (np.array([0.0, 1.0]), np.array([1.0]), np.array([1.0, -1.0]), np.array([-1.0]), np.array([-1.0, 1.0])),
        (np.array([1.0, 0.0]), np.array([-1.0]), np.array([-1.0, 1.0]), np.array([1.0]), np.array([1.0, -1.0])),
    ]
    gradients = [ep_gradients(*sample, beta=0.5) for sample in samples]
    expected = [sum(component[i] for component in gradients) / 2 for i in range(4)]
    actual = [np.mean([component[i] for component in gradients], axis=0) for i in range(4)]
    for expected_part, actual_part in zip(expected, actual, strict=True):
        np.testing.assert_allclose(actual_part, expected_part)


def test_torch_energy_backward_matches_ep_reference() -> None:
    """Autograd of the EP loss matches the handwritten gradient reference."""
    model = IsingEPModel(n_in=3, n_hidden=2, n_classes=1, spins_per_class=3, seed=12)
    x = np.array([0.2, -0.5, 0.9])
    h_free = np.array([1.0, -1.0])
    o_free = np.array([1.0, -1.0, -1.0])
    h_nudged = np.array([-1.0, -1.0])
    o_nudged = np.array([-1.0, 1.0, -1.0])
    beta = 0.4

    loss = (
        model.energy(x, h_nudged, o_nudged)
        - model.energy(x, h_free, o_free)
    ) / beta
    model.zero_grad()
    loss.backward()

    expected = ep_gradients(x, h_free, o_free, h_nudged, o_nudged, beta)
    np.testing.assert_allclose(to_numpy(model.W.grad), expected[0], atol=1e-10)
    np.testing.assert_allclose(to_numpy(model.V.grad), expected[1], atol=1e-10)
    np.testing.assert_allclose(to_numpy(model.b_h.grad), expected[2], atol=1e-10)
    np.testing.assert_allclose(to_numpy(model.b_o.grad), expected[3], atol=1e-10)
    assert model.W.grad.dtype == torch.float64
