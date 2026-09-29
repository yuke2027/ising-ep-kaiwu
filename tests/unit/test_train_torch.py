"""Tests for the Torch autograd EP training loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from examples.mnist_kaiwu import train
from examples.mnist_kaiwu.solve import PairedSolverSettings
from examples.mnist_kaiwu.train import PhasePair, train_epoch
from src.ising_ep.model import IsingEPModel

from tests.conftest import to_numpy


@dataclass
class DummyAudit:
    """Stand-in solve audit that records only the fields the epoch writes."""

    phase: str
    optimizer_kwargs: dict

    def to_json(self) -> dict:
        """Return the ledger payload expected by ``train_epoch``."""
        return {"phase": self.phase, "optimizer_kwargs": dict(self.optimizer_kwargs)}


def _settings() -> PairedSolverSettings:
    return PairedSolverSettings(
        initial_temperature=1.0,
        alpha=0.9,
        cutoff_temperature=0.01,
        iterations_per_t=2,
        candidate_count=2,
        process_num=1,
    )


def test_train_epoch_averages_ep_loss_and_updates_once(tmp_path: Path, monkeypatch) -> None:
    """One batch uses one snapshot and applies one mean-loss optimizer step."""
    model = IsingEPModel(n_in=2, n_hidden=1, n_classes=1, spins_per_class=2, seed=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    calls = []

    def fake_solve(model_arg, x, target, beta, pair_seed, settings, *, run_id, sample_index):
        calls.append((sample_index, pair_seed))
        return PhasePair(
            h_free=torch.ones(1, dtype=torch.float64),
            o_free=torch.ones(2, dtype=torch.float64),
            h_nudged=-torch.ones(1, dtype=torch.float64),
            o_nudged=-torch.ones(2, dtype=torch.float64),
            free_audit=DummyAudit("free", {"seed": pair_seed}),
            nudged_audit=DummyAudit("nudged", {"seed": pair_seed}),
            free_solver_state=np.ones(3),
            nudged_solver_state=np.ones(3),
        )

    monkeypatch.setattr(train, "solve_phases", fake_solve)
    stats = train_epoch(
        model,
        optimizer,
        images=np.array([[1.0, 2.0], [0.5, 1.0]]),
        labels=np.zeros(2, dtype=np.int64),
        order=np.array([0, 1]),
        beta=0.5,
        batch_size=2,
        pair_seed_fn=lambda sample: 100 + sample,
        settings=_settings(),
        run_id="unit",
        seed=0,
        epoch=0,
        ledger_path=tmp_path / "ledger.jsonl",
    )

    expected_w_grad = np.array([[3.0], [6.0]])
    expected_v_grad = np.zeros((1, 2))
    expected_bias_grad = np.array([4.0])
    expected_output_bias_grad = np.full(2, 4.0)
    np.testing.assert_allclose(to_numpy(model.W), to_numpy(initial["W"]) - 0.1 * expected_w_grad)
    np.testing.assert_allclose(to_numpy(model.V), to_numpy(initial["V"]) - 0.1 * expected_v_grad)
    np.testing.assert_allclose(to_numpy(model.b_h), to_numpy(initial["b_h"]) - 0.1 * expected_bias_grad)
    np.testing.assert_allclose(to_numpy(model.b_o), to_numpy(initial["b_o"]) - 0.1 * expected_output_bias_grad)
    assert calls == [(0, 100), (1, 101)]
    assert stats["batch_updates"] == 1.0
    assert stats["n_samples"] == 2.0
    ledger = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger) == 4
    assert [json.loads(line)["phase"] for line in ledger] == ["free", "nudged", "free", "nudged"]
