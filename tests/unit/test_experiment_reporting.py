"""Tests for experiment registration, seed artifacts and report delivery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.mnist_kaiwu import experiment, reporting
from examples.mnist_kaiwu.artifacts import append_jsonl, atomic_write_json
from examples.mnist_kaiwu.experiment import (
    ModelConfig,
    TrainingConfig,
    expected_call_counts,
    freeze_registration,
    ledger_phase_counts,
    run_seed,
)
from examples.mnist_kaiwu.failures import AlignmentError

_HEX = "a" * 64


def _registration(run_kind: str = "smoke"):
    return freeze_registration(
        experiment_id="unit-test",
        run_kind=run_kind,
        manifest_sha256=_HEX,
        method_trace_sha256=_HEX,
        environment_sha256=_HEX,
        seeds=(0,) if run_kind == "smoke" else (0, 1, 2),
        model=ModelConfig(n_hidden=2),
        training=TrainingConfig(epochs=1, batch_size=2, lr=0.1, beta=0.5),
    )


def test_registration_digest_and_primary_seed_requirement() -> None:
    """Registration digests match, and a primary run needs three seeds."""
    registration = _registration("primary")

    assert registration.registration_sha256 == registration.compute_registration_sha256()
    assert registration.checkpoint_policy.epoch == 0

    with pytest.raises(AlignmentError, match="at least 3"):
        freeze_registration(
            experiment_id="bad",
            run_kind="primary",
            manifest_sha256=_HEX,
            method_trace_sha256=_HEX,
            environment_sha256=_HEX,
            seeds=(0,),
            model=ModelConfig(n_hidden=2),
            training=TrainingConfig(epochs=1),
        )


def test_run_seed_writes_checkpointed_artifacts(tmp_path: Path, monkeypatch) -> None:
    """A smoke seed writes a checkpoint and a consistent call ledger."""
    registration = _registration("smoke")

    def fake_train_epoch(model, optimizer, images, labels, order, **kwargs):
        with torch.no_grad():
            model.W.add_(0.05)
            model.V.add_(0.05)
        ledger_path = kwargs["ledger_path"]
        for index in order:
            for phase in ("free", "nudged"):
                append_jsonl(
                    ledger_path,
                    {"phase": phase, "sample_index": int(index), "used_local_backend": False},
                )
        return {
            "hidden_flip": 1.0,
            "output_flip": 1.0,
            "zero_update": 0.0,
            "hidden_zero_update": 0.0,
            "skip_nudge": 0.0,
            "n_samples": float(len(order)),
            "batch_updates": 1.0,
        }

    def fake_evaluate(model, images, labels, settings, **kwargs):
        assert (kwargs["seed_dir"] / "checkpoint-decision.json").is_file()
        append_jsonl(
            kwargs["ledger_path"],
            {"phase": "evaluation", "sample_index": 0, "used_local_backend": False},
        )
        return {
            "accuracy": 1.0,
            "tie_rate": 0.0,
            "per_class": [
                {"class": index, "correct": 1 if index == 0 else 0, "total": 1, "accuracy": 1.0 if index == 0 else 0.0}
                for index in range(10)
            ],
        }

    monkeypatch.setattr(experiment, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(experiment, "evaluate_free_only", fake_evaluate)
    seed_dir = tmp_path / "seed-0"
    status = run_seed(
        registration,
        np.zeros((2, 784)),
        np.zeros(2, dtype=np.int64),
        np.zeros((1, 784)),
        np.zeros(1, dtype=np.int64),
        seed_dir,
        0,
    )

    assert status["state"] == "TEST_EVALUATED"
    assert status["valid"] is True
    assert ledger_phase_counts(seed_dir / "call-ledger.jsonl") == {
        "free": 2,
        "nudged": 2,
        "evaluation": 1,
    }
    assert expected_call_counts(2, 1, 1)["total"] == 5
    assert (seed_dir / "initial-model.npz").is_file()
    assert (seed_dir / "epochs" / "0" / "model.npz").is_file()
    assert (seed_dir / "primary-checkpoint.npz").is_file()
    assert (seed_dir / "checkpoint-decision.json").is_file()


def _write_minimal_valid_seed(root: Path, seed: int, accuracy: float) -> None:
    seed_dir = root / "seeds" / str(seed)
    seed_dir.mkdir(parents=True)
    ledger = {"free": 2, "nudged": 2, "evaluation": 1}
    for phase, count in ledger.items():
        for index in range(count):
            append_jsonl(
                seed_dir / "call-ledger.jsonl",
                {"phase": phase, "sample_index": index, "used_local_backend": False},
            )
    atomic_write_json(seed_dir / "checkpoint-decision.json", {"kind": "final_epoch"})
    atomic_write_json(
        seed_dir / "test" / "metrics.json",
        {"accuracy": accuracy, "tie_rate": 0.0},
    )
    atomic_write_json(
        seed_dir / "status.json",
        {
            "state": "TEST_EVALUATED",
            "valid": True,
            "used_local_backend": False,
            "seed": seed,
            "accuracy": accuracy,
            "tie_rate": 0.0,
            "per_class": [
                {"class": index, "correct": 0, "total": 1, "accuracy": accuracy}
                for index in range(10)
            ],
            "learning_evidence": {"passed": True},
            "ledger": ledger,
        },
    )


def test_deliver_run_writes_aggregate_reports(tmp_path: Path) -> None:
    """Delivery writes the aggregate reports for a completed run tree."""
    root = tmp_path / "run"
    atomic_write_json(
        root / "evidence" / "latest-smoke.json",
        {"status": "PASSED", "used_local_backend": False},
    )
    atomic_write_json(
        root / "evidence" / "attestation.json",
        {"provider": "kaiwu_official", "license_available": True},
    )
    atomic_write_json(
        root / "evidence" / "method-trace.json",
        {
            "trace_sha256": _HEX,
            "rows": [{"source_role": "primary", "source_file_sha256": _HEX}],
        },
    )
    atomic_write_json(root / "dataset" / "mnist100-manifest.json", {"manifest_sha256": _HEX})
    atomic_write_json(
        root / "registration.json",
        {
            "schema_version": "2.0",
            "seeds": [0, 1, 2],
            "model": {"n_hidden": 2},
            "training": {"epochs": 1},
            "registration_sha256": _HEX,
        },
    )
    atomic_write_json(root / "environment.json", {"python": "3.10.11"})
    for seed, accuracy in ((0, 0.6), (1, 0.5), (2, 0.4)):
        _write_minimal_valid_seed(root, seed, accuracy)

    result = reporting.deliver_run(root)

    assert result["verdict"]["verdict"] == "SUCCESS"
    assert result["aggregation"]["accuracy"]["mean"] == pytest.approx(0.5)
    assert (root / "aggregate" / "final-report.md").is_file()
    assert (root / "aggregate" / "failure-report.md").is_file()
    assert (root / "aggregate" / "traceability.json").is_file()
    assert (root / "reproduce.ps1").is_file()
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "SUCCESS"
