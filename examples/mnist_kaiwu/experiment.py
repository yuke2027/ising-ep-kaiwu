"""Registered MNIST/Kaiwu experiment orchestration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Literal
import uuid

import numpy as np
import torch

from src.ising_ep.model import IsingEPModel

from .aggregation import aggregate_official_seeds
from .artifacts import (
    ArtifactIndex,
    atomic_write_json,
    atomic_write_npz,
    canonical_sha256,
    environment_lock,
    utc_now,
)
from .data import build_mnist100_manifest, load_mnist100_from_manifest
from .environment import run_environment_check
from .evaluate import evaluate_free_only
from .failures import (
    ARTIFACT_INTEGRITY_ERROR,
    CALL_COUNT_MISMATCH,
    AlignmentError,
    FailureRecord,
)
from .method_verifier import default_method_trace, require_method_gate
from .metrics import l2_distance, learning_evidence, relative_change
from .reporting import deliver_run
from .solve import PairedSolverSettings
from .train import train_epoch
from .verdict import verdict

REGISTRATION_SCHEMA_VERSION = "2.0"
RunKind = Literal["smoke", "primary"]


def derived_int(experiment_seed: int, domain: str, *parts: int) -> int:
    """Derive a deterministic 31-bit integer from a seed and domain."""
    material = "|".join((str(int(experiment_seed)), domain, *(str(int(part)) for part in parts)))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


@dataclass(frozen=True)
class ModelConfig:
    """Registered Ising EP model dimensions."""

    n_in: int = 784
    n_hidden: int = 64
    n_classes: int = 10
    spins_per_class: int = 4

    def validate(self) -> None:
        """Validate the fixed MNIST architecture and positive hidden width."""
        if self.n_in != 784 or self.n_classes != 10 or self.spins_per_class != 4:
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "model config is not the registered 784-*-40 net")
        if self.n_hidden < 1:
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "n_hidden must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    """Registered EP training hyperparameters."""

    epochs: int = 5
    batch_size: int = 10
    lr: float = 0.01
    beta: float = 1.0

    def validate(self) -> None:
        """Validate positive epoch, batch, learning-rate and beta values."""
        if self.epochs < 1 or self.batch_size < 1:
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "epochs and batch_size must be positive")
        if not (self.lr > 0 and self.beta > 0):
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "lr and beta must be positive")


@dataclass(frozen=True)
class CheckpointPolicy:
    """Checkpoint selection policy; only the final epoch is registered."""

    kind: str = "final_epoch"
    epoch: int = 0

    def validate(self, *, epochs: int) -> None:
        """Require final-epoch checkpoint selection."""
        if self.kind != "final_epoch":
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "only final_epoch checkpoint policy is registered")
        if self.epoch != epochs - 1:
            raise AlignmentError(
                ARTIFACT_INTEGRITY_ERROR,
                "final_epoch policy must select epoch E-1",
                evidence={"epoch": self.epoch, "epochs": epochs},
            )


@dataclass(frozen=True)
class ExperimentRegistration:
    """Frozen registration for a smoke or primary experiment."""

    experiment_id: str
    run_kind: RunKind
    manifest_sha256: str
    method_trace_sha256: str
    environment_sha256: str
    seeds: tuple[int, ...]
    model: ModelConfig
    training: TrainingConfig
    solver: PairedSolverSettings
    checkpoint_policy: CheckpointPolicy
    initialization: str
    candidate_tie_rule: str
    metric_schema_version: str
    created_at_utc: str
    registration_sha256: str

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return the canonical registration payload."""
        return {
            "schema_version": REGISTRATION_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "run_kind": self.run_kind,
            "manifest_sha256": self.manifest_sha256,
            "method_trace_sha256": self.method_trace_sha256,
            "environment_sha256": self.environment_sha256,
            "seeds": list(self.seeds),
            "model": {
                "n_in": self.model.n_in,
                "n_hidden": self.model.n_hidden,
                "n_classes": self.model.n_classes,
                "spins_per_class": self.model.spins_per_class,
            },
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "lr": self.training.lr,
                "beta": self.training.beta,
            },
            "solver": {
                "initial_temperature": self.solver.initial_temperature,
                "alpha": self.solver.alpha,
                "cutoff_temperature": self.solver.cutoff_temperature,
                "iterations_per_t": self.solver.iterations_per_t,
                "candidate_count": self.solver.candidate_count,
                "process_num": self.solver.process_num,
            },
            "checkpoint_policy": {
                "kind": self.checkpoint_policy.kind,
                "epoch": self.checkpoint_policy.epoch,
            },
            "initialization": self.initialization,
            "candidate_tie_rule": self.candidate_tie_rule,
            "metric_schema_version": self.metric_schema_version,
            "created_at_utc": self.created_at_utc,
            "registration_sha256": self.registration_sha256,
        }

    def compute_registration_sha256(self) -> str:
        """Compute a digest over all fields except ``registration_sha256``."""
        payload = dict(self.to_canonical_dict())
        payload.pop("registration_sha256", None)
        return canonical_sha256(payload)

    def locked_projection(self) -> dict[str, Any]:
        """Return fields that must remain identical across seeds."""
        payload = self.to_canonical_dict()
        for field in ("seeds", "created_at_utc", "registration_sha256", "experiment_id"):
            payload.pop(field, None)
        return payload

    def validate(self) -> None:
        """Validate run kind, seeds, configs, checkpoint policy and digest."""
        if self.run_kind not in {"smoke", "primary"}:
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "run_kind must be smoke or primary")
        minimum = 1 if self.run_kind == "smoke" else 3
        if len(self.seeds) < minimum:
            raise AlignmentError(
                ARTIFACT_INTEGRITY_ERROR,
                f"{self.run_kind} registration requires at least {minimum} unique seeds",
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "registration seeds must be unique")
        self.model.validate()
        self.training.validate()
        self.solver.validate()
        self.checkpoint_policy.validate(epochs=self.training.epochs)
        if self.initialization != "torch.uniform_generator":
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "unsupported Torch initialization policy")
        if self.candidate_tie_rule != "original_energy then physical then solver_state then content_sha256":
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "candidate tie rule is not the registered rule")
        if self.compute_registration_sha256() != self.registration_sha256:
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "registration_sha256 does not match contents")


def freeze_registration(
    *,
    experiment_id: str,
    run_kind: RunKind,
    manifest_sha256: str,
    method_trace_sha256: str,
    environment_sha256: str,
    seeds: tuple[int, ...],
    model: ModelConfig | None = None,
    training: TrainingConfig | None = None,
    solver: PairedSolverSettings | None = None,
    checkpoint_policy: CheckpointPolicy | None = None,
) -> ExperimentRegistration:
    """Create, validate and freeze a registration."""
    model = model or ModelConfig()
    training = training or TrainingConfig()
    solver = solver or PairedSolverSettings(
        initial_temperature=1.25,
        alpha=0.95,
        cutoff_temperature=0.001,
        iterations_per_t=10,
        candidate_count=8,
        process_num=1,
    )
    policy = checkpoint_policy or CheckpointPolicy(kind="final_epoch", epoch=training.epochs - 1)
    draft = ExperimentRegistration(
        experiment_id=experiment_id,
        run_kind=run_kind,
        manifest_sha256=manifest_sha256,
        method_trace_sha256=method_trace_sha256,
        environment_sha256=environment_sha256,
        seeds=tuple(int(seed) for seed in seeds),
        model=model,
        training=training,
        solver=solver,
        checkpoint_policy=policy,
        initialization="torch.uniform_generator",
        candidate_tie_rule="original_energy then physical then solver_state then content_sha256",
        metric_schema_version=REGISTRATION_SCHEMA_VERSION,
        created_at_utc=utc_now(),
        registration_sha256="",
    )
    frozen = ExperimentRegistration(
        **{**draft.__dict__, "registration_sha256": draft.compute_registration_sha256()}
    )
    frozen.validate()
    return frozen


def expected_call_counts(n_train: int, n_test: int, epochs: int) -> dict[str, int]:
    """Return the registered official-call formula."""
    train_free = int(n_train) * int(epochs)
    train_nudged = int(n_train) * int(epochs)
    test_eval = int(n_test)
    return {
        "train_free": train_free,
        "train_nudged": train_nudged,
        "test_evaluation": test_eval,
        "total": train_free + train_nudged + test_eval,
    }


def ledger_phase_counts(path: Path) -> dict[str, int]:
    """Count free, nudged and evaluation rows in a call ledger."""
    counts = {"free": 0, "nudged": 0, "evaluation": 0}
    if not Path(path).is_file():
        return counts
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        phase = json.loads(line).get("phase")
        if phase in counts:
            counts[phase] += 1
    return counts


def _write_seed_failure(seed_dir: Path, exc: AlignmentError, stage: str, official_calls: int) -> dict[str, Any]:
    seed_dir.mkdir(parents=True, exist_ok=True)
    record = FailureRecord(
        failure_code=exc.code,
        message=exc.message,
        blocked_stage=stage,
        evidence=exc.evidence,
        evidence_id=exc.evidence_id,
        official_call_count=official_calls,
    )
    atomic_write_json(seed_dir / "failure.json", record.to_json())
    status = {"state": "FAILED", "valid": False, "failure": record.to_json(), "used_local_backend": False}
    atomic_write_json(seed_dir / "status.json", status)
    return status


def run_seed(
    registration: ExperimentRegistration,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    seed_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Run one registered seed through training, checkpointing and evaluation."""
    registration.validate()
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    if (seed_dir / "test").exists() and not (seed_dir / "checkpoint-decision.json").exists():
        return _write_seed_failure(
            seed_dir,
            AlignmentError(ARTIFACT_INTEGRITY_ERROR, "test directory exists before checkpoint freeze"),
            "checkpoint",
            0,
        )
    ledger = seed_dir / "call-ledger.jsonl"
    config = registration.model
    training = registration.training
    settings = registration.solver
    model_seed = derived_int(seed, "model")
    model = IsingEPModel(
        n_in=config.n_in,
        n_hidden=config.n_hidden,
        n_classes=config.n_classes,
        spins_per_class=config.spins_per_class,
        seed=model_seed,
        dtype=torch.float64,
        device="cpu",
    )
    initial = model.parameter_snapshot()
    atomic_write_npz(seed_dir / "initial-model.npz", initial)
    atomic_write_json(seed_dir / "config.json", {"seed": seed, "model_seed": model_seed})
    atomic_write_json(seed_dir / "status.json", {"state": "RUNNING", "valid": False, "used_local_backend": False})
    optimizer = torch.optim.SGD(model.parameters(), lr=training.lr)
    hidden_nonzero_rates: list[float] = []
    try:
        for epoch in range(training.epochs):
            rng = np.random.default_rng(derived_int(seed, "shuffle", epoch))
            order = rng.permutation(int(train_images.shape[0]))

            def pair_seed_fn(sample_index: int, epoch_index: int = epoch) -> int:
                return derived_int(seed, "pair", epoch_index, sample_index)

            stats = train_epoch(
                model,
                optimizer,
                train_images,
                train_labels,
                order,
                beta=training.beta,
                batch_size=training.batch_size,
                pair_seed_fn=pair_seed_fn,
                settings=settings,
                run_id=f"{registration.experiment_id}-{seed}",
                seed=seed,
                epoch=epoch,
                ledger_path=ledger,
            )
            hidden_nonzero_rates.append(1.0 - float(stats["hidden_zero_update"]))
            epoch_dir = seed_dir / "epochs" / str(epoch)
            atomic_write_npz(epoch_dir / "model.npz", model.parameter_snapshot())
            atomic_write_json(epoch_dir / "dynamics.json", stats)
        decision = {
            "kind": registration.checkpoint_policy.kind,
            "selected_epoch": registration.checkpoint_policy.epoch,
            "frozen_at_utc": utc_now(),
            "training_only": True,
        }
        atomic_write_json(seed_dir / "checkpoint-decision.json", decision)
        atomic_write_npz(seed_dir / "primary-checkpoint.npz", model.parameter_snapshot())
        test_metrics = evaluate_free_only(
            model,
            test_images,
            test_labels,
            settings,
            seed=seed,
            run_id=f"{registration.experiment_id}-{seed}",
            seed_dir=seed_dir,
            ledger_path=ledger,
            eval_seed=derived_int(seed, "eval"),
        )
        counts = ledger_phase_counts(ledger)
        expected = expected_call_counts(
            int(train_images.shape[0]), int(test_images.shape[0]), training.epochs
        )
        if (
            counts["free"] != expected["train_free"]
            or counts["nudged"] != expected["train_nudged"]
            or counts["evaluation"] != expected["test_evaluation"]
        ):
            raise AlignmentError(
                CALL_COUNT_MISMATCH,
                "official call ledger does not match the registered formula",
                evidence={"actual": counts, "expected": expected},
            )
        final = model.parameter_snapshot()
        w_change = l2_distance(initial["W"], final["W"])
        v_change = l2_distance(initial["V"], final["V"])
        learn = learning_evidence(
            w_change=w_change,
            v_change=v_change,
            hidden_nonzero_rates=hidden_nonzero_rates,
        )
        status = {
            "state": "TEST_EVALUATED",
            "valid": True,
            "used_local_backend": False,
            "seed": seed,
            "accuracy": test_metrics["accuracy"],
            "tie_rate": test_metrics["tie_rate"],
            "per_class": test_metrics["per_class"],
            "learning_evidence": learn,
            "param_change": {
                "W": relative_change(initial["W"], final["W"]),
                "V": relative_change(initial["V"], final["V"]),
            },
            "ledger": counts,
            "expected_calls": expected,
        }
        atomic_write_json(seed_dir / "status.json", status)
        return status
    except AlignmentError as exc:
        return _write_seed_failure(seed_dir, exc, "seed", sum(ledger_phase_counts(ledger).values()))


def run_registered_seeds(
    registration: ExperimentRegistration,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """Run all seeds in a registration and write a summary."""
    registration.validate()
    output_dir = Path(output_dir)
    seeds_root = output_dir / "seeds"
    results = []
    for seed in registration.seeds:
        result = run_seed(
            registration,
            train_images,
            train_labels,
            test_images,
            test_labels,
            seeds_root / str(seed),
            int(seed),
        )
        results.append({"seed": int(seed), **result})
    summary = {
        "state": "SEEDS_COMPLETE",
        "used_local_backend": False,
        "n_seeds": len(results),
        "n_valid": sum(1 for item in results if item.get("valid")),
        "seeds": results,
    }
    atomic_write_json(output_dir / "seeds-summary.json", summary)
    return summary


def _write_root_failure(output_dir: Path, exc: AlignmentError, stage: str) -> int:
    record = FailureRecord(
        failure_code=exc.code,
        message=exc.message,
        blocked_stage=stage,
        evidence=exc.evidence,
        evidence_id=exc.evidence_id,
    )
    atomic_write_json(output_dir / "failure.json", record.to_json())
    atomic_write_json(output_dir / "status.json", {"state": "FAILED", "failure": record.to_json()})
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment command-line parser."""
    parser = argparse.ArgumentParser(description="Run the MNIST/Kaiwu EP experiment.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=str(Path.cwd() / "data"))
    parser.add_argument("--papers-dir", default=None)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--run-kind", choices=("smoke", "primary"), default="primary")
    parser.add_argument("--gates-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run environment, method, data and registered-seed gates."""
    parser = build_parser()
    args = parser.parse_args(argv)
    raw_args = list(argv if argv is not None else sys.argv[1:])
    if any(flag.startswith("--fallback") or flag.startswith("--backend") for flag in raw_args):
        parser.error("local fallback and backend switches are not accepted")
    if args.gates_only and args.train:
        parser.error("choose either --gates-only or --train")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "seeds").exists():
        parser.error("output already contains a training seeds directory")

    env_status = run_environment_check(output_dir)
    if env_status["state"] != "ATTESTED":
        return 1
    environment_sha256 = canonical_sha256(env_status)

    try:
        trace = default_method_trace(
            papers_dir=None if args.papers_dir is None else Path(args.papers_dir)
        )
        require_method_gate(trace)
    except AlignmentError as exc:
        return _write_root_failure(output_dir, exc, "method")
    atomic_write_json(output_dir / "evidence" / "method-trace.json", json.loads(trace.to_canonical_json()))

    try:
        manifest = build_mnist100_manifest(args.data_root, download_if_missing=False)
    except AlignmentError as exc:
        return _write_root_failure(output_dir, exc, "manifest")
    atomic_write_json(output_dir / "dataset" / "mnist100-manifest.json", manifest.to_canonical_dict())

    seeds = tuple(int(part.strip()) for part in str(args.seeds).split(",") if part.strip())
    try:
        registration = freeze_registration(
            experiment_id=uuid.uuid4().hex,
            run_kind=args.run_kind,
            manifest_sha256=manifest.manifest_sha256,
            method_trace_sha256=trace.trace_sha256,
            environment_sha256=environment_sha256,
            seeds=seeds,
            model=ModelConfig(n_hidden=args.n_hidden),
            training=TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                beta=args.beta,
            ),
        )
    except AlignmentError as exc:
        return _write_root_failure(output_dir, exc, "registration")
    atomic_write_json(output_dir / "registration.json", registration.to_canonical_dict())
    atomic_write_json(output_dir / "environment.json", environment_lock())

    index = ArtifactIndex(output_dir)
    for path, stage in (
        (output_dir / "evidence" / "latest-smoke.json", "environment"),
        (output_dir / "evidence" / "method-trace.json", "method"),
        (output_dir / "dataset" / "mnist100-manifest.json", "manifest"),
        (output_dir / "registration.json", "registration"),
    ):
        if path.is_file():
            index.add(path, producer_stage=stage)
    index.write()

    atomic_write_json(
        output_dir / "status.json",
        {
            "state": "GATES_PASSED",
            "run_kind": args.run_kind,
            "used_local_backend": False,
            "training_started": False,
            "registration_sha256": registration.registration_sha256,
            "manifest_sha256": manifest.manifest_sha256,
            "method_trace_sha256": trace.trace_sha256,
        },
    )
    if not args.train:
        print("GATES_PASSED", output_dir)
        print("pass --train to start official seeds after gates")
        return 0

    try:
        train, test = load_mnist100_from_manifest(args.data_root, manifest)
        summary = run_registered_seeds(
            registration,
            train.images,
            train.labels,
            test.images,
            test.labels,
            output_dir,
        )
        aggregate = None
        aggregate_error = None
        if registration.run_kind == "primary":
            try:
                aggregate = aggregate_official_seeds(summary["seeds"])
            except AlignmentError as exc:
                aggregate_error = exc
            final = verdict(
                latest_smoke_passed=True,
                method_gate_passed=True,
                attested=True,
                aggregation=aggregate,
                seed_records=summary["seeds"],
            )
            state = final["verdict"]
        else:
            passed = bool(summary["n_valid"] == len(registration.seeds))
            final = {
                "verdict": "SMOKE_PASSED" if passed else "SMOKE_FAILED",
                "n_valid": summary["n_valid"],
                "partial_success": False,
            }
            state = final["verdict"]
        atomic_write_json(
            output_dir / "status.json",
            {
                "state": state,
                "run_kind": registration.run_kind,
                "used_local_backend": False,
                "training_started": True,
                "n_valid": summary["n_valid"],
                "verdict": final,
                "aggregate_error": None if aggregate_error is None else aggregate_error.to_record(),
            },
        )
        try:
            deliver_run(output_dir)
        except AlignmentError:
            if state in {"SUCCESS", "SMOKE_PASSED"}:
                raise
        print(state, output_dir)
        return 0 if state in {"SUCCESS", "SMOKE_PASSED"} else 1
    except AlignmentError as exc:
        return _write_root_failure(output_dir, exc, "train")


if __name__ == "__main__":
    raise SystemExit(main())
