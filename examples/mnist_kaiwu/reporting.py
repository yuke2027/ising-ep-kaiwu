"""Final reports, requirement traceability, Gap Register and Failure Report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .aggregation import REFERENCE_HISTORY, aggregate_official_seeds, load_seed_statuses, valid_seed_records
from .artifacts import ArtifactIndex, assert_no_secrets, atomic_write_json, atomic_write_text, utc_now
from .failures import ARTIFACT_INTEGRITY_ERROR, AlignmentError
from .verdict import ACCURACY_TARGET, verdict

GAP_STATUSES = ("已对齐", "部分对齐", "未对齐", "缺少证据")
GAP_PRIORITIES = ("P0", "P1", "P2", "P3")
GAP_DOMAINS = (
    "api-matrix",
    "primary-source",
    "method-tests",
    "training-evidence",
    "metrics-evidence",
)
FAILURE_CHAPTERS = ("accuracy", "learning", "solver", "compatibility", "reproducibility")


def _ids(major: int, start: int, end: int) -> tuple[str, ...]:
    return tuple(f"{major}.{i}" for i in range(start, end + 1))


REQUIRED_REQUIREMENT_IDS: tuple[str, ...] = (
    *_ids(1, 1, 10),
    *_ids(2, 1, 10),
    *_ids(3, 1, 8),
    *_ids(4, 1, 10),
    *_ids(5, 1, 10),
    *_ids(6, 1, 12),
    *_ids(7, 1, 12),
    *_ids(8, 1, 9),
    *_ids(9, 1, 10),
)

_MAJOR_META = {
    1: {
        "design_components": ["EnvironmentGate", "Direct Kaiwu official API"],
        "test_ids": ["test_solve.py", "tests/integration/test_kaiwu_solver.py"],
        "artifact_patterns": ["evidence/latest-smoke.json", "evidence/attestation.json"],
        "verdict_role": "compatibility",
    },
    2: {
        "design_components": ["MethodVerifier", "PyTorch energy core"],
        "test_ids": ["test_model.py", "test_ep_gradients.py", "test_mapping.py"],
        "artifact_patterns": ["evidence/method-trace.json"],
        "verdict_role": "method_gate",
    },
    3: {
        "design_components": ["Torch mapping", "Kaiwu QuboModel/IsingModel boundary"],
        "test_ids": ["test_mapping.py", "tests/integration/test_kaiwu_mapping.py"],
        "artifact_patterns": ["evidence/method-trace.json", "evidence/attestation.json"],
        "verdict_role": "method_gate",
    },
    4: {
        "design_components": ["DatasetManifestBuilder", "ExperimentRegistration", "ArtifactManager"],
        "test_ids": ["test_history.py", "test_experiment.py"],
        "artifact_patterns": ["dataset/mnist100-manifest.json", "registration.json"],
        "verdict_role": "dataset_gate",
    },
    5: {
        "design_components": ["Torch TrainingPipeline", "SimulatedAnnealingOptimizer"],
        "test_ids": ["test_train_torch.py", "test_solve.py"],
        "artifact_patterns": ["seeds/*/call-ledger.jsonl"],
        "verdict_role": "official_backend",
    },
    6: {
        "design_components": ["Free-only EvaluationPipeline", "Metrics"],
        "test_ids": ["test_solve.py", "test_reporting_history.py"],
        "artifact_patterns": ["seeds/*/test/metrics.json", "seeds/*/epochs/*/dynamics.json"],
        "verdict_role": "metrics",
    },
    7: {
        "design_components": ["VerdictEngine", "FailureAnalyzer"],
        "test_ids": ["test_reporting_history.py"],
        "artifact_patterns": ["aggregate/verdict.json", "aggregate/failure-report.json"],
        "verdict_role": "verdict",
    },
    8: {
        "design_components": ["ResultAggregator", "Reference isolation"],
        "test_ids": ["test_reporting_history.py"],
        "artifact_patterns": ["aggregate/metrics.json"],
        "verdict_role": "reference_isolation",
    },
    9: {
        "design_components": ["ReportGenerator", "ArtifactManager", "Gap Register"],
        "test_ids": ["test_reporting_history.py"],
        "artifact_patterns": ["aggregate/traceability.json", "aggregate/gap-register.json", "reproduce.ps1"],
        "verdict_role": "delivery",
    },
}


@dataclass(frozen=True)
class GapItem:
    """One validated gap-register item."""

    gap_id: str
    domain: str
    requirement_ids: tuple[str, ...]
    status: Literal["已对齐", "部分对齐", "未对齐", "缺少证据"]
    current_evidence_ids: tuple[str, ...]
    source_basis: str
    impact: str
    acceptance_method: str
    priority: Literal["P0", "P1", "P2", "P3"]
    owner: str | None = None

    def validate(self) -> None:
        """Validate domain, status and required non-aligned metadata."""
        if self.domain not in GAP_DOMAINS:
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, f"unsupported gap domain: {self.domain}")
        if self.status not in GAP_STATUSES:
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, f"invalid gap status: {self.status}")
        if self.status != "已对齐":
            for field_name in ("source_basis", "impact", "acceptance_method"):
                if not str(getattr(self, field_name)).strip():
                    raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, f"non-aligned gap missing {field_name}")
        if self.priority not in GAP_PRIORITIES:
            raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, f"invalid gap priority: {self.priority}")

    def to_json(self) -> dict[str, Any]:
        """Return the JSON representation."""
        return {
            "gap_id": self.gap_id,
            "domain": self.domain,
            "requirement_ids": list(self.requirement_ids),
            "status": self.status,
            "current_evidence_ids": list(self.current_evidence_ids),
            "source_basis": self.source_basis,
            "impact": self.impact,
            "acceptance_method": self.acceptance_method,
            "priority": self.priority,
            "owner": self.owner,
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _maybe_json(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    return _read_json(path)


def _count_ledger(path: Path) -> dict[str, int]:
    counts = {"free": 0, "nudged": 0, "evaluation": 0, "total": 0}
    if not path.is_file():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        phase = str(rec.get("phase") or "")
        if phase in counts:
            counts[phase] += 1
        counts["total"] += 1
    return counts


def _file_present(run_dir: Path, relative: str) -> bool:
    return (Path(run_dir) / relative).is_file()


def _stored_method_gate_passed(method_trace: Mapping[str, Any]) -> bool:
    if not method_trace.get("trace_sha256"):
        return False
    rows = method_trace.get("rows") or []
    primary_rows = [row for row in rows if row.get("source_role") == "primary"]
    return bool(primary_rows) and all(bool(row.get("source_file_sha256")) for row in primary_rows)


def inspect_run(run_dir: Path) -> dict[str, Any]:
    """Read a completed artifact tree. Never starts training or a local solver."""
    root = Path(run_dir)
    smoke = _maybe_json(root / "evidence" / "latest-smoke.json") or {}
    attestation = _maybe_json(root / "evidence" / "attestation.json") or {}
    method_trace = _maybe_json(root / "evidence" / "method-trace.json") or {}
    registration = _maybe_json(root / "registration.json") or {}
    manifest = _maybe_json(root / "dataset" / "mnist100-manifest.json") or {}
    environment = _maybe_json(root / "environment.json") or {}
    stored_status = _maybe_json(root / "status.json") or {}
    root_failure = _maybe_json(root / "failure.json")
    seed_records = load_seed_statuses(root)
    ledger_ok = True
    for record in seed_records:
        seed = record.get("seed")
        counts = _count_ledger(root / "seeds" / str(seed) / "call-ledger.jsonl")
        record["ledger_recount"] = counts
        expected = record.get("ledger") or {}
        if any(
            int(counts.get(key, -1)) != int(expected.get(key, -2))
            for key in ("free", "nudged", "evaluation")
        ):
            ledger_ok = False
        ckpt = root / "seeds" / str(seed) / "checkpoint-decision.json"
        test_metrics = root / "seeds" / str(seed) / "test" / "metrics.json"
        if test_metrics.is_file() and not ckpt.is_file():
            ledger_ok = False
        if record.get("used_local_backend"):
            ledger_ok = False
    latest_smoke_passed = smoke.get("status") == "PASSED" and not smoke.get("used_local_backend")
    attested = (
        attestation.get("provider") == "kaiwu_official"
        and bool(attestation.get("license_available"))
        and latest_smoke_passed
    )
    return {
        "run_dir": str(root),
        "latest_smoke_passed": bool(latest_smoke_passed),
        "method_gate_passed": _stored_method_gate_passed(method_trace),
        "attested": bool(attested),
        "ledger_ok": ledger_ok,
        "seed_records": seed_records,
        "registration": registration,
        "manifest": manifest,
        "environment": environment,
        "method_trace": method_trace,
        "smoke": smoke,
        "attestation": attestation,
        "stored_status": stored_status,
        "root_failure": root_failure,
        "files": {
            "smoke": _file_present(root, "evidence/latest-smoke.json"),
            "attestation": _file_present(root, "evidence/attestation.json"),
            "method_trace": _file_present(root, "evidence/method-trace.json"),
            "manifest": _file_present(root, "dataset/mnist100-manifest.json"),
            "registration": _file_present(root, "registration.json"),
            "environment": _file_present(root, "environment.json"),
        },
    }


def build_traceability(
    *,
    run_evidence_ids: Sequence[str],
    gap_items: Sequence[GapItem],
) -> list[dict[str, Any]]:
    """Build one traceability row for every required requirement ID."""
    by_req: dict[str, list[str]] = {req_id: [] for req_id in REQUIRED_REQUIREMENT_IDS}
    for item in gap_items:
        for req_id in item.requirement_ids:
            if req_id in by_req:
                by_req[req_id].append(item.gap_id)
    rows = []
    for req_id in REQUIRED_REQUIREMENT_IDS:
        major = int(req_id.split(".", 1)[0])
        meta = _MAJOR_META[major]
        evidence = list(run_evidence_ids)
        gaps = list(by_req[req_id])
        if req_id.startswith(("2.", "3.")) and "GAP-PROPERTY-TESTS" in {
            item.gap_id for item in gap_items
        } and not evidence:
            gaps = gaps or ["GAP-PROPERTY-TESTS"]
        if not evidence and not gaps:
            gaps = ["GAP-MISSING-EVIDENCE"]
        rows.append(
            {
                "requirement_id": req_id,
                "design_components": list(meta["design_components"]),
                "test_ids": list(meta["test_ids"]),
                "artifact_patterns": list(meta["artifact_patterns"]),
                "run_evidence_ids": evidence,
                "gap_ids": gaps,
                "verdict_role": meta["verdict_role"],
            }
        )
    present = {row["requirement_id"] for row in rows}
    missing = [req_id for req_id in REQUIRED_REQUIREMENT_IDS if req_id not in present]
    if missing:
        raise AlignmentError(
            ARTIFACT_INTEGRITY_ERROR,
            "traceability missing requirement ids",
            evidence={"missing": missing},
        )
    return rows


def build_gap_register(inspection: Mapping[str, Any], decision: Mapping[str, Any]) -> list[GapItem]:
    """Create the fixed-domain gap register for a run."""
    files = inspection.get("files") or {}
    conditions = decision.get("conditions") or {}
    mean_acc = float(decision.get("mean_accuracy") or 0.0)
    n_valid = int(decision.get("n_valid") or 0)
    items = [
        GapItem(
            gap_id="GAP-API-MATRIX",
            domain="api-matrix",
            requirement_ids=("1.1", "1.2", "1.3", "1.8", "1.9", "1.10", "9.1"),
            status="已对齐" if files.get("smoke") and files.get("attestation") and conditions.get("latest_smoke_passed") else "缺少证据",
            current_evidence_ids=("evidence/latest-smoke.json", "evidence/attestation.json"),
            source_basis="Kaiwu 1.4.1 direct API smoke/attestation artifacts.",
            impact="Primary_Run cannot be attested without official solve evidence.",
            acceptance_method="environment.py plus latest-smoke.json status=PASSED.",
            priority="P0",
        ),
        GapItem(
            gap_id="GAP-PRIMARY-SOURCE",
            domain="primary-source",
            requirement_ids=("2.1", "2.2", "2.10", "9.1"),
            status="部分对齐",
            current_evidence_ids=("evidence/method-trace.json",),
            source_basis="Laydevant2023_EP_Ising.pdf is hashed, but hidden width, SA schedule and hardware are engineering migrations.",
            impact="The run is a method-alignment experiment, not a paper-scale hardware reproduction.",
            acceptance_method="method-trace rows with engineering_migration marked and stored source hashes.",
            priority="P1",
        ),
        GapItem(
            gap_id="GAP-METHOD-TESTS",
            domain="method-tests",
            requirement_ids=("2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8"),
            status="部分对齐",
            current_evidence_ids=("tests/unit/test_model.py", "tests/unit/test_ep_gradients.py", "tests/unit/test_mapping.py"),
            source_basis="Unit energy/autograd/mapping tests exist; optional property tests are not implemented.",
            impact="Mapping correctness is unit-checked, not property-tested at 100 examples.",
            acceptance_method="Add property tests or keep this gap at P2.",
            priority="P2",
        ),
        GapItem(
            gap_id="GAP-TRAINING-EVIDENCE",
            domain="training-evidence",
            requirement_ids=("4.1", "4.5", "5.1", "5.2", "5.3", "5.8", "6.4", "6.7", "6.8", "6.9"),
            status="已对齐" if n_valid >= 3 and conditions.get("no_local_backend") else "未对齐",
            current_evidence_ids=("registration.json", "seeds-summary.json", "seeds/*/call-ledger.jsonl"),
            source_basis="Three official seeds completed with the registered free/nudged/evaluation call counts.",
            impact="Fewer than three valid official seeds blocks Accuracy_Target.",
            acceptance_method="seed status TEST_EVALUATED, ledger recount, used_local_backend=false.",
            priority="P0",
        ),
        GapItem(
            gap_id="GAP-METRICS-EVIDENCE",
            domain="metrics-evidence",
            requirement_ids=("6.1", "6.5", "6.6", "6.10", "6.11", "6.12", "7.1", "7.2", "8.1"),
            status="部分对齐" if mean_acc < ACCURACY_TARGET else "已对齐",
            current_evidence_ids=("aggregate/metrics.json", "aggregate/verdict.json"),
            source_basis=f"Per-seed metrics and Learning_Evidence are complete; mean accuracy {mean_acc:.4f} vs target {ACCURACY_TARGET:.2f}.",
            impact="Accuracy_Target is unmet, so the registered conjunction is FAILED.",
            acceptance_method="Pre-register a new Primary_Run if a higher-accuracy configuration is attempted.",
            priority="P0",
        ),
        GapItem(
            gap_id="GAP-PROPERTY-TESTS",
            domain="method-tests",
            requirement_ids=("2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "2.10", "4.5", "4.6", "4.7", "4.8", "6.5", "6.6", "6.7", "6.8", "6.9", "8.5", "8.6", "8.7", "8.8"),
            status="缺少证据",
            current_evidence_ids=(),
            source_basis="Property tests remain optional and were not executed.",
            impact="Correctness properties are only partially covered by unit fixtures.",
            acceptance_method="Implement the optional property-test tasks and archive reports.",
            priority="P2",
        ),
    ]
    if not conditions.get("latest_smoke_passed") or not conditions.get("attested"):
        items.append(
            GapItem(
                gap_id="GAP-COMPATIBILITY-BLOCKED",
                domain="api-matrix",
                requirement_ids=("1.10", "5.1", "7.7"),
                status="未对齐",
                current_evidence_ids=("evidence/latest-smoke.json", "evidence/attestation.json"),
                source_basis="Latest smoke or official attestation is missing or failed.",
                impact="Primary_Run is blocked; local/legacy substitution is forbidden.",
                acceptance_method="Restore official Kaiwu 1.4.1 license/smoke and re-run the environment check.",
                priority="P0",
            )
        )
    for item in items:
        item.validate()
    if not set(GAP_DOMAINS).issubset({item.domain for item in items}):
        raise AlignmentError(ARTIFACT_INTEGRITY_ERROR, "gap register missing required domains")
    return items


def analyze_failure(
    inspection: Mapping[str, Any],
    decision: Mapping[str, Any],
    aggregation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the auditable failure report from inspected evidence."""
    conditions = dict(decision.get("conditions") or {})
    unmet = [name for name, ok in conditions.items() if not ok]
    valid = valid_seed_records(list(inspection.get("seed_records") or []))
    dynamics = []
    for record in valid:
        seed = record.get("seed")
        path = Path(inspection["run_dir"]) / "seeds" / str(seed) / "epochs" / "0" / "dynamics.json"
        dynamics.append({"seed": seed, **(_maybe_json(path) or {})})
    per_class = (aggregation or {}).get("per_class") or []
    collapsed = [row for row in per_class if float(row.get("max") or 0.0) == 0.0]
    trigger = "accuracy"
    if not conditions.get("latest_smoke_passed") or not conditions.get("attested"):
        trigger = "compatibility"
    elif not conditions.get("method_gate_passed"):
        trigger = "compatibility"
    elif not conditions.get("learning_evidence"):
        trigger = "learning"
    elif not conditions.get("valid_official_seeds_ge_3") or not conditions.get("no_local_backend"):
        trigger = "reproducibility"
    confirmed = []
    if "mean_accuracy_ge_target" in unmet:
        confirmed.append(
            {
                "id": "ACC-BELOW-TARGET",
                "statement": f"mean Primary_Checkpoint test accuracy {float(decision.get('mean_accuracy') or 0.0):.4f} < {ACCURACY_TARGET:.2f}",
                "evidence_ids": ["aggregate/metrics.json", "aggregate/verdict.json", "seeds-summary.json"],
                "quantified": {
                    "mean": (aggregation or {}).get("accuracy"),
                    "n_valid": decision.get("n_valid"),
                    "per_seed": [{"seed": item.get("seed"), "accuracy": item.get("accuracy")} for item in valid],
                },
            }
        )
    for name in unmet:
        if name == "mean_accuracy_ge_target":
            continue
        confirmed.append(
            {
                "id": f"UNMET-{name.upper()}",
                "statement": f"registered conjunction condition {name} is false",
                "evidence_ids": ["aggregate/verdict.json"],
                "quantified": {"condition": name, "value": False},
            }
        )
    hypotheses = [
        {
            "id": "H-SMALL-HIDDEN",
            "statement": "A reduced hidden width versus paper 120 underfits MNIST/100.",
            "support": "Per-class collapse and low mean accuracy can indicate underfitting.",
            "against": "Learning_Evidence requires non-trivial parameter and hidden-state movement.",
            "next_action": "Pre-register a new experiment with a larger n_hidden; do not mutate this registration.",
        },
        {
            "id": "H-SOLVER-BUDGET",
            "statement": "The official SA iteration budget may be too small for the nudged objective.",
            "support": "Solver settings are an engineering migration from the paper schedule.",
            "against": "Free/nudged call counts and warm starts are still recorded and paired.",
            "next_action": "Change solver settings only in a new registration.",
        },
        {
            "id": "H-CLASS-COLLAPSE",
            "statement": "The 4-spin score plus SA noise collapses some digit classes to never-correct.",
            "support": f"Classes with max accuracy 0 across seeds: {[row.get('class') for row in collapsed]}",
            "against": "Tie-rate and per-class metrics are preserved for inspection.",
            "next_action": "Inspect confusion matrices and tie-rate before changing beta.",
        },
    ]
    chapters = {
        "accuracy": {
            "seed_variance": (aggregation or {}).get("accuracy"),
            "per_class": per_class,
            "tie_rate": (aggregation or {}).get("tie_rate"),
            "training_curve": dynamics,
        },
        "learning": {
            "per_seed": [
                {
                    "seed": item.get("seed"),
                    "learning_evidence": item.get("learning_evidence"),
                    "param_change": item.get("param_change"),
                }
                for item in valid
            ],
            "dynamics": dynamics,
        },
        "solver": {
            "settings": (inspection.get("registration") or {}).get("solver"),
            "beta": ((inspection.get("registration") or {}).get("training") or {}).get("beta"),
            "initial_solution": "nudged uses the complete free solver state via single_process_solve(init_solution=...)",
            "auxiliary_spin": "decoded at the Kaiwu example boundary; candidates reranked by original energy",
        },
        "compatibility": {
            "latest_smoke_passed": inspection.get("latest_smoke_passed"),
            "attested": inspection.get("attested"),
            "provider": (inspection.get("attestation") or {}).get("provider"),
            "version": (inspection.get("attestation") or {}).get("version"),
            "stale_root_failure": inspection.get("root_failure"),
        },
        "reproducibility": {
            "manifest_sha256": (inspection.get("manifest") or {}).get("manifest_sha256")
            or (inspection.get("registration") or {}).get("manifest_sha256"),
            "registration_sha256": (inspection.get("registration") or {}).get("registration_sha256"),
            "environment_sha256": (inspection.get("registration") or {}).get("environment_sha256"),
            "ledger_ok": inspection.get("ledger_ok"),
            "seeds": [item.get("seed") for item in valid],
        },
    }
    return {
        "verdict": decision.get("verdict"),
        "partial_success": False,
        "trigger": trigger,
        "unmet_conditions": unmet,
        "confirmed_causes": confirmed,
        "hypotheses": hypotheses,
        "chapters": chapters,
        "reference_history": [dict(item) for item in REFERENCE_HISTORY],
        "created_at_utc": utc_now(),
    }


def _engineering_migrations(method_trace: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for row in method_trace.get("rows") or []:
        if row.get("source_role") == "supporting" or row.get("alignment_status") != "部分对齐":
            continue
        rows.append(
            {
                "topic": str(row.get("topic") or ""),
                "alignment_status": str(row.get("alignment_status") or ""),
                "engineering_migration": str(row.get("engineering_migration") or ""),
            }
        )
    return rows


def render_failure_markdown(report: Mapping[str, Any]) -> str:
    """Render the machine-readable failure report as Markdown."""
    lines = [
        "# Failure Report",
        "",
        f"- verdict: `{report.get('verdict')}`",
        f"- trigger: `{report.get('trigger')}`",
        f"- partial_success: `{report.get('partial_success')}`",
        f"- unmet: {', '.join(report.get('unmet_conditions') or []) or '(none)'}",
        "",
        "## Confirmed causes",
        "",
    ]
    for item in report.get("confirmed_causes") or []:
        lines.append(f"- **{item.get('id')}**: {item.get('statement')}")
        lines.append(f"  - evidence: {', '.join(item.get('evidence_ids') or [])}")
    lines.extend(["", "## Hypotheses (not conclusions)", ""])
    for item in report.get("hypotheses") or []:
        lines.append(f"- **{item.get('id')}**: {item.get('statement')}")
        lines.append(f"  - support: {item.get('support')}")
        lines.append(f"  - against: {item.get('against')}")
        lines.append(f"  - next: {item.get('next_action')}")
    lines.extend(["", "## Chapters", ""])
    for name in FAILURE_CHAPTERS:
        lines.extend(
            [
                f"### {name}",
                "",
                "```json",
                json.dumps(report.get("chapters", {}).get(name), ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    lines.extend(["## Reference runs (excluded from Accuracy_Target)", ""])
    for item in report.get("reference_history") or []:
        lines.append(f"- {item.get('name')}: accuracy={item.get('accuracy')} kind={item.get('kind')}")
    return "\n".join(lines) + "\n"


def render_final_markdown(
    *,
    inspection: Mapping[str, Any],
    decision: Mapping[str, Any],
    aggregation: Mapping[str, Any] | None,
    gaps: Sequence[GapItem],
) -> str:
    """Render the final Chinese run report."""
    acc = (aggregation or {}).get("accuracy") or {}
    seeds = inspection.get("seed_records") or []
    lines = [
        "# PyTorch/Kaiwu 方法对齐实验 Primary_Run 报告",
        "",
        "本报告描述一次 **方法对齐实验**，不是论文硬件或网络规模的等价复现。",
        "",
        f"- 结论: **{decision.get('verdict')}**",
        f"- 准确率目标: {ACCURACY_TARGET:.2f}",
        f"- 有效官方种子: {decision.get('n_valid')}",
        f"- 平均测试准确率: {float(decision.get('mean_accuracy') or 0.0):.4f}",
        f"- 部分成功: `{decision.get('partial_success')}`",
        "",
        "## 门禁",
        "",
    ]
    for key, value in (decision.get("conditions") or {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Primary_Run 种子", ""])
    for item in seeds:
        lines.append(
            f"- seed {item.get('seed')}: accuracy={item.get('accuracy')} "
            f"tie_rate={item.get('tie_rate')} valid={item.get('valid')} "
            f"local={item.get('used_local_backend')} learning={((item.get('learning_evidence') or {}).get('passed'))}"
        )
    if acc:
        lines.extend(
            [
                "",
                "## 跨种子聚合（仅官方有效种子）",
                "",
                f"- accuracy mean/std/min/max: {acc.get('mean')}/{acc.get('std')}/{acc.get('min')}/{acc.get('max')}",
            ]
        )
    lines.extend(["", "## Reference_Run（不进入 Accuracy_Target）", ""])
    for item in REFERENCE_HISTORY:
        extra = ""
        if item.get("name") == "paper_sa":
            extra = f" learned={item.get('learned')} skip={item.get('skip')} zero={item.get('zero')}"
        lines.append(f"- {item.get('name')}: accuracy={item.get('accuracy')} kind={item.get('kind')}{extra}")
    lines.extend(["", "## 工程迁移", ""])
    for row in _engineering_migrations(inspection.get("method_trace") or {}):
        lines.append(f"- {row['topic']} ({row['alignment_status']}): {row['engineering_migration']}")
    model = (inspection.get("registration") or {}).get("model") or {}
    training = (inspection.get("registration") or {}).get("training") or {}
    n_hidden = int(model.get("n_hidden") or 16)
    epochs = int(training.get("epochs") or 1)
    lines.extend(
        [
            "",
            "## 限制",
            "",
            f"- 隐藏层 {n_hidden}、{epochs} epoch、Kaiwu 官方经典 SA 热启动，不等于论文 120 隐层 / D-Wave reverse annealing。",
            "- 可选属性测试未跑，已记入 Gap Register。",
            "- 根目录 `failure.json` 若仍是早期门禁失败记录，以 `aggregate/verdict.json` 为准。",
            "",
            "## Gap Register 摘要",
            "",
        ]
    )
    for item in gaps:
        lines.append(f"- `{item.gap_id}` [{item.domain}] {item.status} ({item.priority})")
    lines.extend(
        [
            "",
            "## 复现",
            "",
            "- Windows: `reproduce.ps1`（不含凭据）",
            "- Unix: `reproduce.sh`",
            "- 聚合本目录: `python -m examples.mnist_kaiwu.reporting --run-dir <this-run>`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _reproduce_ps1(project_rel: str, n_hidden: int, epochs: int, seeds: Sequence[int]) -> str:
    seed_csv = ",".join(str(seed) for seed in seeds) or "0,1,2"
    return f"""# Generated. No credentials. Do not commit .env.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RunDir = $PSScriptRoot
$ProjectRoot = (Resolve-Path (Join-Path $RunDir "{project_rel}")).Path
$py = Join-Path $ProjectRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $py)) {{
  Write-Error "Python 3.10 venv missing. Create .venv, install kaiwu 1.4.1 wheel + kaiwu-community, put KAIWU_* in .env"
}}
Set-Location $ProjectRoot
& $py -m examples.mnist_kaiwu.reporting --run-dir $RunDir
# Full official rerun MUST use a new output directory:
# & $py -m examples.mnist_kaiwu.experiment --output runs\\alignment-rerun --train --seeds {seed_csv} --n-hidden {n_hidden} --epochs {epochs}
"""


def _reproduce_sh(project_rel: str, n_hidden: int, epochs: int, seeds: Sequence[int]) -> str:
    seed_csv = ",".join(str(seed) for seed in seeds) or "0,1,2"
    return f"""#!/usr/bin/env bash
# Generated. No credentials. Do not commit .env.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$RUN_DIR/{project_rel}" && pwd)"
PY="$PROJECT_ROOT/.venv/bin/python"
cd "$PROJECT_ROOT"
"$PY" -m examples.mnist_kaiwu.reporting --run-dir "$RUN_DIR"
# Full official rerun MUST use a new output directory:
# "$PY" -m examples.mnist_kaiwu.experiment --output runs/alignment-rerun --train --seeds {seed_csv} --n-hidden {n_hidden} --epochs {epochs}
"""


def deliver_run(run_dir: Path) -> dict[str, Any]:
    """Re-validate a finished tree, write reports and reproduce scripts. No training."""
    root = Path(run_dir)
    if not root.is_dir():
        raise AlignmentError(
            ARTIFACT_INTEGRITY_ERROR,
            "run directory does not exist",
            evidence={"run_dir": str(root)},
        )
    inspection = inspect_run(root)
    records = list(inspection["seed_records"])
    aggregate_error = None
    try:
        aggregation = aggregate_official_seeds(records)
    except AlignmentError as exc:
        aggregation = None
        aggregate_error = exc
    decision = verdict(
        latest_smoke_passed=bool(inspection["latest_smoke_passed"]),
        method_gate_passed=bool(inspection["method_gate_passed"]),
        attested=bool(inspection["attested"]),
        aggregation=aggregation,
        seed_records=records,
    )
    if not inspection["ledger_ok"] and decision["verdict"] == "SUCCESS":
        decision = dict(decision)
        decision["verdict"] = "FAILED"
        decision["conditions"] = dict(decision["conditions"])
        decision["conditions"]["ledger_ok"] = False
    gaps = build_gap_register(inspection, decision)
    run_evidence = [
        rel
        for rel, ok in (
            ("evidence/latest-smoke.json", inspection["files"]["smoke"]),
            ("evidence/attestation.json", inspection["files"]["attestation"]),
            ("evidence/method-trace.json", inspection["files"]["method_trace"]),
            ("dataset/mnist100-manifest.json", inspection["files"]["manifest"]),
            ("registration.json", inspection["files"]["registration"]),
            ("environment.json", inspection["files"]["environment"]),
            ("seeds-summary.json", (root / "seeds-summary.json").is_file()),
            ("aggregate/metrics.json", aggregation is not None),
        )
        if ok
    ]
    traceability = build_traceability(run_evidence_ids=run_evidence, gap_items=gaps)
    failure_report = analyze_failure(inspection, decision, aggregation)
    aggregate_dir = root / "aggregate"
    if aggregation is not None:
        atomic_write_json(aggregate_dir / "metrics.json", aggregation)
    atomic_write_json(aggregate_dir / "verdict.json", decision)
    atomic_write_json(
        aggregate_dir / "validity.json",
        {
            "n_seeds": len(records),
            "n_valid": len(valid_seed_records(records)),
            "ledger_ok": inspection["ledger_ok"],
            "files": inspection["files"],
            "aggregate_error": None if aggregate_error is None else aggregate_error.to_record(),
            "stale_root_failure": inspection.get("root_failure"),
        },
    )
    atomic_write_json(aggregate_dir / "gap-register.json", [item.to_json() for item in gaps])
    atomic_write_json(aggregate_dir / "traceability.json", traceability)
    atomic_write_json(aggregate_dir / "failure-report.json", failure_report)
    atomic_write_text(aggregate_dir / "failure-report.md", render_failure_markdown(failure_report))
    atomic_write_text(
        aggregate_dir / "final-report.md",
        render_final_markdown(
            inspection=inspection,
            decision=decision,
            aggregation=aggregation,
            gaps=gaps,
        ),
    )
    model = (inspection.get("registration") or {}).get("model") or {}
    training = (inspection.get("registration") or {}).get("training") or {}
    seeds = (inspection.get("registration") or {}).get("seeds") or [0, 1, 2]
    atomic_write_text(
        root / "reproduce.ps1",
        _reproduce_ps1("..\\..", int(model.get("n_hidden") or 16), int(training.get("epochs") or 1), list(seeds)),
    )
    atomic_write_text(
        root / "reproduce.sh",
        _reproduce_sh("../..", int(model.get("n_hidden") or 16), int(training.get("epochs") or 1), list(seeds)),
    )
    index = ArtifactIndex(root)
    index.hydrate()
    for path, stage, reqs in (
        (aggregate_dir / "metrics.json", "aggregate", ("6.10", "8.1")),
        (aggregate_dir / "verdict.json", "verdict", ("7.1", "7.3")),
        (aggregate_dir / "failure-report.json", "failure", ("7.4", "7.9")),
        (aggregate_dir / "gap-register.json", "report", ("9.6", "9.10")),
        (aggregate_dir / "traceability.json", "report", ("9.5",)),
        (aggregate_dir / "final-report.md", "report", ("9.8", "9.9")),
        (root / "reproduce.ps1", "report", ("9.10",)),
        (root / "reproduce.sh", "report", ("9.10",)),
    ):
        if path.is_file():
            index.add(path, producer_stage=stage, requirement_ids=reqs)
    index.write()
    for path in (
        aggregate_dir / "final-report.md",
        aggregate_dir / "failure-report.md",
        root / "reproduce.ps1",
        root / "reproduce.sh",
    ):
        assert_no_secrets(path.read_text(encoding="utf-8"))
    atomic_write_json(
        root / "status.json",
        {
            "state": decision["verdict"],
            "used_local_backend": False,
            "training_started": True,
            "n_valid": decision.get("n_valid"),
            "verdict": decision,
            "delivered_at_utc": utc_now(),
            "aggregate_error": None if aggregate_error is None else aggregate_error.to_record(),
        },
    )
    return {
        "verdict": decision,
        "aggregation": aggregation,
        "gaps": [item.to_json() for item in gaps],
        "traceability": traceability,
        "failure_report": failure_report,
        "inspection": inspection,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the report-delivery command-line parser."""
    parser = argparse.ArgumentParser(description="Deliver aggregate reports for a finished run directory.")
    parser.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Deliver reports and return 0 unless artifact validation fails."""
    args = build_parser().parse_args(argv)
    result = deliver_run(Path(args.run_dir))
    print(result["verdict"]["verdict"], args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAILURE_CHAPTERS",
    "GAP_DOMAINS",
    "GapItem",
    "REQUIRED_REQUIREMENT_IDS",
    "analyze_failure",
    "build_gap_register",
    "build_traceability",
    "deliver_run",
    "inspect_run",
]
