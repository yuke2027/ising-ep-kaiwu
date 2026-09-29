"""Primary/supporting source method trace for discrete Ising EP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
from typing import Literal, Sequence

from .artifacts import SCHEMA_VERSION, canonical_json, canonical_sha256
from .failures import METHOD_GATE_FAILED, AlignmentError

REQUIRED_METHOD_TOPICS = (
    "state_encoding",
    "network_structure",
    "ep_energy",
    "ep_cost",
    "f_beta",
    "free_phase",
    "nudged_phase",
    "positive_beta",
    "initial_state",
    "candidate_selection",
    "gradient",
    "batch_update",
    "anneal_schedule",
    "network_scale",
    "data_scale",
    "solver_hardware",
)

PRIMARY_SOURCE_NAME = "Laydevant2023_EP_Ising.pdf"
SUPPORTING_SOURCE_NAMES = (
    "Ernoult2020_CEP.pdf",
    "Laborieux2020_EP_DeepCNN.pdf",
    "Scellier2024_QEP.pdf",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
REQUIRED_PAPER_FILES = (PRIMARY_SOURCE_NAME,) + SUPPORTING_SOURCE_NAMES


class MethodTraceError(ValueError):
    """Raised when the method trace schema is incomplete or inconsistent."""


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MethodTraceError(f"{field_name} must not be empty")


def _require_text_sequence(values: Sequence[str], field_name: str) -> None:
    if isinstance(values, str) or not values:
        raise MethodTraceError(f"{field_name} must not be empty")
    for index, value in enumerate(values):
        _require_text(value, f"{field_name}[{index}]")


def _validate_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise MethodTraceError(f"{field_name} must be a lowercase SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise MethodTraceError(f"{field_name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class MethodTraceRow:
    """One method-alignment row tied to a source file and project behavior."""

    method_id: str
    topic: str
    source_role: Literal["primary", "supporting"]
    source_file: str
    source_file_sha256: str | None
    page_or_equation: str
    source_statement_summary: str
    project_symbol_or_behavior: str
    alignment_status: Literal["已对齐", "部分对齐", "未对齐", "缺少证据"]
    engineering_migration: str
    verification_test_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    notes: str

    def validate(self) -> None:
        """Validate required text fields, topic and digest shape."""
        _require_text(self.method_id, "method_id")
        if self.topic not in REQUIRED_METHOD_TOPICS:
            raise MethodTraceError(f"unsupported method topic: {self.topic!r}")
        if self.source_role not in {"primary", "supporting"}:
            raise MethodTraceError(f"invalid source_role: {self.source_role!r}")
        _require_text(self.source_file, "source_file")
        if self.source_file_sha256 is not None:
            _validate_sha256(self.source_file_sha256, "source_file_sha256")
        for name in (
            "page_or_equation",
            "source_statement_summary",
            "project_symbol_or_behavior",
        ):
            _require_text(getattr(self, name), name)
        if self.alignment_status not in {"已对齐", "部分对齐", "未对齐", "缺少证据"}:
            raise MethodTraceError(f"invalid alignment_status: {self.alignment_status!r}")
        if self.source_role == "supporting" and self.alignment_status == "已对齐":
            raise MethodTraceError("supporting sources cannot replace primary alignment")
        _require_text(self.engineering_migration, "engineering_migration")
        _require_text_sequence(self.verification_test_ids, "verification_test_ids")
        _require_text_sequence(self.evidence_ids, "evidence_ids")
        _require_text(self.notes, "notes")


@dataclass(frozen=True)
class MethodTrace:
    """Frozen method trace containing all required primary topics."""

    schema_version: str
    primary_source: str
    supporting_sources: tuple[str, ...]
    rows: tuple[MethodTraceRow, ...]
    trace_sha256: str

    @classmethod
    def freeze(
        cls,
        *,
        primary_source: str,
        supporting_sources: Sequence[str],
        rows: Sequence[MethodTraceRow],
    ) -> "MethodTrace":
        """Validate and freeze a trace with its content digest."""
        trace = cls(
            schema_version=SCHEMA_VERSION,
            primary_source=primary_source,
            supporting_sources=tuple(supporting_sources),
            rows=tuple(rows),
            trace_sha256="",
        )
        trace.validate(check_digest=False)
        frozen = replace(trace, trace_sha256=trace.compute_trace_sha256())
        frozen.validate()
        return frozen

    def compute_trace_sha256(self) -> str:
        """Compute the digest over all trace fields except ``trace_sha256``."""
        payload = asdict(self)
        payload.pop("trace_sha256", None)
        return canonical_sha256(payload)

    def validate(self, *, check_digest: bool = True) -> None:
        """Validate schema, source names, topics and content digest."""
        if self.schema_version != SCHEMA_VERSION:
            raise MethodTraceError("unsupported method trace schema_version")
        if self.primary_source != PRIMARY_SOURCE_NAME:
            raise MethodTraceError("primary_source must be the registered Primary_Source file")
        if tuple(self.supporting_sources) != SUPPORTING_SOURCE_NAMES:
            raise MethodTraceError("supporting_sources must match the registered set")
        topics = []
        for row in self.rows:
            row.validate()
            if row.source_role == "primary":
                topics.append(row.topic)
        missing = [topic for topic in REQUIRED_METHOD_TOPICS if topic not in topics]
        if missing:
            raise MethodTraceError(f"method trace missing primary topics: {missing}")
        if check_digest:
            _validate_sha256(self.trace_sha256, "trace_sha256")
            if self.compute_trace_sha256() != self.trace_sha256:
                raise MethodTraceError("trace_sha256 does not match method trace")

    def to_canonical_json(self) -> str:
        """Serialize the trace using canonical JSON."""
        return canonical_json(self)


def source_file_sha256(path: Path | None) -> str | None:
    """Return the SHA-256 of a source file, or ``None`` if it is absent."""
    if path is None or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_papers_dir(papers_dir: Path | None = None) -> Path:
    """Resolve an explicit papers directory or the usual project/workspace locations."""
    if papers_dir is not None:
        return Path(papers_dir)
    for candidate in (PROJECT_ROOT / "papers", WORKSPACE_ROOT / "papers", PROJECT_ROOT):
        if all((candidate / name).is_file() for name in REQUIRED_PAPER_FILES):
            return candidate
    return WORKSPACE_ROOT / "papers"


def default_method_trace(*, papers_dir: Path | None = None) -> MethodTrace:
    """Build the registered method topics; missing PDFs are marked missing evidence."""
    root = resolve_papers_dir(papers_dir)
    primary_path = root / PRIMARY_SOURCE_NAME
    primary_hash = source_file_sha256(primary_path)
    rows: list[MethodTraceRow] = []

    def primary(
        topic: str,
        page: str,
        summary: str,
        behavior: str,
        migration: str,
        tests: tuple[str, ...],
        notes: str,
        present_status: str = "已对齐",
    ) -> MethodTraceRow:
        return MethodTraceRow(
            method_id=f"primary-{topic}",
            topic=topic,
            source_role="primary",
            source_file=PRIMARY_SOURCE_NAME,
            source_file_sha256=primary_hash,
            page_or_equation=page,
            source_statement_summary=summary,
            project_symbol_or_behavior=behavior,
            alignment_status="缺少证据" if primary_hash is None else present_status,
            engineering_migration=migration,
            verification_test_ids=tests,
            evidence_ids=(f"method-{topic}",),
            notes=notes,
        )

    rows.extend(
        [
            primary(
                "state_encoding",
                "Ising spins s∈{-1,+1}",
                "Hidden and output units are bipolar spins.",
                "IsingEPModel.target_spins / physical states in {-1,+1}",
                "No migration: bipolar encoding is used as specified.",
                ("test_target_spins_encode_four_spin_class_blocks",),
                "Physical state checks require exact ±1.",
            ),
            primary(
                "network_structure",
                "784-120-40 with 4 output spins per class",
                "Visible pixels are clamped; hidden and output spins are solved.",
                "IsingEPModel(n_in=784, n_hidden configurable, n_output=40)",
                "Hidden width is an engineering migration from paper 120.",
                ("test_parameter_shapes_and_same_seed_reproducibility",),
                "Network-scale difference is reported as engineering_migration.",
                present_status="部分对齐",
            ),
            primary(
                "ep_energy",
                "E = -x^T W h - h^T V o - b_h^T h - b_o^T o",
                "Energy is bilinear in clamped inputs and spins.",
                "IsingEPModel.energy / mapping.energy",
                "Unified energy sign; not copied from D-Wave-specific notation.",
                ("test_torch_energy_backward_matches_ep_reference",),
                "Float64 evaluation at the algorithm boundary.",
            ),
            primary(
                "ep_cost",
                "C = 1/2 ||o-y||^2 = 40 - y^T o",
                "Squared cost on bipolar outputs reduces to a linear field.",
                "IsingEPModel.cost / mapping.cost_c",
                "No migration for the algebraic reduction.",
                ("test_ep_and_qubo_energies_differ_only_by_constant_on_tiny_model",),
                "Constant n_output*beta is kept in the QUBO offset.",
            ),
            primary(
                "f_beta",
                "F_β = E + β C",
                "Nudged objective adds a positive multiple of the cost.",
                "IsingEPModel.free_energy / mapping.free_energy",
                "No migration.",
                ("test_ep_and_qubo_energies_differ_only_by_constant_on_tiny_model",),
                "Primary_Run requires β>0.",
            ),
            primary(
                "free_phase",
                "β=0 equilibrium",
                "Free phase solves the unperturbed energy.",
                "solve_kaiwu_problem(phase='free')",
                "Solved on Kaiwu official SA rather than D-Wave.",
                ("test_solve_normalizes_candidates_and_selects_by_original_energy",),
                "Solver hardware is an engineering migration.",
                present_status="部分对齐",
            ),
            primary(
                "nudged_phase",
                "β>0 output-field nudge",
                "Nudge is applied only through output fields.",
                "IsingEPModel.fields / build_qubo",
                "No skip-nudge shortcut in Primary_Run.",
                ("test_ep_and_qubo_energies_differ_only_by_constant_on_tiny_model",),
                "paper_sa skip-nudge remains reference-only.",
            ),
            primary(
                "positive_beta",
                "positive nudge coefficient",
                "Learning uses a strictly positive β.",
                "train_epoch rejects beta <= 0",
                "β scale is chosen from field magnitude, not paper numerics.",
                ("test_train_epoch_averages_ep_loss_and_updates_once",),
                "Zero β is rejected before solving.",
            ),
            primary(
                "initial_state",
                "nudged initialized from free",
                "Paper reverse-anneals from the free state.",
                "single_process_solve(init_solution=complete free solver state)",
                "Warm-start 1.4.1 SA via single_process_solve(init_solution=...); not reverse annealing.",
                ("test_solve_passes_complete_solver_state_for_warm_start",),
                "Initial-solution semantics are verified against the installed SDK.",
                present_status="部分对齐",
            ),
            primary(
                "candidate_selection",
                "lowest-energy sample",
                "Use the lowest-energy configuration among reads.",
                "solve_kaiwu_problem reranks candidates by original EP energy",
                "Rerank with original float64 energy; ignore SDK order.",
                ("test_solve_normalizes_candidates_and_selects_by_original_energy",),
                "Near-ties use physical, solver-state, then content SHA-256.",
            ),
            primary(
                "gradient",
                "two-phase correlation difference / β",
                "Update estimates are free-minus-nudged correlations over β.",
                "loss=(E_nudged-E_free)/beta followed by loss.backward()",
                "No migration of the discrete two-phase estimator.",
                ("test_torch_energy_backward_matches_ep_reference",),
                "Zero when free and nudged states coincide.",
            ),
            primary(
                "batch_update",
                "mean gradient step",
                "Parameters are updated from a batch of estimates.",
                "torch.optim.SGD over the mean EP loss",
                "PyTorch autograd replaces handwritten NumPy gradients.",
                ("test_train_epoch_averages_ep_loss_and_updates_once",),
                "All batch solves use the same parameter snapshot.",
            ),
            primary(
                "anneal_schedule",
                "D-Wave reverse annealing schedule",
                "Paper uses a reverse temperature schedule on hardware.",
                "PairedSolverSettings",
                "Official Kaiwu SA schedule is not claimed equivalent.",
                ("test_solve_passes_complete_solver_state_for_warm_start",),
                "Schedule difference is an engineering migration.",
                present_status="部分对齐",
            ),
            primary(
                "network_scale",
                "784-120-40",
                "Paper hidden width is 120.",
                "IsingEPModel.n_hidden configurable",
                "Smaller hidden layer bounds solver size.",
                ("test_parameter_shapes_and_same_seed_reproducibility",),
                "Must be reported as engineering_migration.",
                present_status="部分对齐",
            ),
            primary(
                "data_scale",
                "MNIST/100",
                "100 train and 10 test images per class.",
                "data.first_per_class_indices",
                "No migration of the MNIST/100 split rule.",
                ("test_historical_registration_and_manifest_are_readable",),
                "Manifest hashing preserves auditability.",
            ),
            primary(
                "solver_hardware",
                "D-Wave QPU + software SA",
                "Paper primary results use D-Wave reverse annealing.",
                "kw.classical.SimulatedAnnealingOptimizer",
                "Kaiwu official classical SA is an engineering migration.",
                ("test_solve_normalizes_candidates_and_selects_by_original_energy",),
                "Local and paper_sa runs are Reference_Run only.",
                present_status="部分对齐",
            ),
        ]
    )
    for name in SUPPORTING_SOURCE_NAMES:
        support_path = root / name
        support_hash = source_file_sha256(support_path)
        rows.append(
            MethodTraceRow(
                method_id=f"supporting-{name}",
                topic="ep_energy",
                source_role="supporting",
                source_file=name,
                source_file_sha256=support_hash,
                page_or_equation="background",
                source_statement_summary="Continuous or convolutional EP background that does not override the discrete Ising primary method.",
                project_symbol_or_behavior="Cited only to interpret Primary_Source energy/nudge discussion.",
                alignment_status="缺少证据" if support_hash is None else "部分对齐",
                engineering_migration="Supporting sources cannot replace Primary_Source constraints.",
                verification_test_ids=("test_method_trace_schema",),
                evidence_ids=(f"supporting-{name}",),
                notes="Used as background for F_β and two-phase learning, not as an alternate algorithm.",
            )
        )
    return MethodTrace.freeze(
        primary_source=PRIMARY_SOURCE_NAME,
        supporting_sources=SUPPORTING_SOURCE_NAMES,
        rows=rows,
    )


def require_method_gate(trace: MethodTrace) -> None:
    """Require a valid trace and a hashed Primary_Source PDF."""
    trace.validate()
    primary_rows = [row for row in trace.rows if row.source_role == "primary"]
    if not primary_rows:
        raise AlignmentError(METHOD_GATE_FAILED, "method trace has no primary rows")
    missing = [
        row.topic
        for row in primary_rows
        if row.source_file_sha256 is None or row.alignment_status == "缺少证据"
    ]
    if missing:
        raise AlignmentError(
            METHOD_GATE_FAILED,
            "Primary_Source PDF is missing or unhashed",
            evidence={"topics": missing, "expected_file": PRIMARY_SOURCE_NAME},
        )


__all__ = [
    "PRIMARY_SOURCE_NAME",
    "REQUIRED_METHOD_TOPICS",
    "SUPPORTING_SOURCE_NAMES",
    "MethodTrace",
    "MethodTraceError",
    "MethodTraceRow",
    "default_method_trace",
    "require_method_gate",
    "resolve_papers_dir",
    "source_file_sha256",
]
