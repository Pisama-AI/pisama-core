"""Typed SDK view of the additive Pisama diagnosis contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


def _probability(value: Any, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _extras(payload: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in known}


@dataclass(frozen=True)
class CoverageRecord:
    status: str
    completeness: float
    reason: Optional[str] = None
    missing_required: tuple[str, ...] = ()
    requirements: tuple[dict[str, Any], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "completeness",
            _probability(self.completeness, "completeness"),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CoverageRecord":
        known = {
            "status",
            "completeness",
            "reason",
            "missing_required",
            "requirements",
        }
        return cls(
            status=str(payload.get("status") or "missing"),
            completeness=float(payload.get("completeness") or 0.0),
            reason=(str(payload["reason"]) if payload.get("reason") else None),
            missing_required=tuple(payload.get("missing_required") or ()),
            requirements=tuple(dict(item) for item in payload.get("requirements") or ()),
            extra=_extras(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.extra,
            "status": self.status,
            "completeness": self.completeness,
            "reason": self.reason,
            "missing_required": list(self.missing_required),
            "requirements": [dict(item) for item in self.requirements],
        }


@dataclass(frozen=True)
class DiagnosisEvidence:
    kind: str
    event_ids: tuple[str, ...] = ()
    description: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiagnosisEvidence":
        known = {"kind", "event_ids", "description", "data"}
        return cls(
            kind=str(payload.get("kind") or "unknown"),
            event_ids=tuple(str(item) for item in payload.get("event_ids") or ()),
            description=str(payload.get("description") or ""),
            data=dict(payload.get("data") or {}),
            extra=_extras(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.extra,
            "kind": self.kind,
            "event_ids": list(self.event_ids),
            "description": self.description,
            "data": self.data,
        }


@dataclass(frozen=True)
class CausalCandidateRecord:
    label: str
    support: float
    evidence_ids: tuple[str, ...] = ()
    rationale: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "support", _probability(self.support, "support"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CausalCandidateRecord":
        known = {"label", "support", "evidence_ids", "rationale"}
        return cls(
            label=str(payload.get("label") or "unknown"),
            support=float(payload.get("support") or 0.0),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids") or ()),
            rationale=(str(payload["rationale"]) if payload.get("rationale") else None),
            extra=_extras(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.extra,
            "label": self.label,
            "support": self.support,
            "evidence_ids": list(self.evidence_ids),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class InterventionRecord:
    kind: str
    status: str
    target_step: Optional[int] = None
    prefix_preserved_through: Optional[int] = None
    outcome_before: Optional[str] = None
    outcome_after: Optional[str] = None
    verifier: Optional[str] = None
    run_id: Optional[str] = None
    evidence: tuple[DiagnosisEvidence, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("target_step", "prefix_preserved_through"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.status == "verified":
            if self.target_step is None or not self.verifier:
                raise ValueError("verified intervention requires target_step and verifier")
            if self.outcome_before is None or self.outcome_after is None:
                raise ValueError("verified intervention requires before and after outcomes")
            if self.outcome_before == self.outcome_after:
                raise ValueError("verified intervention requires an outcome change")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterventionRecord":
        known = {
            "kind",
            "status",
            "target_step",
            "prefix_preserved_through",
            "outcome_before",
            "outcome_after",
            "verifier",
            "run_id",
            "evidence",
        }
        return cls(
            kind=str(payload.get("kind") or "unknown"),
            status=str(payload.get("status") or "not_run"),
            target_step=payload.get("target_step"),
            prefix_preserved_through=payload.get("prefix_preserved_through"),
            outcome_before=payload.get("outcome_before"),
            outcome_after=payload.get("outcome_after"),
            verifier=payload.get("verifier"),
            run_id=payload.get("run_id"),
            evidence=tuple(
                DiagnosisEvidence.from_dict(item) for item in payload.get("evidence") or ()
            ),
            extra=_extras(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.extra,
            "kind": self.kind,
            "status": self.status,
            "target_step": self.target_step,
            "prefix_preserved_through": self.prefix_preserved_through,
            "outcome_before": self.outcome_before,
            "outcome_after": self.outcome_after,
            "verifier": self.verifier,
            "run_id": self.run_id,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class DiagnosisRecord:
    failure_type: str
    applicability: str
    coverage: CoverageRecord
    origin_module: str = "unknown"
    stage: str = "unknown"
    detected_at_step: Optional[int] = None
    intervention_target_step: Optional[int] = None
    boundary_status: str = "unknown"
    agent_id: Optional[str] = None
    evidence: tuple[DiagnosisEvidence, ...] = ()
    causal_candidates: tuple[CausalCandidateRecord, ...] = ()
    intervention: Optional[InterventionRecord] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def causally_verified(self) -> bool:
        return bool(self.intervention and self.intervention.status == "verified")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiagnosisRecord":
        if not payload.get("failure_type"):
            raise ValueError("diagnosis failure_type must not be empty")
        known = {
            "failure_type",
            "applicability",
            "coverage",
            "origin_module",
            "stage",
            "detected_at_step",
            "intervention_target_step",
            "boundary_status",
            "agent_id",
            "evidence",
            "causal_candidates",
            "intervention",
            "causally_verified",
            "metadata",
        }
        intervention = payload.get("intervention")
        return cls(
            failure_type=str(payload["failure_type"]),
            applicability=str(payload.get("applicability") or "unknown"),
            coverage=CoverageRecord.from_dict(payload.get("coverage") or {}),
            origin_module=str(payload.get("origin_module") or "unknown"),
            stage=str(payload.get("stage") or "unknown"),
            detected_at_step=payload.get("detected_at_step"),
            intervention_target_step=payload.get("intervention_target_step"),
            boundary_status=str(payload.get("boundary_status") or "unknown"),
            agent_id=payload.get("agent_id"),
            evidence=tuple(
                DiagnosisEvidence.from_dict(item) for item in payload.get("evidence") or ()
            ),
            causal_candidates=tuple(
                CausalCandidateRecord.from_dict(item)
                for item in payload.get("causal_candidates") or ()
            ),
            intervention=(InterventionRecord.from_dict(intervention) if intervention else None),
            metadata=dict(payload.get("metadata") or {}),
            extra=_extras(payload, known),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.extra,
            "failure_type": self.failure_type,
            "applicability": self.applicability,
            "coverage": self.coverage.to_dict(),
            "origin_module": self.origin_module,
            "stage": self.stage,
            "detected_at_step": self.detected_at_step,
            "intervention_target_step": self.intervention_target_step,
            "boundary_status": self.boundary_status,
            "agent_id": self.agent_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "causal_candidates": [item.to_dict() for item in self.causal_candidates],
            "intervention": self.intervention.to_dict() if self.intervention else None,
            "causally_verified": self.causally_verified,
            "metadata": self.metadata,
        }


__all__ = [
    "CausalCandidateRecord",
    "CoverageRecord",
    "DiagnosisEvidence",
    "DiagnosisRecord",
    "InterventionRecord",
]
