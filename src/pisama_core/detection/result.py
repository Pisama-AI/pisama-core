"""Detection result models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pisama_core.detection.diagnosis import DiagnosisRecord


class FixType(str, Enum):
    """Types of fixes that can be recommended."""

    BREAK_LOOP = "break_loop"
    SWITCH_STRATEGY = "switch_strategy"
    ADD_DELAY = "add_delay"
    CACHE_RESULT = "cache_result"
    ESCALATE = "escalate"
    ROLLBACK = "rollback"
    RESET_CONTEXT = "reset_context"
    TERMINATE = "terminate"

    def __str__(self) -> str:
        return self.value


@dataclass
class Evidence:
    """Evidence supporting a detection."""

    description: str
    span_ids: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    # Location in trace
    start_index: Optional[int] = None
    end_index: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "span_ids": self.span_ids,
            "data": self.data,
            "start_index": self.start_index,
            "end_index": self.end_index,
        }


@dataclass
class FixRecommendation:
    """Recommended fix for a detected issue."""

    fix_type: FixType
    instruction: str
    priority: int = 1  # 1 = highest priority

    # Fix-specific parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Approval requirements
    requires_approval: bool = True
    auto_approved: bool = False

    # Estimated impact
    estimated_success_rate: float = 0.8
    reversible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "fix_type": str(self.fix_type),
            "instruction": self.instruction,
            "priority": self.priority,
            "parameters": self.parameters,
            "requires_approval": self.requires_approval,
            "auto_approved": self.auto_approved,
            "estimated_success_rate": self.estimated_success_rate,
            "reversible": self.reversible,
        }


@dataclass
class DetectionResult:
    """Result from a detector."""

    # Identity
    detector_name: str
    detector_version: str = "1.0.0"

    # Detection outcome
    detected: bool = False
    severity: int = 0  # 0-100
    confidence: float = 1.0  # 0.0-1.0

    # Details
    summary: str = ""
    evidence: list[Evidence] = field(default_factory=list)

    # Recommendations
    recommendation: Optional[FixRecommendation] = None
    alternative_recommendations: list[FixRecommendation] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnosis_record: Optional[dict[str, Any]] = None

    @property
    def has_recommendation(self) -> bool:
        """Whether this result includes a fix recommendation."""
        return self.recommendation is not None

    @property
    def diagnosis(self) -> Optional[DiagnosisRecord]:
        """Typed view of the optional server diagnosis payload."""

        return DiagnosisRecord.from_dict(self.diagnosis_record) if self.diagnosis_record else None

    def set_diagnosis(self, diagnosis: DiagnosisRecord) -> None:
        self.diagnosis_record = diagnosis.to_dict()

    @property
    def all_recommendations(self) -> list[FixRecommendation]:
        """Get all recommendations (primary + alternatives)."""
        recs = []
        if self.recommendation:
            recs.append(self.recommendation)
        recs.extend(self.alternative_recommendations)
        return recs

    def add_evidence(
        self,
        description: str,
        span_ids: Optional[list[str]] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> Evidence:
        """Add evidence to this result."""
        evidence = Evidence(
            description=description,
            span_ids=span_ids or [],
            data=data or {},
        )
        self.evidence.append(evidence)
        return evidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "detected": self.detected,
            "severity": self.severity,
            "confidence": self.confidence,
            "summary": self.summary,
            "evidence": [e.to_dict() for e in self.evidence],
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
            "alternative_recommendations": [r.to_dict() for r in self.alternative_recommendations],
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
            "diagnosis_record": self.diagnosis_record,
        }

    @classmethod
    def no_issue(cls, detector_name: str) -> "DetectionResult":
        """Create a result indicating no issue was detected."""
        return cls(
            detector_name=detector_name,
            detected=False,
            severity=0,
            summary="No issues detected",
        )

    @classmethod
    def issue_found(
        cls,
        detector_name: str,
        severity: int,
        summary: str,
        fix_type: Optional[FixType] = None,
        fix_instruction: Optional[str] = None,
    ) -> "DetectionResult":
        """Create a result indicating an issue was found."""
        result = cls(
            detector_name=detector_name,
            detected=True,
            severity=min(100, max(0, severity)),
            summary=summary,
        )

        if fix_type and fix_instruction:
            result.recommendation = FixRecommendation(
                fix_type=fix_type,
                instruction=fix_instruction,
            )

        return result
