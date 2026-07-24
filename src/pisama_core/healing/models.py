"""Models for the healing engine."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from pisama_core.detection.result import FixRecommendation
from pisama_core.traces.enums import Platform

if TYPE_CHECKING:
    from pisama_core.adapters.base import PlatformAdapter


@dataclass
class FixContext:
    """Context for applying a fix."""

    # Platform
    platform: Platform
    session_id: str

    # Current state
    current_state: dict[str, Any] = field(default_factory=dict)

    # Adapter (injected by platform layer)
    adapter: Optional["PlatformAdapter"] = None

    # Fix tracking
    fix_id: Optional[str] = None
    attempt_number: int = 1

    # Previous attempts
    previous_fixes: list[str] = field(default_factory=list)


@dataclass
class FixResult:
    """Result of applying a fix."""

    success: bool
    fix_type: str
    message: str

    # Details
    changes_made: list[str] = field(default_factory=list)
    rollback_possible: bool = True

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: float = 0.0

    # Errors
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "fix_type": self.fix_type,
            "message": self.message,
            "changes_made": self.changes_made,
            "rollback_possible": self.rollback_possible,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
        }


@dataclass
class RollbackResult:
    """Result of rolling back a fix."""

    success: bool
    message: str
    restored_state: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class HealingPlan:
    """Plan for healing a detected issue."""

    # Primary fix
    primary_fix: FixRecommendation

    # Fallbacks
    fallback_fixes: list[FixRecommendation] = field(default_factory=list)

    # Requirements
    requires_approval: bool = True
    estimated_impact: str = "medium"

    # Context
    detection_summary: str = ""
    severity: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_fix": self.primary_fix.to_dict(),
            "fallback_fixes": [f.to_dict() for f in self.fallback_fixes],
            "requires_approval": self.requires_approval,
            "estimated_impact": self.estimated_impact,
            "detection_summary": self.detection_summary,
            "severity": self.severity,
        }
