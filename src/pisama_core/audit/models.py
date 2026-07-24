"""Audit event models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Detection events
    DETECTION_RUN = "detection_run"
    ISSUE_DETECTED = "issue_detected"

    # Healing events
    FIX_RECOMMENDED = "fix_recommended"
    FIX_APPROVED = "fix_approved"
    FIX_APPLIED = "fix_applied"
    FIX_FAILED = "fix_failed"
    FIX_ROLLED_BACK = "fix_rolled_back"

    # Enforcement events
    DIRECTIVE_ISSUED = "directive_issued"
    COMPLIANCE_RECORDED = "compliance_recorded"
    VIOLATION_RECORDED = "violation_recorded"
    TOOL_BLOCKED = "tool_blocked"

    # Session events
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    SESSION_TERMINATED = "session_terminated"

    # System events
    CONFIG_CHANGED = "config_changed"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


@dataclass
class AuditEvent:
    """An audit event."""

    event_type: AuditEventType
    session_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Event details
    details: dict[str, Any] = field(default_factory=dict)

    # Context
    platform: str = "unknown"
    severity: int = 0

    # Actor
    actor: str = "system"  # system, user, agent

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": str(self.event_type),
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "platform": self.platform,
            "severity": self.severity,
            "actor": self.actor,
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        return cls(
            event_type=AuditEventType(data["event_type"]),
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            details=data.get("details", {}),
            platform=data.get("platform", "unknown"),
            severity=data.get("severity", 0),
            actor=data.get("actor", "system"),
        )
