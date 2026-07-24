"""Audit logger for Pisama."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pisama_core.audit.models import AuditEvent, AuditEventType
from pisama_core.detection.result import DetectionResult
from pisama_core.healing.models import FixResult


class AuditLogger:
    """Logger for audit events.

    Logs to a JSONL file by default, with optional additional backends.

    Example:
        logger = AuditLogger(log_dir=Path("~/.pisama/audit"))

        logger.log_detection(result, session_id="abc123")
        logger.log_fix_applied(fix_result, session_id="abc123")
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_file: str = "audit_log.jsonl",
    ) -> None:
        self.log_dir = log_dir or Path.home() / ".pisama" / "audit"
        self.log_file = self.log_dir / log_file
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Ensure log directory exists."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _write_event(self, event: AuditEvent) -> None:
        """Write an event to the log file."""
        with open(self.log_file, "a") as f:
            f.write(event.to_json() + "\n")

    def log(
        self,
        event_type: AuditEventType,
        session_id: str,
        details: Optional[dict[str, Any]] = None,
        platform: str = "unknown",
        severity: int = 0,
        actor: str = "system",
    ) -> AuditEvent:
        """Log an audit event.

        Args:
            event_type: Type of event
            session_id: Session identifier
            details: Event details
            platform: Platform name
            severity: Severity level
            actor: Who/what triggered the event

        Returns:
            The created AuditEvent
        """
        event = AuditEvent(
            event_type=event_type,
            session_id=session_id,
            details=details or {},
            platform=platform,
            severity=severity,
            actor=actor,
        )
        self._write_event(event)
        return event

    def log_detection(
        self,
        result: DetectionResult,
        session_id: str,
        platform: str = "unknown",
    ) -> AuditEvent:
        """Log a detection result."""
        event_type = (
            AuditEventType.ISSUE_DETECTED if result.detected else AuditEventType.DETECTION_RUN
        )

        return self.log(
            event_type=event_type,
            session_id=session_id,
            details={
                "detector": result.detector_name,
                "detected": result.detected,
                "severity": result.severity,
                "summary": result.summary,
            },
            platform=platform,
            severity=result.severity,
        )

    def log_fix_applied(
        self,
        result: FixResult,
        session_id: str,
        platform: str = "unknown",
    ) -> AuditEvent:
        """Log a fix application."""
        event_type = AuditEventType.FIX_APPLIED if result.success else AuditEventType.FIX_FAILED

        return self.log(
            event_type=event_type,
            session_id=session_id,
            details={
                "fix_type": result.fix_type,
                "success": result.success,
                "message": result.message,
                "changes": result.changes_made,
                "error": result.error,
            },
            platform=platform,
        )

    def log_directive(
        self,
        directive_id: str,
        action: str,
        session_id: str,
        platform: str = "unknown",
    ) -> AuditEvent:
        """Log a directive being issued."""
        return self.log(
            event_type=AuditEventType.DIRECTIVE_ISSUED,
            session_id=session_id,
            details={
                "directive_id": directive_id,
                "action": action,
            },
            platform=platform,
        )

    def log_compliance(
        self,
        directive_id: str,
        complied: bool,
        session_id: str,
        platform: str = "unknown",
    ) -> AuditEvent:
        """Log compliance with a directive."""
        event_type = (
            AuditEventType.COMPLIANCE_RECORDED if complied else AuditEventType.VIOLATION_RECORDED
        )

        return self.log(
            event_type=event_type,
            session_id=session_id,
            details={
                "directive_id": directive_id,
                "complied": complied,
            },
            platform=platform,
        )

    def log_block(
        self,
        tool_name: str,
        reason: str,
        session_id: str,
        platform: str = "unknown",
    ) -> AuditEvent:
        """Log a tool being blocked."""
        return self.log(
            event_type=AuditEventType.TOOL_BLOCKED,
            session_id=session_id,
            details={
                "tool": tool_name,
                "reason": reason,
            },
            platform=platform,
        )

    def get_events(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Get audit events matching criteria."""
        events: list[AuditEvent] = []

        if not self.log_file.exists():
            return events

        import json

        with open(self.log_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    event = AuditEvent.from_dict(data)

                    # Apply filters
                    if session_id and event.session_id != session_id:
                        continue
                    if event_type and event.event_type != event_type:
                        continue
                    if since and event.timestamp < since:
                        continue

                    events.append(event)

                    if len(events) >= limit:
                        break
                except (json.JSONDecodeError, KeyError):
                    continue

        return events
