"""Security contracts for enforcement directives and durable audit records."""

from __future__ import annotations

import json
import stat
from datetime import datetime, timedelta, timezone

from pisama_core.audit.logger import AuditLogger
from pisama_core.audit.models import AuditEventType
from pisama_core.detection.result import DetectionResult
from pisama_core.healing.fixes.break_loop import BreakLoopFix
from pisama_core.healing.models import FixContext, FixResult
from pisama_core.injection.enforcement import EnforcementEngine, EnforcementLevel
from pisama_core.injection.protocol import Directive, FixInjectionProtocol
from pisama_core.traces.enums import Platform


def test_audit_log_records_security_lifecycle_with_owner_only_permissions(tmp_path) -> None:
    logger = AuditLogger(log_dir=tmp_path / "audit")
    clean = DetectionResult.no_issue("injection")
    attack = DetectionResult.issue_found("injection", 95, "Prompt injection detected")
    applied = FixResult(True, "terminate", "Session terminated", ["Stopped agent"])
    failed = FixResult(False, "terminate", "Termination failed", error="adapter unavailable")

    logger.log_detection(clean, "session-clean", "generic")
    logger.log_detection(attack, "session-attack", "generic")
    logger.log_fix_applied(applied, "session-attack", "generic")
    logger.log_fix_applied(failed, "session-attack", "generic")
    logger.log_directive("fix-1234", "terminate", "session-attack", "generic")
    logger.log_compliance("fix-1234", True, "session-attack", "generic")
    logger.log_compliance("fix-5678", False, "session-attack", "generic")
    logger.log_block("Bash", "Prompt injection", "session-attack", "generic")

    assert stat.S_IMODE(logger.log_file.stat().st_mode) == 0o600
    events = logger.get_events(session_id="session-attack")
    assert [event.event_type for event in events] == [
        AuditEventType.ISSUE_DETECTED,
        AuditEventType.FIX_APPLIED,
        AuditEventType.FIX_FAILED,
        AuditEventType.DIRECTIVE_ISSUED,
        AuditEventType.COMPLIANCE_RECORDED,
        AuditEventType.VIOLATION_RECORDED,
        AuditEventType.TOOL_BLOCKED,
    ]
    assert logger.get_events(event_type=AuditEventType.TOOL_BLOCKED)[0].details == {
        "tool": "Bash",
        "reason": "Prompt injection",
    }


def test_audit_reader_tolerates_corruption_filters_time_and_enforces_limit(tmp_path) -> None:
    logger = AuditLogger(log_dir=tmp_path)
    first = logger.log(AuditEventType.DETECTION_RUN, "session", {"index": 1})
    with logger.log_file.open("a") as log:
        log.write("\n")
        log.write("{broken-json\n")
        log.write(json.dumps({"event_type": "unknown", "session_id": "session"}) + "\n")
    second = logger.log(AuditEventType.DETECTION_RUN, "session", {"index": 2})

    assert logger.get_events(since=first.timestamp + timedelta(microseconds=1)) == [second]
    assert logger.get_events(limit=1) == [first]
    assert AuditLogger(log_dir=tmp_path / "empty").get_events() == []


def test_enforcement_engine_escalates_blocks_deescalates_and_resets() -> None:
    engine = EnforcementEngine(max_violations_before_escalation=1)
    session = "session-security"
    engine.add_directive(session, "fix-1")
    engine.add_directive(session, "fix-1")

    assert engine.record_violation(session, "Bash") is EnforcementLevel.DIRECT
    assert engine.get_level(90, session) is EnforcementLevel.TERMINATE
    engine.record_violation(session, "Bash")
    assert engine.should_block(session, "Bash") == (
        True,
        "Tool 'Bash' blocked. Follow pending directives.",
    )
    assert engine.should_block(session, "Read") == (
        True,
        "Pending directives must be addressed first",
    )

    engine.record_violation(session)
    assert engine.should_block(session, "Read") == (
        True,
        "Session terminated due to non-compliance",
    )
    assert engine.record_compliance(session, "fix-1") is EnforcementLevel.BLOCK
    assert engine.get_stats(session) == {
        "level": "block",
        "violations": 3,
        "compliances": 1,
        "pending_directives": 0,
        "blocked_tools": [],
    }
    engine.reset(session)
    assert engine.get_stats(session)["violations"] == 0


def test_fix_protocol_tracks_real_break_loop_directive_and_multiline_format() -> None:
    protocol = FixInjectionProtocol()
    context = FixContext(platform=Platform.GENERIC, session_id="session-security")
    directive = protocol.create_directive(
        BreakLoopFix(),
        context,
        reason="Repeated tool execution",
        level=EnforcementLevel.BLOCK,
    )
    formatted = protocol.format_directive(directive)

    assert directive.priority == "CRITICAL"
    assert directive.action == "break_loop"
    assert protocol.get_directive(directive.directive_id) is directive
    assert directive.directive_id in formatted
    assert "STOP the current loop" in formatted
    assert protocol.clear_directive(directive.directive_id)
    assert protocol.get_directive(directive.directive_id) is None
    assert not protocol.clear_directive(directive.directive_id)

    multiline = protocol.format_simple(
        "switch_strategy",
        "Stop current work.\nTry another strategy.",
        "Loop risk",
    )
    assert "\n║   Try another strategy." in multiline


def test_directive_serialization_includes_expiration() -> None:
    created = datetime(2026, 7, 23, tzinfo=timezone.utc)
    expires = created + timedelta(minutes=5)
    directive = Directive(
        directive_id="fix-expiring",
        priority="HIGH",
        action="break_loop",
        instruction="Stop",
        reason="Loop",
        created_at=created,
        expires_at=expires,
    )

    assert directive.to_dict() == {
        "directive_id": "fix-expiring",
        "priority": "HIGH",
        "action": "break_loop",
        "instruction": "Stop",
        "reason": "Loop",
        "created_at": created.isoformat(),
        "expires_at": expires.isoformat(),
    }
