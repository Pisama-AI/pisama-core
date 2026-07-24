"""Tests for pisama_core.audit module."""

import json
import tempfile
from pathlib import Path

from pisama_core.audit.logger import AuditLogger
from pisama_core.audit.models import AuditEvent, AuditEventType


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_event_types(self):
        """Test event type values."""
        assert AuditEventType.ISSUE_DETECTED.value == "issue_detected"
        assert AuditEventType.FIX_APPLIED.value == "fix_applied"
        assert AuditEventType.FIX_FAILED.value == "fix_failed"
        assert AuditEventType.DETECTION_RUN.value == "detection_run"


class TestAuditEvent:
    """Tests for AuditEvent model."""

    def test_create_event(self):
        """Test basic event creation."""
        event = AuditEvent(
            event_type=AuditEventType.ISSUE_DETECTED,
            session_id="session-1",
            details={"severity": 50},
        )
        assert event.event_type == AuditEventType.ISSUE_DETECTED
        assert event.session_id == "session-1"
        assert event.timestamp is not None

    def test_event_to_dict(self):
        """Test event serialization."""
        event = AuditEvent(
            event_type=AuditEventType.DIRECTIVE_ISSUED,
            session_id="session-1",
            details={"action": "blocked"},
        )
        data = event.to_dict()
        assert data["event_type"] == "directive_issued"
        assert data["session_id"] == "session-1"
        assert "timestamp" in data

    def test_event_to_json(self):
        """Test event JSON serialization."""
        event = AuditEvent(
            event_type=AuditEventType.FIX_APPLIED,
            session_id="session-1",
            details={"fix_type": "break_loop"},
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "fix_applied"


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_create_logger(self):
        """Test creating audit logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = AuditLogger(log_dir=log_dir)
            assert logger is not None

    def test_log_event(self):
        """Test logging an event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = AuditLogger(log_dir=log_dir)

            logger.log(
                event_type=AuditEventType.DETECTION_RUN,
                session_id="session-1",
                details={"severity": 50, "issues": ["Loop detected"]},
            )

            assert logger.log_file.exists()
            with open(logger.log_file) as f:
                line = f.readline()
                event = json.loads(line)
                assert event["event_type"] == "detection_run"

    def test_log_multiple_events(self):
        """Test logging multiple events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = AuditLogger(log_dir=log_dir)

            logger.log(AuditEventType.DETECTION_RUN, "session-1", {"severity": 40})
            logger.log(AuditEventType.DIRECTIVE_ISSUED, "session-1", {"action": "warning"})
            logger.log(AuditEventType.FIX_APPLIED, "session-1", {"fix_type": "break_loop"})

            with open(logger.log_file) as f:
                lines = f.readlines()
                assert len(lines) == 3

    def test_get_events(self):
        """Test getting events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = AuditLogger(log_dir=log_dir)

            for i in range(5):
                logger.log(AuditEventType.DETECTION_RUN, "session-1", {"index": i})

            events = logger.get_events(limit=3)
            assert len(events) <= 3

    def test_get_events_by_session(self):
        """Test getting events for a session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = AuditLogger(log_dir=log_dir)

            logger.log(AuditEventType.ISSUE_DETECTED, "session-1", {"severity": 40})
            logger.log(AuditEventType.ISSUE_DETECTED, "session-2", {"severity": 50})
            logger.log(AuditEventType.DIRECTIVE_ISSUED, "session-1", {"action": "block"})

            session1_events = logger.get_events(session_id="session-1")
            assert len(session1_events) == 2
