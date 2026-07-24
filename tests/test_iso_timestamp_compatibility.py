"""Cross-version ISO 8601 parsing contracts."""

from datetime import timezone

from pisama_core import Span
from pisama_core.traces.models import Event, TraceMetadata


def test_z_suffix_is_supported_on_every_declared_python_version() -> None:
    timestamp = "2026-01-01T10:00:00Z"

    event = Event.from_dict({"name": "started", "timestamp": timestamp})
    span = Span.from_dict(
        {
            "name": "tool",
            "start_time": timestamp,
            "end_time": timestamp,
        }
    )
    metadata = TraceMetadata.from_dict({"created_at": timestamp})

    assert event.timestamp.tzinfo is timezone.utc
    assert span.start_time.tzinfo is timezone.utc
    assert span.end_time is not None
    assert span.end_time.tzinfo is timezone.utc
    assert metadata.created_at.tzinfo is timezone.utc
