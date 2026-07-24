"""Tests for pisama_core.traces module."""

from datetime import datetime, timezone

from pisama_core.traces.enums import Platform, SpanKind, SpanStatus
from pisama_core.traces.models import Event, Span, Trace, TraceMetadata


class TestEvent:
    """Tests for Event model."""

    def test_create_event(self):
        """Test basic event creation."""
        event = Event(name="test_event")
        assert event.name == "test_event"
        assert isinstance(event.timestamp, datetime)
        assert event.attributes == {}

    def test_event_with_attributes(self):
        """Test event with attributes."""
        attrs = {"key": "value", "count": 42}
        event = Event(name="test", attributes=attrs)
        assert event.attributes == attrs

    def test_event_to_dict(self):
        """Test event serialization."""
        event = Event(name="test", attributes={"foo": "bar"})
        data = event.to_dict()
        assert data["name"] == "test"
        assert "timestamp" in data
        assert data["attributes"] == {"foo": "bar"}

    def test_event_from_dict(self):
        """Test event deserialization."""
        data = {
            "name": "test",
            "timestamp": "2025-01-01T12:00:00+00:00",
            "attributes": {"key": "value"},
        }
        event = Event.from_dict(data)
        assert event.name == "test"
        assert event.attributes == {"key": "value"}


class TestSpan:
    """Tests for Span model."""

    def test_create_span(self):
        """Test basic span creation."""
        span = Span(name="Read")
        assert span.name == "Read"
        assert span.kind == SpanKind.SYSTEM
        assert span.status == SpanStatus.UNSET
        assert span.end_time is None

    def test_span_with_kind(self):
        """Test span with specific kind."""
        span = Span(name="tool_call", kind=SpanKind.TOOL)
        assert span.kind == SpanKind.TOOL

    def test_span_end(self):
        """Test ending a span."""
        span = Span(name="test")
        assert not span.is_complete

        span.end(SpanStatus.OK)
        assert span.is_complete
        assert span.status == SpanStatus.OK
        assert span.end_time is not None

    def test_span_end_with_error(self):
        """Test ending a span with error."""
        span = Span(name="test")
        span.end(SpanStatus.ERROR, error="Something went wrong")
        assert span.status == SpanStatus.ERROR
        assert span.error_message == "Something went wrong"

    def test_span_duration(self):
        """Test span duration calculation."""
        span = Span(name="test")
        span.start_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        span.end_time = datetime(2025, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
        assert span.duration_ms == 1000.0

    def test_span_duration_none_if_not_ended(self):
        """Test that duration is None if span not ended."""
        span = Span(name="test")
        assert span.duration_ms is None

    def test_span_add_event(self):
        """Test adding events to span."""
        span = Span(name="test")
        span.add_event("checkpoint", {"status": "ok"})
        assert len(span.events) == 1
        assert span.events[0].name == "checkpoint"

    def test_span_to_dict(self):
        """Test span serialization."""
        span = Span(
            span_id="test-id",
            name="Read",
            kind=SpanKind.TOOL,
            platform=Platform.CLAUDE_CODE,
        )
        data = span.to_dict()
        assert data["span_id"] == "test-id"
        assert data["name"] == "Read"
        assert "tool" in data["kind"].lower()

    def test_span_from_dict(self):
        """Test span deserialization."""
        data = {
            "span_id": "test-id",
            "name": "Edit",
            "kind": "tool",
            "platform": "claude_code",
            "start_time": "2025-01-01T12:00:00+00:00",
        }
        span = Span.from_dict(data)
        assert span.span_id == "test-id"
        assert span.name == "Edit"


class TestTrace:
    """Tests for Trace model."""

    def test_create_trace(self):
        """Test basic trace creation."""
        trace = Trace()
        assert trace.trace_id is not None
        assert trace.spans == []
        assert trace.metadata is not None

    def test_trace_add_span(self):
        """Test adding spans to trace."""
        trace = Trace(trace_id="test-trace")
        span = Span(name="Read")
        trace.add_span(span)

        assert len(trace.spans) == 1
        assert span.trace_id == "test-trace"

    def test_trace_create_span(self):
        """Test creating spans via trace."""
        trace = Trace(trace_id="test-trace")
        span = trace.create_span(name="Read", kind=SpanKind.TOOL)

        assert span.name == "Read"
        assert span.trace_id == "test-trace"
        assert len(trace.spans) == 1

    def test_trace_get_span(self):
        """Test getting span by ID."""
        trace = Trace()
        span = Span(span_id="specific-id", name="test")
        trace.add_span(span)

        found = trace.get_span("specific-id")
        assert found is not None
        assert found.name == "test"

        not_found = trace.get_span("nonexistent")
        assert not_found is None

    def test_trace_get_root_spans(self):
        """Test getting root spans."""
        trace = Trace()
        root1 = Span(span_id="root1", name="root1")
        root2 = Span(span_id="root2", name="root2")
        child = Span(span_id="child", name="child", parent_id="root1")

        trace.add_span(root1)
        trace.add_span(root2)
        trace.add_span(child)

        roots = trace.get_root_spans()
        assert len(roots) == 2
        assert all(s.parent_id is None for s in roots)

    def test_trace_get_children(self):
        """Test getting child spans."""
        trace = Trace()
        parent = Span(span_id="parent", name="parent")
        child1 = Span(span_id="child1", name="child1", parent_id="parent")
        child2 = Span(span_id="child2", name="child2", parent_id="parent")

        trace.add_span(parent)
        trace.add_span(child1)
        trace.add_span(child2)

        children = trace.get_children("parent")
        assert len(children) == 2

    def test_trace_get_tool_sequence(self, sample_trace):
        """Test getting tool sequence."""
        sequence = sample_trace.get_tool_sequence()
        assert sequence == ["Read", "Edit", "Read", "Bash", "Read"]

    def test_trace_span_count(self, sample_trace):
        """Test span count property."""
        assert sample_trace.span_count == 5

    def test_trace_error_count(self, error_trace):
        """Test error count property."""
        assert error_trace.error_count == 1

    def test_trace_to_dict(self, sample_trace):
        """Test trace serialization."""
        data = sample_trace.to_dict()
        assert data["trace_id"] == "trace-001"
        assert len(data["spans"]) == 5
        assert "metadata" in data

    def test_trace_from_dict(self):
        """Test trace deserialization."""
        data = {
            "trace_id": "test-trace",
            "spans": [{"span_id": "s1", "name": "Read", "start_time": "2025-01-01T12:00:00+00:00"}],
            "metadata": {
                "session_id": "sess-1",
                "platform": "claude_code",
            },
        }
        trace = Trace.from_dict(data)
        assert trace.trace_id == "test-trace"
        assert len(trace.spans) == 1
        assert trace.metadata.session_id == "sess-1"

    def test_trace_json_roundtrip(self, sample_trace):
        """Test JSON serialization roundtrip."""
        json_str = sample_trace.to_json()
        restored = Trace.from_json(json_str)
        assert restored.trace_id == sample_trace.trace_id
        assert len(restored.spans) == len(sample_trace.spans)


class TestTraceMetadata:
    """Tests for TraceMetadata model."""

    def test_create_metadata(self):
        """Test basic metadata creation."""
        meta = TraceMetadata()
        assert meta.session_id is not None
        assert meta.platform == Platform.GENERIC
        assert meta.environment == "development"

    def test_metadata_with_platform(self):
        """Test metadata with specific platform."""
        meta = TraceMetadata(platform=Platform.CLAUDE_CODE)
        assert meta.platform == Platform.CLAUDE_CODE

    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        meta = TraceMetadata(
            session_id="sess-1",
            platform=Platform.LANGGRAPH,
            tags={"env": "test"},
        )
        data = meta.to_dict()
        assert data["session_id"] == "sess-1"
        assert data["tags"] == {"env": "test"}

    def test_metadata_from_dict(self):
        """Test metadata deserialization."""
        data = {
            "session_id": "sess-1",
            "platform": "autogen",
            "environment": "production",
        }
        meta = TraceMetadata.from_dict(data)
        assert meta.session_id == "sess-1"
        assert meta.platform == Platform.AUTOGEN
        assert meta.environment == "production"
