"""Pytest fixtures for pisama-core tests."""

import pytest

from pisama_core.traces.enums import Platform, SpanKind, SpanStatus
from pisama_core.traces.models import Span, Trace, TraceMetadata


@pytest.fixture
def sample_span() -> Span:
    """Create a sample span for testing."""
    return Span(
        span_id="span-001",
        name="Read",
        kind=SpanKind.TOOL,
        platform=Platform.CLAUDE_CODE,
        input_data={"file_path": "/tmp/test.py"},
        output_data={"content": "print('hello')"},
    )


@pytest.fixture
def sample_trace() -> Trace:
    """Create a sample trace with multiple spans."""
    trace = Trace(
        trace_id="trace-001",
        metadata=TraceMetadata(
            session_id="session-001",
            platform=Platform.CLAUDE_CODE,
        ),
    )

    # Add some tool spans
    tools = ["Read", "Edit", "Read", "Bash", "Read"]
    for i, tool_name in enumerate(tools):
        span = trace.create_span(
            name=tool_name,
            kind=SpanKind.TOOL,
        )
        span.span_id = f"span-{i:03d}"
        span.end(SpanStatus.OK)

    return trace


@pytest.fixture
def looping_trace() -> Trace:
    """Create a trace with a loop pattern."""
    trace = Trace(
        trace_id="trace-loop",
        metadata=TraceMetadata(
            session_id="session-loop",
            platform=Platform.CLAUDE_CODE,
        ),
    )

    # Create a loop: Read repeated 10 times
    for i in range(10):
        span = trace.create_span(
            name="Read",
            kind=SpanKind.TOOL,
        )
        span.input_data = {"file_path": "/tmp/same.py"}
        span.end(SpanStatus.OK)

    return trace


@pytest.fixture
def error_trace() -> Trace:
    """Create a trace with errors."""
    trace = Trace(
        trace_id="trace-error",
        metadata=TraceMetadata(
            session_id="session-error",
            platform=Platform.CLAUDE_CODE,
        ),
    )

    # Success span
    span1 = trace.create_span(name="Read", kind=SpanKind.TOOL)
    span1.end(SpanStatus.OK)

    # Error span
    span2 = trace.create_span(name="Bash", kind=SpanKind.TOOL)
    span2.end(SpanStatus.ERROR, error="Command failed with exit code 1")

    # Another success
    span3 = trace.create_span(name="Read", kind=SpanKind.TOOL)
    span3.end(SpanStatus.OK)

    return trace
