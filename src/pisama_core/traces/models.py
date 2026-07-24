"""Core trace models for Pisama.

These models provide a universal format for representing agent execution
traces across all supported platforms (Claude Code, LangGraph, AutoGen,
CrewAI, n8n).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pisama_core.traces.enums import Platform, SpanKind, SpanStatus


def _generate_id() -> str:
    """Generate a unique ID."""
    return uuid4().hex[:16]


def _now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass
class Event:
    """An event that occurred during a span.

    Events are discrete occurrences within a span, such as log messages,
    state changes, or notable milestones.
    """

    name: str
    timestamp: datetime = field(default_factory=_now)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            attributes=data.get("attributes", {}),
        )


@dataclass
class Span:
    """A span representing a unit of work in the agent trace.

    Spans are the building blocks of traces. Each span represents a
    discrete operation (tool call, LLM inference, agent turn, etc.)
    and can have child spans for nested operations.
    """

    # Identity
    span_id: str = field(default_factory=_generate_id)
    parent_id: Optional[str] = None
    trace_id: Optional[str] = None

    # Basic info
    name: str = "unnamed"
    kind: SpanKind = SpanKind.SYSTEM

    # Platform
    platform: Platform = Platform.GENERIC
    platform_metadata: dict[str, Any] = field(default_factory=dict)

    # Timing
    start_time: datetime = field(default_factory=_now)
    end_time: Optional[datetime] = None

    # Status
    status: SpanStatus = SpanStatus.UNSET
    error_message: Optional[str] = None

    # Data
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)

    # Input/Output (for tool calls, LLM calls)
    input_data: Optional[dict[str, Any]] = None
    output_data: Optional[dict[str, Any]] = None

    def end(self, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None) -> None:
        """Mark the span as ended."""
        self.end_time = _now()
        self.status = status
        if error:
            self.error_message = error

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> Event:
        """Add an event to the span."""
        event = Event(name=name, attributes=attributes or {})
        self.events.append(event)
        return event

    @property
    def duration_ms(self) -> Optional[float]:
        """Duration in milliseconds, or None if not ended."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    @property
    def is_complete(self) -> bool:
        """Whether the span has ended."""
        return self.end_time is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "kind": str(self.kind),
            "platform": str(self.platform),
            "platform_metadata": self.platform_metadata,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": str(self.status),
            "error_message": self.error_message,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
            "input_data": self.input_data,
            "output_data": self.output_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Span":
        """Create from dictionary."""
        return cls(
            span_id=data.get("span_id", _generate_id()),
            parent_id=data.get("parent_id"),
            trace_id=data.get("trace_id"),
            name=data.get("name", "unnamed"),
            kind=SpanKind(data.get("kind", "system")),
            platform=Platform(data.get("platform", "generic")),
            platform_metadata=data.get("platform_metadata", {}),
            start_time=datetime.fromisoformat(data["start_time"])
            if "start_time" in data
            else _now(),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            status=SpanStatus(data.get("status", "unset")),
            error_message=data.get("error_message"),
            attributes=data.get("attributes", {}),
            events=[Event.from_dict(e) for e in data.get("events", [])],
            input_data=data.get("input_data"),
            output_data=data.get("output_data"),
        )


@dataclass
class TraceMetadata:
    """Metadata about a trace."""

    # Session info
    session_id: str = field(default_factory=_generate_id)
    user_id: Optional[str] = None

    # Platform info
    platform: Platform = Platform.GENERIC
    platform_version: Optional[str] = None

    # Environment
    environment: str = "development"
    host: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=_now)

    # Custom metadata
    tags: dict[str, str] = field(default_factory=dict)
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "platform": str(self.platform),
            "platform_version": self.platform_version,
            "environment": self.environment,
            "host": self.host,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceMetadata":
        """Create from dictionary."""
        return cls(
            session_id=data.get("session_id", _generate_id()),
            user_id=data.get("user_id"),
            platform=Platform(data.get("platform", "generic")),
            platform_version=data.get("platform_version"),
            environment=data.get("environment", "development"),
            host=data.get("host"),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else _now(),
            tags=data.get("tags", {}),
            custom=data.get("custom", {}),
        )


@dataclass
class Trace:
    """A complete trace of agent execution.

    A trace contains multiple spans representing the full execution
    history of an agent session or workflow.
    """

    trace_id: str = field(default_factory=_generate_id)
    spans: list[Span] = field(default_factory=list)
    metadata: TraceMetadata = field(default_factory=TraceMetadata)

    def add_span(self, span: Span) -> Span:
        """Add a span to the trace."""
        span.trace_id = self.trace_id
        self.spans.append(span)
        return span

    def create_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.SYSTEM,
        parent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Span:
        """Create and add a new span."""
        span = Span(
            name=name,
            kind=kind,
            parent_id=parent_id,
            trace_id=self.trace_id,
            platform=self.metadata.platform,
            **kwargs,
        )
        self.spans.append(span)
        return span

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a span by ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_root_spans(self) -> list[Span]:
        """Get all root spans (no parent)."""
        return [s for s in self.spans if s.parent_id is None]

    def get_children(self, span_id: str) -> list[Span]:
        """Get child spans of a span."""
        return [s for s in self.spans if s.parent_id == span_id]

    def get_spans_by_kind(self, kind: SpanKind) -> list[Span]:
        """Get all spans of a specific kind."""
        return [s for s in self.spans if s.kind == kind]

    def get_tool_sequence(self) -> list[str]:
        """Get the sequence of tool names called."""
        tool_spans = self.get_spans_by_kind(SpanKind.TOOL)
        sorted_spans = sorted(tool_spans, key=lambda s: s.start_time)
        return [s.name for s in sorted_spans]

    @property
    def duration_ms(self) -> Optional[float]:
        """Total duration of the trace in milliseconds."""
        if not self.spans:
            return None
        start = min(s.start_time for s in self.spans)
        ends = [s.end_time for s in self.spans if s.end_time]
        if not ends:
            return None
        end = max(ends)
        return (end - start).total_seconds() * 1000

    @property
    def span_count(self) -> int:
        """Number of spans in the trace."""
        return len(self.spans)

    @property
    def error_count(self) -> int:
        """Number of spans with errors."""
        return sum(1 for s in self.spans if s.status.is_failure)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trace":
        """Create from dictionary."""
        return cls(
            trace_id=data.get("trace_id", _generate_id()),
            spans=[Span.from_dict(s) for s in data.get("spans", [])],
            metadata=TraceMetadata.from_dict(data.get("metadata", {})),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "Trace":
        """Create from JSON string."""
        import json

        return cls.from_dict(json.loads(json_str))
