"""Universal trace abstraction for Agent Forensics.

This module provides a framework-agnostic trace representation that works across:
- LangSmith/LangChain traces
- OpenTelemetry spans
- LangFuse observations
- n8n workflow logs
- Raw JSON agent logs
- CrewAI task logs
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class SpanType(Enum):
    """Type of span in the trace."""

    AGENT = "agent"  # Agent reasoning/orchestration
    TOOL_CALL = "tool_call"  # Tool/function invocation
    LLM_CALL = "llm_call"  # Raw LLM API call
    HANDOFF = "handoff"  # Agent-to-agent transfer
    CHAIN = "chain"  # LangChain chain execution
    RETRIEVAL = "retrieval"  # RAG retrieval step
    UNKNOWN = "unknown"


class SpanStatus(Enum):
    """Execution status of a span."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class StateSnapshot:
    """Portable detector input derived from a universal span."""

    agent_id: str
    state_delta: Dict[str, Any]
    content: str
    sequence_num: int = 0


@dataclass
class UniversalSpan:
    """Framework-agnostic span representation for Agent Forensics.

    This abstraction unifies traces from different sources into a common format
    that can be used by all detection algorithms.
    """

    # Core identifiers
    id: str
    trace_id: str
    name: str

    # Type and status
    span_type: SpanType
    status: SpanStatus = SpanStatus.OK

    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: int = 0

    # Hierarchy
    parent_id: Optional[str] = None
    children: List["UniversalSpan"] = field(default_factory=list)

    # Agent context (for multi-agent)
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None

    # Input/Output data
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)

    # LLM-specific fields
    prompt: Optional[str] = None
    response: Optional[str] = None
    model: Optional[str] = None
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0

    # Tool-specific fields
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None

    # Error handling
    error: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None

    # Source metadata
    source_format: str = "unknown"  # langsmith, otel, langfuse, n8n, raw
    raw_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.end_time and self.start_time:
            delta = (self.end_time - self.start_time).total_seconds()
            self.duration_ms = int(delta * 1000)
        if self.tokens_input or self.tokens_output:
            self.tokens_total = self.tokens_input + self.tokens_output

    @property
    def is_single_agent(self) -> bool:
        """Check if this span is from a single-agent system."""
        return self.span_type in [SpanType.TOOL_CALL, SpanType.LLM_CALL, SpanType.CHAIN]

    @property
    def is_multi_agent(self) -> bool:
        """Check if this span involves multi-agent coordination."""
        return self.span_type in [SpanType.AGENT, SpanType.HANDOFF]

    @property
    def has_error(self) -> bool:
        """Check if this span has an error."""
        return self.status == SpanStatus.ERROR or self.error is not None

    @property
    def content_hash(self) -> str:
        """Generate a hash of the span content for deduplication."""
        content = json.dumps(
            {
                "input": self.input_data,
                "output": self.output_data,
                "tool_name": self.tool_name,
                "tool_args": self.tool_args,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:16]  # nosec B324

    def to_state_snapshot(self) -> "StateSnapshot":
        """Convert to StateSnapshot format for detection algorithms.

        This provides compatibility with existing detection modules that
        expect the StateSnapshot dataclass.
        """
        text_fields = (
            ("Prompt", self.prompt),
            ("Response", self.response),
            ("Tool", self.tool_name),
        )
        json_fields = (
            ("Args", self.tool_args),
            ("Result", self.tool_result),
            ("Input", self.input_data),
            ("Output", self.output_data),
        )
        content_parts = [f"{label}: {value}" for label, value in text_fields if value]
        content_parts.extend(
            f"{label}: {json.dumps(value)}" for label, value in json_fields if value
        )

        content = "\n".join(content_parts) if content_parts else str(self.raw_data)

        return StateSnapshot(
            agent_id=self.agent_id or self.agent_name or "default_agent",
            state_delta=self.output_data or {},
            content=content,
            sequence_num=0,  # Will be set by caller
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "name": self.name,
            "span_type": self.span_type.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "parent_id": self.parent_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_total": self.tokens_total,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "error": self.error,
            "error_type": self.error_type,
            "source_format": self.source_format,
            "metadata": self.metadata,
        }


@dataclass
class UniversalTrace:
    """A complete trace containing multiple spans."""

    trace_id: str
    spans: List[UniversalSpan] = field(default_factory=list)

    # Trace-level metadata
    source_format: str = "unknown"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_ms: int = 0
    total_tokens: int = 0

    # Error summary
    has_errors: bool = False
    error_count: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived fields from spans."""
        if self.spans:
            self._calculate_summary()

    def _calculate_summary(self):
        """Calculate summary statistics from spans."""
        self.start_time = min(
            (span.start_time for span in self.spans if span.start_time),
            default=None,
        )
        self.end_time = max(
            (span.end_time for span in self.spans if span.end_time),
            default=None,
        )

        if self.start_time and self.end_time:
            delta = (self.end_time - self.start_time).total_seconds()
            self.total_duration_ms = int(delta * 1000)

        self.total_tokens = sum(s.tokens_total for s in self.spans)
        self.error_count = sum(1 for s in self.spans if s.has_error)
        self.has_errors = self.error_count > 0

    def add_span(self, span: UniversalSpan):
        """Add a span to the trace and recalculate summary."""
        self.spans.append(span)
        self._calculate_summary()

    def get_root_spans(self) -> List[UniversalSpan]:
        """Get top-level spans (no parent)."""
        return [s for s in self.spans if s.parent_id is None]

    def get_span_by_id(self, span_id: str) -> Optional[UniversalSpan]:
        """Find a span by its ID."""
        for span in self.spans:
            if span.id == span_id:
                return span
        return None

    def get_children(self, parent_id: str) -> List[UniversalSpan]:
        """Get all child spans of a parent."""
        return [s for s in self.spans if s.parent_id == parent_id]

    def get_tool_calls(self) -> List[UniversalSpan]:
        """Get all tool call spans."""
        return [s for s in self.spans if s.span_type == SpanType.TOOL_CALL]

    def get_llm_calls(self) -> List[UniversalSpan]:
        """Get all LLM call spans."""
        return [s for s in self.spans if s.span_type == SpanType.LLM_CALL]

    def get_errors(self) -> List[UniversalSpan]:
        """Get all spans with errors."""
        return [s for s in self.spans if s.has_error]

    def to_state_snapshots(self) -> List["StateSnapshot"]:
        """Convert all spans to StateSnapshot format for detection."""
        snapshots = []
        for i, span in enumerate(self.spans):
            snapshot = span.to_state_snapshot()
            # Update sequence number based on position
            snapshot.sequence_num = i
            snapshots.append(snapshot)
        return snapshots

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "source_format": self.source_format,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "has_errors": self.has_errors,
            "error_count": self.error_count,
            "metadata": self.metadata,
        }
