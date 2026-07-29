"""OpenAI Agents SDK trace ingestion adapter.

Converts traces emitted by the OpenAI Agents SDK (the `openai-agents`
package: `Agent`, `Runner`, handoffs, guardrails) into Pisama's universal
Span/Trace format.

This is a different API surface from `adapters/openai.py`, which handles
the Assistants API and the Responses API. The Agents SDK is the migration
target for OpenAI Agent Builder, which shuts down on 30 November 2026, so
the two are expected to coexist rather than one replacing the other in
this adapter set.

Supported input is the SDK's own exported tracing payload, i.e. the dicts
a `TracingProcessor` / `TracingExporter` receives:

- trace:  ``{"object": "trace", "id", "workflow_name", "group_id", "metadata"}``
- span:   ``{"object": "trace.span", "id", "trace_id", "parent_id",
             "started_at", "ended_at", "span_data": {...}, "error": {...}}``

The adapter does not import the `openai-agents` package. It accepts the
exported dicts so callers can pass whatever their SDK version produces,
matching how `adapters/openai.py` avoids a hard `openai` dependency.

Ingestion-only. The Agents SDK exposes input/output guardrails for
blocking, but those are the caller's to configure; this adapter reads
what already happened.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from pisama_core.traces.enums import Platform, SpanKind, SpanStatus
from pisama_core.traces.models import Span, Trace, TraceMetadata
from pisama_core.utils.time_utils import parse_iso_datetime

__all__ = [
    "PisamaTracingProcessor",
    "parse_agents_trace",
    "parse_agents_span",
]


# `span_data.type` -> Pisama SpanKind. The Agents SDK's own type strings
# are the stable contract here; see span_data.py in openai-agents-python.
_SPAN_KIND_BY_TYPE: dict[str, SpanKind] = {
    "agent": SpanKind.AGENT,
    "turn": SpanKind.AGENT_TURN,
    "task": SpanKind.TASK,
    "handoff": SpanKind.HANDOFF,
    "function": SpanKind.TOOL,
    "mcp_tools": SpanKind.TOOL,
    "generation": SpanKind.LLM,
    "response": SpanKind.LLM,
    "guardrail": SpanKind.SYSTEM,
    "custom": SpanKind.SYSTEM,
    "transcription": SpanKind.SYSTEM,
    "speech": SpanKind.SYSTEM,
    "speech_group": SpanKind.SYSTEM,
}


def parse_agents_trace(
    trace: dict[str, Any],
    spans: Optional[Iterable[dict[str, Any]]] = None,
) -> Trace:
    """Parse an Agents SDK exported trace + its spans into a Pisama Trace.

    Args:
        trace: The exported ``{"object": "trace", ...}`` dict. `id` and
            `workflow_name` are read; `group_id` and `metadata` are
            preserved on the trace metadata when present.
        spans: Iterable of exported ``{"object": "trace.span", ...}``
            dicts. Parent/child structure is carried through `parent_id`
            unchanged, so the SDK's own span tree survives ingestion.

    Returns:
        A `Trace` whose spans mirror the SDK's spans one for one. No
        synthetic root span is inserted: the Agents SDK already emits a
        root `agent` span per run, and adding another would change the
        depth that structural detectors measure.
    """
    trace_id = str(trace.get("id") or "") or Trace().trace_id
    workflow_name = trace.get("workflow_name")

    parsed = Trace(
        trace_id=trace_id,
        metadata=TraceMetadata(
            session_id=str(trace.get("group_id") or trace_id),
            platform=Platform.OPENAI,
            # Distinguishes this from "assistants-v2" / "responses-v1" in
            # adapters/openai.py. Same vendor, different API surface.
            platform_version="agents-sdk-v1",
            custom={
                "workflow_name": workflow_name,
                "group_id": trace.get("group_id"),
                "trace_metadata": trace.get("metadata"),
            },
        ),
    )

    for span in spans or []:
        parsed.spans.append(parse_agents_span(span, trace_id))

    return parsed


def parse_agents_span(span: dict[str, Any], trace_id: Optional[str] = None) -> Span:
    """Convert one exported Agents SDK span into a Pisama Span."""
    data = span.get("span_data") or {}
    span_type = str(data.get("type") or "custom")
    kind = _SPAN_KIND_BY_TYPE.get(span_type, SpanKind.SYSTEM)

    error = span.get("error")
    started = _ts(span.get("started_at"))
    ended = _ts(span.get("ended_at"))

    attributes: dict[str, Any] = {
        "openai_agents.span.type": span_type,
        "gen_ai.system": "openai",
    }
    input_data: Optional[dict[str, Any]] = None
    output_data: Optional[dict[str, Any]] = None

    if span_type == "agent":
        # `handoffs` and `tools` are the declared capability of the agent,
        # not calls that happened. Keeping them distinct from the handoff
        # spans below is what lets a detector ask "was a declared handoff
        # never taken?" rather than only "which handoffs fired?".
        attributes["openai_agents.agent.name"] = data.get("name")
        attributes["openai_agents.agent.handoffs"] = data.get("handoffs") or []
        attributes["openai_agents.agent.tools"] = data.get("tools") or []
        attributes["openai_agents.agent.output_type"] = data.get("output_type")

    elif span_type == "handoff":
        attributes["openai_agents.handoff.from"] = data.get("from_agent")
        attributes["openai_agents.handoff.to"] = data.get("to_agent")

    elif span_type == "function":
        attributes["openai_agents.tool.name"] = data.get("name")
        input_data = {"arguments": data.get("input")}
        output_data = {"output": data.get("output")}
        if data.get("mcp_data") is not None:
            attributes["openai_agents.tool.mcp_data"] = data.get("mcp_data")

    elif span_type == "generation":
        attributes["openai_agents.model"] = data.get("model")
        attributes["openai_agents.model_config"] = data.get("model_config") or {}
        attributes.update(_gen_ai_usage_attrs(data.get("usage"), data.get("model")))
        input_data = {"input": data.get("input")}
        output_data = {"output": data.get("output")}

    elif span_type == "response":
        attributes["openai_agents.response.id"] = data.get("response_id")
        attributes.update(_gen_ai_usage_attrs(data.get("usage")))

    elif span_type == "guardrail":
        # `triggered` is the whole point of a guardrail span: it records
        # that the SDK stopped the run. Surface it as a first-class
        # attribute rather than burying it in output_data.
        attributes["openai_agents.guardrail.name"] = data.get("name")
        attributes["openai_agents.guardrail.triggered"] = bool(data.get("triggered"))

    elif span_type == "mcp_tools":
        attributes["openai_agents.mcp.server"] = data.get("server")
        output_data = {"result": data.get("result")}

    elif span_type in ("task", "turn", "custom"):
        attributes["openai_agents.name"] = data.get("name")
        if data.get("data") is not None:
            input_data = {"data": data.get("data")}

    else:
        # Unknown span type. Keep the raw payload so a newer SDK version
        # does not silently drop information on ingest.
        input_data = {"span_data": data}

    name = _span_name(span_type, data)

    # Span.span_id is a non-optional str with a default factory, so an
    # absent SDK id must fall through to that factory rather than be set
    # to None. Passing the key conditionally is the only way to do that.
    identity: dict[str, Any] = {}
    if span.get("id"):
        identity["span_id"] = str(span["id"])

    return Span(
        **identity,
        trace_id=str(span.get("trace_id") or trace_id or ""),
        parent_id=span.get("parent_id"),
        name=name,
        kind=kind,
        platform=Platform.OPENAI,
        start_time=started or _now(),
        end_time=ended,
        status=_map_status(error, ended),
        error_message=_error_message(error),
        attributes=attributes,
        input_data=input_data,
        output_data=output_data,
    )


# --- Utilities ---


def _span_name(span_type: str, data: dict[str, Any]) -> str:
    """Build a readable span name, mirroring `openai.<thing>:<id>` style."""
    if span_type == "handoff":
        return f"openai_agents.handoff:{data.get('from_agent')}->{data.get('to_agent')}"
    label = data.get("name") or data.get("model") or data.get("server")
    if label:
        return f"openai_agents.{span_type}:{label}"
    return f"openai_agents.{span_type}"


def _gen_ai_usage_attrs(usage: Any, model: Any = None) -> dict[str, Any]:
    """Translate an Agents SDK `usage` payload into OTEL `gen_ai.usage.*`.

    The Agents SDK reports `input_tokens` / `output_tokens`; the older
    Assistants and Responses payloads use `prompt_tokens` /
    `completion_tokens`. Accept both so token-budget detectors do not
    need a per-API branch.
    """
    if not isinstance(usage, dict):
        return {}
    out: dict[str, Any] = {}
    prompt = usage.get("input_tokens", usage.get("prompt_tokens"))
    completion = usage.get("output_tokens", usage.get("completion_tokens"))
    if prompt is not None:
        out["gen_ai.usage.input_tokens"] = prompt
    if completion is not None:
        out["gen_ai.usage.output_tokens"] = completion
    total = usage.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    if total is not None:
        out["gen_ai.usage.total_tokens"] = total
    if model is not None:
        out["gen_ai.request.model"] = str(model)
    return out


def _map_status(error: Any, ended_at: Optional[datetime]) -> SpanStatus:
    """Agents SDK spans carry no status field, only an optional error.

    Absence of an error plus a set `ended_at` is the success signal; a
    span with neither is still running rather than successful.
    """
    if error:
        return SpanStatus.ERROR
    if ended_at is not None:
        return SpanStatus.OK
    return SpanStatus.IN_PROGRESS


def _error_message(error: Any) -> Optional[str]:
    if not error:
        return None
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        return error.get("message") or str(error)
    return str(error)


def _ts(value: Any) -> Optional[datetime]:
    """Coerce an Agents SDK timestamp to a datetime.

    The SDK writes ISO-8601 strings. Unix seconds and datetimes are also
    accepted so a caller that pre-normalised its payload still works.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            return parse_iso_datetime(value)
        except ValueError:
            return None
    return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PisamaTracingProcessor:
    """A `TracingProcessor` for the OpenAI Agents SDK.

    Register it once and every completed run is delivered to your callback as a
    Pisama `Trace`::

        from agents.tracing import add_trace_processor
        from pisama_core.adapters import PisamaTracingProcessor

        add_trace_processor(PisamaTracingProcessor(on_trace=my_handler))

    `add_trace_processor` keeps the SDK's own exporter registered alongside this
    one, so OpenAI tracing continues to receive the same spans. Use
    `set_trace_processors` instead to make Pisama the only destination.

    This class deliberately does **not** import or subclass
    `agents.tracing.TracingProcessor`. The SDK dispatches by duck typing, so
    implementing the six methods is sufficient, and `pisama-core` stays free of a
    vendor dependency. The trade-off is that a future SDK release could widen the
    interface without a type error here; `on_trace` would simply stop being
    called, which the tests guard by asserting the method set.

    Spans are buffered per trace id rather than in one list, because concurrent
    runs interleave `on_span_end` calls and a single buffer would attribute one
    run's spans to another. The buffer is mutated under a lock: the SDK may
    dispatch from more than one thread.
    """

    def __init__(self, on_trace: Optional[Callable[[Trace], None]] = None) -> None:
        """
        Args:
            on_trace: Called with the parsed `Trace` when a run completes. If
                omitted, traces are parsed and dropped, which is only useful for
                measuring overhead.
        """
        self._on_trace = on_trace
        self._spans: Dict[str, List[dict]] = {}
        self._lock = threading.Lock()

    # --- TracingProcessor interface ---

    def on_trace_start(self, trace: Any) -> None:
        return None

    def on_span_start(self, span: Any) -> None:
        return None

    def on_span_end(self, span: Any) -> None:
        exported = _export(span)
        if not exported:
            return
        trace_id = str(exported.get("trace_id") or "")
        with self._lock:
            self._spans.setdefault(trace_id, []).append(exported)

    def on_trace_end(self, trace: Any) -> None:
        exported = _export(trace)
        if not exported:
            return
        trace_id = str(exported.get("id") or "")
        with self._lock:
            spans = self._spans.pop(trace_id, [])
        parsed = parse_agents_trace(exported, spans)
        if self._on_trace is not None:
            self._on_trace(parsed)

    def force_flush(self) -> None:
        return None

    def shutdown(self) -> None:
        """Drop any spans whose trace never ended, so a long process cannot leak."""
        with self._lock:
            self._spans.clear()


def _export(obj: Any) -> Optional[dict]:
    """Call `.export()` defensively.

    The SDK's own exporters tolerate a span or trace that declines to export
    (`export()` may return None), and a processor that raised here would take
    down the caller's run for the sake of telemetry.
    """
    export = getattr(obj, "export", None)
    if export is None:
        return None
    try:
        result = export()
    except Exception:
        return None
    return result if isinstance(result, dict) else None
