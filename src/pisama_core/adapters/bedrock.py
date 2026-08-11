"""AWS Bedrock Agents trace ingestion adapter.

Converts Bedrock Agent `InvokeAgent` trace output into Pisama's universal
Span/Trace format. Ingestion-only — Bedrock Agents does not expose a
pre-execution hook API, so blocking and fix injection are not supported.

Input shape expected: the trace objects from `InvokeAgentResponse.completion`.
Each chunk in the streamed response may contain a `trace` field with one of:

- `preProcessingTrace`
- `orchestrationTrace`
- `postProcessingTrace`
- `failureTrace`
- `guardrailTrace`

Within `orchestrationTrace`, the interesting sub-events are:

- `modelInvocationInput` / `modelInvocationOutput` — LLM call boundaries
- `rationale` — reasoning step
- `invocationInput.actionGroupInvocationInput` — tool call
- `observation.actionGroupInvocationOutput` — tool result
- `observation.knowledgeBaseLookupOutput` — retrieval result
- `observation.finalResponse` — final user-facing answer

The adapter does NOT import boto3. Accepts dicts that match the documented
JSON shape so callers can pass whatever their SDK version returns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from pisama_core.traces.enums import Platform, SpanKind, SpanStatus
from pisama_core.traces.models import Span, Trace, TraceMetadata

__all__ = ["parse_invoke_agent"]


def _gen_ai_usage_attrs(usage: Any, model: Any = None) -> dict[str, Any]:
    """Translate a Bedrock `usage` payload into OTEL `gen_ai.usage.*` attrs.

    Bedrock returns usage as `{inputTokens, outputTokens}` on
    `modelInvocationOutput.metadata.usage`. Detectors across the codebase
    read `gen_ai.usage.*`, not vendor-specific keys, so every adapter must
    translate to the OTEL naming to keep detectors vendor-neutral.
    """
    if not isinstance(usage, dict):
        return {}
    out: dict[str, Any] = {}
    if (it := usage.get("inputTokens")) is not None:
        out["gen_ai.usage.input_tokens"] = it
    if (ot := usage.get("outputTokens")) is not None:
        out["gen_ai.usage.output_tokens"] = ot
    total = usage.get("totalTokens")
    if (
        total is None
        and (it := usage.get("inputTokens")) is not None
        and (ot := usage.get("outputTokens")) is not None
    ):
        total = int(it) + int(ot)
    if total is not None:
        out["gen_ai.usage.total_tokens"] = total
    if model is not None:
        out["gen_ai.request.model"] = str(model)
    out["gen_ai.system"] = "bedrock"
    return out


def _gen_ai_model_attrs(model: Any) -> dict[str, Any]:
    """Emit `gen_ai.request.model` when Bedrock names the foundation model."""
    if model in (None, ""):
        return {}
    return {"gen_ai.request.model": str(model), "gen_ai.system": "bedrock"}


def parse_invoke_agent(
    session_id: str,
    agent_id: str,
    traces: Iterable[dict[str, Any]],
    final_answer: Optional[str] = None,
    agent_alias_id: Optional[str] = None,
) -> Trace:
    """Parse a Bedrock Agent `InvokeAgent` trace stream into a Pisama Trace.

    Args:
        session_id: Bedrock session ID.
        agent_id: Bedrock agent ID.
        traces: Iterable of trace chunk dicts extracted from the streaming
            `InvokeAgentResponse.completion`. Pass the `trace` field of
            each chunk, not the chunk envelope.
        final_answer: Optional final text answer extracted from
            `observation.finalResponse.text` or the top-level `completion`
            aggregate. Recorded as a USER_OUTPUT span for convenience.
        agent_alias_id: Optional alias for the deployed agent version.

    Returns:
        A `Trace` with one AGENT root span and child spans for each
        orchestration step.
    """
    trace = Trace(
        metadata=TraceMetadata(
            session_id=session_id,
            platform=Platform.BEDROCK,
            platform_version="agents-v1",
            custom={"agent_id": agent_id, "agent_alias_id": agent_alias_id},
        ),
    )

    # Bedrock's InvokeAgent response doesn't carry an explicit trace start
    # time; we approximate it with the first child span's start_time once
    # every chunk has been processed. Using _now() at construction time
    # would make root.start_time the ingestion time, not the actual trace
    # time — that skews every downstream latency metric. Set to None here
    # and backfill below.
    root = Span(
        trace_id=trace.trace_id,
        name=f"bedrock.agent:{agent_id}",
        kind=SpanKind.AGENT,
        platform=Platform.BEDROCK,
        platform_metadata={"agent_id": agent_id, "session_id": session_id},
        start_time=_now(),  # provisional; overwritten below if a child has an earlier start_time
        status=SpanStatus.OK,
        attributes={"bedrock.agent.id": agent_id, "bedrock.session.id": session_id},
    )
    trace.spans.append(root)

    for chunk in traces:
        # Chunks may wrap trace data in `{'trace': {...}}` or be raw.
        node = chunk.get("trace") if "trace" in chunk else chunk
        if not isinstance(node, dict):
            continue

        # `eventTime` lives on the TracePart envelope, not inside the trace
        # node. Without it every child span falls back to ingestion time, which
        # made trace duration microseconds against a multi-second invocation.
        event_time = _ts_iso(chunk.get("eventTime"))
        started, ended = _node_times(node)
        first_new_span = len(trace.spans)

        if "orchestrationTrace" in node:
            trace.spans.extend(
                _parse_orchestration(node["orchestrationTrace"], trace.trace_id, root.span_id)
            )
        elif "preProcessingTrace" in node:
            trace.spans.append(
                _generic_step_span(
                    node["preProcessingTrace"],
                    "bedrock.pre_processing",
                    SpanKind.SYSTEM,
                    trace.trace_id,
                    root.span_id,
                )
            )
        elif "postProcessingTrace" in node:
            trace.spans.append(
                _generic_step_span(
                    node["postProcessingTrace"],
                    "bedrock.post_processing",
                    SpanKind.SYSTEM,
                    trace.trace_id,
                    root.span_id,
                )
            )
        elif "customOrchestrationTrace" in node:
            # AWS added customOrchestrationTrace in 2025 for agents that
            # swap in a custom Lambda orchestrator. The payload shape is
            # user-defined, so we preserve the whole node via the generic
            # span's `input_data=node` behavior rather than trying to map
            # documented sub-fields that may not exist.
            trace.spans.append(
                _generic_step_span(
                    node["customOrchestrationTrace"],
                    "bedrock.custom_orchestration",
                    SpanKind.SYSTEM,
                    trace.trace_id,
                    root.span_id,
                )
            )
        elif "failureTrace" in node:
            fail = node["failureTrace"]
            trace.spans.append(
                Span(
                    trace_id=trace.trace_id,
                    parent_id=root.span_id,
                    name="bedrock.failure",
                    kind=SpanKind.SYSTEM,
                    platform=Platform.BEDROCK,
                    status=SpanStatus.ERROR,
                    error_message=str(fail.get("failureReason") or "unknown"),
                    attributes={"bedrock.failure.reason": fail.get("failureReason")},
                )
            )
            root.status = SpanStatus.ERROR
            root.error_message = root.error_message or str(fail.get("failureReason") or "")
        elif "guardrailTrace" in node:
            gr = node["guardrailTrace"]
            trace.spans.append(
                Span(
                    trace_id=trace.trace_id,
                    parent_id=root.span_id,
                    name="bedrock.guardrail",
                    kind=SpanKind.SYSTEM,
                    platform=Platform.BEDROCK,
                    status=SpanStatus.BLOCKED
                    if gr.get("action") == "INTERVENED"
                    else SpanStatus.OK,
                    attributes={"bedrock.guardrail.action": gr.get("action")},
                    input_data=gr.get("inputAssessments"),
                    output_data=gr.get("outputAssessments"),
                )
            )

        # Bedrock times every trace part, and several nodes carry a finer
        # `metadata.startTime`/`endTime` pair. The span constructors above leave
        # both fields at their dataclass default, so stamp whatever this chunk
        # produced rather than thread a timestamp through every constructor.
        for span in trace.spans[first_new_span:]:
            if started or event_time:
                span.start_time = started or event_time
            if ended:
                span.end_time = ended

    if final_answer is not None:
        trace.spans.append(
            Span(
                trace_id=trace.trace_id,
                parent_id=root.span_id,
                name="bedrock.final_response",
                kind=SpanKind.USER_OUTPUT,
                platform=Platform.BEDROCK,
                status=SpanStatus.OK,
                output_data={"text": final_answer},
            )
        )

    # Backfill root.start_time to the earliest child start_time so latency
    # metrics match the real trace duration. If no child has a start_time
    # (e.g. the chunk stream was empty), leave the provisional _now() value.
    child_starts = [
        s.start_time for s in trace.spans if s.span_id != root.span_id and s.start_time is not None
    ]
    if child_starts:
        earliest = min(child_starts)
        if earliest < root.start_time:
            root.start_time = earliest
    # The root must end when the trace ended, not when Pisama parsed it. Once
    # child spans carry real Bedrock times, closing the root at _now() reports
    # the gap between the invocation and its ingestion as the agent's latency.
    child_ends = [
        s.end_time for s in trace.spans if s.span_id != root.span_id and s.end_time is not None
    ]
    if child_ends:
        root.end_time = max(child_ends + child_starts) if child_starts else max(child_ends)
    elif child_starts:
        root.end_time = max(child_starts)
    else:
        root.end_time = _now()
    return trace


def _parse_orchestration(
    node: dict[str, Any],
    trace_id: str,
    parent_id: str,
) -> list[Span]:
    """Convert one orchestrationTrace entry into one or more spans.

    Bedrock emits one orchestrationTrace per step, and each step may
    contain several of these sub-fields. We emit one span per sub-field
    that is present.
    """
    spans: list[Span] = []

    if "modelInvocationInput" in node:
        mi = node["modelInvocationInput"]
        spans.append(
            Span(
                trace_id=trace_id,
                parent_id=parent_id,
                name="bedrock.llm.input",
                kind=SpanKind.LLM,
                platform=Platform.BEDROCK,
                status=SpanStatus.IN_PROGRESS,
                # `foundationModel` is the only place the model id appears in
                # an orchestration stream, and it sits on the INPUT node.
                # `modelInvocationOutput` cannot carry it: OrchestrationTrace is
                # a union, so input and output always arrive as separate nodes.
                attributes={
                    "bedrock.invocation.type": mi.get("type"),
                    **_gen_ai_model_attrs(mi.get("foundationModel")),
                },
                input_data={"text": mi.get("text"), "parameters": mi.get("inferenceConfiguration")},
            )
        )

    if "modelInvocationOutput" in node:
        mo = node["modelInvocationOutput"]
        raw = (mo.get("rawResponse") or {}).get("content")
        usage = mo.get("metadata", {}).get("usage")
        spans.append(
            Span(
                trace_id=trace_id,
                parent_id=parent_id,
                name="bedrock.llm.output",
                kind=SpanKind.LLM,
                platform=Platform.BEDROCK,
                status=SpanStatus.OK,
                attributes={
                    "bedrock.usage": usage or {},
                    **_gen_ai_usage_attrs(usage),
                },
                output_data={"text": raw},
            )
        )

    if "rationale" in node:
        rat = node["rationale"]
        spans.append(
            Span(
                trace_id=trace_id,
                parent_id=parent_id,
                name="bedrock.rationale",
                kind=SpanKind.SYSTEM,
                platform=Platform.BEDROCK,
                status=SpanStatus.OK,
                output_data={"text": rat.get("text")},
            )
        )

    # Track the invocation-input span IDs so the matching observation
    # spans can be parented to them. Each orchestrationTrace entry carries
    # at most one of each pair, so two local variables suffice.
    tool_input_span_id: Optional[str] = None
    kb_input_span_id: Optional[str] = None

    if "invocationInput" in node:
        ii = node["invocationInput"]
        action = ii.get("actionGroupInvocationInput") or {}
        kb = ii.get("knowledgeBaseLookupInput")
        if action:
            tool_span = Span(
                trace_id=trace_id,
                parent_id=parent_id,
                name=f"bedrock.tool:{action.get('function') or action.get('apiPath') or 'action'}",
                kind=SpanKind.TOOL,
                platform=Platform.BEDROCK,
                status=SpanStatus.IN_PROGRESS,
                attributes={
                    "bedrock.action_group.name": action.get("actionGroupName"),
                    "bedrock.action_group.api_path": action.get("apiPath"),
                    "bedrock.action_group.function": action.get("function"),
                },
                input_data={"parameters": action.get("parameters")},
            )
            tool_input_span_id = tool_span.span_id
            spans.append(tool_span)
        if kb:
            kb_span = Span(
                trace_id=trace_id,
                parent_id=parent_id,
                name="bedrock.kb.lookup",
                kind=SpanKind.RETRIEVAL,
                platform=Platform.BEDROCK,
                status=SpanStatus.IN_PROGRESS,
                attributes={"bedrock.kb.id": kb.get("knowledgeBaseId")},
                input_data={"text": kb.get("text")},
            )
            kb_input_span_id = kb_span.span_id
            spans.append(kb_span)

    if "observation" in node:
        obs = node["observation"]
        action_out = obs.get("actionGroupInvocationOutput")
        kb_out = obs.get("knowledgeBaseLookupOutput")
        final = obs.get("finalResponse")
        if action_out:
            spans.append(
                Span(
                    trace_id=trace_id,
                    # Parent the observation to its invocation input when we
                    # saw one in the same orchestrationTrace; fall back to
                    # the orchestration parent otherwise.
                    parent_id=tool_input_span_id or parent_id,
                    name="bedrock.tool.output",
                    kind=SpanKind.TOOL,
                    platform=Platform.BEDROCK,
                    status=SpanStatus.OK,
                    output_data={"text": action_out.get("text")},
                )
            )
        if kb_out:
            spans.append(
                Span(
                    trace_id=trace_id,
                    parent_id=kb_input_span_id or parent_id,
                    name="bedrock.kb.output",
                    kind=SpanKind.RETRIEVAL,
                    platform=Platform.BEDROCK,
                    status=SpanStatus.OK,
                    output_data={"references": kb_out.get("retrievedReferences")},
                )
            )
        if final:
            spans.append(
                Span(
                    trace_id=trace_id,
                    parent_id=parent_id,
                    name="bedrock.final_response",
                    kind=SpanKind.USER_OUTPUT,
                    platform=Platform.BEDROCK,
                    status=SpanStatus.OK,
                    output_data={"text": final.get("text")},
                )
            )

    return spans


def _generic_step_span(
    node: dict[str, Any],
    name: str,
    kind: SpanKind,
    trace_id: str,
    parent_id: str,
) -> Span:
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=name,
        kind=kind,
        platform=Platform.BEDROCK,
        status=SpanStatus.OK,
        attributes={"bedrock.trace.keys": list(node.keys())},
        input_data=node,
    )


def _node_times(node: dict[str, Any]) -> tuple[Optional[datetime], Optional[datetime]]:
    """Find the finest start/end times a trace node offers.

    Model-invocation, action-group, final-response, failure and guardrail nodes
    all nest a `metadata` object carrying `startTime` and `endTime`. They sit one
    or two levels below the trace node depending on the variant, so search
    rather than hard-code a path.
    """

    def walk(value: Any, depth: int) -> tuple[Optional[datetime], Optional[datetime]]:
        if depth > 3 or not isinstance(value, dict):
            return None, None
        meta = value.get("metadata")
        if isinstance(meta, dict):
            start = _ts_iso(meta.get("startTime"))
            end = _ts_iso(meta.get("endTime"))
            if start or end:
                return start, end
        for nested in value.values():
            start, end = walk(nested, depth + 1)
            if start or end:
                return start, end
        return None, None

    return walk(node, 0)


def _ts_iso(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp as emitted by Bedrock.

    `TracePart.eventTime` and the per-node `Metadata` start/end times are
    `SyntheticTimestamp_date_time`, serialised with a trailing `Z`.
    """
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)
