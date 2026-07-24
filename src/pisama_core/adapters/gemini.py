"""Gemini Interactions API trace ingestion adapter.

Converts a Gemini Interactions API response envelope into Pisama's universal
Span/Trace format. Ingestion-only — the Interactions API is a managed runtime
with no pre-execution hook surface, so blocking and fix injection are not
supported (same stance as bedrock.py).

The Interactions API is Beta; the response envelope tracks
https://ai.google.dev/gemini-api/docs/interactions. The parser reads these
top-level fields when present and ignores anything it doesn't recognise:

- ``id`` / ``name``                    interaction identifier
- ``session_id`` / ``session``         session identifier (falls back to caller arg)
- ``model``                            model name, e.g. ``gemini-3.1-pro``
- ``created_at`` / ``start_time``      interaction start (ISO 8601)
- ``finished_at`` / ``end_time``       interaction end   (ISO 8601)
- ``status``                           ``COMPLETED`` | ``RUNNING`` | ``FAILED``
- ``operation_name``                   long-running op handle (mark root IN_PROGRESS)
- ``messages[]``                       conversation messages (user/model/tool)
- ``tool_calls[]``                     tool invocations with args/result
- ``state.tasks[]``                    decomposed goals (feeds decomposition detector)
- ``candidates[]``                     model completions with safety_ratings
- ``usage_metadata``                   ``{prompt,candidates,total}_token_count``

The adapter does NOT import ``google-genai``. Accepts dicts that match the
documented JSON shape so callers can pass whatever their SDK version returns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from pisama_core.traces.enums import Platform, SpanKind, SpanStatus
from pisama_core.traces.models import Event, Span, Trace, TraceMetadata

__all__ = ["parse_interactions_response"]


_PLATFORM_VERSION = "interactions-beta-v1"


def _gen_ai_usage_attrs(usage: Any, model: Any = None) -> dict[str, Any]:
    """Translate a Gemini ``usage_metadata`` payload into OTEL ``gen_ai.usage.*``.

    Gemini reports tokens as ``prompt_token_count`` / ``candidates_token_count``
    / ``total_token_count``. Detectors across the codebase read ``gen_ai.usage.*``
    only, so every adapter must translate to the OTEL naming to keep
    detectors vendor-neutral.
    """
    if not isinstance(usage, dict):
        return {}
    out: dict[str, Any] = {}
    if (pt := usage.get("prompt_token_count")) is not None:
        out["gen_ai.usage.input_tokens"] = pt
    if (ct := usage.get("candidates_token_count")) is not None:
        out["gen_ai.usage.output_tokens"] = ct
    total = usage.get("total_token_count")
    if (
        total is None
        and (pt := usage.get("prompt_token_count")) is not None
        and (ct := usage.get("candidates_token_count")) is not None
    ):
        total = int(pt) + int(ct)
    if total is not None:
        out["gen_ai.usage.total_tokens"] = total
    if model is not None:
        out["gen_ai.request.model"] = str(model)
    out["gen_ai.system"] = "gemini"
    return out


def parse_interactions_response(
    response: dict[str, Any],
    session_id: Optional[str] = None,
) -> Trace:
    """Parse a Gemini Interactions API response envelope into a Pisama Trace.

    Args:
        response: The decoded JSON response from the Interactions API (or a
            poll response for a long-running operation).
        session_id: Session ID to record on ``TraceMetadata``. Falls back to
            ``response['session_id']`` then ``response['session']``.

    Returns:
        A ``Trace`` with one AGENT_TURN root span and child spans for each
        message, tool call, decomposed task, and candidate completion.
    """
    if not isinstance(response, dict):
        raise TypeError("response must be a dict")

    interaction_id = response.get("id") or response.get("name") or "interaction"
    session = session_id or response.get("session_id") or response.get("session")
    model = response.get("model")
    start = _parse_iso(response.get("created_at") or response.get("start_time"))
    end = _parse_iso(response.get("finished_at") or response.get("end_time"))
    operation_name = response.get("operation_name")
    status_str = (response.get("status") or "").upper()

    trace = Trace(
        metadata=TraceMetadata(
            session_id=session or _ensure_session_id(interaction_id),
            platform=Platform.GEMINI,
            platform_version=_PLATFORM_VERSION,
            custom={
                "interaction_id": interaction_id,
                "model": model,
                "operation_name": operation_name,
            },
        ),
    )

    root_status = _root_status(status_str, operation_name)
    root = Span(
        trace_id=trace.trace_id,
        name=f"gemini.interaction:{interaction_id}",
        kind=SpanKind.AGENT_TURN,
        platform=Platform.GEMINI,
        platform_metadata={"interaction_id": interaction_id, "model": model},
        start_time=start or _now(),
        end_time=end,
        status=root_status,
        attributes={
            "gemini.interaction.id": interaction_id,
            "gemini.model": model,
            "gen_ai.system": "gemini",
            "gen_ai.request.model": str(model) if model else None,
        },
    )
    trace.spans.append(root)

    # state.tasks → TASK spans (decomposition detector reads `goals`)
    state = response.get("state") or {}
    for task in _as_iterable(state.get("tasks")):
        trace.spans.append(_task_span(task, trace.trace_id, root.span_id))

    # messages → MESSAGE / USER_INPUT / USER_OUTPUT spans
    for message in _as_iterable(response.get("messages")):
        span = _message_span(message, trace.trace_id, root.span_id)
        if span is not None:
            trace.spans.append(span)

    # tool_calls → TOOL spans (one pair: call + result)
    for call in _as_iterable(response.get("tool_calls")):
        trace.spans.extend(_tool_call_spans(call, trace.trace_id, root.span_id))

    # candidates → LLM spans with safety events
    usage = response.get("usage_metadata")
    for idx, candidate in enumerate(_as_iterable(response.get("candidates"))):
        trace.spans.append(
            _candidate_span(candidate, idx, usage, model, trace.trace_id, root.span_id)
        )
        usage = (
            None  # Attach usage to the first candidate only; downstream detectors sum per trace.
        )

    return trace


def _task_span(task: Any, trace_id: str, parent_id: str) -> Span:
    task = task if isinstance(task, dict) else {"goal": str(task)}
    goal = task.get("goal") or task.get("description") or task.get("name")
    status = _task_status(task.get("status"))
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"gemini.task:{task.get('id') or goal or 'task'}",
        kind=SpanKind.TASK,
        platform=Platform.GEMINI,
        status=status,
        attributes={
            "goals": [goal] if goal else [],
            "gemini.task.status": task.get("status"),
        },
        input_data={"goal": goal},
        output_data={"result": task.get("result")} if "result" in task else None,
    )


def _message_span(message: Any, trace_id: str, parent_id: str) -> Optional[Span]:
    if not isinstance(message, dict):
        return None
    role = (message.get("role") or "").lower()
    text = _extract_text(message.get("parts") or message.get("content"))
    if role == "user":
        kind = SpanKind.USER_INPUT
    elif role in ("model", "assistant"):
        kind = SpanKind.USER_OUTPUT
    else:
        kind = SpanKind.MESSAGE
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"gemini.message:{role or 'unknown'}",
        kind=kind,
        platform=Platform.GEMINI,
        status=SpanStatus.OK,
        start_time=_parse_iso(message.get("timestamp")) or _now(),
        attributes={"gemini.message.role": role},
        output_data={"text": text} if text else None,
    )


def _tool_call_spans(call: Any, trace_id: str, parent_id: str) -> list[Span]:
    if not isinstance(call, dict):
        return []
    call_id = call.get("id") or call.get("tool_call_id")
    name = call.get("name") or call.get("function_name") or "tool"
    args = call.get("args") or call.get("arguments")
    result = call.get("result") or call.get("response")
    started = _parse_iso(call.get("started_at") or call.get("start_time"))
    finished = _parse_iso(call.get("finished_at") or call.get("end_time"))
    errored = bool(call.get("error"))

    call_span = Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"gemini.tool:{name}",
        kind=SpanKind.TOOL,
        platform=Platform.GEMINI,
        status=SpanStatus.ERROR if errored else SpanStatus.OK,
        error_message=call.get("error") if errored else None,
        start_time=started or _now(),
        end_time=finished,
        attributes={
            "gemini.tool.name": name,
            "gemini.tool.call_id": call_id,
        },
        input_data={"arguments": args} if args is not None else None,
        output_data={"result": result} if result is not None else None,
    )
    return [call_span]


def _candidate_span(
    candidate: Any,
    index: int,
    usage: Any,
    model: Any,
    trace_id: str,
    parent_id: str,
) -> Span:
    candidate = candidate if isinstance(candidate, dict) else {}
    content = candidate.get("content") or {}
    text = _extract_text(content.get("parts"))
    finish_reason = candidate.get("finish_reason")
    usage_attrs = _gen_ai_usage_attrs(usage, model=model) if usage else {}

    span = Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"gemini.candidate:{index}",
        kind=SpanKind.LLM,
        platform=Platform.GEMINI,
        status=_candidate_status(finish_reason),
        attributes={
            "gemini.candidate.index": index,
            "gemini.finish_reason": finish_reason,
            **usage_attrs,
        },
        output_data={"text": text} if text else None,
    )
    for rating in _as_iterable(candidate.get("safety_ratings")):
        if not isinstance(rating, dict):
            continue
        span.events.append(
            Event(
                name="safety_check",
                attributes={
                    "category": rating.get("category"),
                    "probability": rating.get("probability"),
                    "blocked": rating.get("blocked"),
                },
            )
        )
    return span


def _extract_text(parts: Any) -> Optional[str]:
    if parts is None:
        return None
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        chunks = []
        for part in parts:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict) and "text" in part:
                chunks.append(str(part["text"]))
        return "".join(chunks) or None
    if isinstance(parts, dict) and "text" in parts:
        return str(parts["text"])
    return None


def _as_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _parse_iso(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _root_status(status_str: str, operation_name: Any) -> SpanStatus:
    if operation_name and status_str in ("RUNNING", "PENDING", ""):
        return SpanStatus.IN_PROGRESS
    if status_str in ("FAILED", "ERROR"):
        return SpanStatus.ERROR
    if status_str == "CANCELLED":
        return SpanStatus.CANCELLED
    if status_str == "TIMEOUT":
        return SpanStatus.TIMEOUT
    return SpanStatus.OK


def _task_status(value: Any) -> SpanStatus:
    if not value:
        return SpanStatus.UNSET
    v = str(value).upper()
    if v in ("DONE", "COMPLETED", "SUCCESS"):
        return SpanStatus.OK
    if v in ("IN_PROGRESS", "RUNNING", "PENDING"):
        return SpanStatus.IN_PROGRESS
    if v in ("FAILED", "ERROR"):
        return SpanStatus.ERROR
    return SpanStatus.UNSET


def _candidate_status(finish_reason: Any) -> SpanStatus:
    if not finish_reason:
        return SpanStatus.OK
    fr = str(finish_reason).upper()
    if fr in ("STOP", "END_TURN", "FINISHED"):
        return SpanStatus.OK
    if fr in ("SAFETY", "BLOCKED", "RECITATION"):
        return SpanStatus.BLOCKED
    if fr in ("MAX_TOKENS",):
        return SpanStatus.TIMEOUT
    return SpanStatus.OK


def _ensure_session_id(interaction_id: str) -> str:
    return f"gemini-{interaction_id}"


def _now() -> datetime:
    return datetime.now(timezone.utc)
