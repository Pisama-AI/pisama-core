"""Gemini Interactions API trace ingestion adapter.

Converts a Gemini Interactions API ``Interaction`` into Pisama's universal
Span/Trace format. Ingestion-only — the Interactions API is a managed runtime
with no pre-execution hook surface, so blocking and fix injection are not
supported (same stance as bedrock.py).

The shape below is the vendor's, read from ``google.genai.interactions`` in
google-genai 2.17.0 (inner ``_gaos`` SDK 2.4.1-preview.5, OpenAPI v1beta) and
pinned by the fixtures under ``tests/fixtures/``. It is deliberately spelled out
because an earlier version of this adapter targeted an envelope the API never
returned — ``messages[]``, ``tool_calls[]``, ``candidates[]``, ``state.tasks[]``,
``session_id``, ``created_at``/``finished_at``, ``usage_metadata`` — of which the
real ``Interaction`` has none. A real response parsed to almost nothing.

Top-level fields read here:

- ``id``                    interaction identifier (a resource path, not a bare id)
- ``created`` / ``updated`` ISO-8601 strings, not epoch numbers
- ``status``                lowercase: ``in_progress`` | ``requires_action`` |
                            ``completed`` | ``failed`` | ``cancelled`` |
                            ``incomplete`` | ``budget_exceeded`` | ``queued``
- ``model``                 model name
- ``usage``                 ``total_input_tokens`` / ``total_output_tokens`` /
                            ``total_tokens`` (+ thought / cached / tool-use)
- ``errors[]``              ``{code, message}``; ``code`` is a URI string.
                            There is no top-level ``error``.
- ``steps[]``               the conversation, as a union discriminated on ``type``
- ``environment_id``, ``previous_interaction_id``, ``system_instruction``

``steps[]`` variants and their payload keys:

- ``user_input``       ``content``
- ``model_output``     ``content``, ``error`` (a google.rpc ``Status``:
                       ``{code:int, message, details}`` — NOT the interaction-level
                       ``Error``)
- ``thought``          ``summary`` (a list of content blocks, not a string),
                       ``signature``
- ``function_call``    ``name``, ``id``, ``arguments`` (a dict, not a JSON string)
- ``function_result``  ``name``, ``call_id``, ``result``, ``is_error``
- ``*_call`` / ``*_result`` for code execution, URL context, MCP server tools,
  Google Search, File Search and Google Maps

Two shape notes that drive decisions below. **Steps carry no timestamps** — the
interaction has ``created``/``updated`` and nothing per step. And ``output_text``
is recomputed by the SDK from the trailing ``model_output`` on every validation,
so it is a client-side convenience rather than a wire field; the trailing
``model_output`` step is the source of truth.

The adapter does NOT import ``google-genai``. It accepts dicts matching the
serialized shape so callers can pass ``.model_dump(mode="json")`` or the raw
HTTP payload.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from pisama_core.traces.enums import Platform, SpanKind, SpanStatus
from pisama_core.traces.models import Span, Trace, TraceMetadata

__all__ = ["parse_interactions_response"]


_PLATFORM_VERSION = "interactions-v1beta"

# Interaction.status -> span status. The API's terminal states are richer than
# ok/error: `cancelled` is a caller action, and `incomplete` / `budget_exceeded`
# are the runtime stopping short of the goal, which is a failure to complete
# rather than a crash. Everything unfinished maps to IN_PROGRESS so a polled
# interaction is never mistaken for a finished one.
_STATUS_MAP: dict[str, SpanStatus] = {
    "completed": SpanStatus.OK,
    "failed": SpanStatus.ERROR,
    "cancelled": SpanStatus.CANCELLED,
    "incomplete": SpanStatus.ERROR,
    "budget_exceeded": SpanStatus.ERROR,
    "in_progress": SpanStatus.IN_PROGRESS,
    "queued": SpanStatus.IN_PROGRESS,
    "requires_action": SpanStatus.IN_PROGRESS,
}

# Step type -> span kind. Retrieval-shaped tools are separated from function
# calls so retrieval detectors see them; `thought` is reasoning rather than
# conversation, and is mapped to TASK to match how the ADK adapter records
# planner output.
_STEP_KINDS: dict[str, SpanKind] = {
    "user_input": SpanKind.USER_INPUT,
    "model_output": SpanKind.LLM,
    "thought": SpanKind.TASK,
    "function_call": SpanKind.TOOL,
    "function_result": SpanKind.TOOL,
    "code_execution_call": SpanKind.TOOL,
    "code_execution_result": SpanKind.TOOL,
    "mcp_server_tool_call": SpanKind.TOOL,
    "mcp_server_tool_result": SpanKind.TOOL,
    "google_search_call": SpanKind.RETRIEVAL,
    "google_search_result": SpanKind.RETRIEVAL,
    "file_search_call": SpanKind.RETRIEVAL,
    "file_search_result": SpanKind.RETRIEVAL,
    "url_context_call": SpanKind.RETRIEVAL,
    "url_context_result": SpanKind.RETRIEVAL,
    "google_maps_call": SpanKind.RETRIEVAL,
    "google_maps_result": SpanKind.RETRIEVAL,
}

_RESULT_SUFFIX = "_result"


def _gen_ai_usage_attrs(usage: Any, model: Any = None) -> dict[str, Any]:
    """Translate an Interactions ``usage`` payload into OTEL ``gen_ai.usage.*``.

    The Interactions API reports ``total_input_tokens`` / ``total_output_tokens``
    / ``total_tokens``. It does NOT use ``prompt_token_count`` /
    ``candidates_token_count``; those belong to ``generateContent``, a different
    surface, and reading them here yielded no usage at all.

    Thought, cached and tool-use tokens have no OTEL equivalent, so they are kept
    under ``gemini.usage.*`` rather than being folded into the standard keys and
    silently inflating them.
    """
    if not isinstance(usage, dict):
        return {}
    out: dict[str, Any] = {}
    if (it := usage.get("total_input_tokens")) is not None:
        out["gen_ai.usage.input_tokens"] = it
    if (ot := usage.get("total_output_tokens")) is not None:
        out["gen_ai.usage.output_tokens"] = ot
    total = usage.get("total_tokens")
    if total is None and it is not None and ot is not None:
        total = int(it) + int(ot)
    if total is not None:
        out["gen_ai.usage.total_tokens"] = total
    for key, attr in (
        ("total_thought_tokens", "gemini.usage.thought_tokens"),
        ("total_cached_tokens", "gemini.usage.cached_tokens"),
        ("total_tool_use_tokens", "gemini.usage.tool_use_tokens"),
    ):
        if (value := usage.get(key)) is not None:
            out[attr] = value
    if model is not None:
        out["gen_ai.request.model"] = str(model)
    out["gen_ai.system"] = "gemini"
    return out


def parse_interactions_response(
    response: dict[str, Any],
    session_id: Optional[str] = None,
) -> Trace:
    """Parse a Gemini Interactions API ``Interaction`` into a Pisama Trace.

    Args:
        response: The decoded JSON of an ``Interaction`` (or a poll response for
            one still running). Accepts ``.model_dump(mode="json")`` output.
        session_id: Session ID to record on ``TraceMetadata``. The Interactions
            API has no session field, so this falls back to ``environment_id``
            (the nearest equivalent) and then to the interaction id.

    Returns:
        A ``Trace`` with one AGENT_TURN root span and one child span per step,
        with result steps parented to the call they answer.
    """
    if not isinstance(response, dict):
        raise TypeError("response must be a dict")

    interaction_id = str(response.get("id") or "interaction")
    model = response.get("model")
    started = _parse_iso(response.get("created"))
    ended = _parse_iso(response.get("updated"))
    status_raw = str(response.get("status") or "")
    errors = [e for e in _as_iterable(response.get("errors")) if isinstance(e, dict)]

    trace = Trace(
        metadata=TraceMetadata(
            session_id=(session_id or response.get("environment_id") or interaction_id),
            platform=Platform.GEMINI,
            platform_version=_PLATFORM_VERSION,
            custom={
                "interaction_id": interaction_id,
                "model": model,
                "previous_interaction_id": response.get("previous_interaction_id"),
                "environment_id": response.get("environment_id"),
            },
        ),
    )

    root = Span(
        trace_id=trace.trace_id,
        name=f"gemini.interaction:{interaction_id}",
        kind=SpanKind.AGENT_TURN,
        platform=Platform.GEMINI,
        platform_metadata={"interaction_id": interaction_id, "model": model},
        start_time=started or _now(),
        end_time=ended,
        status=_STATUS_MAP.get(status_raw, SpanStatus.UNSET),
        error_message=_first_error_message(errors),
        attributes={
            "gemini.interaction.status": status_raw,
            "gemini.errors": errors,
            **_gen_ai_usage_attrs(response.get("usage"), model),
            **(
                {"gemini.system_instruction": response["system_instruction"]}
                if response.get("system_instruction")
                else {}
            ),
        },
        input_data=response.get("input"),
    )
    trace.spans.append(root)

    # Steps carry no timestamps of their own, so they inherit the interaction's
    # start. That is honest about ordering without inventing per-step durations:
    # falling back to _now() would stamp them with ingestion time instead.
    step_start = started or root.start_time
    call_span_by_id: dict[str, str] = {}

    for index, step in enumerate(_as_iterable(response.get("steps"))):
        if not isinstance(step, dict):
            continue
        span = _step_span(step, index, trace.trace_id, root.span_id, step_start, call_span_by_id)
        if span is not None:
            trace.spans.append(span)

    return trace


def _step_span(
    step: dict[str, Any],
    index: int,
    trace_id: str,
    root_id: str,
    start_time: datetime,
    call_span_by_id: dict[str, str],
) -> Optional[Span]:
    """Convert one step into a Span, parenting results to their call."""
    step_type = str(step.get("type") or "unknown")
    kind = _STEP_KINDS.get(step_type, SpanKind.SYSTEM)

    # A result step answers a call step; `call_id` on the result matches `id` on
    # the call. Parenting them makes the tool round trip legible instead of two
    # unrelated siblings.
    parent_id = root_id
    if step_type.endswith(_RESULT_SUFFIX):
        call_id = step.get("call_id")
        if isinstance(call_id, str):
            parent_id = call_span_by_id.get(call_id, root_id)

    name = step.get("name") or step_type
    span = Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"gemini.step:{name}",
        kind=kind,
        platform=Platform.GEMINI,
        start_time=start_time,
        status=_step_status(step),
        error_message=_step_error_message(step),
        attributes={
            "gemini.step.type": step_type,
            "gemini.step.index": index,
            **({"gemini.step.name": step["name"]} if step.get("name") else {}),
        },
        input_data=_step_input(step, step_type),
        output_data=_step_output(step, step_type),
    )

    if not step_type.endswith(_RESULT_SUFFIX) and isinstance(step.get("id"), str):
        call_span_by_id[step["id"]] = span.span_id
    return span


def _step_input(step: dict[str, Any], step_type: str) -> Optional[dict[str, Any]]:
    if step_type == "user_input":
        return {"content": step.get("content")}
    if step_type == "thought":
        # `summary` is a list of content blocks, not a string.
        return {"summary": step.get("summary"), "signature": step.get("signature")}
    if step_type.endswith(_RESULT_SUFFIX):
        return None
    # Call steps: `arguments` is already a dict on this API, unlike OpenAI's
    # JSON-string arguments, so it is passed through rather than parsed.
    args = step.get("arguments")
    return {"arguments": args} if args is not None else None


def _step_output(step: dict[str, Any], step_type: str) -> Optional[dict[str, Any]]:
    if step_type == "model_output":
        return {"content": step.get("content")}
    if step_type.endswith(_RESULT_SUFFIX):
        # `result` is a union: a free-form object, a list of content blocks, or a
        # plain string. It is preserved as-is rather than coerced to text.
        return {"result": step.get("result")}
    return None


def _step_status(step: dict[str, Any]) -> SpanStatus:
    if step.get("is_error") is True:
        return SpanStatus.ERROR
    # `model_output.error` is a google.rpc Status ({code:int, message, details}),
    # a different type from the interaction-level Error ({code:str, message}).
    error = step.get("error")
    if isinstance(error, dict) and (error.get("code") or error.get("message")):
        return SpanStatus.ERROR
    return SpanStatus.OK


def _step_error_message(step: dict[str, Any]) -> Optional[str]:
    error = step.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if step.get("is_error") is True:
        return _stringify_result(step.get("result"))
    return None


def _stringify_result(result: Any) -> Optional[str]:
    if result is None:
        return None
    if isinstance(result, str):
        return result
    return str(result)


def _first_error_message(errors: list[dict[str, Any]]) -> Optional[str]:
    for error in errors:
        if error.get("message"):
            return str(error["message"])
    return None


def _as_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return value
    return ()


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse the ISO-8601 strings the Interactions API uses for time.

    `created` / `updated` are strings such as ``2026-08-12T09:14:02Z``. An
    earlier version accepted epoch numbers, which this API never sends.
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
