"""OpenAI trace ingestion adapter.

Converts OpenAI Assistants API and Responses API traces into Pisama's
universal Span/Trace format. Ingestion-only — OpenAI does not expose a
pre-execution hook API comparable to Claude Code, so blocking and fix
injection are not supported.

Supported inputs:
- OpenAI Assistants API: `Run` object + `run.steps` list
- OpenAI Responses API: `Response` object with `output` array

The adapter does not import the `openai` package. It accepts dicts that
match the documented JSON shape so callers can pass whatever their SDK
version returns (`.model_dump()`, `.dict()`, or the raw HTTP payload).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from pisama_core.traces.enums import Platform, SpanKind, SpanStatus
from pisama_core.traces.models import Span, Trace, TraceMetadata

__all__ = [
    "parse_assistants_run",
    "parse_response",
]


def _gen_ai_usage_attrs(usage: Any, model: Any = None) -> dict[str, Any]:
    """Translate an OpenAI `usage` payload into OTEL `gen_ai.usage.*` attrs.

    Detectors read `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`
    / `gen_ai.usage.total_tokens` regardless of vendor, so every adapter
    that carries token usage must emit these keys. Leaving only the
    platform-native `openai.usage` dict forces every detector to add a
    vendor branch, which is how "vendor-neutral" turns into per-vendor
    bridge code.
    """
    if not isinstance(usage, dict):
        return {}
    out: dict[str, Any] = {}
    # The two OpenAI surfaces name these differently: Assistants reports
    # prompt_tokens/completion_tokens, the Responses API reports
    # input_tokens/output_tokens. Accepting only the former silently dropped
    # every Responses-API token split.
    if (pt := usage.get("prompt_tokens", usage.get("input_tokens"))) is not None:
        out["gen_ai.usage.input_tokens"] = pt
    if (ct := usage.get("completion_tokens", usage.get("output_tokens"))) is not None:
        out["gen_ai.usage.output_tokens"] = ct
    if (tt := usage.get("total_tokens")) is not None:
        out["gen_ai.usage.total_tokens"] = tt
    if model is not None:
        out["gen_ai.request.model"] = str(model)
    out["gen_ai.system"] = "openai"
    return out


# --- Assistants API ---


def parse_assistants_run(
    run: dict[str, Any],
    steps: Optional[Iterable[dict[str, Any]]] = None,
    thread_messages: Optional[Iterable[dict[str, Any]]] = None,
) -> Trace:
    """Parse an OpenAI Assistants API `Run` + its steps into a Pisama Trace.

    Args:
        run: The `thread.run` object. Must include `id`, `assistant_id`,
            `thread_id`, `status`. `created_at`, `completed_at`,
            `failed_at`, and `usage` are read when present.
        steps: Iterable of `thread.run.step` objects. Each step becomes a
            child span. `tool_calls` inside `step_details` produce TOOL
            spans; `message_creation` produces MESSAGE spans.
        thread_messages: Optional iterable of `thread.message` objects to
            attach to the trace as USER_INPUT / USER_OUTPUT spans.

    Returns:
        A `Trace` with:
        - root `AGENT` span for the run
        - child spans for each run step
        - additional spans for thread messages when provided
    """
    run_id = str(run.get("id") or "")
    assistant_id = str(run.get("assistant_id") or "")
    thread_id = str(run.get("thread_id") or "")
    status_raw = str(run.get("status") or "unset")

    trace = Trace(
        trace_id=run_id or Trace().trace_id,
        metadata=TraceMetadata(
            session_id=thread_id or run_id,
            platform=Platform.OPENAI,
            platform_version="assistants-v2",
            custom={
                "assistant_id": assistant_id,
                "thread_id": thread_id,
                "model": run.get("model"),
                "usage": run.get("usage"),
            },
        ),
    )

    root = Span(
        trace_id=trace.trace_id,
        name=f"openai.run:{run_id}" if run_id else "openai.run",
        kind=SpanKind.AGENT,
        platform=Platform.OPENAI,
        platform_metadata={
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "model": run.get("model"),
        },
        start_time=_ts(run.get("created_at")) or _now(),
        end_time=_ts(run.get("completed_at") or run.get("failed_at") or run.get("cancelled_at")),
        status=_map_run_status(status_raw),
        error_message=_extract_error(run),
        attributes={
            "openai.run.status": status_raw,
            "openai.usage": run.get("usage") or {},
            **_gen_ai_usage_attrs(run.get("usage"), run.get("model")),
        },
    )
    trace.spans.append(root)

    for message in thread_messages or []:
        trace.spans.append(_parse_thread_message(message, trace.trace_id, root.span_id))

    for step in steps or []:
        trace.spans.extend(_parse_run_step(step, trace.trace_id, root.span_id))

    return trace


def _parse_run_step(
    step: dict[str, Any],
    trace_id: str,
    parent_id: str,
) -> list[Span]:
    """Convert one `thread.run.step` into one or more Spans."""
    step_id = str(step.get("id") or "")
    step_type = str(step.get("type") or "unknown")
    details = step.get("step_details") or {}

    spans: list[Span] = []

    if step_type == "tool_calls":
        for call in details.get("tool_calls") or []:
            spans.append(_parse_tool_call(call, trace_id, parent_id, step))
        return spans

    if step_type == "code_interpreter":
        # A code_interpreter step has one source code input and zero or more
        # structured outputs (type="logs" with `logs` text, or type="image"
        # with `image.file_id`). Flattening outputs to `str(...)` loses the
        # type/file-id distinction, so preserve the structured list.
        ci = details.get("code_interpreter") or {}
        spans.append(
            Span(
                trace_id=trace_id,
                parent_id=parent_id,
                name=f"openai.code_interpreter:{step_id}",
                kind=SpanKind.TOOL,
                platform=Platform.OPENAI,
                start_time=_ts(step.get("created_at")) or _now(),
                end_time=_ts(step.get("completed_at") or step.get("failed_at")),
                status=_map_run_status(str(step.get("status") or "unset")),
                attributes={
                    "openai.step.id": step_id,
                    "openai.step.type": step_type,
                    "openai.tool.type": "code_interpreter",
                },
                input_data={"input": ci.get("input")},
                output_data={"outputs": ci.get("outputs") or []},
            )
        )
        return spans

    if step_type == "message_creation":
        msg_id = (details.get("message_creation") or {}).get("message_id")
        spans.append(
            Span(
                trace_id=trace_id,
                parent_id=parent_id,
                name=f"openai.message_creation:{msg_id or step_id}",
                kind=SpanKind.MESSAGE,
                platform=Platform.OPENAI,
                start_time=_ts(step.get("created_at")) or _now(),
                end_time=_ts(step.get("completed_at") or step.get("failed_at")),
                status=_map_run_status(str(step.get("status") or "unset")),
                attributes={
                    "openai.step.id": step_id,
                    "openai.message.id": msg_id,
                },
                output_data={"message_id": msg_id},
            )
        )
        return spans

    # Fallback — keep the step as a generic system span so the trace does
    # not silently lose information for unknown step types.
    spans.append(
        Span(
            trace_id=trace_id,
            parent_id=parent_id,
            name=f"openai.step:{step_type}",
            kind=SpanKind.SYSTEM,
            platform=Platform.OPENAI,
            start_time=_ts(step.get("created_at")) or _now(),
            end_time=_ts(step.get("completed_at") or step.get("failed_at")),
            status=_map_run_status(str(step.get("status") or "unset")),
            attributes={"openai.step.id": step_id, "openai.step.type": step_type},
            input_data=details,
        )
    )
    return spans


def _parse_tool_call(
    call: dict[str, Any],
    trace_id: str,
    parent_id: str,
    step: dict[str, Any],
) -> Span:
    call_type = str(call.get("type") or "function")
    # Every tool-call variant nests its payload under a key named after its own
    # variant name: `function` holds {name, arguments, output}, while
    # `code_interpreter` holds {input, outputs} and `file_search` holds
    # {ranking_options, results}. Reading only `function` silently reduced every
    # other variant to nulls, discarding the executed code and its outputs.
    #
    # (Keep "type:" off the start of any line here: mypy reads `# type:` as a
    # PEP 484 type comment and fails the file with a syntax error.)
    payload = call.get(call_type)
    payload = payload if isinstance(payload, dict) else {}
    name = str(payload.get("name") or call_type)
    if call_type == "function":
        input_data: dict[str, Any] = {"arguments": payload.get("arguments")}
        output_data: dict[str, Any] = {"output": payload.get("output")}
    elif call_type == "code_interpreter":
        input_data = {"input": payload.get("input")}
        output_data = {"outputs": payload.get("outputs")}
    elif call_type == "file_search":
        input_data = {"ranking_options": payload.get("ranking_options")}
        output_data = {"results": payload.get("results")}
    else:
        # An unrecognised variant keeps its payload verbatim rather than being
        # dropped; a future OpenAI tool type degrades to "unparsed" not "lost".
        input_data = {"payload": payload}
        output_data = {}
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"openai.tool:{name}",
        kind=SpanKind.TOOL,
        platform=Platform.OPENAI,
        start_time=_ts(step.get("created_at")) or _now(),
        end_time=_ts(step.get("completed_at") or step.get("failed_at")),
        # Assistants API v2 does not expose per-tool-call `status` or
        # `last_error`; failure for an individual call surfaces only via
        # the parent run's `last_error` and the overall step status. Once
        # OpenAI adds per-call failure fields, switch to reading them from
        # `call` directly to stop masking partial failures.
        status=_map_run_status(str(step.get("status") or "unset")),
        attributes={
            "openai.tool.type": call_type,
            "openai.tool.name": name,
        },
        input_data=input_data,
        output_data=output_data,
    )


def _parse_thread_message(
    message: dict[str, Any],
    trace_id: str,
    parent_id: str,
) -> Span:
    role = str(message.get("role") or "assistant")
    kind = SpanKind.USER_INPUT if role == "user" else SpanKind.USER_OUTPUT
    content = message.get("content") or []
    text = _extract_message_text(content)
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"openai.message:{role}",
        kind=kind,
        platform=Platform.OPENAI,
        start_time=_ts(message.get("created_at")) or _now(),
        status=SpanStatus.OK,
        attributes={"openai.message.role": role, "openai.message.id": message.get("id")},
        output_data={"text": text} if text else None,
    )


# --- Responses API ---


def parse_response(response: dict[str, Any]) -> Trace:
    """Parse an OpenAI Responses API `Response` into a Pisama Trace.

    Args:
        response: The Responses API response object. Expected keys:
            `id`, `model`, `created_at`, `status`, `output` (list of
            items with `type` in {"message", "function_call",
            "function_call_output", "reasoning"}).

    Returns:
        A `Trace` with one root span per response and one child span per
        output item.
    """
    resp_id = str(response.get("id") or "")
    status_raw = str(response.get("status") or "completed")

    trace = Trace(
        trace_id=resp_id or Trace().trace_id,
        metadata=TraceMetadata(
            session_id=resp_id,
            platform=Platform.OPENAI,
            platform_version="responses-v1",
            custom={"model": response.get("model"), "usage": response.get("usage")},
        ),
    )

    root = Span(
        trace_id=trace.trace_id,
        name=f"openai.response:{resp_id}" if resp_id else "openai.response",
        kind=SpanKind.AGENT_TURN,
        platform=Platform.OPENAI,
        platform_metadata={"model": response.get("model")},
        start_time=_ts(response.get("created_at")) or _now(),
        end_time=_ts(response.get("completed_at")),
        status=_map_response_status(status_raw),
        error_message=_extract_error(response),
        attributes={
            "openai.response.status": status_raw,
            "openai.usage": response.get("usage") or {},
            **_gen_ai_usage_attrs(response.get("usage"), response.get("model")),
        },
    )
    trace.spans.append(root)

    for item in response.get("output") or []:
        trace.spans.append(_parse_response_output_item(item, trace.trace_id, root.span_id))

    return trace


def _parse_response_output_item(
    item: dict[str, Any],
    trace_id: str,
    parent_id: str,
) -> Span:
    item_type = str(item.get("type") or "message")

    if item_type == "function_call":
        return Span(
            trace_id=trace_id,
            parent_id=parent_id,
            name=f"openai.tool:{item.get('name') or 'function'}",
            kind=SpanKind.TOOL,
            platform=Platform.OPENAI,
            status=SpanStatus.OK,
            attributes={
                "openai.tool.name": item.get("name"),
                "openai.call.id": item.get("call_id"),
            },
            input_data={"arguments": item.get("arguments")},
        )

    if item_type == "function_call_output":
        return Span(
            trace_id=trace_id,
            parent_id=parent_id,
            name=f"openai.tool_output:{item.get('call_id') or ''}",
            kind=SpanKind.TOOL,
            platform=Platform.OPENAI,
            status=SpanStatus.OK,
            attributes={"openai.call.id": item.get("call_id")},
            output_data={"output": item.get("output")},
        )

    if item_type == "reasoning":
        return Span(
            trace_id=trace_id,
            parent_id=parent_id,
            name="openai.reasoning",
            kind=SpanKind.LLM,
            platform=Platform.OPENAI,
            status=SpanStatus.OK,
            output_data={"summary": item.get("summary")},
        )

    # Default: message
    text = _extract_message_text(item.get("content") or [])
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name="openai.message",
        kind=SpanKind.MESSAGE,
        platform=Platform.OPENAI,
        status=SpanStatus.OK,
        attributes={"openai.message.role": item.get("role")},
        output_data={"text": text} if text else None,
    )


# --- Utilities ---


def _ts(value: Any) -> Optional[datetime]:
    """Coerce an OpenAI timestamp to a datetime.

    OpenAI uses Unix seconds in most places and ISO-8601 strings in a
    few (Responses API `completed_at`). Accept both; return None for
    missing values.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _map_run_status(status: str) -> SpanStatus:
    mapping = {
        "queued": SpanStatus.UNSET,
        "in_progress": SpanStatus.IN_PROGRESS,
        "requires_action": SpanStatus.IN_PROGRESS,
        "cancelling": SpanStatus.IN_PROGRESS,
        "completed": SpanStatus.OK,
        "incomplete": SpanStatus.ERROR,
        "failed": SpanStatus.ERROR,
        "cancelled": SpanStatus.CANCELLED,
        "expired": SpanStatus.TIMEOUT,
    }
    return mapping.get(status, SpanStatus.UNSET)


def _map_response_status(status: str) -> SpanStatus:
    mapping = {
        "completed": SpanStatus.OK,
        "in_progress": SpanStatus.IN_PROGRESS,
        "failed": SpanStatus.ERROR,
        "incomplete": SpanStatus.ERROR,
        "cancelled": SpanStatus.CANCELLED,
    }
    return mapping.get(status, SpanStatus.UNSET)


def _extract_error(obj: dict[str, Any]) -> Optional[str]:
    err = obj.get("last_error") or obj.get("error")
    if not err:
        return None
    if isinstance(err, str):
        return err
    if isinstance(err, dict):
        return err.get("message") or str(err)
    return str(err)


def _extract_message_text(content: Any) -> str:
    """Flatten OpenAI content blocks (list of {type, text/...}) to a string."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in ("text", "output_text", "input_text"):
            val = block.get("text")
            if isinstance(val, dict):
                val = val.get("value")
            if val:
                parts.append(str(val))
    return "\n".join(parts)
