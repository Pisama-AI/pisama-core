"""Google Agent Development Kit (ADK) trace ingestion adapter.

ADK (https://google.github.io/adk-docs) is Google's open-source agent
framework that runs on Vertex AI Agent Engine. It exposes three primitives
Pisama cares about:

- ``InvocationContext`` — the root execution scope that groups every event
  a single user turn produces (LLM calls, tool calls, sub-agent spawns,
  state writes).
- ``session.state`` — a key/value store that agents read/write between
  steps. State transitions are emitted as deltas.
- ``before_*_callback`` / ``after_*_callback`` hooks — real pre-execution
  points where a developer can intercept or rewrite a request. This is
  what makes fix injection feasible for ADK (unlike Deep Agents, which is
  ingestion-only).

Pisama ingests ADK traces as a sequence of event dicts. Each dict may
carry one of the following ``type`` values:

- ``agent.invocation.start`` / ``agent.invocation.end`` — root bookends
- ``agent.transfer``                                    — handoff to a peer
- ``sub_agent.start`` / ``sub_agent.end``               — nested agent run
- ``tool.function_call`` / ``tool.function_response``   — tool I/O pair
- ``llm.request`` / ``llm.response``                    — model call boundaries
- ``session.state_delta``                               — state-change event
- ``planner.plan`` / ``planner.replan``                 — plan/todo emission
- ``code_executor.*``                                   — code-interpreter call
- ``retrieval.search``                                  — RAG lookup
- ``before_*_callback`` / ``after_*_callback``          — hook fires

The adapter does NOT import the ``google-adk`` package. It accepts dicts
matching the documented payload so callers can pass whatever their runtime
exposes through its OTEL exporter or callback payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from pisama_core.adapters.base import (
    InjectionMethod,
    InjectionResult,
    PlatformAdapter,
)
from pisama_core.injection.enforcement import EnforcementLevel
from pisama_core.traces.enums import Platform, SpanKind, SpanStatus
from pisama_core.traces.models import Event, Span, Trace, TraceMetadata

__all__ = [
    "GoogleAdkAdapter",
    "parse_adk_trace",
]


_PLATFORM_VERSION = "adk-v1"


def _gen_ai_usage_attrs(usage: Any, model: Any = None) -> dict[str, Any]:
    """Translate an ADK ``usage_metadata`` payload to OTEL ``gen_ai.usage.*``.

    ADK follows the Gemini SDK shape (``prompt_token_count``,
    ``candidates_token_count``, ``total_token_count``). Detectors across the
    codebase read ``gen_ai.usage.*`` only, so every adapter must translate
    to the OTEL naming to keep detectors vendor-neutral.
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
    out["gen_ai.system"] = "google_adk"
    return out


def parse_adk_trace(
    events: Iterable[dict[str, Any]],
    invocation_id: Optional[str] = None,
    agent_name: str = "adk_agent",
    session_id: Optional[str] = None,
) -> Trace:
    """Parse an ADK event stream into a Pisama Trace.

    Args:
        events: Iterable of ADK event dicts (see module docstring for the
            supported ``type`` values).
        invocation_id: The ``InvocationContext`` id; becomes the root span
            attribute ``google.adk.invocation.id``.
        agent_name: Root agent name (used in the root span name).
        session_id: Optional ADK session id; falls back to ``invocation_id``.

    Returns:
        A ``Trace`` with one AGENT root span and child spans per event.
    """
    trace = Trace(
        metadata=TraceMetadata(
            session_id=session_id or invocation_id or "adk-session",
            platform=Platform.GOOGLE_ADK,
            platform_version=_PLATFORM_VERSION,
            custom={
                "invocation_id": invocation_id,
                "agent_name": agent_name,
            },
        ),
    )

    root = Span(
        trace_id=trace.trace_id,
        name=f"adk.agent:{agent_name}",
        kind=SpanKind.AGENT,
        platform=Platform.GOOGLE_ADK,
        platform_metadata={"agent_name": agent_name, "invocation_id": invocation_id},
        start_time=_now(),
        status=SpanStatus.OK,
        attributes={
            "google.adk.agent.name": agent_name,
            "google.adk.invocation.id": invocation_id or "",
        },
    )
    trace.spans.append(root)

    # Per-agent parent stack: lets nested sub-agents parent their children
    # correctly. Root is always at the bottom.
    agent_stack: list[str] = [root.span_id]
    # Pending tool call span_ids keyed by function_call_id so responses can
    # parent to their call.
    pending_tool_calls: dict[str, str] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = str(event.get("type") or event.get("event") or "").lower()
        parent_id = agent_stack[-1]

        if etype in ("agent.invocation.start",):
            # Root already emitted; record a state-delta-like event to the root
            # so detectors see the start marker in timeline order.
            root.events.append(
                Event(
                    name="adk.invocation.start",
                    attributes={"agent": event.get("agent") or agent_name},
                )
            )

        elif etype == "agent.invocation.end":
            # Finalize root.
            root.end_time = _parse_iso(event.get("timestamp")) or _now()
            if event.get("error"):
                root.status = SpanStatus.ERROR
                root.error_message = str(event["error"])

        elif etype == "agent.transfer":
            trace.spans.append(_handoff_span(event, trace.trace_id, parent_id))

        elif etype == "sub_agent.start":
            sub = _sub_agent_span(event, trace.trace_id, parent_id)
            trace.spans.append(sub)
            agent_stack.append(sub.span_id)

        elif etype == "sub_agent.end":
            # Close the top sub-agent span. Never pop the root.
            if len(agent_stack) > 1:
                closed_id = agent_stack.pop()
                for span in trace.spans:
                    if span.span_id == closed_id:
                        span.end_time = _parse_iso(event.get("timestamp")) or _now()
                        if event.get("error"):
                            span.status = SpanStatus.ERROR
                            span.error_message = str(event["error"])
                        break

        elif etype == "tool.function_call":
            call = _tool_call_span(event, trace.trace_id, parent_id)
            trace.spans.append(call)
            call_id = event.get("function_call_id") or event.get("call_id")
            if call_id:
                pending_tool_calls[str(call_id)] = call.span_id

        elif etype == "tool.function_response":
            call_id = event.get("function_call_id") or event.get("call_id")
            call_span_id = pending_tool_calls.pop(str(call_id), None) if call_id else None
            trace.spans.append(
                _tool_response_span(event, trace.trace_id, call_span_id or parent_id)
            )

        elif etype in ("llm.request", "llm.response"):
            trace.spans.append(_llm_span(event, etype, trace.trace_id, parent_id))

        elif etype == "session.state_delta":
            root.events.append(
                Event(
                    name="state_delta",
                    attributes={
                        "changed_keys": sorted(list((event.get("delta") or {}).keys())),
                        "delta": event.get("delta") or {},
                        "agent": event.get("agent"),
                    },
                )
            )

        elif etype in ("planner.plan", "planner.replan"):
            trace.spans.append(_plan_span(event, etype, trace.trace_id, parent_id))

        elif etype.startswith("code_executor."):
            trace.spans.append(_code_executor_span(event, trace.trace_id, parent_id))

        elif etype.startswith("retrieval."):
            trace.spans.append(_retrieval_span(event, trace.trace_id, parent_id))

        elif (
            etype.endswith("_callback") or etype.startswith("before_") or etype.startswith("after_")
        ):
            trace.spans.append(_hook_span(event, etype, trace.trace_id, parent_id))

    # Ensure root has an end_time for latency metrics.
    if root.end_time is None:
        root.end_time = _now()
    return trace


def _handoff_span(event: dict[str, Any], trace_id: str, parent_id: str) -> Span:
    target = str(event.get("to") or event.get("target") or "unknown")
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"adk.handoff:{target}",
        kind=SpanKind.HANDOFF,
        platform=Platform.GOOGLE_ADK,
        start_time=_parse_iso(event.get("timestamp")) or _now(),
        end_time=_parse_iso(event.get("timestamp")) or _now(),
        status=SpanStatus.OK,
        attributes={
            "google.adk.handoff.from": event.get("from"),
            "google.adk.handoff.to": target,
        },
        output_data={"message": event.get("message")} if event.get("message") else None,
    )


def _sub_agent_span(event: dict[str, Any], trace_id: str, parent_id: str) -> Span:
    name = str(event.get("agent") or event.get("name") or "sub_agent")
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"adk.sub_agent:{name}",
        kind=SpanKind.AGENT,
        platform=Platform.GOOGLE_ADK,
        start_time=_parse_iso(event.get("timestamp")) or _now(),
        status=SpanStatus.IN_PROGRESS,
        attributes={
            "google.adk.agent.name": name,
            "google.adk.sub_agent.isolated_context": bool(event.get("isolated_context", True)),
        },
        input_data={"task": event.get("task")} if event.get("task") else None,
    )


def _tool_call_span(event: dict[str, Any], trace_id: str, parent_id: str) -> Span:
    name = str(event.get("name") or event.get("function") or "tool")
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"adk.tool:{name}",
        kind=SpanKind.TOOL,
        platform=Platform.GOOGLE_ADK,
        start_time=_parse_iso(event.get("timestamp")) or _now(),
        status=SpanStatus.IN_PROGRESS,
        attributes={
            "adk.tool.name": name,
            "adk.tool.call_id": event.get("function_call_id") or event.get("call_id"),
        },
        input_data={"arguments": event.get("args") or event.get("arguments")},
    )


def _tool_response_span(event: dict[str, Any], trace_id: str, parent_id: str) -> Span:
    name = str(event.get("name") or "tool")
    errored = bool(event.get("error"))
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"adk.tool_output:{name}",
        kind=SpanKind.TOOL,
        platform=Platform.GOOGLE_ADK,
        start_time=_parse_iso(event.get("timestamp")) or _now(),
        end_time=_parse_iso(event.get("timestamp")) or _now(),
        status=SpanStatus.ERROR if errored else SpanStatus.OK,
        error_message=str(event["error"]) if errored else None,
        attributes={"adk.tool.name": name},
        output_data={"result": event.get("response") or event.get("result")},
    )


def _llm_span(event: dict[str, Any], etype: str, trace_id: str, parent_id: str) -> Span:
    model = event.get("model")
    usage = event.get("usage_metadata") or event.get("usage")
    errored = bool(event.get("error"))
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"adk.{etype}",
        kind=SpanKind.LLM,
        platform=Platform.GOOGLE_ADK,
        start_time=_parse_iso(event.get("timestamp")) or _now(),
        end_time=_parse_iso(event.get("timestamp")) or _now() if etype == "llm.response" else None,
        status=SpanStatus.ERROR if errored else SpanStatus.OK,
        error_message=str(event["error"]) if errored else None,
        attributes={
            "adk.llm.model": model,
            **_gen_ai_usage_attrs(usage, model=model),
        },
        input_data={"prompt": event.get("prompt"), "messages": event.get("messages")}
        if etype == "llm.request"
        else None,
        output_data={"text": event.get("text"), "candidates": event.get("candidates")}
        if etype == "llm.response"
        else None,
    )


def _plan_span(event: dict[str, Any], etype: str, trace_id: str, parent_id: str) -> Span:
    plan = event.get("plan") or event.get("todos") or []
    goals = [_goal_text(item) for item in plan if item is not None]
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"adk.{etype}",
        kind=SpanKind.TASK,
        platform=Platform.GOOGLE_ADK,
        start_time=_parse_iso(event.get("timestamp")) or _now(),
        end_time=_parse_iso(event.get("timestamp")) or _now(),
        status=SpanStatus.OK,
        attributes={
            "adk.planner.event": etype,
            "adk.plan.size": len(plan),
            # `goals` is the canonical attribute the decomposition /
            # completion detectors read.
            "goals": goals,
        },
        output_data={"plan": plan},
    )


def _code_executor_span(event: dict[str, Any], trace_id: str, parent_id: str) -> Span:
    name = str(event.get("name") or "code_executor")
    errored = bool(event.get("error"))
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"adk.{name}",
        kind=SpanKind.TOOL,
        platform=Platform.GOOGLE_ADK,
        start_time=_parse_iso(event.get("timestamp")) or _now(),
        end_time=_parse_iso(event.get("timestamp")) or _now(),
        status=SpanStatus.ERROR if errored else SpanStatus.OK,
        error_message=str(event["error"]) if errored else None,
        attributes={
            "adk.tool.type": "code_executor",
            "adk.tool.name": name,
        },
        input_data={"code": event.get("code")} if event.get("code") else None,
        output_data={"output": event.get("output")} if event.get("output") else None,
    )


def _retrieval_span(event: dict[str, Any], trace_id: str, parent_id: str) -> Span:
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"adk.{event.get('type') or 'retrieval.search'}",
        kind=SpanKind.RETRIEVAL,
        platform=Platform.GOOGLE_ADK,
        start_time=_parse_iso(event.get("timestamp")) or _now(),
        end_time=_parse_iso(event.get("timestamp")) or _now(),
        status=SpanStatus.OK,
        attributes={"adk.retrieval.source": event.get("source")},
        input_data={"query": event.get("query")},
        output_data={"results": event.get("results")} if event.get("results") is not None else None,
    )


def _hook_span(event: dict[str, Any], etype: str, trace_id: str, parent_id: str) -> Span:
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"adk.{etype}",
        kind=SpanKind.HOOK,
        platform=Platform.GOOGLE_ADK,
        start_time=_parse_iso(event.get("timestamp")) or _now(),
        end_time=_parse_iso(event.get("timestamp")) or _now(),
        status=SpanStatus.OK,
        attributes={"adk.hook.event": etype},
        input_data=event.get("input"),
        output_data=event.get("output"),
    )


def _goal_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("content") or item.get("text") or item.get("goal") or item)
    return str(item)


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


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# PlatformAdapter class (supports fix injection via ADK callbacks)
# ─────────────────────────────────────────────────────────────────────────────


class GoogleAdkAdapter(PlatformAdapter):
    """Adapter for Google Agent Development Kit traces.

    ADK exposes real pre-execution hooks (``before_agent_callback``,
    ``before_model_callback``, ``before_tool_callback``). Unlike
    ``DeepAgentsAdapter`` and the parse-only OpenAI/Bedrock adapters,
    ``GoogleAdkAdapter.inject_fix`` returns a structured ``InjectionResult``
    that a developer-registered callback (installed on the ADK ``LlmAgent``)
    can consume. Pisama does not itself execute the callback; it hands the
    directive across the boundary.
    """

    def __init__(
        self,
        invocation_id: Optional[str] = None,
        agent_name: str = "adk_agent",
        session_id: Optional[str] = None,
    ) -> None:
        self._invocation_id = invocation_id
        self._agent_name = agent_name
        self._session_id = session_id
        self._last_event: dict[str, Any] = {}
        # The inbox a developer-installed `before_*_callback` reads to fetch
        # pending directives. The callback should pop from here and apply.
        self._pending_directives: list[InjectionResult] = []

    # -- identity -------------------------------------------------------------

    @property
    def platform_name(self) -> Platform:
        return Platform.GOOGLE_ADK

    @property
    def platform_version(self) -> Optional[str]:
        return _PLATFORM_VERSION

    # -- trace capture --------------------------------------------------------

    def parse_trace(
        self,
        events: Iterable[dict[str, Any]],
        invocation_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Trace:
        return parse_adk_trace(
            events,
            invocation_id=invocation_id or self._invocation_id,
            agent_name=agent_name or self._agent_name,
            session_id=session_id or self._session_id,
        )

    def capture_span(self, raw_data: Any) -> Span:
        """Convert a single ADK event dict into a Span.

        Used for realtime capture (one event at a time). For full-trace
        ingestion, prefer ``parse_trace``.
        """
        if not isinstance(raw_data, dict):
            raise TypeError("GoogleAdkAdapter.capture_span expects a dict event")
        trace = parse_adk_trace(
            [raw_data],
            invocation_id=self._invocation_id,
            agent_name=self._agent_name,
            session_id=self._session_id,
        )
        self._last_event = raw_data
        # Return the first non-root span if present; otherwise return root.
        return trace.spans[1] if len(trace.spans) > 1 else trace.spans[0]

    # -- state / context ------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        return {
            "invocation_id": self._invocation_id,
            "agent_name": self._agent_name,
            "session_id": self._session_id,
            "last_event": self._last_event,
            "pending_directives": len(self._pending_directives),
        }

    def get_session_context(self) -> dict[str, Any]:
        return {
            "invocation_id": self._invocation_id,
            "pending_directives": [d.directive_id for d in self._pending_directives],
        }

    # -- injection / blocking -------------------------------------------------

    def get_supported_injection_methods(self) -> list[InjectionMethod]:
        return [
            InjectionMethod.CALLBACK,
            InjectionMethod.STATE,
            InjectionMethod.MESSAGE,
        ]

    def inject_fix(
        self,
        directive: str,
        level: EnforcementLevel,
        directive_id: Optional[str] = None,
    ) -> InjectionResult:
        """Queue a directive for the next ADK callback to consume.

        The developer is expected to register a ``before_agent_callback``
        (or ``before_model_callback`` / ``before_tool_callback``) that
        drains ``adapter._pending_directives`` and applies the directives
        — typically by prepending ``directive`` to the request messages or
        by setting ``session.state`` keys before the model is invoked.
        Pisama does not execute the callback itself; ADK runs it in-process.
        """
        result = InjectionResult(
            success=True,
            method=InjectionMethod.CALLBACK,
            directive_id=directive_id,
            message=directive,
            blocked=level >= EnforcementLevel.BLOCK,
        )
        self._pending_directives.append(result)
        return result

    def can_block(self) -> bool:
        return True

    def block_action(self, reason: str) -> bool:
        """Signal that the next callback should raise to block the action.

        Returns True; ADK callbacks can return ``None``/raise to short-circuit
        the LLM or tool call. The caller's ``before_*_callback`` reads
        ``adapter._pending_directives`` and enforces the block.
        """
        self._pending_directives.append(
            InjectionResult(
                success=True,
                method=InjectionMethod.CALLBACK,
                message=reason,
                blocked=True,
            )
        )
        return True

    # -- developer helper -----------------------------------------------------

    def drain_pending_directives(self) -> list[InjectionResult]:
        """Pop all pending directives for the callback to apply.

        Meant to be called from within a user-registered ``before_*_callback``.
        Returns the queued directives in FIFO order and clears the inbox.
        """
        out = list(self._pending_directives)
        self._pending_directives.clear()
        return out
