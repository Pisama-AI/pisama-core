"""LangChain Deep Agents trace ingestion adapter.

Deep Agents (announced 2026-03-15) is a structured multi-step agent runtime
built on LangGraph that adds three primitives:

- ``write_todos`` — a planning tool whose arguments become the agent's
  todo-list / plan state. Each call produces a new AgentState plan.
- ``task`` — a subagent spawn tool that runs a child agent in an isolated
  context window. Child traces are represented here as nested spans.
- LangGraph Memory Store — persistent key-value state across runs; state
  transitions between graph nodes are captured as state-delta events.

Pisama ingests Deep Agents traces as a sequence of state dicts (one per
graph checkpoint). Each state dict may contain:

- ``messages`` — LangChain message objects (or dicts).
- ``todos`` — the current plan, populated by ``write_todos`` tool calls.
- ``subagents`` — list of child-agent spawn events (populated by ``task``).
- ``node`` — the graph node name that produced this state (``agent``,
  ``tools``, ``planner``, ``supervisor`` etc.).

The adapter does NOT import ``langchain``, ``langgraph`` or ``deepagents``.
It accepts dicts matching the documented payload shape so callers can pass
whatever their runtime exposes via ``.get_state()`` / checkpointers.

Ingestion-only — Deep Agents' runtime does not expose a pre-execution hook
API comparable to Claude Code, so blocking and fix injection are not
supported here. The Deep Agents runtime provides its own guardrails; Pisama
complements those with external forensics after execution.
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
    "DeepAgentsAdapter",
    "parse_deep_agents_trace",
]


# ─────────────────────────────────────────────────────────────────────────────
# Functional parser (mirrors openai.parse_response / bedrock.parse_invoke_agent)
# ─────────────────────────────────────────────────────────────────────────────


def parse_deep_agents_trace(
    states: Iterable[dict[str, Any]],
    session_id: Optional[str] = None,
    agent_name: str = "deep_agent",
) -> Trace:
    """Parse a Deep Agents trace into a Pisama Trace.

    Args:
        states: Iterable of state-dict checkpoints from the Deep Agents
            runtime. Each dict may contain ``messages``, ``todos``,
            ``subagents``, and ``node`` fields.
        session_id: Optional LangGraph thread / session id.
        agent_name: Name of the root agent (shown as root span name).

    Returns:
        A ``Trace`` with:
        - root ``AGENT`` span for the run
        - ``TASK`` spans for each ``write_todos`` plan
        - ``AGENT`` child spans for each subagent spawn (isolated context)
        - ``TOOL`` / ``MESSAGE`` / ``LLM`` spans for per-node activity
        - state-delta events attached to the root span
    """
    trace = Trace(
        metadata=TraceMetadata(
            session_id=session_id or Trace().trace_id,
            platform=Platform.LANGGRAPH,
            platform_version="deep-agents-v1",
            custom={"runtime": "deep_agents", "agent_name": agent_name},
        ),
    )

    root = Span(
        trace_id=trace.trace_id,
        name=f"deep_agents.agent:{agent_name}",
        kind=SpanKind.AGENT,
        platform=Platform.LANGGRAPH,
        platform_metadata={"runtime": "deep_agents", "agent_name": agent_name},
        start_time=_now(),
        status=SpanStatus.OK,
        attributes={"deep_agents.session_id": session_id or ""},
    )
    trace.spans.append(root)

    prev_state: dict[str, Any] = {}
    step_index = 0

    for state in states:
        if not isinstance(state, dict):
            continue

        node_name = str(state.get("node") or f"step_{step_index}")

        # 1. write_todos → plan/goals span (TASK kind).
        todos = _extract_todos(state)
        if todos and todos != prev_state.get("todos"):
            trace.spans.append(_todos_span(todos, node_name, trace.trace_id, root.span_id))

        # 2. subagent spawns → child AGENT spans (isolated context).
        for sub in _extract_subagents(state):
            trace.spans.extend(_subagent_spans(sub, trace.trace_id, root.span_id))

        # 3. messages → tool / llm / message spans for this graph node.
        for msg_span in _message_spans(state, node_name, trace.trace_id, root.span_id):
            trace.spans.append(msg_span)

        # 4. state delta → event on root span (LangGraph-style transitions).
        delta = _state_delta(prev_state, state)
        if delta:
            root.events.append(
                Event(
                    name=f"state_delta:{node_name}",
                    attributes={
                        "node": node_name,
                        "changed_keys": sorted(delta.keys()),
                        "delta": delta,
                    },
                )
            )

        prev_state = state
        step_index += 1

    root.end_time = _now()
    return trace


# ─────────────────────────────────────────────────────────────────────────────
# PlatformAdapter class (parity with base.PlatformAdapter API)
# ─────────────────────────────────────────────────────────────────────────────


class DeepAgentsAdapter(PlatformAdapter):
    """Adapter for LangChain Deep Agents traces.

    Deep Agents runs on LangGraph, so we reuse ``Platform.LANGGRAPH`` as the
    platform tag and mark the variant via ``platform_version="deep-agents-v1"``.
    Ingestion-only: the Deep Agents runtime has no externally-exposed
    pre-execution hook API, so ``inject_fix`` / ``block_action`` report
    unsupported.
    """

    def __init__(self, session_id: Optional[str] = None, agent_name: str = "deep_agent") -> None:
        self._session_id = session_id
        self._agent_name = agent_name
        self._last_state: dict[str, Any] = {}

    # -- identity -------------------------------------------------------------

    @property
    def platform_name(self) -> Platform:
        return Platform.LANGGRAPH

    @property
    def platform_version(self) -> Optional[str]:
        return "deep-agents-v1"

    # -- trace capture --------------------------------------------------------

    def parse_trace(
        self,
        states: Iterable[dict[str, Any]],
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> Trace:
        """Parse a full Deep Agents trace (sequence of state checkpoints)."""
        return parse_deep_agents_trace(
            states,
            session_id=session_id or self._session_id,
            agent_name=agent_name or self._agent_name,
        )

    def capture_span(self, raw_data: Any) -> Span:
        """Convert a single Deep Agents state checkpoint into a Span.

        Used by the realtime path (one checkpoint at a time). For full-trace
        ingestion, prefer ``parse_trace``.
        """
        if not isinstance(raw_data, dict):
            raise TypeError("DeepAgentsAdapter.capture_span expects a dict state")

        node_name = str(raw_data.get("node") or "deep_agents.step")
        span = Span(
            name=f"deep_agents.node:{node_name}",
            kind=SpanKind.CHAIN,
            platform=Platform.LANGGRAPH,
            platform_metadata={"runtime": "deep_agents", "node": node_name},
            start_time=_now(),
            status=SpanStatus.OK,
            attributes={
                "deep_agents.node": node_name,
                "deep_agents.has_todos": bool(_extract_todos(raw_data)),
                "deep_agents.has_subagents": bool(_extract_subagents(raw_data)),
            },
            input_data={"messages": _extract_message_dicts(raw_data)},
            output_data={"todos": _extract_todos(raw_data)},
        )
        delta = _state_delta(self._last_state, raw_data)
        if delta:
            span.events.append(Event(name="state_delta", attributes={"delta": delta}))
        span.end_time = _now()
        self._last_state = raw_data
        return span

    # -- state / context ------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id,
            "agent_name": self._agent_name,
            "last_state": self._last_state,
        }

    def get_session_context(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id,
            "todos": _extract_todos(self._last_state),
            "subagents": _extract_subagents(self._last_state),
        }

    # -- injection / blocking (unsupported) -----------------------------------

    def get_supported_injection_methods(self) -> list[InjectionMethod]:
        return []

    def inject_fix(
        self,
        directive: str,
        level: EnforcementLevel,
        directive_id: Optional[str] = None,
    ) -> InjectionResult:
        return InjectionResult(
            success=False,
            method=InjectionMethod.MESSAGE,
            directive_id=directive_id,
            error="Deep Agents runtime does not expose a pre-execution hook API; "
            "fix injection is not supported. Use post-hoc detection only.",
        )

    def can_block(self) -> bool:
        return False

    def block_action(self, reason: str) -> bool:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _todos_span(
    todos: list[Any],
    node_name: str,
    trace_id: str,
    parent_id: str,
) -> Span:
    """Build a TASK span representing a write_todos plan."""
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name="deep_agents.plan:write_todos",
        kind=SpanKind.TASK,
        platform=Platform.LANGGRAPH,
        start_time=_now(),
        end_time=_now(),
        status=SpanStatus.OK,
        attributes={
            "deep_agents.tool": "write_todos",
            "deep_agents.node": node_name,
            "deep_agents.plan.size": len(todos),
            # Carry the plan as the detectors' canonical "goals" attribute so
            # the decomposition / completion detectors can read it without a
            # Deep-Agents branch.
            "goals": [_todo_to_goal(t) for t in todos],
        },
        input_data={"node": node_name},
        output_data={"todos": todos},
    )


def _todo_to_goal(todo: Any) -> str:
    """Flatten a todo entry to a plain goal string.

    Deep Agents stores todos as ``{content, status}`` dicts (see the
    reference implementation); older versions use plain strings. We accept
    both so downstream detectors see a uniform list[str].
    """
    if isinstance(todo, dict):
        return str(todo.get("content") or todo.get("text") or todo)
    return str(todo)


def _subagent_spans(
    sub: dict[str, Any],
    trace_id: str,
    parent_id: str,
) -> list[Span]:
    """Build an AGENT span (plus optional MESSAGE children) for a subagent spawn.

    The child span uses a fresh span_id so its context is explicitly
    isolated from the parent in trace-tree traversal — mirroring Deep
    Agents' subagent isolation semantics.
    """
    name = str(sub.get("name") or sub.get("agent") or "subagent")
    task_text = sub.get("task") or sub.get("description")
    result = sub.get("result") or sub.get("output")

    agent_span = Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name=f"deep_agents.subagent:{name}",
        kind=SpanKind.AGENT,
        platform=Platform.LANGGRAPH,
        start_time=_now(),
        end_time=_now(),
        status=SpanStatus.OK,
        attributes={
            "deep_agents.tool": "task",
            "deep_agents.subagent.name": name,
            # Flag so downstream tooling can distinguish isolated-context
            # children from the main agent's own spans.
            "deep_agents.subagent.isolated_context": True,
        },
        input_data={"task": task_text} if task_text else None,
        output_data={"result": result} if result is not None else None,
    )
    spans = [agent_span]

    # Optional handoff span if the caller recorded a pre-spawn message.
    if handoff := sub.get("handoff"):
        spans.append(
            Span(
                trace_id=trace_id,
                parent_id=agent_span.span_id,
                name="deep_agents.handoff",
                kind=SpanKind.HANDOFF,
                platform=Platform.LANGGRAPH,
                start_time=_now(),
                end_time=_now(),
                status=SpanStatus.OK,
                output_data={"text": handoff if isinstance(handoff, str) else str(handoff)},
            )
        )
    return spans


def _message_spans(
    state: dict[str, Any],
    node_name: str,
    trace_id: str,
    parent_id: str,
) -> list[Span]:
    """Emit spans for new messages/tool-calls in this state checkpoint.

    We emit one span per message, mirroring the openai.py tool-call shape so
    detectors reading ``openai.tool.name`` / ``openai.tool.arguments`` style
    attributes also work against Deep Agents traces without a vendor branch.
    """
    spans: list[Span] = []
    for msg in state.get("messages") or []:
        span = _message_to_span(msg, node_name, trace_id, parent_id)
        if span is not None:
            spans.append(span)
    return spans


def _message_to_span(
    msg: Any,
    node_name: str,
    trace_id: str,
    parent_id: str,
) -> Optional[Span]:
    """Convert a single LangChain message (or dict) into a Span.

    Tool calls become TOOL spans, tool messages become TOOL output spans,
    AI/human/system messages become LLM/USER_INPUT spans accordingly.
    ``write_todos`` calls are skipped here — they are already captured as a
    plan span by ``_todos_span``.
    """
    msg_dict = _message_as_dict(msg)
    if not msg_dict:
        return None

    msg_type = str(msg_dict.get("type") or msg_dict.get("role") or "").lower()
    tool_calls = msg_dict.get("tool_calls") or []

    if tool_calls:
        call = tool_calls[0]
        tool_name = str(call.get("name") or "tool")
        if tool_name == "write_todos":
            return None
        return Span(
            trace_id=trace_id,
            parent_id=parent_id,
            name=f"deep_agents.tool:{tool_name}",
            kind=SpanKind.TOOL,
            platform=Platform.LANGGRAPH,
            start_time=_now(),
            end_time=_now(),
            status=SpanStatus.OK,
            attributes={
                "deep_agents.tool.name": tool_name,
                "deep_agents.node": node_name,
            },
            input_data={"arguments": call.get("args") or call.get("arguments")},
        )

    if msg_type in ("tool", "tool_message"):
        return Span(
            trace_id=trace_id,
            parent_id=parent_id,
            name=f"deep_agents.tool_output:{msg_dict.get('name') or ''}",
            kind=SpanKind.TOOL,
            platform=Platform.LANGGRAPH,
            start_time=_now(),
            end_time=_now(),
            status=SpanStatus.OK,
            attributes={
                "deep_agents.tool.name": msg_dict.get("name"),
                "deep_agents.node": node_name,
            },
            output_data={"output": msg_dict.get("content")},
        )

    if msg_type in ("human", "user"):
        return Span(
            trace_id=trace_id,
            parent_id=parent_id,
            name="deep_agents.user_input",
            kind=SpanKind.USER_INPUT,
            platform=Platform.LANGGRAPH,
            start_time=_now(),
            status=SpanStatus.OK,
            attributes={"deep_agents.node": node_name},
            output_data={"text": msg_dict.get("content")},
        )

    # Default: treat as an LLM/assistant message.
    return Span(
        trace_id=trace_id,
        parent_id=parent_id,
        name="deep_agents.llm",
        kind=SpanKind.LLM,
        platform=Platform.LANGGRAPH,
        start_time=_now(),
        end_time=_now(),
        status=SpanStatus.OK,
        attributes={"deep_agents.node": node_name, "deep_agents.message.role": msg_type or "ai"},
        output_data={"text": msg_dict.get("content")},
    )


def _message_as_dict(msg: Any) -> dict[str, Any]:
    """Normalize a LangChain message or dict into a plain dict.

    Avoids importing langchain_core. Supports:
    - plain dict messages with {type/role, content, tool_calls, name}
    - LangChain BaseMessage instances via duck-typing (type, content attrs)
    """
    if isinstance(msg, dict):
        return msg
    out: dict[str, Any] = {}
    for attr in ("type", "role", "content", "name", "tool_calls"):
        if hasattr(msg, attr):
            out[attr] = getattr(msg, attr)
    return out


def _extract_message_dicts(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [_message_as_dict(m) for m in state.get("messages") or []]


def _extract_todos(state: dict[str, Any]) -> list[Any]:
    """Return the current todos list from a state dict (or []).

    Deep Agents' write_todos tool writes to ``state["todos"]``. Earlier
    experimental builds used ``state["plan"]``; accept both for resilience.
    """
    if not isinstance(state, dict):
        return []
    todos = state.get("todos")
    if todos is None:
        todos = state.get("plan")
    return list(todos) if todos else []


def _extract_subagents(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return subagent spawn events from a state dict (or [])."""
    if not isinstance(state, dict):
        return []
    subs = state.get("subagents") or state.get("subagent_calls") or []
    return [s for s in subs if isinstance(s, dict)]


def _state_delta(prev: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Compute a shallow delta between two state dicts.

    Only top-level keys are diffed — that is sufficient for state-transition
    detection (corruption / loop / coordination detectors key off root-level
    keys). Deep equality is not attempted; values are compared via ``==``.
    """
    if not isinstance(prev, dict) or not isinstance(current, dict):
        return {}
    delta: dict[str, Any] = {}
    for key in set(prev) | set(current):
        if prev.get(key) != current.get(key):
            delta[key] = {"before": prev.get(key), "after": current.get(key)}
    return delta


def _now() -> datetime:
    return datetime.now(timezone.utc)
