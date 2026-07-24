"""Omnigent session events -> ATIF v1.7 trajectory.

Maps the SSE event stream of an Omnigent session (the ``ServerStreamEvent``
union emitted by ``GET /v1/sessions/{id}/stream``) into an ATIF v1.7
trajectory dict that ``AtifParser.parse`` accepts. Child (sub-agent) session
streams embed as ``subagent_trajectories`` and are linked from the spawning
tool call's observation result via ``subagent_trajectory_ref``.

Wire shapes were verified against a real omnigent 0.4.0 session
(``backend/tests/fixtures/omnigent/`` in the monorepo, exercised by
``backend/tests/test_omnigent_parser.py``). The events consumed here:

- ``session.input.consumed``          -> user step (text under data.data.content)
- ``response.output_item.done``       -> items: ``message`` (assistant text),
  ``function_call`` (name/arguments/call_id), ``function_call_output``
  (call_id/output)
- ``response.reasoning_text.delta``   -> accumulated reasoning_content
- ``response.completed``              -> agent step boundary
- ``session.usage``                   -> cumulative cost/tokens (final_metrics)
- ``session.created``                 -> child_session_id (sub-agent link)

Unknown event types are skipped, not errors: omnigent is alpha and its event
union grows; the mapper must tolerate new variants.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

ATIF_SCHEMA_VERSION = "ATIF-v1.7"


def _content_text(content: Any) -> str:
    """Join the text of an OpenResponses-style content block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    """Omnigent serializes tool arguments as a JSON string; ATIF wants a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"_raw": parsed}
        except (json.JSONDecodeError, ValueError):
            return {"_raw": raw}
    return {}


class _StepBuilder:
    """Accumulates one agent turn (message, reasoning, tool calls, results)."""

    def __init__(self) -> None:
        self.message: str = ""
        self.reasoning: List[str] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []
        self.model_name: Optional[str] = None
        self._seen_call_ids: set = set()

    def add_tool_call(self, call_id: str, name: str, arguments: Dict[str, Any]) -> None:
        """Append a tool call, deduping by call_id within this step.

        Omnigent 0.4.0 emits each ``function_call`` item twice on the stream
        with an identical ``call_id`` (once ``in_progress``, once on the
        follow-up). Appending both doubles the tool-call spans downstream and
        inflates the workflow graph / loop matches, so a repeated call_id is
        dropped. A falsy call_id is never deduped (can't correlate).
        """
        if call_id and call_id in self._seen_call_ids:
            return
        if call_id:
            self._seen_call_ids.add(call_id)
        self.tool_calls.append(
            {
                "tool_call_id": call_id,
                "function_name": name,
                "arguments": arguments,
            }
        )

    @property
    def empty(self) -> bool:
        return not (self.message or self.reasoning or self.tool_calls or self.results)

    def build(self, step_id: int) -> Dict[str, Any]:
        step: Dict[str, Any] = {
            "step_id": step_id,
            "source": "agent",
            "message": self.message,
        }
        if self.model_name:
            step["model_name"] = self.model_name
        if self.reasoning:
            step["reasoning_content"] = "".join(self.reasoning)
        if self.tool_calls:
            step["tool_calls"] = self.tool_calls
        if self.results:
            step["observation"] = {"results": self.results}
        return step


def events_to_atif(
    events: Iterable[Dict[str, Any]],
    *,
    child_streams: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    agent_name: str = "omnigent",
    agent_version: str = "unknown",
    session_id: Optional[str] = None,
    trajectory_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Map an Omnigent session event stream to an ATIF trajectory dict.

    :param events: Parsed SSE event dicts for one session, in arrival order.
    :param child_streams: Optional ``{child_session_id: [events]}`` for
        sub-agent sessions spawned by this one. Each maps recursively and
        embeds under ``subagent_trajectories``; the spawning tool call's
        observation result gets a ``subagent_trajectory_ref``.
    :param agent_name: ATIF ``agent.name`` (the Omnigent agent name).
    :param agent_version: ATIF ``agent.version`` (e.g. omnigent version).
    :param session_id: Session id override; defaults to the first
        ``conversation_id`` seen in the stream.
    :param trajectory_id: Set when this trajectory embeds in a parent's
        ``subagent_trajectories`` (required by ATIF there).
    :returns: ATIF trajectory dict, or ``None`` when no steps are mappable.
    """
    child_streams = child_streams or {}

    steps: List[Dict[str, Any]] = []
    current = _StepBuilder()
    seen_session_id = session_id
    child_ids: List[str] = []
    last_usage: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    # call_id -> delegated agent name, from sys_session_send arguments. Used to
    # name sub-agent trajectories after the real worker (e.g. "researcher")
    # instead of a "<parent>/subagent" clone, so the delegation edge and the
    # sub-agent-boundary detectors attribute to a truthful identity.
    delegated_agent_by_call: Dict[str, str] = {}
    delegated_agent_by_child: Dict[str, str] = {}

    def flush_agent_step() -> None:
        nonlocal current
        if not current.empty:
            steps.append(current.build(len(steps) + 1))
        current = _StepBuilder()

    for event in events:
        etype = event.get("type")
        if seen_session_id is None and event.get("conversation_id"):
            seen_session_id = event["conversation_id"]

        if etype == "session.input.consumed":
            payload = (event.get("data") or {}).get("data") or {}
            if payload.get("role") == "user":
                flush_agent_step()
                text = _content_text(payload.get("content"))
                if text:
                    # Omnigent injects sub-agent inbox notifications into the
                    # parent's input queue as role=user with a "[System:"
                    # prefix (its own runtime convention; 0.4.0 has no
                    # structural discriminator). Those are system steps in
                    # ATIF terms, not human turns.
                    source = "system" if text.startswith("[System:") else "user"
                    steps.append({"step_id": len(steps) + 1, "source": source, "message": text})

        elif etype == "response.output_item.done":
            item = event.get("item") or {}
            item_type = item.get("type")
            if item_type == "message" and item.get("role") == "assistant":
                current.message = _content_text(item.get("content")) or current.message
            elif item_type == "function_call":
                call_id = str(item.get("call_id") or item.get("id") or "")
                name = str(item.get("name") or "unknown")
                arguments = _parse_arguments(item.get("arguments"))
                if name == "sys_session_send":
                    delegated = arguments.get("agent")
                    if isinstance(delegated, str) and delegated and call_id:
                        delegated_agent_by_call[call_id] = delegated
                current.add_tool_call(call_id, name, arguments)
            elif item_type == "function_call_output":
                result: Dict[str, Any] = {
                    "source_call_id": item.get("call_id"),
                    "content": item.get("output") or "",
                }
                output_str = str(item.get("output", ""))
                for child_id in child_streams:
                    if child_id not in child_ids and child_id in output_str:
                        result["subagent_trajectory_ref"] = [
                            {"trajectory_id": child_id, "session_id": child_id}
                        ]
                        child_ids.append(child_id)
                        output_call_id = item.get("call_id")
                        if (
                            isinstance(output_call_id, str)
                            and output_call_id in delegated_agent_by_call
                        ):
                            delegated_agent_by_child[child_id] = delegated_agent_by_call[
                                output_call_id
                            ]
                current.results.append(result)

        elif etype == "response.reasoning_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                current.reasoning.append(delta)

        elif etype == "response.completed":
            flush_agent_step()

        elif etype == "session.usage":
            if event.get("total_cost_usd") is not None or event.get("usage_by_model"):
                last_usage = event
                by_model = event.get("usage_by_model") or {}
                if by_model and model_name is None:
                    model_name = next(iter(by_model))

        elif etype == "session.created":
            child = event.get("child_session_id")
            if child and child in child_streams and child not in child_ids:
                # Linked here only if no function_call_output referenced it.
                pass  # linkage happens on the output item above

    flush_agent_step()

    if not steps:
        return None

    trajectory: Dict[str, Any] = {
        "schema_version": ATIF_SCHEMA_VERSION,
        "agent": {
            "name": agent_name,
            "version": agent_version,
            **({"model_name": model_name} if model_name else {}),
        },
        "steps": steps,
        "extra": {"source": "omnigent"},
    }
    if seen_session_id:
        trajectory["session_id"] = seen_session_id
    if trajectory_id:
        trajectory["trajectory_id"] = trajectory_id

    if last_usage:
        final: Dict[str, Any] = {}
        cost = last_usage.get("total_cost_usd")
        if cost is not None:
            final["total_cost_usd"] = cost
        by_model = last_usage.get("usage_by_model") or {}
        if by_model:
            final["total_prompt_tokens"] = sum(
                int(u.get("input_tokens") or 0) for u in by_model.values()
            )
            final["total_completion_tokens"] = sum(
                int(u.get("output_tokens") or 0) for u in by_model.values()
            )
        if final:
            final["total_steps"] = len(steps)
            trajectory["final_metrics"] = final

    subagents = []
    for child_id, child_events in child_streams.items():
        sub = events_to_atif(
            child_events,
            agent_name=delegated_agent_by_child.get(child_id, "subagent"),
            agent_version=agent_version,
            session_id=child_id,
            trajectory_id=child_id,
        )
        if sub is not None:
            subagents.append(sub)
        else:
            logger.warning("omnigent child session %s had no mappable steps", child_id)
    if subagents:
        trajectory["subagent_trajectories"] = subagents

    return trajectory


def load_event_stream(path: str) -> List[Dict[str, Any]]:
    """Load a captured SSE stream (one JSON event per line) from disk."""
    events: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
