"""Tests for the LangChain Deep Agents trace ingestion adapter.

Deep Agents runs on LangGraph and adds three primitives:
- ``write_todos`` for planning (mapped to TASK goals),
- ``task`` for subagent spawns (mapped to nested AGENT spans),
- LangGraph Memory Store transitions (mapped to state-delta events).

These tests feed the adapter documented state-dict shapes and assert the
resulting Trace has the expected structure. No langchain / langgraph imports.
"""

import pytest

from pisama_core.adapters.deep_agents import (
    DeepAgentsAdapter,
    parse_deep_agents_trace,
)
from pisama_core.traces.enums import Platform, SpanKind, SpanStatus
from pisama_core.traces.models import Trace


class TestWriteTodosPlanning:
    def test_write_todos_produces_task_span_not_tool_span(self):
        """Key anti-regression: write_todos is a PLANNING primitive, not a tool
        call. Its output must surface as a TASK span with goals[], so the
        decomposition / completion detectors can consume it without a
        Deep-Agents-specific branch.
        """
        states = [
            {
                "node": "planner",
                "todos": [
                    {"content": "Research topic X", "status": "pending"},
                    {"content": "Draft summary", "status": "pending"},
                    {"content": "Publish report", "status": "pending"},
                ],
            },
        ]
        trace = parse_deep_agents_trace(states, session_id="s1")

        task_spans = [s for s in trace.spans if s.kind == SpanKind.TASK]
        assert len(task_spans) == 1, "write_todos must produce exactly one TASK span"

        plan = task_spans[0]
        assert plan.name == "deep_agents.plan:write_todos"
        # Explicit anti-regression guard — write_todos MUST NOT be a TOOL span.
        assert plan.kind != SpanKind.TOOL
        assert plan.attributes["deep_agents.tool"] == "write_todos"
        assert plan.attributes["goals"] == [
            "Research topic X",
            "Draft summary",
            "Publish report",
        ]
        assert plan.attributes["deep_agents.plan.size"] == 3

    def test_write_todos_accepts_plain_string_todos(self):
        """Older Deep Agents builds used plain strings for todos — accept both."""
        states = [{"node": "planner", "todos": ["step one", "step two"]}]
        trace = parse_deep_agents_trace(states)
        task_spans = [s for s in trace.spans if s.kind == SpanKind.TASK]
        assert len(task_spans) == 1
        assert task_spans[0].attributes["goals"] == ["step one", "step two"]


class TestSubagentSpawn:
    def test_subagent_spawn_becomes_child_agent_span(self):
        states = [
            {
                "node": "supervisor",
                "subagents": [
                    {
                        "name": "researcher",
                        "task": "Look up recent papers on agent failure modes",
                        "result": "Found 12 papers",
                    }
                ],
            }
        ]
        trace = parse_deep_agents_trace(states)

        # Root + one subagent span
        agent_spans = [s for s in trace.spans if s.kind == SpanKind.AGENT]
        assert len(agent_spans) == 2
        root = agent_spans[0]
        subagent = agent_spans[1]

        assert subagent.parent_id == root.span_id, "subagent must be a child of root"
        assert subagent.name == "deep_agents.subagent:researcher"
        assert subagent.attributes["deep_agents.tool"] == "task"
        assert subagent.attributes["deep_agents.subagent.name"] == "researcher"
        assert subagent.attributes["deep_agents.subagent.isolated_context"] is True
        assert subagent.input_data == {"task": "Look up recent papers on agent failure modes"}
        assert subagent.output_data == {"result": "Found 12 papers"}


class TestLangGraphStateTransitions:
    def test_state_transitions_produce_delta_events_on_root(self):
        """Each graph checkpoint that changes top-level state keys should
        append a ``state_delta:<node>`` event to the root span. The
        corruption / loop / coordination detectors rely on these.
        """
        states = [
            {"node": "agent", "counter": 1, "messages": []},
            {"node": "tools", "counter": 2, "messages": []},
            {"node": "agent", "counter": 2, "messages": [{"type": "ai", "content": "hi"}]},
        ]
        trace = parse_deep_agents_trace(states)

        root = trace.spans[0]
        assert root.kind == SpanKind.AGENT

        delta_events = [e for e in root.events if e.name.startswith("state_delta:")]
        # First state introduces keys vs empty prev → 1 event
        # Second state changes counter → 1 event
        # Third state changes counter + messages → 1 event
        assert len(delta_events) == 3

        # Second transition should record counter change
        second = delta_events[1]
        assert second.name == "state_delta:tools"
        assert "counter" in second.attributes["changed_keys"]
        assert second.attributes["delta"]["counter"]["before"] == 1
        assert second.attributes["delta"]["counter"]["after"] == 2


class TestPlatformReporting:
    def test_platform_langgraph_with_deep_agents_version_tag(self):
        trace = parse_deep_agents_trace([{"node": "agent"}], session_id="thread-123")
        assert trace.metadata.platform == Platform.LANGGRAPH
        assert trace.metadata.platform_version == "deep-agents-v1"
        assert trace.metadata.custom["runtime"] == "deep_agents"
        assert trace.metadata.custom["agent_name"] == "deep_agent"


class TestConvenienceFunctionShape:
    def test_parse_deep_agents_trace_returns_trace(self):
        """The convenience function parallels parse_openai_response /
        parse_invoke_agent: take raw state dicts, return a fully-populated
        Trace with a root AGENT span.
        """
        trace = parse_deep_agents_trace(
            [{"node": "agent", "messages": [{"type": "ai", "content": "hello"}]}],
            session_id="sess-abc",
            agent_name="my_agent",
        )
        assert isinstance(trace, Trace)
        assert len(trace.spans) >= 1
        root = trace.spans[0]
        assert root.kind == SpanKind.AGENT
        assert root.name == "deep_agents.agent:my_agent"
        assert root.platform == Platform.LANGGRAPH
        assert root.status == SpanStatus.OK
        # Root span carries the session id as an attribute.
        assert root.attributes["deep_agents.session_id"] == "sess-abc"


class TestEdgeCases:
    def test_empty_state_iterable_still_produces_root_span(self):
        """Empty trace must not crash; returns a Trace with only the root."""
        trace = parse_deep_agents_trace([], session_id="s0")
        assert len(trace.spans) == 1
        root = trace.spans[0]
        assert root.kind == SpanKind.AGENT
        assert root.events == []

    def test_malformed_non_dict_state_entries_are_skipped(self):
        """The adapter defensively skips non-dict entries rather than raising —
        this matches the documented behaviour (see the ``isinstance(state,
        dict)`` guard in parse_deep_agents_trace).
        """
        states = [
            None,
            "not a dict",
            42,
            {"node": "agent", "todos": [{"content": "real step"}]},
        ]
        # Must not raise.
        trace = parse_deep_agents_trace(states)
        # Only the single valid dict produced a TASK span.
        task_spans = [s for s in trace.spans if s.kind == SpanKind.TASK]
        assert len(task_spans) == 1
        assert task_spans[0].attributes["goals"] == ["real step"]

    def test_capture_span_rejects_non_dict_input(self):
        """DeepAgentsAdapter.capture_span raises TypeError on non-dict input —
        the realtime path has no fallback and must surface the error loudly.
        """
        adapter = DeepAgentsAdapter(session_id="s1")
        with pytest.raises(TypeError):
            adapter.capture_span("not a dict")


class TestSessionPropagation:
    def test_session_id_propagates_from_input_to_trace_metadata(self):
        trace = parse_deep_agents_trace([{"node": "agent"}], session_id="thread-xyz-42")
        assert trace.metadata.session_id == "thread-xyz-42"
        # And to the root span attribute.
        assert trace.spans[0].attributes["deep_agents.session_id"] == "thread-xyz-42"

    def test_adapter_parse_trace_uses_constructor_session_id_as_default(self):
        """DeepAgentsAdapter(session_id=...) is used when parse_trace is
        called without an explicit override — mirrors the constructor /
        instance-state pattern used by OpenAI / Bedrock adapters.
        """
        adapter = DeepAgentsAdapter(session_id="ctor-session", agent_name="worker")
        trace = adapter.parse_trace([{"node": "agent"}])
        assert trace.metadata.session_id == "ctor-session"
        assert trace.metadata.custom["agent_name"] == "worker"
        assert trace.spans[0].name == "deep_agents.agent:worker"
