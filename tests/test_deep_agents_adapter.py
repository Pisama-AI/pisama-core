"""Tests for the LangChain Deep Agents adapter.

Deep Agents traces are sequences of state-dict checkpoints (from the
LangGraph checkpointer). The adapter must:

- Map ``write_todos`` calls to TASK / goals spans.
- Map ``task`` subagent spawns to child AGENT spans with isolated context.
- Emit state-delta events as the graph advances between nodes.

Fixtures here mirror the documented Deep Agents state shape so we do not
depend on the ``deepagents`` package at test time.
"""

from __future__ import annotations

import pytest

from pisama_core.adapters.base import PlatformAdapter
from pisama_core.adapters.deep_agents import (
    DeepAgentsAdapter,
    parse_deep_agents_trace,
)
from pisama_core.traces.enums import Platform, SpanKind

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def trace_with_write_todos() -> list[dict]:
    """A trace where the agent plans via ``write_todos`` then executes."""
    return [
        {
            "node": "agent",
            "messages": [{"type": "human", "content": "Research agent testing tools."}],
            "todos": [],
        },
        {
            "node": "planner",
            "messages": [
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "write_todos",
                            "args": {
                                "todos": [
                                    {"content": "Survey the landscape", "status": "pending"},
                                    {"content": "Draft the report", "status": "pending"},
                                ]
                            },
                        }
                    ],
                }
            ],
            "todos": [
                {"content": "Survey the landscape", "status": "pending"},
                {"content": "Draft the report", "status": "pending"},
            ],
        },
    ]


@pytest.fixture
def trace_with_subagent_spawn() -> list[dict]:
    """A trace that spawns a ``research`` subagent via the ``task`` tool."""
    return [
        {
            "node": "supervisor",
            "messages": [{"type": "human", "content": "Summarize today's AI news."}],
        },
        {
            "node": "supervisor",
            "subagents": [
                {
                    "name": "research",
                    "task": "Pull 5 news stories about AI agents.",
                    "handoff": "delegating research to subagent",
                    "result": "Here are 5 stories: ...",
                }
            ],
        },
    ]


@pytest.fixture
def trace_with_state_transitions() -> list[dict]:
    """LangGraph-style state transitions between nodes."""
    return [
        {"node": "agent", "messages": [], "counter": 0, "phase": "init"},
        {"node": "tools", "messages": [], "counter": 1, "phase": "working"},
        {"node": "agent", "messages": [], "counter": 2, "phase": "done"},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Import cleanliness
# ─────────────────────────────────────────────────────────────────────────────


def test_adapter_import_is_clean():
    """Importing the adapter must not require langchain/langgraph/deepagents."""
    import importlib

    mod = importlib.import_module("pisama_core.adapters.deep_agents")
    assert hasattr(mod, "DeepAgentsAdapter")
    assert hasattr(mod, "parse_deep_agents_trace")


def test_adapter_inherits_platform_adapter():
    adapter = DeepAgentsAdapter()
    assert isinstance(adapter, PlatformAdapter)
    assert adapter.platform_name == Platform.LANGGRAPH
    assert adapter.platform_version == "deep-agents-v1"


# ─────────────────────────────────────────────────────────────────────────────
# write_todos → goals/plan span
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteTodos:
    def test_parse_extracts_todos_as_goals(self, trace_with_write_todos):
        trace = parse_deep_agents_trace(trace_with_write_todos, session_id="s1")

        plan_spans = [s for s in trace.spans if s.kind == SpanKind.TASK]
        assert len(plan_spans) == 1
        plan = plan_spans[0]
        assert plan.attributes["deep_agents.tool"] == "write_todos"
        assert plan.attributes["deep_agents.plan.size"] == 2
        assert plan.attributes["goals"] == [
            "Survey the landscape",
            "Draft the report",
        ]

    def test_write_todos_tool_call_is_not_double_counted_as_tool(self, trace_with_write_todos):
        """The write_todos tool call must become a plan span, not a TOOL span."""
        trace = parse_deep_agents_trace(trace_with_write_todos)
        tool_spans = [
            s
            for s in trace.spans
            if s.kind == SpanKind.TOOL
            and s.attributes.get("deep_agents.tool.name") == "write_todos"
        ]
        assert tool_spans == []

    def test_adapter_class_parse_trace_matches_function(self, trace_with_write_todos):
        adapter = DeepAgentsAdapter(session_id="s1", agent_name="researcher")
        trace = adapter.parse_trace(trace_with_write_todos)
        assert trace.metadata.platform == Platform.LANGGRAPH
        assert trace.metadata.platform_version == "deep-agents-v1"
        assert trace.metadata.custom["runtime"] == "deep_agents"
        assert trace.metadata.custom["agent_name"] == "researcher"


# ─────────────────────────────────────────────────────────────────────────────
# Subagent spawn → child span (isolated context)
# ─────────────────────────────────────────────────────────────────────────────


class TestSubagentSpawn:
    def test_subagent_becomes_child_agent_span(self, trace_with_subagent_spawn):
        trace = parse_deep_agents_trace(trace_with_subagent_spawn)
        subagent_spans = [s for s in trace.spans if s.name.startswith("deep_agents.subagent:")]
        assert len(subagent_spans) == 1
        sub = subagent_spans[0]
        assert sub.kind == SpanKind.AGENT
        assert sub.attributes["deep_agents.subagent.name"] == "research"
        assert sub.attributes["deep_agents.subagent.isolated_context"] is True
        assert sub.input_data["task"].startswith("Pull 5")
        assert sub.output_data["result"].startswith("Here are 5")

    def test_subagent_span_is_parented_to_root_not_nested_in_parent_message(
        self, trace_with_subagent_spawn
    ):
        trace = parse_deep_agents_trace(trace_with_subagent_spawn)
        root = trace.spans[0]
        assert root.kind == SpanKind.AGENT
        sub = [s for s in trace.spans if s.name.startswith("deep_agents.subagent:")][0]
        # Isolated context = parented to root, not to an intermediate message span.
        assert sub.parent_id == root.span_id
        # Handoff span should be a child of the subagent span.
        handoff = [s for s in trace.spans if s.kind == SpanKind.HANDOFF]
        assert len(handoff) == 1
        assert handoff[0].parent_id == sub.span_id


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph state transitions → state deltas
# ─────────────────────────────────────────────────────────────────────────────


class TestStateDeltas:
    def test_state_transitions_produce_delta_events(self, trace_with_state_transitions):
        trace = parse_deep_agents_trace(trace_with_state_transitions)
        root = trace.spans[0]
        delta_events = [e for e in root.events if e.name.startswith("state_delta:")]
        # 3 checkpoints → 3 deltas (first one is vs empty prev_state).
        assert len(delta_events) == 3
        # Second transition: counter 0→1, phase init→working, node agent→tools.
        second = delta_events[1]
        changed = second.attributes["changed_keys"]
        assert "counter" in changed
        assert "phase" in changed
        assert "node" in changed
        assert second.attributes["delta"]["counter"]["before"] == 0
        assert second.attributes["delta"]["counter"]["after"] == 1

    def test_unchanged_state_emits_no_delta(self):
        trace = parse_deep_agents_trace(
            [
                {"node": "agent", "x": 1},
                {"node": "agent", "x": 1},
            ]
        )
        root = trace.spans[0]
        delta_events = [e for e in root.events if e.name.startswith("state_delta:")]
        # First checkpoint diffs against empty dict (produces a delta).
        # Second checkpoint is identical to the first — no delta.
        assert len(delta_events) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Injection / blocking surface (must report unsupported, like other ingesters)
# ─────────────────────────────────────────────────────────────────────────────


class TestAdapterSurface:
    def test_injection_is_unsupported(self):
        from pisama_core.injection.enforcement import EnforcementLevel

        adapter = DeepAgentsAdapter()
        result = adapter.inject_fix("do X", EnforcementLevel.SUGGEST)
        assert result.success is False
        assert result.error is not None
        assert adapter.get_supported_injection_methods() == []

    def test_blocking_is_unsupported(self):
        adapter = DeepAgentsAdapter()
        assert adapter.can_block() is False
        assert adapter.block_action("nope") is False

    def test_capture_span_builds_single_node_span(self):
        adapter = DeepAgentsAdapter()
        span = adapter.capture_span(
            {
                "node": "planner",
                "messages": [{"type": "ai", "content": "planning"}],
                "todos": [{"content": "step 1", "status": "pending"}],
            }
        )
        assert span.platform == Platform.LANGGRAPH
        assert span.name == "deep_agents.node:planner"
        assert span.attributes["deep_agents.has_todos"] is True
        assert span.attributes["deep_agents.has_subagents"] is False

    def test_capture_span_rejects_non_dict(self):
        adapter = DeepAgentsAdapter()
        with pytest.raises(TypeError):
            adapter.capture_span("not a dict")
