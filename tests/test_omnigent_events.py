"""Contract tests for the Omnigent event mapper.

The JSONL fixtures are excerpts from a real omnigent 0.4.0 session captured
on 2026-07-10. They are deliberately checked into this standalone package so
the ingestion contract does not depend on the private application repository.
"""

from __future__ import annotations

from pathlib import Path

from pisama_core.ingestion.conversation_trace import ConversationTrace, ConversationTurnData
from pisama_core.ingestion.omnigent_events import (
    _content_text,
    _parse_arguments,
    events_to_atif,
    load_event_stream,
)
from pisama_core.ingestion.universal_trace import (
    SpanStatus,
    SpanType,
    UniversalSpan,
    UniversalTrace,
)

FIXTURES = Path(__file__).parent / "fixtures" / "omnigent"
CHILD_ID = "conv_e29d5b7e80744c0ea5cc9c46b73d47e8"


def test_real_captured_child_stream_maps_cost_tokens_and_messages() -> None:
    events = load_event_stream(str(FIXTURES / "child_stream.jsonl"))

    trajectory = events_to_atif(
        events,
        agent_name="researcher",
        agent_version="omnigent-0.4.0",
    )

    assert trajectory is not None
    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert trajectory["session_id"] == CHILD_ID
    assert trajectory["agent"] == {
        "name": "researcher",
        "version": "omnigent-0.4.0",
        "model_name": "claude-opus-4-8",
    }
    assert [step["source"] for step in trajectory["steps"]] == ["user", "agent"]
    assert trajectory["steps"][0]["message"] == "What is 17 * 23?"
    assert trajectory["steps"][1]["message"] == "17 * 23 = 391"
    assert trajectory["steps"][1]["model_name"] == "pisama_fixture_agent"
    assert trajectory["final_metrics"] == {
        "total_cost_usd": 0.058257500000000004,
        "total_prompt_tokens": 2489,
        "total_completion_tokens": 11,
        "total_steps": 2,
    }


def test_real_delegation_excerpt_deduplicates_and_embeds_child() -> None:
    parent = load_event_stream(str(FIXTURES / "parent_delegation_excerpt.jsonl"))
    child = load_event_stream(str(FIXTURES / "child_stream.jsonl"))

    trajectory = events_to_atif(
        parent,
        child_streams={CHILD_ID: child},
        agent_name="pisama_fixture_agent",
        agent_version="omnigent-0.4.0",
        session_id="parent-session",
        trajectory_id="parent-trajectory",
    )

    assert trajectory is not None
    assert trajectory["session_id"] == "parent-session"
    assert trajectory["trajectory_id"] == "parent-trajectory"
    agent_step = next(step for step in trajectory["steps"] if step["source"] == "agent")
    assert agent_step["reasoning_content"] == "The user is asking me to use"
    assert len(agent_step["tool_calls"]) == 1
    assert agent_step["tool_calls"][0]["function_name"] == "sys_session_send"
    assert agent_step["tool_calls"][0]["arguments"]["agent"] == "researcher"
    reference = agent_step["observation"]["results"][0]["subagent_trajectory_ref"]
    assert reference == [{"trajectory_id": CHILD_ID, "session_id": CHILD_ID}]

    subagent = trajectory["subagent_trajectories"][0]
    assert subagent["trajectory_id"] == CHILD_ID
    assert subagent["agent"]["name"] == "researcher"
    assert subagent["steps"][-1]["message"] == "17 * 23 = 391"
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 2545
    assert trajectory["final_metrics"]["total_completion_tokens"] == 707


def test_mapper_is_forward_compatible_and_handles_unmappable_children(caplog) -> None:
    trajectory = events_to_atif(
        [
            {"type": "session.future.variant", "conversation_id": "forward-compatible"},
            {
                "type": "session.input.consumed",
                "data": {
                    "data": {
                        "role": "user",
                        "content": "[System: worker completed]",
                    }
                },
            },
            {"type": "session.created", "child_session_id": "empty-child"},
        ],
        child_streams={"empty-child": [{"type": "session.heartbeat"}]},
    )

    assert trajectory is not None
    assert trajectory["session_id"] == "forward-compatible"
    assert trajectory["steps"] == [
        {"step_id": 1, "source": "system", "message": "[System: worker completed]"}
    ]
    assert "subagent_trajectories" not in trajectory
    assert "empty-child had no mappable steps" in caplog.text
    assert events_to_atif([{"type": "session.heartbeat"}]) is None


def test_wire_value_normalizers_tolerate_alpha_schema_variants() -> None:
    assert _content_text("plain") == "plain"
    assert _content_text([{"text": "one"}, {"ignored": True}, {"text": " two"}]) == "one two"
    assert _content_text({"text": "not-a-list"}) == ""

    assert _parse_arguments({"path": "README.md"}) == {"path": "README.md"}
    assert _parse_arguments('{"path": "README.md"}') == {"path": "README.md"}
    assert _parse_arguments("[1, 2]") == {"_raw": [1, 2]}
    assert _parse_arguments("{broken") == {"_raw": "{broken"}
    assert _parse_arguments("") == {}


def test_conversation_helpers_preserve_real_captured_turn_semantics() -> None:
    trace = ConversationTrace(
        trace_id="conv-trace",
        conversation_id=CHILD_ID,
        framework="omnigent",
        source_format="omnigent-0.4.0",
    )
    user = ConversationTurnData(
        turn_id="turn-user",
        turn_number=0,
        role="user",
        participant_id="human",
        content="What is 17 * 23?",
    )
    tool = ConversationTurnData(
        turn_id="turn-tool",
        turn_number=0,
        role="tool",
        participant_id="researcher",
        content="Delegated arithmetic request",
        tool_calls=[{"name": "sys_session_send", "args": {"agent": "researcher"}}],
    )
    system = ConversationTurnData(
        turn_id="turn-system",
        turn_number=0,
        role="system",
        participant_id="omnigent",
        content="[System: researcher completed the delegated arithmetic request]",
    )
    for turn in (user, tool, system):
        trace.add_turn(turn)

    assert [turn.turn_number for turn in trace.turns] == [1, 2, 3]
    assert trace.total_tokens == sum(len(turn.content) // 4 for turn in trace.turns)
    assert trace.get_context_up_to_turn(2).startswith("[user:human]\nWhat is 17 * 23?")
    assert trace.get_user_turns() == [user]
    assert trace.get_agent_turns() == []
    assert trace.get_initial_task() == system.content
    assert list(trace.iter_turn_pairs()) == [(user, tool), (tool, system)]

    spans = trace.to_universal_spans()
    assert [span.span_type for span in spans] == [
        SpanType.UNKNOWN,
        SpanType.TOOL_CALL,
        SpanType.CHAIN,
    ]
    assert spans[1].tool_name == "sys_session_send"
    assert spans[1].tool_args == {"agent": "researcher"}


def test_universal_trace_queries_hashing_and_snapshots() -> None:
    root = UniversalSpan(
        id="root",
        trace_id="trace",
        name="researcher",
        span_type=SpanType.AGENT,
        agent_name="researcher",
        raw_data={"captured": True},
    )
    tool = UniversalSpan(
        id="tool",
        trace_id="trace",
        name="sys_session_send",
        span_type=SpanType.TOOL_CALL,
        parent_id="root",
        agent_id="researcher",
        input_data={"question": "What is 17 * 23?"},
        output_data={"answer": 391},
        tool_name="sys_session_send",
        tool_args={"agent": "researcher"},
        tokens_input=12,
        tokens_output=3,
    )
    error = UniversalSpan(
        id="llm",
        trace_id="trace",
        name="provider-response",
        span_type=SpanType.LLM_CALL,
        parent_id="root",
        status=SpanStatus.ERROR,
        error="provider timeout",
    )
    universal = UniversalTrace(trace_id="trace", spans=[root, tool, error])

    assert root.is_multi_agent
    assert not root.is_single_agent
    assert tool.is_single_agent
    assert len(tool.content_hash) == 16
    assert universal.get_root_spans() == [root]
    assert universal.get_span_by_id("tool") is tool
    assert universal.get_span_by_id("missing") is None
    assert universal.get_children("root") == [tool, error]
    assert universal.get_tool_calls() == [tool]
    assert universal.get_llm_calls() == [error]
    assert universal.get_errors() == [error]
    assert universal.error_count == 1
    assert universal.has_errors
    snapshots = universal.to_state_snapshots()
    assert [snapshot.sequence_num for snapshot in snapshots] == [0, 1, 2]
    assert snapshots[0].content == "{'captured': True}"
    assert snapshots[1].state_delta == {"answer": 391}
