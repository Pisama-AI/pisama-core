"""Round-trip / fidelity tests for the framework-agnostic ingestion spine.

Every adapter (OpenAI/Bedrock/Gemini/ADK/DeepAgents) funnels traffic through
`UniversalSpan` / `UniversalTrace` / `ConversationTrace`, yet these abstractions
had zero direct test coverage. These tests pin the serialization contract:

- `UniversalSpan` / `UniversalTrace` expose `to_dict()` but NO `from_dict()`, so we
  assert *field fidelity* (construct -> `to_dict()` emits the right keys/values) and
  that the parent/child hierarchy survives via `parent_id`.
- `ConversationTrace` has both `to_dict()` + `from_dict()`, so we assert a true
  identity round-trip at the dict level, plus the `to_universal_span()` mapping.

Pure data fixtures only: no mocks, no network, no LLM. Datetimes are passed
explicitly so `to_dict()` output is deterministic.
"""

from datetime import datetime, timezone

import pytest

from pisama_core.ingestion import (
    ConversationTrace,
    ConversationTurnData,
    SpanStatus,
    SpanType,
    UniversalSpan,
    UniversalTrace,
)

# The exact key set UniversalSpan.to_dict() is contracted to emit (see audit notes).
EXPECTED_SPAN_KEYS = {
    "id",
    "trace_id",
    "name",
    "span_type",
    "status",
    "start_time",
    "end_time",
    "duration_ms",
    "parent_id",
    "agent_id",
    "agent_name",
    "input_data",
    "output_data",
    "prompt",
    "response",
    "model",
    "tokens_input",
    "tokens_output",
    "tokens_total",
    "tool_name",
    "tool_args",
    "tool_result",
    "error",
    "error_type",
    "source_format",
    "metadata",
}


def _make_llm_span(**overrides) -> UniversalSpan:
    """Build a fully-populated LLM_CALL span with deterministic timing."""
    defaults = dict(
        id="span-llm-1",
        trace_id="trace-1",
        name="claude-call",
        span_type=SpanType.LLM_CALL,
        status=SpanStatus.OK,
        start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 1, 12, 0, 2, 500000, tzinfo=timezone.utc),
        agent_id="agent-7",
        agent_name="Researcher",
        input_data={"role": "user"},
        output_data={"content": "hi"},
        prompt="What is 2+2?",
        response="4",
        model="claude-opus-4-8",
        tokens_input=120,
        tokens_output=80,
        tool_name=None,
        source_format="anthropic",
        metadata={"k": "v"},
    )
    defaults.update(overrides)
    return UniversalSpan(**defaults)


def _make_conversation_trace() -> ConversationTrace:
    """Build a clean 3-turn conversation (no per-turn extra) for round-tripping."""
    trace = ConversationTrace(
        trace_id="conv-trace-1",
        conversation_id="conv-1",
        framework="anthropic",
        source_format="claude",
        extra={"ground_truth": "no_failure"},
    )
    trace.add_turn(
        ConversationTurnData(
            turn_id="t1",
            turn_number=1,
            role="user",
            participant_id="user-1",
            content="Plan a trip to Helsinki.",
            token_count=12,
        )
    )
    trace.add_turn(
        ConversationTurnData(
            turn_id="t2",
            turn_number=2,
            role="agent",
            participant_id="agent-1",
            content="Sure, here is a three-day itinerary for Helsinki.",
            token_count=20,
        )
    )
    trace.add_turn(
        ConversationTurnData(
            turn_id="t3",
            turn_number=3,
            role="user",
            participant_id="user-1",
            content="Add a museum on day two.",
            token_count=8,
        )
    )
    return trace


class TestUniversalSpanFidelity:
    """UniversalSpan exposes to_dict() only -> assert field fidelity."""

    def test_to_dict_emits_all_expected_keys_with_values(self):
        """Test 1: construct -> to_dict() emits the full key set with correct values."""
        span = _make_llm_span()
        d = span.to_dict()

        # Exact contract: no missing keys, no surprise keys.
        assert set(d.keys()) == EXPECTED_SPAN_KEYS

        # Scalar identifiers pass through verbatim.
        assert d["id"] == "span-llm-1"
        assert d["trace_id"] == "trace-1"
        assert d["name"] == "claude-call"
        assert d["agent_id"] == "agent-7"
        assert d["agent_name"] == "Researcher"
        assert d["model"] == "claude-opus-4-8"
        assert d["prompt"] == "What is 2+2?"
        assert d["response"] == "4"
        assert d["source_format"] == "anthropic"

        # Dict-valued fields are preserved (not stringified).
        assert d["input_data"] == {"role": "user"}
        assert d["output_data"] == {"content": "hi"}
        assert d["metadata"] == {"k": "v"}

        # Timestamps are ISO-8601 strings, not datetime objects.
        assert d["start_time"] == "2025-01-01T12:00:00+00:00"
        assert d["end_time"] == "2025-01-01T12:00:02.500000+00:00"

    def test_to_dict_handles_none_timestamps_and_defaults(self):
        """Test 1b: optional fields default cleanly and end_time=None serializes to None."""
        span = UniversalSpan(
            id="s",
            trace_id="t",
            name="bare",
            span_type=SpanType.UNKNOWN,
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        d = span.to_dict()
        assert set(d.keys()) == EXPECTED_SPAN_KEYS
        assert d["end_time"] is None
        assert d["duration_ms"] == 0  # no end_time -> __post_init__ leaves it at 0
        assert d["parent_id"] is None
        assert d["tool_name"] is None
        assert d["tokens_total"] == 0
        assert d["input_data"] == {}
        assert d["metadata"] == {}


class TestUniversalTraceHierarchy:
    """Parent/child structure must survive to_dict() via parent_id."""

    def test_hierarchy_preserved_through_to_dict(self):
        """Test 2: root + children resolve correctly through to_dict(), no orphans."""
        root = UniversalSpan(
            id="root",
            trace_id="tr",
            name="orchestrator",
            span_type=SpanType.AGENT,
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        child_a = UniversalSpan(
            id="child-a",
            trace_id="tr",
            name="tool-a",
            span_type=SpanType.TOOL_CALL,
            parent_id="root",
            start_time=datetime(2025, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        )
        child_b = UniversalSpan(
            id="child-b",
            trace_id="tr",
            name="tool-b",
            span_type=SpanType.TOOL_CALL,
            parent_id="root",
            start_time=datetime(2025, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        )
        trace = UniversalTrace(trace_id="tr", spans=[root, child_a, child_b])

        # Object-level hierarchy helpers.
        roots = trace.get_root_spans()
        assert [s.id for s in roots] == ["root"]
        assert {s.id for s in trace.get_children("root")} == {"child-a", "child-b"}

        d = trace.to_dict()
        assert d["trace_id"] == "tr"
        assert len(d["spans"]) == 3

        # Reconstruct the tree purely from serialized parent_id values.
        span_dicts = d["spans"]
        ids = {s["id"] for s in span_dicts}
        dict_roots = [s for s in span_dicts if s["parent_id"] is None]
        assert [s["id"] for s in dict_roots] == ["root"]

        dict_children = [s for s in span_dicts if s["parent_id"] == "root"]
        assert {s["id"] for s in dict_children} == {"child-a", "child-b"}

        # No orphan: every non-null parent_id points at a real span in the trace.
        for s in span_dicts:
            if s["parent_id"] is not None:
                assert s["parent_id"] in ids


class TestTokenAndDurationFidelity:
    """Numeric derived fields must survive serialization."""

    def test_tokens_and_duration_survive_to_dict(self):
        """Test 3: tokens_input/output/total and duration_ms survive to_dict()."""
        span = _make_llm_span()  # 120 in / 80 out, 2.5s wall clock
        d = span.to_dict()
        assert d["tokens_input"] == 120
        assert d["tokens_output"] == 80
        assert d["tokens_total"] == 200  # __post_init__ sums the two
        assert d["duration_ms"] == 2500  # 2.5s -> 2500ms

    def test_trace_aggregates_token_totals(self):
        """Test 3b: UniversalTrace rolls up per-span token totals."""
        s1 = _make_llm_span(id="a", tokens_input=10, tokens_output=5)
        s2 = _make_llm_span(id="b", tokens_input=30, tokens_output=20)
        trace = UniversalTrace(trace_id="tr", spans=[s1, s2])
        d = trace.to_dict()
        assert d["total_tokens"] == 15 + 50
        assert d["has_errors"] is False
        assert d["error_count"] == 0


class TestEnumSerialization:
    """Enum-valued fields must serialize to their scalar `.value`, not the enum."""

    def test_span_type_and_status_serialize_to_scalars(self):
        """Test 4: SpanType / SpanStatus serialize to their string values."""
        span = _make_llm_span(span_type=SpanType.HANDOFF, status=SpanStatus.ERROR)
        d = span.to_dict()
        assert d["span_type"] == "handoff"
        assert d["status"] == "error"
        assert isinstance(d["span_type"], str)
        assert isinstance(d["status"], str)

    def test_every_enum_member_round_trips_to_its_value(self):
        """Test 4b: each SpanType/SpanStatus member serializes to exactly its .value."""
        for st in SpanType:
            d = _make_llm_span(span_type=st).to_dict()
            assert d["span_type"] == st.value
        for status in SpanStatus:
            d = _make_llm_span(status=status).to_dict()
            assert d["status"] == status.value


class TestConversationTraceRoundTrip:
    """ConversationTrace has to_dict() + from_dict() -> assert identity round-trip."""

    def test_to_dict_from_dict_is_identity(self):
        """Test 5: to_dict() -> from_dict() -> to_dict() is an exact identity."""
        original = _make_conversation_trace()
        d = original.to_dict()

        restored = ConversationTrace.from_dict(d)

        # Dict-level identity: serialize the reconstruction, compare byte-for-structure.
        assert restored.to_dict() == d

        # Object-level field fidelity.
        assert restored.trace_id == original.trace_id
        assert restored.conversation_id == original.conversation_id
        assert restored.framework == original.framework
        assert restored.source_format == original.source_format
        assert restored.total_turns == original.total_turns == 3
        assert restored.total_tokens == original.total_tokens
        assert restored.participants == original.participants == ["user-1", "agent-1"]
        assert restored.extra == {"ground_truth": "no_failure"}

        # Per-turn fidelity, including the recomputed accumulated_tokens chain.
        assert [t.turn_id for t in restored.turns] == ["t1", "t2", "t3"]
        assert [t.role for t in restored.turns] == ["user", "agent", "user"]
        assert [t.accumulated_tokens for t in restored.turns] == [12, 32, 40]
        # content_hash is recomputed in __post_init__ and must match the original.
        assert [t.content_hash for t in restored.turns] == [t.content_hash for t in original.turns]

    def test_turn_to_universal_span_maps_role_content_tokens(self):
        """Test 6: ConversationTurnData.to_universal_span() maps role/content/tokens."""
        user_turn = ConversationTurnData(
            turn_id="u1",
            turn_number=1,
            role="user",
            participant_id="user-1",
            content="What is the capital of Finland?",
            token_count=9,
        )
        agent_turn = ConversationTurnData(
            turn_id="a1",
            turn_number=2,
            role="agent",
            participant_id="agent-1",
            content="The capital of Finland is Helsinki.",
            token_count=11,
        )

        user_span = user_turn.to_universal_span("conv-trace-1")
        assert user_span.trace_id == "conv-trace-1"
        assert user_span.id == "u1"
        assert user_span.name == "turn:1:user"
        assert user_span.span_type == SpanType.UNKNOWN  # user role -> UNKNOWN
        assert user_span.agent_id == "user-1"
        assert user_span.agent_name == "user-1"
        assert user_span.input_data == {"role": "user", "turn_number": 1}
        assert user_span.output_data == {"content": "What is the capital of Finland?"}
        assert user_span.prompt == "What is the capital of Finland?"  # set for user role
        assert user_span.response is None
        assert user_span.tokens_total == 9
        assert user_span.source_format == "conversation"

        agent_span = agent_turn.to_universal_span("conv-trace-1")
        assert agent_span.name == "turn:2:agent"
        assert agent_span.span_type == SpanType.AGENT  # agent role -> AGENT
        assert agent_span.response == "The capital of Finland is Helsinki."
        assert agent_span.prompt is None
        assert agent_span.tokens_total == 11

    def test_to_universal_spans_covers_every_turn(self):
        """Test 6b: ConversationTrace.to_universal_spans() yields one span per turn."""
        trace = _make_conversation_trace()
        spans = trace.to_universal_spans()
        assert len(spans) == len(trace.turns) == 3
        assert all(s.trace_id == "conv-trace-1" for s in spans)
        assert [s.id for s in spans] == ["t1", "t2", "t3"]
        # Span types follow the role mapping: user->UNKNOWN, agent->AGENT, user->UNKNOWN.
        assert [s.span_type for s in spans] == [
            SpanType.UNKNOWN,
            SpanType.AGENT,
            SpanType.UNKNOWN,
        ]


class TestConversationTracePerTurnFidelity:
    """Per-turn fields must survive a to_dict()->from_dict() round-trip.

    These previously XFAIL'd: to_dict() dropped per-turn `extra` (which from_dict()
    was already prepared to read), plus `tool_calls`/`tool_results` (which
    to_universal_span() depends on) and `timestamp`. The source now emits and re-reads
    all of them, so these are real passing asserts. If a future change re-drops any of
    these fields, these tests catch it.
    """

    def test_per_turn_extra_survives_round_trip(self):
        """Per-turn `extra` round-trips (was the asymmetric read-but-not-written gap)."""
        trace = ConversationTrace(
            trace_id="rt-extra", conversation_id="rt-extra", framework="anthropic"
        )
        trace.add_turn(
            ConversationTurnData(
                turn_id="t1",
                turn_number=1,
                role="user",
                participant_id="user-1",
                content="hello",
                token_count=2,
                extra={"annotation": "gold"},
            )
        )
        restored = ConversationTrace.from_dict(trace.to_dict())
        assert restored.turns[0].extra == {"annotation": "gold"}

    def test_tool_calls_survive_round_trip(self):
        """A tool turn keeps its tool_calls/tool_results across a round-trip."""
        trace = ConversationTrace(
            trace_id="rt-tools", conversation_id="rt-tools", framework="anthropic"
        )
        trace.add_turn(
            ConversationTurnData(
                turn_id="t1",
                turn_number=1,
                role="tool",
                participant_id="tool-1",
                content="search result",
                token_count=2,
                tool_calls=[{"name": "search", "args": {"q": "helsinki"}}],
                tool_results=[{"name": "search", "output": "Helsinki, Finland"}],
            )
        )
        restored = ConversationTrace.from_dict(trace.to_dict())
        assert restored.turns[0].tool_calls == [{"name": "search", "args": {"q": "helsinki"}}]
        assert restored.turns[0].tool_results == [{"name": "search", "output": "Helsinki, Finland"}]
        # And the recovered tool identity flows through to the span conversion.
        span = restored.turns[0].to_universal_span(restored.trace_id)
        assert span.tool_name == "search"
        assert span.tool_args == {"q": "helsinki"}

    def test_timestamp_survives_round_trip(self):
        """Per-turn `timestamp` round-trips via ISO-8601 string."""
        ts = datetime(2025, 1, 1, 12, 30, 45, tzinfo=timezone.utc)
        trace = ConversationTrace(trace_id="rt-ts", conversation_id="rt-ts", framework="anthropic")
        trace.add_turn(
            ConversationTurnData(
                turn_id="t1",
                turn_number=1,
                role="user",
                participant_id="user-1",
                content="hi",
                token_count=1,
                timestamp=ts,
            )
        )
        d = trace.to_dict()
        assert d["turns"][0]["timestamp"] == "2025-01-01T12:30:45+00:00"
        restored = ConversationTrace.from_dict(d)
        assert restored.turns[0].timestamp == ts

    def test_rich_turn_is_identity_at_dict_level(self):
        """A turn carrying every per-turn field is a dict-level round-trip fixed point."""
        trace = ConversationTrace(
            trace_id="rt-rich", conversation_id="rt-rich", framework="anthropic"
        )
        trace.add_turn(
            ConversationTurnData(
                turn_id="t1",
                turn_number=1,
                role="tool",
                participant_id="tool-1",
                content="result",
                token_count=3,
                timestamp=datetime(2025, 6, 1, 9, 0, 0, tzinfo=timezone.utc),
                tool_calls=[{"name": "lookup", "args": {"id": 1}}],
                tool_results=[{"name": "lookup", "output": "ok"}],
                extra={"label": "x"},
            )
        )
        d = trace.to_dict()
        assert ConversationTrace.from_dict(d).to_dict() == d


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
