"""Tests for the OpenAI Agents SDK trace ingestion adapter.

The fixture is not hand-written. `tests/fixtures/openai_agents_handoff_trace.json`
was captured from openai-agents 0.19.0 by driving the SDK's own span context
managers through a capturing `TracingProcessor`, so every payload asserted here
is the verbatim output of the SDK's `TraceImpl.export()` / `SpanImpl.export()`.
No OpenAI model was called to produce it, and the test suite gains no dependency
on the `openai-agents` package.

Scenario in the fixture: TriageAgent hands off to RefundAgent, the refund tool
fails with a gateway timeout, and an output guardrail then trips. That covers the
handoff-architecture failure shape the adapter exists to carry.
"""

import json
from pathlib import Path

import pytest

from pisama_core.adapters.openai_agents import parse_agents_span, parse_agents_trace
from pisama_core.traces.enums import Platform, SpanKind, SpanStatus

FIXTURE = Path(__file__).parent / "fixtures" / "openai_agents_handoff_trace.json"


@pytest.fixture(scope="module")
def payload():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def parsed(payload):
    return parse_agents_trace(payload["traces"][0], payload["spans"])


class TestTraceLevel:
    def test_trace_identity_comes_from_the_sdk(self, parsed, payload):
        assert parsed.trace_id == payload["traces"][0]["id"]
        assert parsed.trace_id.startswith("trace_")

    def test_platform_version_distinguishes_agents_sdk(self, parsed):
        """Must not collide with the Assistants/Responses adapter.

        Both surfaces are OpenAI, so Platform alone cannot tell them apart;
        platform_version is what routes them.
        """
        assert parsed.metadata.platform == Platform.OPENAI
        assert parsed.metadata.platform_version == "agents-sdk-v1"

    def test_workflow_name_and_group_id_preserved(self, parsed):
        assert parsed.metadata.custom["workflow_name"] == "support-triage"
        assert parsed.metadata.session_id == "thread_abc123"

    def test_every_sdk_span_survives_ingestion(self, parsed, payload):
        assert len(parsed.spans) == len(payload["spans"]) == 8

    def test_no_synthetic_root_is_inserted(self, parsed, payload):
        """Adding a root would change the depth structural detectors measure."""
        sdk_ids = {s["id"] for s in payload["spans"]}
        assert {s.span_id for s in parsed.spans} == sdk_ids

    def test_parent_child_structure_is_carried_through(self, parsed, payload):
        by_id = {s.span_id: s for s in parsed.spans}
        for raw in payload["spans"]:
            assert by_id[raw["id"]].parent_id == raw["parent_id"]

    def test_agent_spans_are_roots_of_their_subtrees(self, parsed):
        agents = [s for s in parsed.spans if s.kind == SpanKind.AGENT]
        assert {s.attributes["openai_agents.agent.name"] for s in agents} == {
            "TriageAgent",
            "RefundAgent",
        }


class TestSpanKindMapping:
    @pytest.mark.parametrize(
        "span_type,expected_kind",
        [
            ("agent", SpanKind.AGENT),
            ("handoff", SpanKind.HANDOFF),
            ("function", SpanKind.TOOL),
            ("generation", SpanKind.LLM),
            ("guardrail", SpanKind.SYSTEM),
            ("custom", SpanKind.SYSTEM),
        ],
    )
    def test_type_maps_to_kind(self, parsed, span_type, expected_kind):
        matching = [
            s for s in parsed.spans if s.attributes["openai_agents.span.type"] == span_type
        ]
        assert matching, f"fixture has no {span_type} span"
        assert all(s.kind == expected_kind for s in matching)

    def test_unknown_span_type_keeps_its_payload(self):
        """A newer SDK type must not silently drop data on ingest."""
        span = parse_agents_span(
            {
                "object": "trace.span",
                "id": "span_future",
                "trace_id": "trace_1",
                "parent_id": None,
                "started_at": "2026-07-28T10:00:00+00:00",
                "ended_at": "2026-07-28T10:00:01+00:00",
                "span_data": {"type": "some_future_type", "novel_field": 42},
                "error": None,
            }
        )
        assert span.kind == SpanKind.SYSTEM
        assert span.input_data["span_data"]["novel_field"] == 42


class TestHandoff:
    def test_handoff_edge_is_preserved(self, parsed):
        handoffs = [s for s in parsed.spans if s.kind == SpanKind.HANDOFF]
        assert len(handoffs) == 1
        h = handoffs[0]
        assert h.attributes["openai_agents.handoff.from"] == "TriageAgent"
        assert h.attributes["openai_agents.handoff.to"] == "RefundAgent"

    def test_handoff_span_is_named_readably(self, parsed):
        h = next(s for s in parsed.spans if s.kind == SpanKind.HANDOFF)
        assert h.name == "openai_agents.handoff:TriageAgent->RefundAgent"

    def test_declared_handoffs_stay_distinct_from_taken_ones(self, parsed):
        """TriageAgent declares two handoffs but only one fired.

        Keeping the declared list separate from the handoff spans is what
        lets a detector ask whether a declared route was never taken.
        """
        triage = next(
            s
            for s in parsed.spans
            if s.attributes.get("openai_agents.agent.name") == "TriageAgent"
        )
        assert triage.attributes["openai_agents.agent.handoffs"] == [
            "RefundAgent",
            "BillingAgent",
        ]
        taken = {
            s.attributes["openai_agents.handoff.to"]
            for s in parsed.spans
            if s.kind == SpanKind.HANDOFF
        }
        assert taken == {"RefundAgent"}


class TestToolsAndErrors:
    def test_successful_tool_call_carries_io(self, parsed):
        lookup = next(
            s for s in parsed.spans if s.attributes.get("openai_agents.tool.name") == "lookup_order"
        )
        assert lookup.kind == SpanKind.TOOL
        assert lookup.status == SpanStatus.OK
        assert lookup.input_data["arguments"] == '{"order_id": "4417"}'
        assert "shipped" in lookup.output_data["output"]

    def test_failed_tool_call_becomes_error_status(self, parsed):
        refund = next(
            s for s in parsed.spans if s.attributes.get("openai_agents.tool.name") == "issue_refund"
        )
        assert refund.status == SpanStatus.ERROR
        assert refund.error_message == "payment gateway timeout"

    def test_only_the_failing_span_is_errored(self, parsed):
        errored = [s for s in parsed.spans if s.status == SpanStatus.ERROR]
        assert len(errored) == 1

    def test_running_span_is_not_reported_as_success(self):
        """No error plus no ended_at means in-progress, not OK."""
        span = parse_agents_span(
            {
                "id": "span_live",
                "trace_id": "trace_1",
                "parent_id": None,
                "started_at": "2026-07-28T10:00:00+00:00",
                "ended_at": None,
                "span_data": {"type": "function", "name": "slow_tool"},
                "error": None,
            }
        )
        assert span.status == SpanStatus.IN_PROGRESS


class TestGuardrail:
    def test_triggered_guardrail_is_first_class(self, parsed):
        g = next(
            s
            for s in parsed.spans
            if s.attributes.get("openai_agents.guardrail.name") == "refund_limit_guardrail"
        )
        assert g.attributes["openai_agents.guardrail.triggered"] is True


class TestUsage:
    def test_agents_sdk_token_keys_map_to_gen_ai(self, parsed):
        """The Agents SDK reports input_tokens/output_tokens, not prompt/completion."""
        gen = next(s for s in parsed.spans if s.kind == SpanKind.LLM)
        assert gen.attributes["gen_ai.usage.input_tokens"] == 412
        assert gen.attributes["gen_ai.usage.output_tokens"] == 37
        assert gen.attributes["gen_ai.request.model"] == "gpt-4.1"

    def test_total_is_derived_when_absent(self, parsed):
        gen = next(s for s in parsed.spans if s.kind == SpanKind.LLM)
        assert gen.attributes["gen_ai.usage.total_tokens"] == 449

    def test_legacy_token_keys_still_accepted(self):
        span = parse_agents_span(
            {
                "id": "span_legacy",
                "trace_id": "t",
                "span_data": {
                    "type": "generation",
                    "model": "gpt-4.1",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
                "error": None,
                "ended_at": "2026-07-28T10:00:01+00:00",
            }
        )
        assert span.attributes["gen_ai.usage.input_tokens"] == 10
        assert span.attributes["gen_ai.usage.output_tokens"] == 5
        assert span.attributes["gen_ai.usage.total_tokens"] == 15

    def test_gen_ai_system_is_set_on_every_span(self, parsed):
        assert all(s.attributes.get("gen_ai.system") == "openai" for s in parsed.spans)


class TestRobustness:
    def test_timestamps_parse_to_aware_datetimes(self, parsed):
        for s in parsed.spans:
            assert s.start_time.tzinfo is not None

    def test_empty_span_list_yields_empty_trace(self, payload):
        t = parse_agents_trace(payload["traces"][0], [])
        assert t.spans == []
        assert t.trace_id == payload["traces"][0]["id"]

    def test_span_without_id_gets_a_generated_one(self):
        """span_id is a non-optional str, so it must never come back None."""
        span = parse_agents_span({"span_data": {"type": "custom", "name": "x"}})
        assert isinstance(span.span_id, str)
        assert span.span_id
