"""Tests for the Gemini Interactions API trace ingestion adapter.

These run against committed fixtures whose every key comes from the vendor:
they are built by instantiating google-genai's own ``Interaction`` model and
dumping it, so a shape the SDK cannot produce cannot appear here. Regenerate
with ``tests/fixtures/capture/capture_gemini.py``; the recipe and its
guarantees are in ``tests/fixtures/capture/README.md``.

This file previously tested a different API. The adapter was written against an
envelope the Interactions API never returned — ``messages[]``, ``tool_calls[]``,
``candidates[]``, ``state.tasks[]``, ``session_id``, ``created_at`` /
``finished_at``, ``usage_metadata.prompt_token_count`` — and the tests supplied
exactly that envelope, so they passed while a real interaction parsed to a bare
root span with no usage and no conversation. Those tests were removed rather
than kept: unlike a hand-written test that pins a real-but-unreachable branch,
they asserted capabilities the adapter no longer has, because the fields they
drove do not exist.

One caveat inherited from the vendor, recorded so these tests are not read as
stronger than they are: ``_gaos.BaseModel`` sets ``extra="allow"`` and parses
steps leniently, so ``Interaction.model_validate`` alone would accept some
malformed payloads. The capture script's ``--check`` mode compensates by
re-validating every step strictly through its concrete class, which does raise.
"""

import json
from pathlib import Path

import pytest

from pisama_core.adapters.gemini import parse_interactions_response
from pisama_core.traces.enums import Platform, SpanKind, SpanStatus

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def completed():
    return _load("gemini_interaction_completed.json")


@pytest.fixture
def failed():
    return _load("gemini_interaction_failed.json")


class TestGeminiInteractionsAgainstRealSDKPayloads:
    def test_root_span_carries_interaction_identity(self, completed):
        trace = parse_interactions_response(completed)
        root = trace.spans[0]
        assert trace.metadata.platform == Platform.GEMINI
        assert root.kind == SpanKind.AGENT_TURN
        assert root.status == SpanStatus.OK
        assert completed["id"] in root.name
        assert root.platform_metadata["model"] == completed["model"]

    def test_timestamps_come_from_created_and_updated(self, completed):
        """The API sends ISO-8601 `created` / `updated`, not `created_at` epochs."""
        trace = parse_interactions_response(completed)
        root = trace.spans[0]
        assert root.start_time is not None
        assert root.start_time.isoformat().startswith(completed["created"][:16])
        if completed.get("updated"):
            assert root.end_time is not None

    def test_usage_uses_the_interactions_token_names(self, completed):
        """`total_input_tokens`, not the generateContent `prompt_token_count`."""
        usage = completed["usage"]
        attrs = parse_interactions_response(completed).spans[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == usage["total_input_tokens"]
        assert attrs["gen_ai.usage.output_tokens"] == usage["total_output_tokens"]
        assert attrs["gen_ai.system"] == "gemini"

    def test_thought_and_tool_tokens_are_not_folded_into_the_standard_keys(self, completed):
        """They have no OTEL equivalent, so they must not inflate the standard counts."""
        usage = completed["usage"]
        attrs = parse_interactions_response(completed).spans[0].attributes
        assert attrs["gemini.usage.thought_tokens"] == usage["total_thought_tokens"]
        assert attrs["gen_ai.usage.input_tokens"] != usage["total_thought_tokens"]

    def test_every_step_becomes_a_span(self, completed):
        trace = parse_interactions_response(completed)
        assert len(trace.spans) == 1 + len(completed["steps"])
        types = [s.attributes["gemini.step.type"] for s in trace.spans[1:]]
        assert types == [s["type"] for s in completed["steps"]]

    def test_step_types_map_to_meaningful_kinds(self, completed):
        by_type = {
            s.attributes["gemini.step.type"]: s.kind
            for s in parse_interactions_response(completed).spans[1:]
        }
        assert by_type["user_input"] == SpanKind.USER_INPUT
        assert by_type["model_output"] == SpanKind.LLM
        assert by_type["function_call"] == SpanKind.TOOL
        assert by_type["thought"] == SpanKind.TASK

    def test_function_result_is_parented_to_its_call(self, completed):
        """`call_id` on the result matches `id` on the call."""
        trace = parse_interactions_response(completed)
        spans = {s.span_id: s for s in trace.spans}
        call = next(
            s for s in trace.spans if s.attributes.get("gemini.step.type") == "function_call"
        )
        result = next(
            s for s in trace.spans if s.attributes.get("gemini.step.type") == "function_result"
        )
        assert result.parent_id == call.span_id
        assert spans[result.parent_id] is call

    def test_function_call_arguments_are_a_dict_not_a_json_string(self, completed):
        """This API sends `arguments` as an object, unlike OpenAI's JSON string."""
        call_step = next(s for s in completed["steps"] if s["type"] == "function_call")
        trace = parse_interactions_response(completed)
        span = next(
            s for s in trace.spans if s.attributes.get("gemini.step.type") == "function_call"
        )
        assert isinstance(call_step["arguments"], dict)
        assert span.input_data["arguments"] == call_step["arguments"]

    def test_session_falls_back_when_the_api_supplies_none(self, completed):
        """The Interactions API has no session field; environment_id is the nearest."""
        assert "session_id" not in completed and "session" not in completed
        assert parse_interactions_response(completed, session_id="s-1").metadata.session_id == "s-1"
        without = parse_interactions_response(completed).metadata.session_id
        assert without == (completed.get("environment_id") or completed["id"])


class TestGeminiInteractionsFailures:
    def test_failed_interaction_maps_to_error_with_its_message(self, failed):
        trace = parse_interactions_response(failed)
        root = trace.spans[0]
        assert root.status == SpanStatus.ERROR
        assert root.error_message == failed["errors"][0]["message"]

    def test_errors_is_a_list_and_there_is_no_top_level_error(self, failed):
        assert isinstance(failed["errors"], list)
        assert "error" not in failed
        assert parse_interactions_response(failed).spans[0].attributes["gemini.errors"]

    def test_failing_tool_result_is_flagged(self, failed):
        result = next(
            s
            for s in parse_interactions_response(failed).spans
            if s.attributes.get("gemini.step.type") == "function_result"
        )
        assert result.status == SpanStatus.ERROR

    def test_model_output_error_is_an_rpc_status_not_an_interaction_error(self, failed):
        """Step-level `error` is google.rpc Status ({code:int}); the interaction's is {code:str}."""
        step = next(s for s in failed["steps"] if s["type"] == "model_output" and s.get("error"))
        assert isinstance(step["error"]["code"], int)
        assert isinstance(failed["errors"][0]["code"], str)
        span = next(
            s
            for s in parse_interactions_response(failed).spans
            if s.attributes.get("gemini.step.type") == "model_output"
        )
        assert span.status == SpanStatus.ERROR
        assert span.error_message == step["error"]["message"]


class TestGeminiInteractionsStatusMapping:
    @pytest.mark.parametrize(
        "status,expected",
        [
            ("completed", SpanStatus.OK),
            ("failed", SpanStatus.ERROR),
            ("cancelled", SpanStatus.CANCELLED),
            ("in_progress", SpanStatus.IN_PROGRESS),
            ("queued", SpanStatus.IN_PROGRESS),
            ("requires_action", SpanStatus.IN_PROGRESS),
            ("incomplete", SpanStatus.ERROR),
            ("budget_exceeded", SpanStatus.ERROR),
        ],
    )
    def test_every_real_status_value_is_mapped(self, completed, status, expected):
        """All eight are real API values. The adapter previously uppercased them."""
        payload = dict(completed, status=status)
        assert parse_interactions_response(payload).spans[0].status == expected

    def test_unknown_status_is_unset_not_ok(self, completed):
        payload = dict(completed, status="something_new")
        assert parse_interactions_response(payload).spans[0].status == SpanStatus.UNSET

    def test_non_dict_input_is_rejected(self):
        with pytest.raises(TypeError):
            parse_interactions_response([])
