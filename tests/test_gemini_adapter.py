"""Tests for the Gemini Interactions API trace ingestion adapter.

These tests feed ``parse_interactions_response`` payloads that match the
documented JSON shape of the Gemini Interactions API Beta and assert that
the resulting Trace has the expected structure. No ``google-genai`` imports
required.
"""

from pisama_core.adapters.gemini import parse_interactions_response
from pisama_core.traces.enums import Platform, SpanKind, SpanStatus


class TestGeminiInteractionsAdapter:
    def test_parse_single_turn_with_usage(self):
        response = {
            "id": "interaction_abc",
            "session_id": "sess-1",
            "model": "gemini-3.1-pro",
            "created_at": "2026-04-19T12:00:00+00:00",
            "finished_at": "2026-04-19T12:00:02+00:00",
            "status": "COMPLETED",
            "messages": [
                {
                    "role": "user",
                    "parts": [{"text": "Hello"}],
                    "timestamp": "2026-04-19T12:00:00+00:00",
                },
                {
                    "role": "model",
                    "parts": [{"text": "Hi!"}],
                    "timestamp": "2026-04-19T12:00:02+00:00",
                },
            ],
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hi!"}], "role": "model"},
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
            },
        }

        trace = parse_interactions_response(response)

        assert trace.metadata.platform == Platform.GEMINI
        assert trace.metadata.platform_version == "interactions-beta-v1"
        assert trace.metadata.session_id == "sess-1"

        root = trace.spans[0]
        assert root.kind == SpanKind.AGENT_TURN
        assert root.platform == Platform.GEMINI
        assert root.status == SpanStatus.OK
        assert root.attributes["gen_ai.system"] == "gemini"
        assert root.attributes["gen_ai.request.model"] == "gemini-3.1-pro"

        kinds = [s.kind for s in trace.spans]
        assert kinds.count(SpanKind.USER_INPUT) == 1
        assert kinds.count(SpanKind.USER_OUTPUT) == 1
        assert kinds.count(SpanKind.LLM) == 1

        candidate = [s for s in trace.spans if s.kind == SpanKind.LLM][0]
        assert candidate.attributes["gen_ai.usage.input_tokens"] == 10
        assert candidate.attributes["gen_ai.usage.output_tokens"] == 5
        assert candidate.attributes["gen_ai.usage.total_tokens"] == 15
        assert candidate.attributes["gen_ai.system"] == "gemini"

    def test_multi_turn_with_tool_call(self):
        response = {
            "id": "interaction_tool",
            "session_id": "sess-2",
            "model": "gemini-3.1-pro",
            "status": "COMPLETED",
            "messages": [
                {"role": "user", "parts": [{"text": "Weather in SF?"}]},
                {"role": "model", "parts": [{"text": "It's 68F"}]},
            ],
            "tool_calls": [
                {
                    "id": "tc_1",
                    "name": "get_weather",
                    "args": {"city": "SF"},
                    "result": {"temp": 68},
                    "started_at": "2026-04-19T12:00:01+00:00",
                    "finished_at": "2026-04-19T12:00:02+00:00",
                }
            ],
            "candidates": [{"content": {"parts": [{"text": "It's 68F"}]}, "finish_reason": "STOP"}],
        }

        trace = parse_interactions_response(response)
        tool_spans = [s for s in trace.spans if s.kind == SpanKind.TOOL]
        assert len(tool_spans) == 1
        tool = tool_spans[0]
        assert tool.name == "gemini.tool:get_weather"
        assert tool.attributes["gemini.tool.name"] == "get_weather"
        assert tool.attributes["gemini.tool.call_id"] == "tc_1"
        assert tool.input_data == {"arguments": {"city": "SF"}}
        assert tool.output_data == {"result": {"temp": 68}}
        assert tool.status == SpanStatus.OK

    def test_task_decomposition_populates_goals(self):
        """Tasks under state.tasks must emit TASK spans with `goals` so the
        decomposition detector keeps firing vendor-neutrally."""
        response = {
            "id": "interaction_tasks",
            "model": "gemini-3.1-pro",
            "status": "COMPLETED",
            "state": {
                "tasks": [
                    {"id": "t1", "goal": "Fetch weather", "status": "DONE"},
                    {"id": "t2", "goal": "Summarize", "status": "IN_PROGRESS"},
                ]
            },
        }

        trace = parse_interactions_response(response)
        task_spans = [s for s in trace.spans if s.kind == SpanKind.TASK]
        assert len(task_spans) == 2
        assert task_spans[0].attributes["goals"] == ["Fetch weather"]
        assert task_spans[0].status == SpanStatus.OK
        assert task_spans[1].status == SpanStatus.IN_PROGRESS

    def test_long_running_operation_marks_root_in_progress(self):
        response = {
            "id": "interaction_lr",
            "model": "gemini-3.1-pro",
            "status": "RUNNING",
            "operation_name": "operations/abc-123",
        }
        trace = parse_interactions_response(response)
        root = trace.spans[0]
        assert root.status == SpanStatus.IN_PROGRESS
        assert trace.metadata.custom["operation_name"] == "operations/abc-123"

    def test_safety_block_maps_to_blocked_status(self):
        response = {
            "id": "interaction_safety",
            "model": "gemini-3.1-pro",
            "status": "COMPLETED",
            "candidates": [
                {
                    "content": {"parts": []},
                    "finish_reason": "SAFETY",
                    "safety_ratings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "probability": "HIGH",
                            "blocked": True,
                        },
                    ],
                }
            ],
        }
        trace = parse_interactions_response(response)
        candidate = [s for s in trace.spans if s.kind == SpanKind.LLM][0]
        assert candidate.status == SpanStatus.BLOCKED
        assert len(candidate.events) == 1
        assert candidate.events[0].name == "safety_check"
        assert candidate.events[0].attributes["category"] == "HARM_CATEGORY_HARASSMENT"
        assert candidate.events[0].attributes["blocked"] is True

    def test_failed_status_maps_to_error(self):
        response = {
            "id": "interaction_fail",
            "model": "gemini-3.1-pro",
            "status": "FAILED",
        }
        trace = parse_interactions_response(response)
        assert trace.spans[0].status == SpanStatus.ERROR

    def test_session_id_falls_back_to_argument(self):
        """When response lacks session_id, caller-supplied session_id wins."""
        response = {"id": "i", "model": "gemini-3.1-pro", "status": "COMPLETED"}
        trace = parse_interactions_response(response, session_id="explicit-session")
        assert trace.metadata.session_id == "explicit-session"

    def test_tool_call_with_error_maps_to_error_status(self):
        response = {
            "id": "interaction_tool_err",
            "model": "gemini-3.1-pro",
            "status": "COMPLETED",
            "tool_calls": [
                {"id": "tc", "name": "bad_tool", "error": "timeout"},
            ],
        }
        trace = parse_interactions_response(response)
        tool = [s for s in trace.spans if s.kind == SpanKind.TOOL][0]
        assert tool.status == SpanStatus.ERROR
        assert tool.error_message == "timeout"
