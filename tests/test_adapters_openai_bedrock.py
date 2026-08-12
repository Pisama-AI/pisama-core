"""Tests for OpenAI and Bedrock trace ingestion adapters.

Two kinds of test live here, and the difference matters.

The hand-written classes below feed the adapter a payload an author believed
matched the source API. That belief is not always correct: two of them are now
known to assert shapes their vendor cannot emit (see the warnings on
`test_code_interpreter_step_preserves_structured_outputs` and
`test_tool_observation_is_parented_to_invocation`). A passing hand-written test
is evidence about the adapter's internals, not about the integration.

`TestOpenAIAgainstRealSDKPayloads` and `TestBedrockAgainstRealServiceModel` are
the ones that speak to the integration. They run against committed fixtures
whose every key comes from the vendor itself — the `openai` package's own
pydantic models, and the `bedrock-agent-runtime` service model that botocore
ships. Regenerate them with `tests/fixtures/capture/`; the recipe and its
guarantees are documented in `tests/fixtures/capture/README.md`.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

from pisama_core.adapters.bedrock import parse_invoke_agent
from pisama_core.adapters.openai import parse_assistants_run, parse_response
from pisama_core.traces.enums import Platform, SpanKind, SpanStatus

FIXTURES = Path(__file__).parent / "fixtures"


class TestOpenAIAssistantsAdapter:
    def test_parse_completed_run_with_tool_call(self):
        run = {
            "id": "run_abc",
            "assistant_id": "asst_1",
            "thread_id": "thread_1",
            "status": "completed",
            "created_at": 1_700_000_000,
            "completed_at": 1_700_000_030,
            "model": "gpt-4.1",
            "usage": {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
        }
        steps = [
            {
                "id": "step_1",
                "type": "tool_calls",
                "status": "completed",
                "created_at": 1_700_000_005,
                "completed_at": 1_700_000_010,
                "step_details": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"SF"}',
                                "output": '{"temp":68}',
                            },
                        }
                    ]
                },
            },
            {
                "id": "step_2",
                "type": "message_creation",
                "status": "completed",
                "created_at": 1_700_000_011,
                "completed_at": 1_700_000_015,
                "step_details": {"message_creation": {"message_id": "msg_9"}},
            },
        ]

        trace = parse_assistants_run(run, steps=steps)

        assert trace.metadata.platform == Platform.OPENAI
        assert trace.metadata.platform_version == "assistants-v2"
        # Root + 2 step spans
        assert len(trace.spans) == 3
        root = trace.spans[0]
        assert root.kind == SpanKind.AGENT
        assert root.status == SpanStatus.OK
        assert root.attributes["openai.usage"]["total_tokens"] == 180
        tool_span = trace.spans[1]
        assert tool_span.kind == SpanKind.TOOL
        assert tool_span.name == "openai.tool:get_weather"
        assert tool_span.input_data["arguments"] == '{"city":"SF"}'
        msg_span = trace.spans[2]
        assert msg_span.kind == SpanKind.MESSAGE
        assert msg_span.output_data["message_id"] == "msg_9"

    def test_code_interpreter_step_preserves_structured_outputs(self):
        """Code-interpreter outputs are structured {type: logs|image, ...}
        blocks; the adapter must preserve them rather than collapse to str().

        WARNING: this payload cannot occur. `RunStep.type` is
        `Literal["message_creation", "tool_calls"]`, so a step whose `type` is
        `"code_interpreter"` is rejected by the SDK's own model. Real code
        interpreter usage arrives as a `CodeInterpreterToolCall` *inside* a
        `tool_calls` step, which the adapter drops entirely.
        `TestOpenAIAgainstRealSDKPayloads` covers the real shape. This test is
        retained only because it pins the behaviour of the (unreachable) branch
        at openai.py:151-176; it is not evidence the integration works.
        """
        run = {
            "id": "run_ci",
            "assistant_id": "a",
            "thread_id": "t",
            "status": "completed",
        }
        steps = [
            {
                "id": "step_ci",
                "type": "code_interpreter",
                "status": "completed",
                "created_at": 1_700_000_000,
                "completed_at": 1_700_000_002,
                "step_details": {
                    "code_interpreter": {
                        "input": "print(1+1)",
                        "outputs": [
                            {"type": "logs", "logs": "2\n"},
                            {"type": "image", "image": {"file_id": "file_abc"}},
                        ],
                    }
                },
            },
        ]
        trace = parse_assistants_run(run, steps=steps)
        ci_spans = [s for s in trace.spans if s.name.startswith("openai.code_interpreter")]
        assert len(ci_spans) == 1
        span = ci_spans[0]
        assert span.kind == SpanKind.TOOL
        assert span.input_data["input"] == "print(1+1)"
        outputs = span.output_data["outputs"]
        assert outputs[0]["type"] == "logs"
        assert outputs[0]["logs"] == "2\n"
        assert outputs[1]["type"] == "image"
        assert outputs[1]["image"]["file_id"] == "file_abc"

    def test_failed_run_maps_to_error_status(self):
        run = {
            "id": "run_fail",
            "assistant_id": "a",
            "thread_id": "t",
            "status": "failed",
            "last_error": {"code": "rate_limit", "message": "slow down"},
        }
        trace = parse_assistants_run(run)
        root = trace.spans[0]
        assert root.status == SpanStatus.ERROR
        assert root.error_message == "slow down"


class TestOpenAIResponsesAdapter:
    def test_parse_response_with_function_call_and_message(self):
        response = {
            "id": "resp_1",
            "model": "gpt-5.4",
            "status": "completed",
            "created_at": 1_700_000_000,
            "completed_at": 1_700_000_008,
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "output": [
                {
                    "type": "function_call",
                    "name": "search",
                    "call_id": "call_1",
                    "arguments": '{"q":"pisama"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '[{"title":"Pisama docs"}]',
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Here is the result."}],
                },
            ],
        }
        trace = parse_response(response)
        assert trace.metadata.platform == Platform.OPENAI
        assert trace.metadata.platform_version == "responses-v1"
        kinds = [s.kind for s in trace.spans]
        assert SpanKind.AGENT_TURN in kinds
        assert kinds.count(SpanKind.TOOL) == 2
        # Last span is the assistant message
        assistant_msg = trace.spans[-1]
        assert assistant_msg.kind == SpanKind.MESSAGE
        assert assistant_msg.output_data["text"] == "Here is the result."


class TestBedrockAdapter:
    def test_parse_orchestration_trace_with_tool_and_final_answer(self):
        session_id = "sess-42"
        agent_id = "AGENT123"
        traces = [
            {
                "trace": {
                    "orchestrationTrace": {"rationale": {"text": "I need to call the weather API."}}
                }
            },
            {
                "trace": {
                    "orchestrationTrace": {
                        "invocationInput": {
                            "actionGroupInvocationInput": {
                                "actionGroupName": "weather",
                                "apiPath": "/get",
                                "function": "get_weather",
                                "parameters": [{"name": "city", "value": "SF"}],
                            }
                        }
                    }
                }
            },
            {
                "trace": {
                    "orchestrationTrace": {
                        "observation": {"actionGroupInvocationOutput": {"text": '{"temp":68}'}}
                    }
                }
            },
            {
                "trace": {
                    "orchestrationTrace": {
                        "observation": {"finalResponse": {"text": "It's 68 in SF."}}
                    }
                }
            },
        ]

        trace = parse_invoke_agent(
            session_id=session_id,
            agent_id=agent_id,
            traces=traces,
            final_answer="It's 68 in SF.",
        )
        assert trace.metadata.platform == Platform.BEDROCK
        assert trace.metadata.session_id == session_id
        kinds = [s.kind for s in trace.spans]
        assert kinds.count(SpanKind.TOOL) == 2  # invocation input + output
        assert kinds.count(SpanKind.USER_OUTPUT) >= 1

    def test_failure_trace_sets_root_error(self):
        traces = [{"trace": {"failureTrace": {"failureReason": "model not available"}}}]
        trace = parse_invoke_agent(session_id="s", agent_id="a", traces=traces)
        root = trace.spans[0]
        assert root.status == SpanStatus.ERROR
        assert "model not available" in (root.error_message or "")

    def test_guardrail_intervention_is_blocked(self):
        traces = [{"trace": {"guardrailTrace": {"action": "INTERVENED"}}}]
        trace = parse_invoke_agent(session_id="s", agent_id="a", traces=traces)
        gr = [s for s in trace.spans if s.name == "bedrock.guardrail"][0]
        assert gr.status == SpanStatus.BLOCKED

    def test_tool_observation_is_parented_to_invocation(self):
        """bedrock.tool.output should be a child of bedrock.tool, not a sibling.

        WARNING: this payload cannot occur. `OrchestrationTrace` is declared
        `"union": true` in the `bedrock-agent-runtime` service model, so
        `invocationInput` and `observation` can never share one trace node.
        Against real payloads they arrive as separate nodes and the parenting
        this test asserts is unreachable. Retained to pin the branch's
        behaviour, not as evidence the integration works.
        """
        traces = [
            {
                "trace": {
                    "orchestrationTrace": {
                        "invocationInput": {
                            "actionGroupInvocationInput": {
                                "actionGroupName": "weather",
                                "function": "get_weather",
                                "parameters": [],
                            }
                        },
                        "observation": {
                            "actionGroupInvocationOutput": {"text": '{"temp":68}'},
                        },
                    }
                }
            }
        ]
        trace = parse_invoke_agent(session_id="s", agent_id="a", traces=traces)
        tool_input = [s for s in trace.spans if s.name == "bedrock.tool:get_weather"][0]
        tool_output = [s for s in trace.spans if s.name == "bedrock.tool.output"][0]
        assert tool_output.parent_id == tool_input.span_id

    def test_kb_observation_is_parented_to_lookup(self):
        traces = [
            {
                "trace": {
                    "orchestrationTrace": {
                        "invocationInput": {
                            "knowledgeBaseLookupInput": {
                                "knowledgeBaseId": "kb_1",
                                "text": "what is pisama?",
                            }
                        },
                        "observation": {
                            "knowledgeBaseLookupOutput": {
                                "retrievedReferences": [{"content": "..."}],
                            },
                        },
                    }
                }
            }
        ]
        trace = parse_invoke_agent(session_id="s", agent_id="a", traces=traces)
        kb_input = [s for s in trace.spans if s.name == "bedrock.kb.lookup"][0]
        kb_output = [s for s in trace.spans if s.name == "bedrock.kb.output"][0]
        assert kb_output.parent_id == kb_input.span_id

    def test_custom_orchestration_trace_is_captured(self):
        """customOrchestrationTrace events (AWS 2025) must not be dropped."""
        traces = [{"trace": {"customOrchestrationTrace": {"event": {"stage": "planner"}}}}]
        trace = parse_invoke_agent(session_id="s", agent_id="a", traces=traces)
        custom = [s for s in trace.spans if s.name == "bedrock.custom_orchestration"]
        assert len(custom) == 1


class TestOpenAIAgainstRealSDKPayloads:
    """Run the adapter against payloads the `openai` package itself produced.

    Every key here came from instantiating the SDK's own pydantic models and
    dumping them, so pydantic rejected any shape the API cannot return. That is
    the difference between these tests and the hand-written ones above.
    """

    @staticmethod
    def _assistants():
        return json.loads((FIXTURES / "openai_assistants_run.json").read_text())

    @staticmethod
    def _responses():
        return json.loads((FIXTURES / "openai_responses_api.json").read_text())

    def test_real_assistants_run_parses(self):
        payload = self._assistants()
        trace = parse_assistants_run(
            payload["run"], steps=payload["steps"], thread_messages=payload["thread_messages"]
        )
        assert trace.metadata.platform == Platform.OPENAI
        assert trace.spans[0].kind == SpanKind.AGENT
        # The run failed, so the root must carry that.
        assert trace.spans[0].status == SpanStatus.ERROR
        assert any(s.kind == SpanKind.TOOL for s in trace.spans)

    def test_real_assistants_run_step_types_are_only_what_the_sdk_can_emit(self):
        """RunStep.type is Literal["message_creation", "tool_calls"].

        The adapter also branches on "code_interpreter" at step level, which the
        SDK's model rejects. This pins the real vocabulary so that branch cannot
        be mistaken for a supported path.
        """
        steps = self._assistants()["steps"]
        assert {s["type"] for s in steps} <= {"message_creation", "tool_calls"}

    def test_real_code_interpreter_tool_call_keeps_its_payload(self):
        payload = self._assistants()
        trace = parse_assistants_run(payload["run"], steps=payload["steps"])
        ci = [s for s in trace.spans if "code_interpreter" in s.name]
        assert ci, "expected a code_interpreter tool span"
        # The executed source and its structured outputs must both survive.
        assert ci[0].input_data.get("input")
        assert ci[0].output_data.get("outputs")

    def test_real_responses_api_parses(self):
        trace = parse_response(self._responses())
        assert trace.metadata.platform == Platform.OPENAI
        assert any(s.kind == SpanKind.TOOL for s in trace.spans)

    def test_function_call_output_really_can_appear_in_response_output(self):
        """Guards a branch a prior audit suspected was dead. It is not.

        ResponseFunctionToolCallOutputItem is a member of the ResponseOutputItem
        union, so openai.py's "function_call_output" branch is reachable.
        """
        output_types = {item["type"] for item in self._responses()["output"]}
        assert "function_call_output" in output_types
        trace = parse_response(self._responses())
        assert any(s.name.startswith("openai.tool_output") for s in trace.spans)

    def test_real_responses_api_token_usage_is_recorded(self):
        payload = self._responses()
        trace = parse_response(payload)
        attrs = trace.spans[0].attributes
        assert attrs.get("gen_ai.usage.input_tokens") == payload["usage"]["input_tokens"]


class TestBedrockAgainstRealServiceModel:
    """Run the adapter against a payload built from botocore's service model.

    Every key comes from the `bedrock-agent-runtime` shape definitions that
    botocore ships, and the capture round-trips through botocore's own
    eventstream decoder, so an invented key would be dropped and the round-trip
    would not match. See tests/fixtures/capture/README.md.
    """

    @staticmethod
    def _fixture():
        return json.loads((FIXTURES / "bedrock_invoke_agent_trace.json").read_text())

    def _parse(self, scenario):
        payload = self._fixture()
        request = payload["invoke_agent_request"]
        events = payload["scenarios"][scenario]["completion_events"]
        return parse_invoke_agent(
            session_id=request["sessionId"],
            agent_id=request["agentId"],
            traces=[event["trace"] for event in events],
        )

    def test_real_orchestration_stream_parses(self):
        trace = self._parse("success")
        assert trace.metadata.platform == Platform.BEDROCK
        kinds = {s.kind for s in trace.spans}
        assert SpanKind.TOOL in kinds
        assert SpanKind.LLM in kinds

    def test_real_failure_trace_sets_root_error(self):
        trace = self._parse("failure")
        assert trace.spans[0].status == SpanStatus.ERROR

    def test_real_spans_carry_the_timestamps_the_service_model_provides(self):
        """Spans must be stamped from the payload, not from ingestion time.

        Not every node carries a duration: `TracePart.eventTime` is a point in
        time, and only some nodes nest a `metadata` start/end pair. So end_time
        is legitimately None for point events. What must never happen is a span
        timed to when Pisama happened to parse it.
        """
        payload = self._fixture()
        events = payload["scenarios"]["success"]["completion_events"]
        stamps = sorted(e["trace"]["eventTime"] for e in events)
        earliest = datetime.fromisoformat(stamps[0].replace("Z", "+00:00"))
        latest = datetime.fromisoformat(stamps[-1].replace("Z", "+00:00"))

        trace = self._parse("success")
        for span in trace.spans:
            assert span.start_time is not None
            assert earliest <= span.start_time <= latest + timedelta(seconds=30), span.name

        root = trace.spans[0]
        assert root.start_time == earliest
        # The root closes on the trace, not on ingestion: a stale _now() here
        # reported the gap between invocation and parse as agent latency.
        assert root.end_time is not None
        assert root.end_time <= latest + timedelta(seconds=30)
        assert 0 < (trace.duration_ms or 0) < 60_000

    def test_real_model_identifier_is_recorded(self):
        trace = self._parse("success")
        assert any("gen_ai.request.model" in s.attributes for s in trace.spans)
