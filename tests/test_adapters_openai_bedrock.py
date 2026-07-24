"""Tests for OpenAI and Bedrock trace ingestion adapters.

These adapters restore Pisama's "vendor neutral" claim. Each test feeds the
adapter a payload matching the documented JSON shape of its source API and
asserts that the resulting Trace has the expected structure. No SDK imports
required.
"""

from pisama_core.adapters.bedrock import parse_invoke_agent
from pisama_core.adapters.openai import parse_assistants_run, parse_response
from pisama_core.traces.enums import Platform, SpanKind, SpanStatus


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
        """bedrock.tool.output should be a child of bedrock.tool, not a sibling."""
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
