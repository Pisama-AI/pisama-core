"""Tests for the Google ADK trace ingestion adapter.

Feeds ``parse_adk_trace`` / ``GoogleAdkAdapter`` event-dict payloads shaped
like ADK's runtime events and asserts the resulting Trace. No ``google-adk``
imports required.
"""

from pisama_core.adapters.base import InjectionMethod
from pisama_core.adapters.google_adk import GoogleAdkAdapter, parse_adk_trace
from pisama_core.injection.enforcement import EnforcementLevel
from pisama_core.traces.enums import Platform, SpanKind, SpanStatus


class TestGoogleAdkParser:
    def test_parse_single_agent_single_tool_turn(self):
        events = [
            {
                "type": "agent.invocation.start",
                "agent": "weather",
                "timestamp": "2026-04-19T10:00:00+00:00",
            },
            {
                "type": "tool.function_call",
                "name": "get_weather",
                "function_call_id": "fc_1",
                "args": {"city": "SF"},
                "timestamp": "2026-04-19T10:00:01+00:00",
            },
            {
                "type": "tool.function_response",
                "function_call_id": "fc_1",
                "name": "get_weather",
                "response": {"temp": 68},
                "timestamp": "2026-04-19T10:00:02+00:00",
            },
            {
                "type": "llm.response",
                "model": "gemini-3.1-pro",
                "text": "It's 68F in SF.",
                "usage_metadata": {
                    "prompt_token_count": 100,
                    "candidates_token_count": 15,
                    "total_token_count": 115,
                },
                "timestamp": "2026-04-19T10:00:03+00:00",
            },
            {"type": "agent.invocation.end", "timestamp": "2026-04-19T10:00:04+00:00"},
        ]

        trace = parse_adk_trace(events, invocation_id="inv-1", agent_name="weather")

        assert trace.metadata.platform == Platform.GOOGLE_ADK
        assert trace.metadata.platform_version == "adk-v1"
        root = trace.spans[0]
        assert root.kind == SpanKind.AGENT
        assert root.attributes["google.adk.invocation.id"] == "inv-1"

        tool_call = [
            s for s in trace.spans if s.name == "adk.tool:get_weather" and s.kind == SpanKind.TOOL
        ][0]
        tool_response = [s for s in trace.spans if s.name == "adk.tool_output:get_weather"][0]
        # The response span must be parented to the call span (OpenAI/Bedrock parity).
        assert tool_response.parent_id == tool_call.span_id
        assert tool_response.output_data == {"result": {"temp": 68}}

        llm_span = [s for s in trace.spans if s.kind == SpanKind.LLM][0]
        assert llm_span.attributes["gen_ai.system"] == "google_adk"
        assert llm_span.attributes["gen_ai.usage.input_tokens"] == 100
        assert llm_span.attributes["gen_ai.usage.output_tokens"] == 15
        assert llm_span.attributes["gen_ai.usage.total_tokens"] == 115

    def test_agent_transfer_emits_handoff(self):
        events = [
            {"type": "agent.invocation.start", "agent": "dispatcher"},
            {"type": "agent.transfer", "from": "dispatcher", "to": "coder", "message": "go code"},
        ]
        trace = parse_adk_trace(events, agent_name="dispatcher")
        handoff = [s for s in trace.spans if s.kind == SpanKind.HANDOFF]
        assert len(handoff) == 1
        assert handoff[0].attributes["google.adk.handoff.to"] == "coder"
        assert handoff[0].output_data == {"message": "go code"}

    def test_sub_agent_nested_context_is_isolated(self):
        """A sub-agent's child spans must be parented to the sub-agent span,
        not to the root — matching ADK's isolated-context semantics."""
        events = [
            {"type": "agent.invocation.start", "agent": "root"},
            {"type": "sub_agent.start", "agent": "researcher", "task": "find sources"},
            {"type": "tool.function_call", "name": "search", "function_call_id": "fc_s"},
            {
                "type": "tool.function_response",
                "function_call_id": "fc_s",
                "name": "search",
                "response": {"hits": 3},
            },
            {"type": "sub_agent.end"},
        ]
        trace = parse_adk_trace(events, agent_name="root")
        sub_spans = [s for s in trace.spans if s.name == "adk.sub_agent:researcher"]
        assert len(sub_spans) == 1
        sub = sub_spans[0]
        assert sub.attributes["google.adk.sub_agent.isolated_context"] is True

        tool_call = [s for s in trace.spans if s.name == "adk.tool:search"][0]
        # Tool call emitted between sub_agent.start and sub_agent.end must be
        # parented to the sub-agent, not the root.
        assert tool_call.parent_id == sub.span_id

    def test_state_delta_event_on_root(self):
        events = [
            {"type": "agent.invocation.start", "agent": "a"},
            {
                "type": "session.state_delta",
                "delta": {"counter": {"before": 0, "after": 1}},
                "agent": "a",
            },
        ]
        trace = parse_adk_trace(events, agent_name="a")
        root = trace.spans[0]
        delta_events = [e for e in root.events if e.name == "state_delta"]
        assert len(delta_events) == 1
        assert delta_events[0].attributes["changed_keys"] == ["counter"]

    def test_planner_plan_populates_goals(self):
        """planner.plan events must surface `goals` so the decomposition
        detector fires vendor-neutrally."""
        events = [
            {"type": "agent.invocation.start", "agent": "a"},
            {
                "type": "planner.plan",
                "plan": [
                    {"content": "Fetch weather", "status": "pending"},
                    {"content": "Summarize", "status": "pending"},
                ],
            },
        ]
        trace = parse_adk_trace(events, agent_name="a")
        plan_spans = [
            s for s in trace.spans if s.kind == SpanKind.TASK and s.name == "adk.planner.plan"
        ]
        assert len(plan_spans) == 1
        assert plan_spans[0].attributes["goals"] == ["Fetch weather", "Summarize"]
        assert plan_spans[0].attributes["adk.plan.size"] == 2

    def test_code_executor_is_tool(self):
        events = [
            {"type": "agent.invocation.start", "agent": "a"},
            {"type": "code_executor.run", "code": "print(1+1)", "output": "2"},
        ]
        trace = parse_adk_trace(events, agent_name="a")
        code_spans = [
            s
            for s in trace.spans
            if s.kind == SpanKind.TOOL and s.attributes.get("adk.tool.type") == "code_executor"
        ]
        assert len(code_spans) == 1

    def test_retrieval_search_emits_retrieval_span(self):
        events = [
            {"type": "agent.invocation.start", "agent": "a"},
            {"type": "retrieval.search", "query": "what is pisama?", "results": [{"doc": "..."}]},
        ]
        trace = parse_adk_trace(events, agent_name="a")
        retrieval = [s for s in trace.spans if s.kind == SpanKind.RETRIEVAL]
        assert len(retrieval) == 1
        assert retrieval[0].input_data == {"query": "what is pisama?"}

    def test_tool_error_flags_response_span(self):
        events = [
            {"type": "agent.invocation.start", "agent": "a"},
            {"type": "tool.function_call", "name": "bad", "function_call_id": "fc"},
            {
                "type": "tool.function_response",
                "function_call_id": "fc",
                "name": "bad",
                "error": "boom",
            },
        ]
        trace = parse_adk_trace(events, agent_name="a")
        response = [s for s in trace.spans if s.name == "adk.tool_output:bad"][0]
        assert response.status == SpanStatus.ERROR
        assert response.error_message == "boom"

    def test_invocation_end_with_error_marks_root(self):
        events = [
            {"type": "agent.invocation.start", "agent": "a"},
            {"type": "agent.invocation.end", "error": "model unavailable"},
        ]
        trace = parse_adk_trace(events, agent_name="a")
        assert trace.spans[0].status == SpanStatus.ERROR
        assert trace.spans[0].error_message == "model unavailable"


class TestGoogleAdkAdapter:
    def test_platform_identity(self):
        adapter = GoogleAdkAdapter(invocation_id="inv-1", agent_name="a")
        assert adapter.platform_name == Platform.GOOGLE_ADK
        assert adapter.platform_version == "adk-v1"

    def test_supports_callback_injection(self):
        adapter = GoogleAdkAdapter()
        methods = adapter.get_supported_injection_methods()
        assert InjectionMethod.CALLBACK in methods
        assert InjectionMethod.STATE in methods
        assert InjectionMethod.MESSAGE in methods
        assert adapter.can_block() is True

    def test_inject_fix_queues_directive_for_callback(self):
        adapter = GoogleAdkAdapter()
        result = adapter.inject_fix(
            directive="avoid unsafe tools",
            level=EnforcementLevel.DIRECT,
            directive_id="dir-1",
        )
        assert result.success is True
        assert result.method == InjectionMethod.CALLBACK
        assert result.directive_id == "dir-1"
        assert result.blocked is False

        pending = adapter.drain_pending_directives()
        assert len(pending) == 1
        assert pending[0].message == "avoid unsafe tools"

    def test_block_level_sets_blocked_flag(self):
        adapter = GoogleAdkAdapter()
        result = adapter.inject_fix(
            directive="halt",
            level=EnforcementLevel.BLOCK,
            directive_id="dir-block",
        )
        assert result.blocked is True

    def test_block_action_appends_blocked_directive(self):
        adapter = GoogleAdkAdapter()
        assert adapter.block_action("safety violation") is True
        pending = adapter.drain_pending_directives()
        assert len(pending) == 1
        assert pending[0].blocked is True
        assert pending[0].message == "safety violation"

    def test_capture_span_returns_first_child_for_event(self):
        adapter = GoogleAdkAdapter(invocation_id="inv", agent_name="a")
        span = adapter.capture_span(
            {"type": "tool.function_call", "name": "s", "function_call_id": "f"}
        )
        assert span.kind == SpanKind.TOOL

    def test_parse_trace_uses_adapter_state(self):
        adapter = GoogleAdkAdapter(
            invocation_id="inv-42", agent_name="planner", session_id="sess-99"
        )
        trace = adapter.parse_trace([{"type": "agent.invocation.start", "agent": "planner"}])
        assert trace.metadata.session_id == "sess-99"
        assert trace.spans[0].attributes["google.adk.invocation.id"] == "inv-42"
