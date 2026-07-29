"""Tests for PisamaTracingProcessor.

The processor is duck-typed against the Agents SDK rather than subclassing
`agents.tracing.TracingProcessor`, which keeps pisama-core free of a vendor
dependency. That trade-off is only safe if the method set stays correct, so the
first test pins it: if the SDK widens the interface, this fails loudly instead of
the processor silently never being called.

Most tests drive the processor with the SDK's export shapes directly, so the
suite runs without `openai-agents` installed. One test registers it with the real
SDK and is skipped when the package is absent.
"""

import threading

import pytest

from pisama_core.adapters import PisamaTracingProcessor
from pisama_core.traces.enums import SpanKind


class _FakeExportable:
    """Stands in for an SDK span or trace, which expose .export()."""

    def __init__(self, payload, raises=False):
        self._payload = payload
        self._raises = raises

    def export(self):
        if self._raises:
            raise RuntimeError("export failed")
        return self._payload


def _span(span_id, trace_id, span_data=None):
    return _FakeExportable(
        {
            "object": "trace.span",
            "id": span_id,
            "trace_id": trace_id,
            "parent_id": None,
            "started_at": "2026-07-28T10:00:00Z",
            "ended_at": "2026-07-28T10:00:01Z",
            "span_data": span_data or {"type": "agent", "name": "A", "handoffs": [], "tools": []},
            "error": None,
        }
    )


def _trace(trace_id, workflow="wf"):
    return _FakeExportable(
        {"object": "trace", "id": trace_id, "workflow_name": workflow, "group_id": None}
    )


class TestInterfaceContract:
    def test_implements_every_method_the_sdk_dispatches(self):
        """Duck typing only holds while the method set matches the SDK's ABC.

        Hard-coded rather than imported so the check runs without the SDK; the
        live test below catches drift against the installed version.
        """
        required = {
            "on_trace_start",
            "on_trace_end",
            "on_span_start",
            "on_span_end",
            "force_flush",
            "shutdown",
        }
        assert required <= {m for m in dir(PisamaTracingProcessor) if not m.startswith("_")}


class TestDelivery:
    def test_completed_run_is_delivered_as_a_trace(self):
        got = []
        p = PisamaTracingProcessor(on_trace=got.append)
        p.on_span_end(_span("span_1", "trace_1"))
        p.on_trace_end(_trace("trace_1", "support"))
        assert len(got) == 1
        assert got[0].trace_id == "trace_1"
        assert got[0].metadata.custom["workflow_name"] == "support"
        assert [s.kind for s in got[0].spans] == [SpanKind.AGENT]

    def test_no_callback_is_valid_and_does_not_raise(self):
        p = PisamaTracingProcessor()
        p.on_span_end(_span("span_1", "trace_1"))
        p.on_trace_end(_trace("trace_1"))

    def test_trace_with_no_spans_still_delivers(self):
        got = []
        PisamaTracingProcessor(on_trace=got.append).on_trace_end(_trace("trace_empty"))
        assert len(got) == 1 and got[0].spans == []


class TestConcurrentRuns:
    def test_spans_are_attributed_to_their_own_trace(self):
        """Interleaved runs must not pool into one buffer.

        A single span list would hand trace_a's spans to whichever trace ended
        first, which is the bug this keying exists to prevent.
        """
        got = {}
        p = PisamaTracingProcessor(on_trace=lambda t: got.__setitem__(t.trace_id, t))
        p.on_span_end(_span("a1", "trace_a"))
        p.on_span_end(_span("b1", "trace_b"))
        p.on_span_end(_span("a2", "trace_a"))
        p.on_trace_end(_trace("trace_b"))
        p.on_trace_end(_trace("trace_a"))
        assert {s.span_id for s in got["trace_a"].spans} == {"a1", "a2"}
        assert {s.span_id for s in got["trace_b"].spans} == {"b1"}

    def test_buffer_is_released_after_the_trace_ends(self):
        p = PisamaTracingProcessor()
        p.on_span_end(_span("s", "trace_1"))
        p.on_trace_end(_trace("trace_1"))
        assert p._spans == {}

    def test_concurrent_span_ends_do_not_lose_spans(self):
        p = PisamaTracingProcessor()
        def emit(n):
            for i in range(50):
                p.on_span_end(_span(f"{n}_{i}", "trace_1"))
        threads = [threading.Thread(target=emit, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(p._spans["trace_1"]) == 200

    def test_shutdown_drops_orphaned_spans(self):
        """A trace that never ends must not leak its buffer for the process life."""
        p = PisamaTracingProcessor()
        p.on_span_end(_span("s", "trace_never_ends"))
        p.shutdown()
        assert p._spans == {}


class TestRobustness:
    def test_a_failing_export_does_not_break_the_caller(self):
        """Telemetry must never take down the run it is observing."""
        got = []
        p = PisamaTracingProcessor(on_trace=got.append)
        p.on_span_end(_FakeExportable(None, raises=True))
        p.on_trace_end(_FakeExportable(None, raises=True))
        assert got == []

    def test_export_returning_none_is_ignored(self):
        got = []
        p = PisamaTracingProcessor(on_trace=got.append)
        p.on_span_end(_FakeExportable(None))
        p.on_trace_end(_FakeExportable(None))
        assert got == []

    def test_object_without_export_is_ignored(self):
        p = PisamaTracingProcessor(on_trace=lambda t: pytest.fail("should not fire"))
        p.on_span_end(object())
        p.on_trace_end(object())


class TestAgainstTheRealSDK:
    def test_registers_and_receives_a_real_run(self):
        """The duck-typing bet, verified against the installed SDK."""
        agents_tracing = pytest.importorskip("agents.tracing")
        got = []
        processor = PisamaTracingProcessor(on_trace=got.append)
        agents_tracing.set_trace_processors([processor])
        with agents_tracing.trace(workflow_name="live-check"):
            with agents_tracing.agent_span(
                name="Triage", handoffs=["Refund"], tools=[], output_type="str"
            ):
                with agents_tracing.handoff_span(from_agent="Triage", to_agent="Refund"):
                    pass

        assert len(got) == 1, "the SDK did not dispatch to a duck-typed processor"
        parsed = got[0]
        assert parsed.metadata.custom["workflow_name"] == "live-check"
        kinds = {s.kind for s in parsed.spans}
        assert SpanKind.HANDOFF in kinds and SpanKind.AGENT in kinds
        handoff = next(s for s in parsed.spans if s.kind == SpanKind.HANDOFF)
        assert handoff.attributes["openai_agents.handoff.to"] == "Refund"
