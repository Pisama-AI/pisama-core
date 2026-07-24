"""Tests for EscalationLoopDetector."""

import asyncio

from pisama_core import SpanKind, Trace
from pisama_core.detection.detectors.escalation import EscalationLoopDetector
from pisama_core.traces.enums import Platform
from pisama_core.traces.models import TraceMetadata


def _make_handoff(trace: Trace, src: str, tgt: str, output: str = "") -> None:
    span = trace.create_span(name=f"{src}->{tgt}", kind=SpanKind.HANDOFF)
    span.attributes["source_agent"] = src
    span.attributes["target_agent"] = tgt
    if output:
        span.output_data = {"output": output}


def _run(trace: Trace):
    d = EscalationLoopDetector()
    return asyncio.run(d.detect(trace))


def _base_trace() -> Trace:
    return Trace(metadata=TraceMetadata(session_id="s1", platform=Platform.LANGGRAPH))


class TestRoundTripLoop:
    def test_fires_on_repeated_roundtrip(self):
        t = _base_trace()
        for _ in range(4):
            _make_handoff(t, "A", "B")
            _make_handoff(t, "B", "A")
        r = _run(t)
        assert r.detected
        assert "escalation_loop" in r.detector_name
        assert r.severity >= 40

    def test_no_issue_below_threshold(self):
        """Only 2 round trips — at max_round_trips, not above."""
        t = _base_trace()
        for _ in range(2):
            _make_handoff(t, "A", "B")
            _make_handoff(t, "B", "A")
        r = _run(t)
        assert not r.detected

    def test_no_double_count_single_loop(self):
        """(A,B) and (B,A) must count as one loop, not two."""
        t = _base_trace()
        for _ in range(4):
            _make_handoff(t, "X", "Y")
            _make_handoff(t, "Y", "X")
        r = _run(t)
        assert r.detected
        # severity should be 40 (one loop), not 80 (double-counted)
        assert r.severity == 40

    def test_stale_output_bonus(self):
        stale = "the request cannot be fulfilled as specified"
        t = _base_trace()
        _make_handoff(t, "A", "B", output=stale)
        for _ in range(3):
            _make_handoff(t, "A", "B")
            _make_handoff(t, "B", "A")
        _make_handoff(t, "B", "A", output=stale)
        r = _run(t)
        assert r.detected
        assert r.severity >= 60  # 40 + 20 stale bonus


class TestApprovalShopping:
    def test_fires_on_three_plus_targets(self):
        t = _base_trace()
        for tgt in ("mgr1", "mgr2", "mgr3"):
            _make_handoff(t, "worker", tgt)
        r = _run(t)
        assert r.detected
        assert r.severity >= 30

    def test_no_issue_two_targets(self):
        t = _base_trace()
        for tgt in ("mgr1", "mgr2"):
            _make_handoff(t, "worker", tgt)
        r = _run(t)
        assert not r.detected


class TestEdgeCases:
    def test_empty_trace_no_crash(self):
        r = _run(_base_trace())
        assert not r.detected

    def test_fewer_than_three_handoffs_no_issue(self):
        t = _base_trace()
        _make_handoff(t, "A", "B")
        _make_handoff(t, "B", "A")
        r = _run(t)
        assert not r.detected

    def test_one_directional_chain_no_issue(self):
        t = _base_trace()
        for i in range(5):
            _make_handoff(t, f"agent{i}", f"agent{i + 1}")
        r = _run(t)
        assert not r.detected
