"""Exercise every registered detector at least once.

Catches the bug class that the cost.py / quickstart issue belonged to:
detectors that import successfully but crash when actually called. The
synth-agents harness covers ingest/detect/query end-to-end, but this
test exercises the SDK code path directly so a regression is reported
inside `pytest packages/pisama-core/tests/` rather than only in the
backend integration suite.

Each detector is instantiated and its `run()` is awaited against a small
synthetic trace. We assert:
  - the call does not raise
  - it returns a `DetectionResult`
  - the returned `detector_name` matches `detector.name`

We do NOT assert anything about `detected` or `severity` — that's
detector-specific calibration, not a smoke property. Many detectors
correctly return `no_issue` on a generic trace; that's still pass.
"""

from __future__ import annotations

import asyncio

import pytest

from pisama_core import SpanKind, Trace
from pisama_core.detection.registry import registry
from pisama_core.detection.result import DetectionResult


def _build_synthetic_trace() -> Trace:
    """A trace with a mix of LLM, tool, and message spans.

    Generic enough that every detector should accept it as input even if
    it returns `no_issue`. We deliberately include a tight loop so the
    loop detector has something real to find.
    """
    trace = Trace()
    for _ in range(8):
        trace.create_span(name="Read", kind=SpanKind.TOOL)
    for _ in range(2):
        trace.create_span(name="anthropic.complete", kind=SpanKind.LLM)
    trace.create_span(name="agent_b.send", kind=SpanKind.AGENT)
    return trace


@pytest.fixture(scope="module")
def synthetic_trace() -> Trace:
    return _build_synthetic_trace()


@pytest.mark.parametrize(
    "detector",
    list(registry.get_enabled()),
    ids=lambda d: d.name,
)
def test_detector_runs_without_crashing(detector, synthetic_trace):
    result = asyncio.run(detector.run(synthetic_trace))
    assert isinstance(result, DetectionResult), (
        f"{detector.name} returned {type(result).__name__}, expected DetectionResult"
    )
    assert result.detector_name == detector.name, (
        f"{detector.name} returned a result naming itself {result.detector_name!r}"
    )


def test_at_least_one_detector_fires_on_loop():
    """If the loop detector is shipping, an 8x consecutive Read trace must trip it.

    This is the canary we use everywhere else (README quickstart, publish
    workflow smoke) and it belongs in the test suite too. If this stops
    firing, someone changed loop semantics or rolled the loop detector
    out of the default registry.
    """
    trace = Trace()
    for _ in range(8):
        trace.create_span(name="Read", kind=SpanKind.TOOL)
    fired = [asyncio.run(d.run(trace)) for d in registry.get_enabled() if d.name == "loop"]
    assert fired, "loop detector is not registered"
    assert any(r.detected for r in fired), "loop detector did not fire on 8x consecutive Read"
