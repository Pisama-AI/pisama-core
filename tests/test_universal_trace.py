"""Regression tests for portable universal trace ingestion."""

from pisama_core.ingestion.universal_trace import SpanType, UniversalSpan, UniversalTrace


def test_state_snapshots_do_not_depend_on_backend_modules():
    trace = UniversalTrace(
        trace_id="trace-1",
        spans=[
            UniversalSpan(
                id="span-1",
                trace_id="trace-1",
                name="agent turn",
                span_type=SpanType.AGENT,
                agent_id="researcher",
                output_data={"answer": "42"},
            )
        ],
    )

    snapshots = trace.to_state_snapshots()

    assert snapshots[0].agent_id == "researcher"
    assert snapshots[0].state_delta == {"answer": "42"}
    assert snapshots[0].sequence_num == 0
