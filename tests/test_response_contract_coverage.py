"""Authored coverage-accounting controls, not semantic accuracy validation."""

import asyncio

from pisama_core.detection.detectors.communication import CommunicationDetector
from pisama_core.traces.models import Trace


def run(spans):
    return asyncio.run(CommunicationDetector().detect(Trace.from_dict({"spans": spans})))


def pair(prompt, response, **extra):
    return {
        "kind": "llm",
        "input_data": {"prompt": prompt},
        "output_data": {"content": response},
        **extra,
    }


def test_mixed_multiple_contracts_and_conditional_unsupported_accounted():
    result = run(
        [
            pair("Reply with OK.", "NO"),
            pair("Return JSON only.", "{}"),
            pair("If ready, reply with OK.", "not ready"),
            {"kind": "tool", "name": "synthetic"},
        ]
    )
    coverage = result.metadata["response_contract_coverage"]
    assert result.detected
    assert coverage["trace_span_count"] == 4
    assert coverage["considered_count"] == 3
    assert coverage["checked_count"] == 2
    assert coverage["unsupported_count"] == 1
    assert coverage["outside_scope_count"] == 1
    assert [row["status"] for row in coverage["records"]] == [
        "violated",
        "satisfied",
        "unsupported",
        "outside_scope",
    ]
    assert coverage["business_semantics_assessed"] is False


def test_duplicate_parent_ids_never_resolve_arbitrarily():
    result = run(
        [
            {
                "kind": "message",
                "span_id": "duplicate",
                "output_data": {"content": "Reply with OK."},
            },
            {
                "kind": "message",
                "span_id": "duplicate",
                "output_data": {"content": "Reply with NO."},
            },
            {"kind": "message", "parent_id": "duplicate", "output_data": {"content": "NO"}},
        ]
    )
    rows = result.metadata["response_contract_coverage"]["records"]
    assert not result.detected
    assert rows[0]["identity_ambiguous"] and rows[1]["identity_ambiguous"]
    assert rows[2]["reason"] == "ambiguous_parent_identity"
    assert [row["span_index"] for row in rows] == [0, 1, 2]


def test_missing_ids_keep_distinct_positional_accounting():
    result = run(
        [pair("Reply with OK.", "OK", span_id=None), pair("Explain why.", "because", span_id="")]
    )
    coverage = result.metadata["response_contract_coverage"]
    assert coverage["checked_count"] == 1
    assert coverage["unsupported_count"] == 1
    assert all(row["identity_ambiguous"] for row in coverage["records"])
    assert [row["span_index"] for row in coverage["records"]] == [0, 1]
