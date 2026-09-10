"""Authored synthetic controls, not calibration or representative validation."""

import asyncio
import json
from pathlib import Path

import pytest

from pisama_core.detection.detectors.communication import CommunicationDetector
from pisama_core.traces.models import Trace

CASES = json.loads((Path(__file__).parent / "fixtures/communication_contracts.json").read_text())


@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize("suffix", ["or", "if ready, otherwise"])
def test_quoted_alternatives_and_conditions_abstain(quote, suffix):
    request = f"Return {quote}OK{quote} {suffix} {quote}NO{quote}."
    detector = CommunicationDetector()
    assert detector._literal_contract(request) is None
    assert detector._detect_single(request, "NO") is None


@pytest.mark.parametrize("quote", ["'", '"'])
def test_single_quoted_literal_still_checked(quote):
    request = f"Return {quote}DONE{quote}."
    detector = CommunicationDetector()
    assert detector._detect_single(request, "DONE") is None
    assert detector._detect_single(request, "NO") is not None


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_constants_rejected(value):
    assert CommunicationDetector()._detect_single("Return JSON only.", value) is not None


@pytest.mark.parametrize("case", CASES)
def test_explicit_contracts(case):
    trace = Trace.from_dict(
        {
            "spans": [
                {
                    "kind": "llm",
                    "input_data": {"prompt": case["request"]},
                    "output_data": {"content": case["response"]},
                }
            ]
        }
    )
    result = asyncio.run(CommunicationDetector().detect(trace))
    assert result.detected == case["detected"]
    if result.detected:
        assert result.confidence < 1


@pytest.mark.parametrize("linked", [True, False])
@pytest.mark.parametrize("receiver_kind", ["llm", "message", "handoff"])
def test_message_relationship_not_adjacency(linked, receiver_kind):
    trace = Trace.from_dict(
        {
            "spans": [
                {
                    "span_id": "sender",
                    "kind": "message",
                    "output_data": {"content": "Reply with OK."},
                },
                {
                    "span_id": "receiver",
                    "parent_id": "sender" if linked else None,
                    "kind": receiver_kind,
                    "output_data": {"content": "NO"},
                },
            ]
        }
    )
    assert asyncio.run(CommunicationDetector().detect(trace)).detected == linked


@pytest.mark.parametrize(
    "prompt,status", [("Explain JSON.", "abstained"), ("Reply with OK.", "contract_satisfied")]
)
def test_machine_readable_assessment(prompt, status):
    trace = Trace.from_dict(
        {
            "spans": [
                {
                    "kind": "llm",
                    "input_data": {"content": prompt},
                    "output_data": {"content": "OK"},
                }
            ]
        }
    )
    result = asyncio.run(CommunicationDetector().detect(trace))
    assert result.to_dict()["metadata"]["assessment"] == status
    assert result.confidence < 1
    assert result.metadata["checked_contracts"] == (0 if status == "abstained" else 1)
