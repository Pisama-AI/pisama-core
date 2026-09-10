"""Authored synthetic controls, not calibration or representative validation."""

import asyncio
import json
from pathlib import Path

import pytest

from pisama_core.detection.detectors.communication import CommunicationDetector
from pisama_core.traces.models import Trace

CASES = json.loads((Path(__file__).parent / "fixtures/communication_contracts.json").read_text())


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
def test_message_relationship_not_adjacency(linked):
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
                    "kind": "llm",
                    "output_data": {"content": "NO"},
                },
            ]
        }
    )
    assert asyncio.run(CommunicationDetector().detect(trace)).detected == linked
