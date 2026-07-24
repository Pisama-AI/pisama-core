from __future__ import annotations

import pytest

from pisama_core import DiagnosisRecord
from pisama_core.detection.diagnosis import InterventionRecord
from pisama_core.detection.result import DetectionResult


def _payload() -> dict:
    return {
        "failure_type": "silent_failure",
        "applicability": "applicable",
        "coverage": {
            "status": "complete",
            "completeness": 1.0,
            "reason": None,
            "missing_required": [],
            "requirements": [
                {
                    "name": "mutation_evidence",
                    "present": True,
                    "required": True,
                    "source": "tool_result",
                    "detail": None,
                }
            ],
            "future_coverage_field": "preserved",
        },
        "origin_module": "action",
        "stage": "evidence_use",
        "detected_at_step": 11,
        "intervention_target_step": 10,
        "boundary_status": "stable",
        "agent_id": "mcp",
        "evidence": [],
        "causal_candidates": [
            {
                "label": "required_state_change_missing",
                "support": 0.94,
                "evidence_ids": ["add_transaction-10-0"],
                "rationale": "The expected mutation was absent.",
            }
        ],
        "intervention": {
            "kind": "CounterfactualReplay",
            "status": "verified",
            "target_step": 10,
            "prefix_preserved_through": 9,
            "outcome_before": "sha256:before",
            "outcome_after": "sha256:after",
            "verifier": "personal-finance-add-transaction",
            "run_id": "mt-fin2-personal-finance__j4775oHo",
            "evidence": [],
        },
        "causally_verified": True,
        "metadata": {"contract_id": "personal-finance-add-transaction"},
        "future_top_level_field": {"version": 2},
    }


def test_sdk_diagnosis_round_trip_preserves_additive_fields() -> None:
    payload = _payload()

    diagnosis = DiagnosisRecord.from_dict(payload)

    assert diagnosis.causally_verified is True
    assert diagnosis.causal_candidates[0].support == 0.94
    assert diagnosis.coverage.extra["future_coverage_field"] == "preserved"
    assert diagnosis.extra["future_top_level_field"] == {"version": 2}
    assert diagnosis.to_dict() == payload


def test_detection_result_exposes_typed_diagnosis() -> None:
    result = DetectionResult(
        detector_name="silent_failure",
        detected=True,
        diagnosis_record=_payload(),
    )

    assert result.diagnosis is not None
    assert result.diagnosis.stage == "evidence_use"
    result.set_diagnosis(result.diagnosis)
    assert result.to_dict()["diagnosis_record"] == _payload()


def test_sdk_verified_intervention_fails_closed_without_outcome_change() -> None:
    with pytest.raises(ValueError, match="outcome change"):
        InterventionRecord(
            kind="CounterfactualReplay",
            status="verified",
            target_step=10,
            outcome_before="same",
            outcome_after="same",
            verifier="task-contract",
        )
