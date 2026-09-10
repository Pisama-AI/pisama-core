"""Explicit communication-contract violations, not semantic intent inference.

Missing instruction verbs are not evidence of failure. This heuristic abstains
unless a narrow literal or JSON contract can be checked. Confidence is a
heuristic weight, not a calibrated probability.
"""

import json
import re
from typing import Any, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import Platform, SpanKind
from pisama_core.traces.models import Trace


def _reject_non_json_constant(value: str) -> None:
    raise ValueError("Non-JSON numeric constant")


class CommunicationDetector(BaseDetector):
    """Check explicit response contracts on attributable request/response pairs."""

    name = "communication"
    description = "Detects explicit communication contract violations"
    version = "2.0.0"
    platforms: list[Platform] = []
    severity_range = (0, 100)
    realtime_capable = False

    @staticmethod
    def _json_contract(request: str) -> bool:
        return (
            re.fullmatch(
                r"(?:return|reply|respond|output)(?:\s+(?:with|in|only|valid))*\s+json(?:\s+only)?\s*\.?",
                request.strip(),
                re.IGNORECASE,
            )
            is not None
        )

    @staticmethod
    def _literal_contract(request: str) -> Optional[str]:
        # Full-match avoids interpreting examples or conditional prose as an
        # unconditional contract. Unquoted literals must be uppercase tokens.
        prefix = r"(?:reply|respond|return|output|say)\s+(?:(?:with|exactly|only)\s+)*"
        # Exactly one quoted operand; nested/escaped delimiters are outside
        # this conservative grammar, not additional text inside the literal.
        quoted = re.fullmatch(
            prefix + r"""(?:"([^"']+)"|'([^"']+)')\s*\.?""", request.strip(), re.IGNORECASE
        )
        if quoted:
            return quoted.group(1) if quoted.group(1) is not None else quoted.group(2)
        command = re.fullmatch(prefix + r"([A-Z][A-Z0-9_-]*)\s*\.?", request.strip(), re.IGNORECASE)
        if command and command.group(1).isupper() and command.group(1) != "JSON":
            return command.group(1)
        return None

    def _detect_single(
        self,
        sender_message: str,
        receiver_response: str,
        receiver_action: Optional[str] = None,
        sender_name: Optional[str] = None,
        receiver_name: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Check only explicit contracts; abstention does not mean success."""
        if not isinstance(sender_message, str) or not isinstance(receiver_response, str):
            return None
        expected = self._literal_contract(sender_message)
        kind = "literal_mismatch"
        if expected is not None:
            # Literal contracts compare the exact captured string, including
            # significant leading/trailing whitespace inside the operand.
            if receiver_response == expected:
                return None
            explanation = "Response does not match the explicitly requested literal."
        elif self._json_contract(sender_message):
            try:
                json.loads(receiver_response, parse_constant=_reject_non_json_constant)
                return None
            except (json.JSONDecodeError, ValueError):
                kind = "format_mismatch"
                explanation = (
                    "Explicit JSON response contract violated: response is not valid JSON."
                )
        else:
            return None
        return {
            "severity": 55,
            "confidence": 0.9,
            "summary": explanation,
            "evidence": {
                "breakdown_type": kind,
                "contract": "literal" if expected is not None else "json",
                "confidence_basis": "uncalibrated deterministic contract heuristic",
            },
        }

    async def detect(self, trace: Trace) -> DetectionResult:
        # Own input/output or a message parent establishes a relationship;
        # chronological adjacency alone does not establish communication.
        id_counts: dict[str, int] = {}
        for span in trace.spans:
            if isinstance(span.span_id, str) and span.span_id:
                id_counts[span.span_id] = id_counts.get(span.span_id, 0) + 1
        by_id = {
            span.span_id: span
            for span in trace.spans
            if isinstance(span.span_id, str) and id_counts.get(span.span_id) == 1
        }
        records: list[dict[str, Any]] = []
        detected_result: Optional[DetectionResult] = None
        checked = 0
        for index, span in enumerate(trace.spans):
            record: dict[str, Any] = {
                "span_index": index,
                "span_id": span.span_id if isinstance(span.span_id, str) else None,
                "identity_ambiguous": not span.span_id or id_counts.get(span.span_id, 0) != 1,
                "status": "unsupported",
                "contract_kind": None,
                "reason": "missing_attributable_pair",
            }
            records.append(record)
            if span.kind not in {
                SpanKind.LLM,
                SpanKind.AGENT,
                SpanKind.AGENT_TURN,
                SpanKind.CHAIN,
                SpanKind.USER_OUTPUT,
                SpanKind.MESSAGE,
                SpanKind.HANDOFF,
            }:
                record.update(status="outside_scope", reason="span_kind_outside_response_scope")
                continue
            incoming = span.input_data or {}
            outgoing = span.output_data or {}
            request = incoming.get("prompt", incoming.get("content"))
            response = outgoing.get("content", outgoing.get("response"))
            relation = "captured_input_output"
            if not isinstance(request, str):
                parent = by_id.get(span.parent_id) if span.parent_id is not None else None
                if parent is None or parent.kind not in {SpanKind.MESSAGE, SpanKind.HANDOFF}:
                    if span.parent_id and id_counts.get(span.parent_id, 0) > 1:
                        record["reason"] = "ambiguous_parent_identity"
                    continue
                request = (parent.output_data or {}).get("content")
                relation = "explicit_message_parent"
            if not isinstance(request, str) or not isinstance(response, str):
                continue
            if self._literal_contract(request) is None and not self._json_contract(request):
                record["reason"] = "unsupported_or_ambiguous_contract"
                continue
            checked += 1
            finding = self._detect_single(request, response)
            record.update(
                status="violated" if finding else "satisfied",
                contract_kind="literal" if self._literal_contract(request) is not None else "json",
                reason="explicit_contract_checked",
                relationship=relation,
            )
            if finding is None:
                continue
            result = DetectionResult.issue_found(
                detector_name=self.name,
                severity=finding["severity"],
                summary=finding["summary"],
                fix_type=FixType.SWITCH_STRATEGY,
                fix_instruction="Honor the explicit response contract or clarify it before execution.",
            )
            result.confidence = finding["confidence"]
            result.detector_version = self.version
            result.metadata.update(
                assessment="contract_violated",
                checked_contracts=checked,
                confidence_basis="uncalibrated contract heuristic",
            )
            result.add_evidence(
                description=finding["summary"],
                data={
                    **finding["evidence"],
                    "relationship": relation,
                    "span_id": span.span_id,
                    "span_index": index,
                    "identity_ambiguous": record["identity_ambiguous"],
                },
            )
            if detected_result is None:
                detected_result = result
        result = detected_result or DetectionResult.no_issue(self.name)
        result.detector_version = self.version
        result.confidence = 0.9 if checked else 0.0
        if not result.detected:
            result.summary = (
                "Checked explicit contracts satisfied; other intent is unassessed"
                if checked
                else "Abstained: no supported attributable response contract"
            )
        result.metadata.update(
            assessment="contract_violated"
            if result.detected
            else "contract_satisfied"
            if checked
            else "abstained",
            checked_contracts=checked,
            confidence_basis="uncalibrated contract heuristic",
        )
        result.metadata["response_contract_coverage"] = {
            "version": 1,
            "scope": "eligible_captured_response_pairs",
            "trace_span_count": len(records),
            "considered_count": sum(row["status"] != "outside_scope" for row in records),
            "checked_count": checked,
            "unsupported_count": sum(row["status"] == "unsupported" for row in records),
            "outside_scope_count": sum(row["status"] == "outside_scope" for row in records),
            "records": records,
            "business_semantics_assessed": False,
        }
        return result
