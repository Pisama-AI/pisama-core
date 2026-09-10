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
    def _literal_contract(request: str) -> Optional[str]:
        # Full-match avoids interpreting examples or conditional prose as an
        # unconditional contract. Unquoted literals must be uppercase tokens.
        prefix = r"(?:reply|respond|return|output|say)\s+(?:(?:with|exactly|only)\s+)*"
        quoted = re.fullmatch(prefix + r"""(["'])(.+?)\1\s*\.?""", request.strip(), re.IGNORECASE)
        if quoted:
            return quoted.group(2)
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
            if receiver_response.strip() == expected:
                return None
            explanation = "Response does not match the explicitly requested literal."
        elif re.fullmatch(
            r"(?:return|reply|respond|output)(?:\s+(?:with|in|only|valid))*\s+json(?:\s+only)?\s*\.?",
            sender_message.strip(),
            re.IGNORECASE,
        ):
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
        by_id = {span.span_id: span for span in trace.spans}
        for span in trace.spans:
            if span.kind not in {
                SpanKind.LLM,
                SpanKind.AGENT,
                SpanKind.AGENT_TURN,
                SpanKind.CHAIN,
                SpanKind.USER_OUTPUT,
            }:
                continue
            incoming = span.input_data or {}
            outgoing = span.output_data or {}
            request = incoming.get("prompt", incoming.get("content"))
            response = outgoing.get("content", outgoing.get("response"))
            relation = "captured_input_output"
            if not isinstance(request, str):
                parent = by_id.get(span.parent_id)
                if parent is None or parent.kind not in {SpanKind.MESSAGE, SpanKind.HANDOFF}:
                    continue
                request = (parent.output_data or {}).get("content")
                relation = "explicit_message_parent"
            finding = self._detect_single(request, response)
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
            result.add_evidence(
                description=finding["summary"],
                data={
                    **finding["evidence"],
                    "relationship": relation,
                    "span_id": span.span_id,
                },
            )
            return result
        return DetectionResult.no_issue(self.name)
