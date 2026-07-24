"""Communication breakdown detector for inter-agent message failures.

F10: Communication Breakdown Detection (MAST Taxonomy)

Detects when a message between agents is misunderstood or
misinterpreted, leading to incorrect behavior downstream.

This includes:
- Intent misalignment (sender meant X, receiver understood Y)
- Format mismatches (expected JSON, got prose)
- Semantic misinterpretation (ambiguous language)

Ported from backend/app/detection/communication.py.
"""

import json
import logging
import re
from typing import Any, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import Platform
from pisama_core.traces.models import Trace

logger = logging.getLogger(__name__)

# Raw execution trace/log patterns -- these are NOT structured messages
_TRACE_LOG_PATTERNS: list[str] = [
    r"\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}",
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
    r"\b(?:INFO|DEBUG|WARNING|ERROR)\b\]?\s+",
    r"RUN\.SH STARTING",
    r"AUTOGEN_TESTBED_SETTING",
    r"\*\*\[Preprocessing\]\*\*",
    r"=== (?:Test write|MetaGPT|Communication Log)",
]

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "and",
        "or",
        "but",
        "not",
        "no",
        "if",
        "it",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "this",
        "that",
        "will",
        "can",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "must",
        "need",
    }
)


class CommunicationDetector(BaseDetector):
    """Detects communication breakdown between agents.

    Analyzes message intent, format compliance, and semantic clarity
    to detect communication failures in inter-agent messaging.

    Span convention:
        The detector examines consecutive span pairs, treating the first
        span's ``output_data.content`` as the sender message and the second
        span's ``output_data.content`` as the receiver response. Span names
        are used for agent identification.
    """

    name = "communication"
    description = "Detects inter-agent communication breakdown"
    version = "1.2.0"
    platforms: list[Platform] = []  # All platforms
    severity_range = (0, 100)
    realtime_capable = False

    # Default thresholds
    intent_threshold: float = 0.45
    check_format: bool = True
    check_ambiguity: bool = True

    # --- Internal helpers (ported faithfully from backend) ---

    @staticmethod
    def _detect_expected_format(message: str) -> Optional[str]:
        """Detect expected response format from the sender's message."""
        format_hints: dict[str, list[str]] = {
            "json": [r"\bjson\b", r"\{.*\}", r"format.*json", r"return.*json"],
            "list": [r"\blist\b", r"enumerate", r"bullet.*point", r"\d+\.\s"],
            "code": [r"```", r"\bcode\b", r"implement", r"function.*def", r"class\s+\w+"],
            "markdown": [r"#\s+", r"\*\*.*\*\*", r"##\s+"],
            "csv": [r"\bcsv\b", r"comma.*separated", r",.*,.*,"],
        }

        message_lower = message.lower()
        for fmt, patterns in format_hints.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return fmt
        return None

    @staticmethod
    def _check_format_compliance(
        expected_format: Optional[str],
        response: str,
    ) -> tuple[bool, str]:
        """Check if response complies with expected format."""
        if not expected_format:
            return True, "No specific format expected"

        if expected_format == "json":
            try:
                json.loads(response)
                return True, "Valid JSON"
            except json.JSONDecodeError:
                json_match = re.search(r"\{[^{}]*\}|\[[^\[\]]*\]", response)
                if json_match:
                    try:
                        json.loads(json_match.group())
                        return True, "JSON found in response"
                    except (json.JSONDecodeError, ValueError):
                        pass
                return False, "Expected JSON but response is not valid JSON"

        if expected_format == "list":
            list_patterns = [r"^\s*[-\u2022*]\s+", r"^\s*\d+[.)]\s+"]
            for pattern in list_patterns:
                if re.search(pattern, response, re.MULTILINE):
                    return True, "List format detected"
            return False, "Expected list format but none detected"

        if expected_format == "code":
            if "```" in response or re.search(r"\bdef\s+\w+|class\s+\w+|function\s+\w+", response):
                return True, "Code format detected"
            return False, "Expected code but none detected"

        return True, f"Format check passed for {expected_format}"

    @staticmethod
    def _detect_ambiguous_language(message: str) -> list[str]:
        """Detect ambiguous language patterns in a message."""
        ambiguous_patterns: list[tuple[str, str]] = [
            (r"\b(it|this|that|these|those)\b(?!\s+is|\s+are|\s+was)", "ambiguous pronoun"),
            (r"\bsome\s+\w+", "vague quantifier"),
            (r"\bmaybe|perhaps|possibly|probably\b", "uncertain language"),
            (r"\betc\.?|and\s+so\s+on|and\s+more\b", "incomplete enumeration"),
            (r"\bsoon|later|eventually\b", "vague timeline"),
            (r"\b(good|bad|nice|fine|okay)\b", "subjective descriptor"),
        ]

        issues: list[str] = []
        for pattern, issue_type in ambiguous_patterns:
            if re.search(pattern, message.lower()):
                issues.append(issue_type)

        return issues

    @staticmethod
    def _compute_intent_alignment(
        request: str,
        response: str,
        action_taken: Optional[str] = None,
    ) -> float:
        """Compute intent alignment between request and response."""
        request_words = set(request.lower().split())
        response_words = set(response.lower().split())

        action_verbs = {
            "create",
            "update",
            "delete",
            "get",
            "fetch",
            "send",
            "process",
            "analyze",
            "generate",
            "search",
            "find",
            "calculate",
            "compare",
            "summarize",
            "extract",
            "transform",
            "validate",
            "verify",
            "confirm",
            "acknowledge",
            "respond",
            "reply",
            "escalate",
            "delegate",
            "forward",
            "submit",
            "report",
            "transfer",
            "approve",
            "reject",
            "notify",
            "announce",
            "broadcast",
            "check",
            "monitor",
            "review",
            "implement",
            "deploy",
            "configure",
            "install",
            "migrate",
            "test",
            "debug",
            "fix",
            "resolve",
            "handle",
            "execute",
            "run",
        }

        request_actions = request_words & action_verbs
        response_actions = response_words & action_verbs

        if not request_actions:
            keyword_overlap = len(request_words & response_words) / max(len(request_words), 1)
            return min(keyword_overlap * 2, 1.0)

        action_match = len(request_actions & response_actions) / len(request_actions)

        negative_indicators = {"error", "failed", "cannot", "unable", "refused", "sorry"}
        if response_words & negative_indicators:
            action_match *= 0.5

        return action_match

    @staticmethod
    def _is_raw_trace(message: str) -> bool:
        """Check if message is a raw execution trace/log rather than a structured message."""
        hit_count = sum(1 for p in _TRACE_LOG_PATTERNS if re.search(p, message[:500]))
        return hit_count >= 2

    # --- Core single-pair detection ---

    def _detect_single(
        self,
        sender_message: str,
        receiver_response: str,
        receiver_action: Optional[str] = None,
        sender_name: Optional[str] = None,
        receiver_name: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Run communication breakdown detection on a single message pair.

        Returns a dict with detection results if breakdown found, else None.
        """
        is_raw = self._is_raw_trace(sender_message)

        # Format check
        if self.check_format and not is_raw:
            expected_format = self._detect_expected_format(sender_message)
            format_ok, format_msg = self._check_format_compliance(
                expected_format, receiver_response
            )
        else:
            expected_format = None
            format_ok, format_msg = True, "Format check disabled"

        # Intent alignment
        intent_alignment = self._compute_intent_alignment(
            sender_message,
            receiver_response,
            receiver_action,
        )
        if is_raw and intent_alignment < self.intent_threshold:
            intent_alignment = self.intent_threshold  # Neutralize for raw traces

        # Ambiguity
        ambiguities = self._detect_ambiguous_language(sender_message)

        # Detection logic
        breakdown_type: Optional[str] = None
        detected = False

        if not format_ok:
            detected = True
            breakdown_type = "format_mismatch"
        elif intent_alignment < self.intent_threshold:
            detected = True
            breakdown_type = "intent_mismatch"
        elif len(ambiguities) >= 4:
            detected = True
            breakdown_type = "semantic_ambiguity"

        if not detected:
            return None

        # Determine severity, confidence, explanation
        sender_label = f"'{sender_name}'" if sender_name else "sender"
        receiver_label = f"'{receiver_name}'" if receiver_name else "receiver"

        if breakdown_type == "format_mismatch":
            severity = 55
            confidence = 0.9
            explanation = format_msg
            fix = f"Ensure response follows {expected_format} format. Add explicit format instructions."
        elif breakdown_type == "intent_mismatch":
            if intent_alignment < 0.2:
                severity = 75
            else:
                severity = 55
            confidence = 1 - intent_alignment

            # Reduce confidence when keyword overlap is high despite verb mismatch
            request_words = set(sender_message.lower().split())
            response_words = set(receiver_response.lower().split())
            req_content = request_words - _STOP_WORDS
            resp_content = response_words - _STOP_WORDS
            if req_content:
                content_overlap = len(req_content & resp_content) / len(req_content)
                if content_overlap > 0.3:
                    confidence *= max(0.4, 1.0 - content_overlap)

            explanation = (
                f"Response does not align with request intent. "
                f"Alignment score: {intent_alignment:.1%}"
            )
            fix = "Clarify request with specific action verbs and expected outcomes."
        else:
            severity = 30
            confidence = 0.6
            explanation = f"Ambiguous language detected: {', '.join(ambiguities)}"
            fix = "Replace ambiguous language with specific references."

        full_explanation = f"Communication from {sender_label} to {receiver_label}: {explanation}"

        return {
            "detected": True,
            "severity": severity,
            "confidence": confidence,
            "summary": full_explanation,
            "fix_instruction": fix,
            "evidence": {
                "breakdown_type": breakdown_type,
                "intent_alignment": round(intent_alignment, 4),
                "format_ok": format_ok,
                "format_message": format_msg,
                "expected_format": expected_format,
                "ambiguities": ambiguities,
                "is_raw_trace": is_raw,
            },
        }

    # --- Trace-level detect (BaseDetector interface) ---

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect communication breakdown across consecutive span pairs.

        Treats the output of span N as the sender message and the output of
        span N+1 as the receiver response. Returns the highest-severity finding.
        """
        worst: Optional[dict[str, Any]] = None
        sorted_spans = sorted(trace.spans, key=lambda s: s.start_time)

        for i in range(len(sorted_spans) - 1):
            sender = sorted_spans[i]
            receiver = sorted_spans[i + 1]

            sender_output = (sender.output_data or {}).get("content", "")
            receiver_output = (receiver.output_data or {}).get("content", "")

            if not sender_output or not receiver_output:
                continue

            finding = self._detect_single(
                sender_message=sender_output,
                receiver_response=receiver_output,
                sender_name=sender.name,
                receiver_name=receiver.name,
            )
            if finding and (worst is None or finding["severity"] > worst["severity"]):
                worst = finding

        if worst is None:
            return DetectionResult.no_issue(self.name)

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=worst["severity"],
            summary=worst["summary"],
            fix_type=FixType.SWITCH_STRATEGY,
            fix_instruction=worst["fix_instruction"],
        )
        result.confidence = worst["confidence"]
        result.add_evidence(
            description=worst["summary"],
            data=worst["evidence"],
        )
        return result
