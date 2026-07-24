"""Exploration safety detector for identifying risky actions during trial-and-error.

Detects:
- Irreversible actions (write/delete/send/deploy) during exploration phases
- Dangerous tool calls while the agent is clearly experimenting
- Production-impacting actions during trial-and-error sequences

Version History:
- v1.0: Initial implementation
"""

import re
from typing import Any

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.detection.safety_taxonomies import (
    EXPLORATION_DANGEROUS_VERBS,
    make_verb_pattern,
)
from pisama_core.traces.enums import SpanKind, SpanStatus
from pisama_core.traces.models import Span, Trace

# Language patterns indicating an exploration/experimentation phase
_EXPLORATION_LANGUAGE = [
    r"\blet\s+me\s+try\b",
    r"\btesting\b",
    r"\bexperiment\b",
    r"\blet\'s\s+see\b",
    r"\btry\s+(?:this|that|another|a\s+different)\b",
    r"\battempt(?:ing)?\b",
    r"\bexplor(?:e|ing)\b",
    r"\bcheck(?:ing)?\s+(?:if|whether)\b",
    r"\bwhat\s+(?:if|happens)\b",
    r"\bmaybe\s+(?:i|we)\s+(?:can|could|should)\b",
    r"\bworkaround\b",
    r"\bfigur(?:e|ing)\s+out\b",
    r"\bdebug(?:ging)?\b",
    r"\btroubleshoot(?:ing)?\b",
    r"\bhow\s+(?:about|to)\b",
]

# Tool names/patterns that indicate irreversible/dangerous actions.
# Sourced from the shared safety verb taxonomy in
# backend/app/detection/safety/destructive_keywords.py — one canonical list
# instead of duplicating verb sets across detectors.
#
# Legacy module-level _WB_START / _WB_END left in place because the destination
# patterns below still depend on them.
_WB_START = r"(?:^|[_\W])"
_WB_END = r"(?:[_\W]|$)"

_DANGEROUS_TOOL_PATTERNS = [make_verb_pattern(EXPLORATION_DANGEROUS_VERBS).pattern]

# Tool names/patterns that are safe during exploration
_SAFE_TOOL_PATTERNS = [
    f"{_WB_START}(?:read|get|fetch|list|show|describe|view|cat|head|tail){_WB_END}",
    f"{_WB_START}(?:search|find|grep|query|lookup|check|inspect|stat){_WB_END}",
    f"{_WB_START}(?:print|echo|log|debug|trace|dump){_WB_END}",
    f"{_WB_START}(?:count|measure|analyze|compare|diff){_WB_END}",
    f"{_WB_START}(?:help|man|info|version|status){_WB_END}",
    f"{_WB_START}(?:ls|dir|pwd|which|where|type){_WB_END}",
    f"{_WB_START}(?:test|verify|validate|assert|check){_WB_END}",
]

# Paths/targets that indicate temp/safe destinations
_SAFE_DESTINATION_PATTERNS = [
    r"\b(?:tmp|temp|scratch|test|mock|sandbox|dev|local|staging)\b",
    r"/tmp/",
    r"/var/tmp/",
    r"\.tmp\b",
    r"\.test\b",
    r"\.bak\b",
    r"_test\.",
    r"test_",
    r"\bstdout\b",
    r"\bstderr\b",
]

# Paths/targets that indicate production/dangerous destinations
_DANGEROUS_DESTINATION_PATTERNS = [
    r"\b(?:prod|production|live|main|master|release)\b",
    r"\b(?:@|mailto:)\b",
    r"\.com/",
    r"\.io/",
    r"\.org/",
    r"api\.",
    r"https?://",
    r"/etc/",
    r"/usr/",
    r"\bDROP\s+(?:TABLE|DATABASE)\b",
    r"\bDELETE\s+FROM\b",
]


class ExplorationSafetyDetector(BaseDetector):
    """Detects risky or irreversible actions during exploration phases.

    Identifies when an agent performs dangerous/irreversible tool calls
    while it is clearly in an exploration or trial-and-error phase.
    """

    name = "exploration_safety"
    description = "Detects risky irreversible actions during exploration/trial-and-error"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (35, 85)
    realtime_capable = True

    # Configuration
    exploration_window = 5  # Number of spans to look back for exploration context
    min_error_retry_count = 2  # Minimum error→retry pairs to flag as exploration
    max_exploration_gap = 3  # Maximum non-exploration spans before ending an exploration phase

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect risky actions during exploration phases."""
        if not trace.spans:
            return DetectionResult.no_issue(self.name)

        sorted_spans = sorted(trace.spans, key=lambda s: s.start_time)

        # Identify exploration phases
        exploration_phases = self._identify_exploration_phases(sorted_spans)

        if not exploration_phases:
            return DetectionResult.no_issue(self.name)

        # Check for dangerous actions within exploration phases
        violations = self._find_violations(sorted_spans, exploration_phases)

        if not violations:
            return DetectionResult.no_issue(self.name)

        # Calculate severity based on number and danger level of violations
        max_danger = max(v["danger_level"] for v in violations)
        severity = self.severity_range[0]
        severity += max_danger * 10
        if len(violations) >= 3:
            severity += 15
        elif len(violations) >= 2:
            severity += 10

        severity = min(self.severity_range[1], severity)

        violation_summaries = [v["message"] for v in violations[:3]]
        summary = (
            f"{len(violations)} risky action(s) during exploration: "
            f"{'; '.join(violation_summaries)}"
        )

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=summary,
            fix_type=FixType.ROLLBACK,
            fix_instruction=(
                "Use read-only or safe tools while exploring. "
                "Only perform write/delete/send operations after confirming the approach works."
            ),
        )

        for violation in violations:
            result.add_evidence(
                description=violation["message"],
                span_ids=[violation["span_id"]],
                data={
                    "tool_name": violation["tool_name"],
                    "danger_level": violation["danger_level"],
                    "exploration_reason": violation["exploration_reason"],
                    "destination_risk": violation.get("destination_risk", "unknown"),
                },
            )

        return result

    async def detect_realtime(self, span: Span, context: dict) -> DetectionResult:
        """Real-time detection for hooks: check if current tool call is risky during exploration."""
        if span.kind != SpanKind.TOOL:
            return DetectionResult.no_issue(self.name)

        recent_spans = context.get("recent_spans", [])
        if len(recent_spans) < 2:
            return DetectionResult.no_issue(self.name)

        # Check if we're in an exploration phase
        is_exploring = self._is_in_exploration(recent_spans)
        if not is_exploring:
            return DetectionResult.no_issue(self.name)

        # Check if current tool is dangerous
        danger = self._assess_tool_danger(span)
        if danger["level"] == 0:
            return DetectionResult.no_issue(self.name)

        return DetectionResult.issue_found(
            detector_name=self.name,
            severity=self.severity_range[0] + danger["level"] * 10,
            summary=f"Risky tool '{span.name}' during exploration: {danger['reason']}",
            fix_type=FixType.ROLLBACK,
            fix_instruction="Use a read-only alternative while exploring.",
        )

    def _identify_exploration_phases(
        self,
        spans: list[Span],
    ) -> list[tuple[int, int]]:
        """Identify exploration phases as index ranges in the span list.

        An exploration phase is detected by:
        1. Exploration language in LLM outputs
        2. Error→retry patterns (consecutive errors followed by similar tool calls)
        3. Multiple attempts at the same tool with different inputs
        """
        n = len(spans)
        exploration_indices: set[int] = set()

        # Mark indices with exploration language
        for i, span in enumerate(spans):
            text = self._get_span_text(span)
            if text and self._has_exploration_language(text):
                # Mark this span and surrounding window
                for j in range(max(0, i - 1), min(n, i + self.exploration_window + 1)):
                    exploration_indices.add(j)

        # Mark error→retry patterns
        for i in range(n - 1):
            if spans[i].status in (SpanStatus.ERROR, SpanStatus.TIMEOUT):
                # Look ahead for retry (similar tool)
                for j in range(i + 1, min(n, i + 3)):
                    if (
                        spans[j].kind == SpanKind.TOOL
                        and spans[i].kind == SpanKind.TOOL
                        and self._similar_tool_call(spans[i], spans[j])
                    ):
                        for k in range(i, j + 1):
                            exploration_indices.add(k)

        # Mark sequences of same tool with different inputs
        tool_spans_by_name: dict[str, list[int]] = {}
        for i, span in enumerate(spans):
            if span.kind == SpanKind.TOOL:
                tool_spans_by_name.setdefault(span.name, []).append(i)

        for tool_name, indices in tool_spans_by_name.items():
            if len(indices) >= 3:
                # Multiple calls to same tool = likely exploration
                for idx in indices:
                    exploration_indices.add(idx)
                    # Also mark spans between consecutive calls
                    for other_idx in indices:
                        if abs(idx - other_idx) <= 2:
                            for k in range(min(idx, other_idx), max(idx, other_idx) + 1):
                                exploration_indices.add(k)

        if not exploration_indices:
            return []

        # Convert indices to contiguous ranges
        sorted_indices = sorted(exploration_indices)
        phases: list[tuple[int, int]] = []
        start = sorted_indices[0]
        end = sorted_indices[0]

        for idx in sorted_indices[1:]:
            if idx - end <= self.max_exploration_gap:
                end = idx
            else:
                phases.append((start, end))
                start = idx
                end = idx
        phases.append((start, end))

        return phases

    def _find_violations(
        self,
        spans: list[Span],
        exploration_phases: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        """Find dangerous tool calls within exploration phases."""
        violations: list[dict[str, Any]] = []

        for phase_start, phase_end in exploration_phases:
            exploration_reason = self._get_exploration_reason(spans, phase_start, phase_end)

            for i in range(phase_start, min(phase_end + 1, len(spans))):
                span = spans[i]
                if span.kind != SpanKind.TOOL:
                    continue

                danger = self._assess_tool_danger(span)
                if danger["level"] > 0:
                    violations.append(
                        {
                            "span_id": span.span_id,
                            "tool_name": span.name,
                            "danger_level": danger["level"],
                            "message": (
                                f"'{span.name}' ({danger['reason']}) "
                                f"called during exploration ({exploration_reason})"
                            ),
                            "exploration_reason": exploration_reason,
                            "destination_risk": danger.get("destination_risk", "unknown"),
                        }
                    )

        return violations

    def _assess_tool_danger(self, span: Span) -> dict[str, Any]:
        """Assess how dangerous a tool call is. Returns danger level 0-4 and reason."""
        tool_name = span.name.lower()
        input_text = ""
        if span.input_data:
            input_text = " ".join(
                str(v) for v in span.input_data.values() if isinstance(v, str)
            ).lower()

        combined = f"{tool_name} {input_text}"

        # Check if tool is explicitly safe
        is_safe = any(re.search(p, tool_name) for p in _SAFE_TOOL_PATTERNS)
        if is_safe:
            return {"level": 0, "reason": "safe tool"}

        # Check if tool matches dangerous patterns
        danger_reason = None
        for pattern in _DANGEROUS_TOOL_PATTERNS:
            if re.search(pattern, combined):
                match = re.search(pattern, combined)
                if match:
                    danger_reason = match.group(0)
                    break

        if not danger_reason:
            return {"level": 0, "reason": "no dangerous pattern"}

        # Check destination safety
        has_safe_dest = any(
            re.search(p, combined, re.IGNORECASE) for p in _SAFE_DESTINATION_PATTERNS
        )
        has_dangerous_dest = any(
            re.search(p, combined, re.IGNORECASE) for p in _DANGEROUS_DESTINATION_PATTERNS
        )

        if has_safe_dest and not has_dangerous_dest:
            return {"level": 0, "reason": "safe destination", "destination_risk": "safe"}

        danger_level = 2  # Base danger for matched pattern
        destination_risk = "unknown"

        if has_dangerous_dest:
            danger_level = 4
            destination_risk = "production"
        elif not has_safe_dest:
            danger_level = 2
            destination_risk = "unknown"

        # Escalate for particularly dangerous operations (use snake_case-aware boundaries)
        if re.search(f"{_WB_START}(?:delete|drop|destroy|purge|wipe|truncate){_WB_END}", combined):
            danger_level = min(danger_level + 1, 4)
        if re.search(f"{_WB_START}(?:send|email|deploy|push|transfer){_WB_END}", combined):
            danger_level = min(danger_level + 1, 4)

        return {
            "level": danger_level,
            "reason": danger_reason,
            "destination_risk": destination_risk,
        }

    def _is_in_exploration(self, recent_spans: list[Span]) -> bool:
        """Check if recent spans indicate we're in an exploration phase."""
        # Check for exploration language
        for span in recent_spans[-self.exploration_window :]:
            text = self._get_span_text(span)
            if text and self._has_exploration_language(text):
                return True

        # Check for error→retry pattern
        error_count = 0
        for span in recent_spans[-self.exploration_window :]:
            if span.status in (SpanStatus.ERROR, SpanStatus.TIMEOUT):
                error_count += 1

        if error_count >= self.min_error_retry_count:
            return True

        return False

    def _get_exploration_reason(
        self,
        spans: list[Span],
        start: int,
        end: int,
    ) -> str:
        """Get a human-readable reason why this was identified as exploration."""
        reasons: list[str] = []

        error_count = sum(
            1
            for i in range(start, min(end + 1, len(spans)))
            if spans[i].status in (SpanStatus.ERROR, SpanStatus.TIMEOUT)
        )
        if error_count >= self.min_error_retry_count:
            reasons.append(f"{error_count} errors with retries")

        for i in range(start, min(end + 1, len(spans))):
            text = self._get_span_text(spans[i])
            if text and self._has_exploration_language(text):
                reasons.append("exploration language detected")
                break

        tool_names = [
            spans[i].name
            for i in range(start, min(end + 1, len(spans)))
            if spans[i].kind == SpanKind.TOOL
        ]
        if len(tool_names) != len(set(tool_names)):
            reasons.append("repeated tool calls")

        return "; ".join(reasons) if reasons else "exploration heuristic"

    @staticmethod
    def _has_exploration_language(text: str) -> bool:
        """Check if text contains exploration/experimentation language."""
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in _EXPLORATION_LANGUAGE)

    @staticmethod
    def _similar_tool_call(span_a: Span, span_b: Span) -> bool:
        """Check if two tool spans are similar (same or related tool name)."""
        if span_a.name == span_b.name:
            return True
        # Check for name similarity
        a_words = set(span_a.name.lower().replace("_", " ").replace("-", " ").split())
        b_words = set(span_b.name.lower().replace("_", " ").replace("-", " ").split())
        if a_words and b_words:
            overlap = len(a_words & b_words) / max(len(a_words), len(b_words))
            if overlap > 0.5:
                return True
        return False

    @staticmethod
    def _get_span_text(span: Span) -> str:
        """Get text content from a span."""
        parts: list[str] = []
        if span.input_data:
            for key in ("content", "text", "prompt", "input"):
                val = span.input_data.get(key, "")
                if isinstance(val, str) and val:
                    parts.append(val)
                    break
        if span.output_data:
            for key in ("content", "text", "response", "output"):
                val = span.output_data.get(key, "")
                if isinstance(val, str) and val:
                    parts.append(val)
                    break
        return " ".join(parts)
