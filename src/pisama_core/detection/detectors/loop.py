"""Loop detector for identifying infinite loops and stuck patterns."""

from collections import Counter
from typing import Any

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import Platform, SpanKind
from pisama_core.traces.models import Span, Trace


class LoopDetector(BaseDetector):
    """Detects infinite loops, retry storms, and stuck patterns.

    This detector identifies:
    - Consecutive repetition of the same tool/action
    - Cyclic patterns (A -> B -> A -> B)
    - Search spirals (expanding repeated searches)
    - Retry storms (repeated failures)
    """

    name = "loop"
    description = "Detects infinite loops, retry storms, and stuck patterns"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (20, 100)
    realtime_capable = True

    # Configuration (defaults for interactive sessions)
    min_consecutive_for_warning = 3
    min_consecutive_for_critical = 5
    min_cycle_repetitions = 3
    max_cycle_length = 5

    # OpenClaw-specific thresholds (24/7 autonomous agents)
    PLATFORM_OVERRIDES = {
        Platform.OPENCLAW: {
            "min_consecutive_for_warning": 8,
            "min_consecutive_for_critical": 15,
            "max_cycle_length": 8,
        },
        Platform.DIFY: {
            "min_consecutive_for_warning": 6,
            "min_consecutive_for_critical": 10,
            "max_cycle_length": 6,
        },
    }

    def _get_threshold(self, trace: Trace, key: str) -> int:
        """Get platform-aware threshold value."""
        if hasattr(trace, "platform") and trace.platform in self.PLATFORM_OVERRIDES:
            overrides = self.PLATFORM_OVERRIDES[trace.platform]
            if key in overrides:
                return overrides[key]
        return getattr(self, key)

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect loop patterns in a trace."""
        tool_spans = trace.get_spans_by_kind(SpanKind.TOOL)

        warn_threshold = self._get_threshold(trace, "min_consecutive_for_warning")
        crit_threshold = self._get_threshold(trace, "min_consecutive_for_critical")

        if len(tool_spans) < warn_threshold:
            return DetectionResult.no_issue(self.name)

        # Sort by time
        sorted_spans = sorted(tool_spans, key=lambda s: s.start_time)
        tool_sequence = [s.name for s in sorted_spans]

        issues: list[str] = []
        severity = 0
        evidence_data: dict[str, Any] = {}

        # Check consecutive repetitions
        consecutive = self._check_consecutive(tool_sequence)
        if consecutive["count"] >= warn_threshold:
            if consecutive["count"] >= crit_threshold:
                severity += 50
            else:
                severity += 25
            issues.append(
                f"Tool '{consecutive['tool']}' repeated {consecutive['count']}x consecutively"
            )
            evidence_data["consecutive"] = consecutive

        # Check for cyclic patterns
        cycle = self._detect_cycle(tool_sequence)
        if cycle:
            severity += 30
            issues.append(f"Loop pattern: {' -> '.join(cycle['pattern'])} ({cycle['count']}x)")
            evidence_data["cycle"] = cycle

        # Check tool diversity
        diversity = self._check_diversity(tool_sequence)
        if diversity["ratio"] < 0.2 and len(tool_sequence) >= 5:
            severity += 20
            issues.append(f"Low tool diversity ({diversity['ratio']:.0%})")
            evidence_data["diversity"] = diversity

        if not issues:
            return DetectionResult.no_issue(self.name)

        # Cap severity
        severity = min(100, severity)

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=issues[0],
            fix_type=FixType.BREAK_LOOP,
            fix_instruction="Stop the current loop. Try a different approach or ask the user for guidance.",
        )

        # Add evidence
        for issue in issues:
            result.add_evidence(
                description=issue,
                span_ids=[s.span_id for s in sorted_spans[-10:]],
                data=evidence_data,
            )

        return result

    async def detect_realtime(self, span: Span, context: dict) -> DetectionResult:
        """Real-time loop detection for hooks."""
        recent_spans = context.get("recent_spans", [])
        recent_tools = [s.name for s in recent_spans if s.kind == SpanKind.TOOL]

        # Add current span
        if span.kind == SpanKind.TOOL:
            recent_tools.insert(0, span.name)

        if len(recent_tools) < self.min_consecutive_for_warning:
            return DetectionResult.no_issue(self.name)

        # Quick consecutive check
        consecutive = self._check_consecutive(recent_tools)

        if consecutive["count"] >= self.min_consecutive_for_critical:
            return DetectionResult.issue_found(
                detector_name=self.name,
                severity=60 + (consecutive["count"] - self.min_consecutive_for_critical) * 5,
                summary=f"Tool '{consecutive['tool']}' repeated {consecutive['count']}x consecutively",
                fix_type=FixType.BREAK_LOOP,
                fix_instruction="Stop repeating this tool. Try a different approach.",
            )
        elif consecutive["count"] >= self.min_consecutive_for_warning:
            return DetectionResult.issue_found(
                detector_name=self.name,
                severity=30,
                summary=f"Tool '{consecutive['tool']}' repeated {consecutive['count']}x",
                fix_type=FixType.SWITCH_STRATEGY,
                fix_instruction="Consider trying a different approach.",
            )

        return DetectionResult.no_issue(self.name)

    def _check_consecutive(self, sequence: list[str]) -> dict[str, Any]:
        """Check for consecutive repetitions."""
        if not sequence:
            return {"count": 0, "tool": None}

        max_count = 1
        max_tool = sequence[0]
        current_count = 1

        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                current_count += 1
                if current_count > max_count:
                    max_count = current_count
                    max_tool = sequence[i]
            else:
                current_count = 1

        return {"count": max_count, "tool": max_tool}

    def _detect_cycle(self, sequence: list[str]) -> dict[str, Any] | None:
        """Detect cyclic patterns (A -> B -> A -> B)."""
        if len(sequence) < 4:
            return None

        for length in range(2, min(self.max_cycle_length + 1, len(sequence) // 2 + 1)):
            pattern = tuple(sequence[:length])
            matches = 0
            i = 0

            while i + length <= len(sequence):
                if tuple(sequence[i : i + length]) == pattern:
                    matches += 1
                    i += length
                else:
                    break

            if matches >= self.min_cycle_repetitions:
                return {
                    "pattern": list(pattern),
                    "count": matches,
                    "length": length,
                }

        return None

    def _check_diversity(self, sequence: list[str]) -> dict[str, Any]:
        """Check tool diversity."""
        if not sequence:
            return {"ratio": 1.0, "unique": 0, "total": 0}

        counts = Counter(sequence)
        unique = len(counts)
        total = len(sequence)

        return {
            "ratio": unique / total if total > 0 else 1.0,
            "unique": unique,
            "total": total,
            "most_common": counts.most_common(3),
        }
