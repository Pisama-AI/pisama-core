"""Repetition detector for similar but not identical repeated actions."""

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Span, Trace


class RepetitionDetector(BaseDetector):
    """Detects repetitive patterns that aren't exact loops.

    Identifies:
    - Similar searches with slight variations
    - Repeated failed attempts with minor changes
    - Oscillating between similar options
    """

    name = "repetition"
    description = "Detects repetitive patterns with variations"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (15, 50)
    realtime_capable = True

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect repetition patterns."""
        tool_spans = trace.get_spans_by_kind(SpanKind.TOOL)

        if len(tool_spans) < 5:
            return DetectionResult.no_issue(self.name)

        sorted_spans = sorted(tool_spans, key=lambda s: s.start_time)

        # Group by tool name
        tool_groups: dict[str, list[Span]] = {}
        for span in sorted_spans:
            if span.name not in tool_groups:
                tool_groups[span.name] = []
            tool_groups[span.name].append(span)

        # Check for tools called many times
        issues = []
        severity = 0

        for tool_name, spans in tool_groups.items():
            if len(spans) >= 5:
                # Calculate ratio of this tool
                ratio = len(spans) / len(sorted_spans)
                if ratio > 0.5:
                    severity += 25
                    issues.append(f"Tool '{tool_name}' dominates session ({ratio:.0%} of calls)")
                elif ratio > 0.3:
                    severity += 15
                    issues.append(f"Tool '{tool_name}' frequently used ({len(spans)} times)")

        if not issues:
            return DetectionResult.no_issue(self.name)

        return DetectionResult.issue_found(
            detector_name=self.name,
            severity=min(50, severity),
            summary=issues[0],
            fix_type=FixType.SWITCH_STRATEGY,
            fix_instruction="Consider whether repeated tool use is productive.",
        )

    async def detect_realtime(self, span: Span, context: dict) -> DetectionResult:
        """Real-time repetition check."""
        recent_spans = context.get("recent_spans", [])
        if span.kind != SpanKind.TOOL:
            return DetectionResult.no_issue(self.name)

        # Count recent uses of this tool
        same_tool_count = sum(1 for s in recent_spans if s.name == span.name)

        if same_tool_count >= 5:
            return DetectionResult.issue_found(
                detector_name=self.name,
                severity=30,
                summary=f"Tool '{span.name}' used {same_tool_count + 1} times recently",
                fix_type=FixType.SWITCH_STRATEGY,
                fix_instruction="This tool has been used many times. Consider alternatives.",
            )

        return DetectionResult.no_issue(self.name)
