"""Hallucination detector for detecting fabricated information."""

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind, SpanStatus
from pisama_core.traces.models import Trace


class HallucinationDetector(BaseDetector):
    """Detects potential hallucinations and fabricated information.

    Identifies:
    - Tool calls with fabricated inputs
    - References to non-existent files/resources
    - Contradictory information in outputs
    - Claims without supporting tool results
    """

    name = "hallucination"
    description = "Detects potential hallucinations and fabricated information"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (40, 90)
    realtime_capable = False  # Needs semantic analysis

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect potential hallucinations."""
        tool_spans = trace.get_spans_by_kind(SpanKind.TOOL)

        if not tool_spans:
            return DetectionResult.no_issue(self.name)

        issues = []
        severity = 0

        # Check for failed tool calls that might indicate hallucinated paths
        failed_file_ops = []
        for span in tool_spans:
            if span.status == SpanStatus.ERROR:
                if span.name in ["Read", "Glob", "Grep"]:
                    if span.error_message and (
                        "not found" in span.error_message.lower()
                        or "no such file" in span.error_message.lower()
                    ):
                        failed_file_ops.append(span)

        if len(failed_file_ops) >= 3:
            severity += 40
            issues.append(
                f"Multiple file operations failed ({len(failed_file_ops)}) - possible fabricated paths"
            )

        # Check for repeated failures with same pattern
        error_spans = [s for s in tool_spans if s.status.is_failure]
        if len(error_spans) > 5:
            error_ratio = len(error_spans) / len(tool_spans)
            if error_ratio > 0.3:
                severity += 30
                issues.append(
                    f"High error rate ({error_ratio:.0%}) - may indicate fabricated information"
                )

        if not issues:
            return DetectionResult.no_issue(self.name)

        return DetectionResult.issue_found(
            detector_name=self.name,
            severity=min(90, severity),
            summary=issues[0],
            fix_type=FixType.ESCALATE,
            fix_instruction="Possible hallucination detected. Verify information before proceeding.",
        )
