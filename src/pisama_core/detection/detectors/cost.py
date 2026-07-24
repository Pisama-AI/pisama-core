"""Cost detector for monitoring resource usage."""

from typing import ClassVar

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import Platform, SpanKind
from pisama_core.traces.models import Span, Trace


class CostDetector(BaseDetector):
    """Detects excessive resource usage and cost anomalies.

    Identifies:
    - Excessive LLM calls
    - Long-running sessions
    - High tool call volume
    - Potential runaway costs
    """

    name = "cost"
    description = "Detects excessive resource usage and cost anomalies"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (20, 70)
    realtime_capable = True

    # Thresholds (defaults for interactive sessions)
    max_llm_calls = 50
    max_tool_calls = 100
    max_duration_minutes = 30

    # OpenClaw-specific: 24/7 agents use per-hour budgets instead of session limits
    PLATFORM_OVERRIDES: ClassVar[dict[Platform, dict[str, int | None]]] = {
        Platform.OPENCLAW: {
            "max_llm_calls": 500,
            "max_tool_calls": 1000,
            "max_duration_minutes": None,  # No duration limit for persistent agents
        },
        Platform.DIFY: {
            "max_llm_calls": 100,
            "max_tool_calls": 200,
            "max_duration_minutes": 60,
        },
    }

    def _get_limit(self, trace: Trace, key: str) -> int | None:
        """Get platform-aware limit value."""
        if hasattr(trace, "platform") and trace.platform in self.PLATFORM_OVERRIDES:
            overrides = self.PLATFORM_OVERRIDES[trace.platform]
            if key in overrides:
                return overrides[key]
        return getattr(self, key)

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect cost anomalies."""
        llm_spans = trace.get_spans_by_kind(SpanKind.LLM)
        tool_spans = trace.get_spans_by_kind(SpanKind.TOOL)

        max_llm = self._get_limit(trace, "max_llm_calls")
        max_tool = self._get_limit(trace, "max_tool_calls")
        max_duration = self._get_limit(trace, "max_duration_minutes")

        issues = []
        severity = 0

        # Check LLM call count
        if max_llm is not None and len(llm_spans) > max_llm:
            severity += 30
            issues.append(f"Excessive LLM calls ({len(llm_spans)})")

        # Check tool call count
        if max_tool is not None and len(tool_spans) > max_tool:
            severity += 25
            issues.append(f"Excessive tool calls ({len(tool_spans)})")

        # Check duration (skip for persistent agent platforms like OpenClaw)
        duration_ms = trace.duration_ms
        if duration_ms and max_duration is not None:
            duration_min = duration_ms / 60000
            if duration_min > max_duration:
                severity += 30
                issues.append(f"Long-running session ({duration_min:.1f} minutes)")

        if not issues:
            return DetectionResult.no_issue(self.name)

        return DetectionResult.issue_found(
            detector_name=self.name,
            severity=min(70, severity),
            summary=issues[0],
            fix_type=FixType.ESCALATE,
            fix_instruction="Resource usage is high. Consider wrapping up or optimizing approach.",
        )

    async def detect_realtime(self, span: Span, context: dict) -> DetectionResult:
        """Real-time cost monitoring."""
        stats = context.get("session_stats", {})

        tool_count = stats.get("tool_count", 0)
        llm_count = stats.get("llm_count", 0)

        if tool_count > self.max_tool_calls:
            return DetectionResult.issue_found(
                detector_name=self.name,
                severity=40,
                summary=f"Tool call limit exceeded ({tool_count})",
                fix_type=FixType.ESCALATE,
                fix_instruction="Many tools called. Consider whether to continue.",
            )

        if llm_count > self.max_llm_calls:
            return DetectionResult.issue_found(
                detector_name=self.name,
                severity=40,
                summary=f"LLM call limit exceeded ({llm_count})",
                fix_type=FixType.ESCALATE,
                fix_instruction="Many LLM calls made. Session may be too long.",
            )

        return DetectionResult.no_issue(self.name)
