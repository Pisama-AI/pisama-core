"""Base detector class."""

import time
from abc import ABC, abstractmethod
from typing import Optional

from pisama_core.detection.result import DetectionResult, FixRecommendation
from pisama_core.traces.enums import Platform
from pisama_core.traces.models import Span, Trace


class BaseDetector(ABC):
    """Abstract base class for all detectors.

    To create a new detector:
    1. Subclass BaseDetector
    2. Set name, description, and platforms
    3. Implement the detect() method
    4. Optionally implement get_fix_recommendation()
    5. Register with the DetectorRegistry

    Example:
        class MyDetector(BaseDetector):
            name = "my_detector"
            description = "Detects my specific pattern"
            platforms = [Platform.CLAUDE_CODE, Platform.LANGGRAPH]
            severity_range = (20, 60)

            async def detect(self, trace: Trace) -> DetectionResult:
                # Detection logic here
                return DetectionResult.no_issue(self.name)
    """

    # Detector identity
    name: str = "base_detector"
    description: str = "Base detector"
    version: str = "1.0.0"

    # Which platforms this detector applies to
    # Empty list means all platforms
    platforms: list[Platform] = []

    # Severity range this detector can produce
    severity_range: tuple[int, int] = (0, 100)

    # Detection settings
    enabled: bool = True
    realtime_capable: bool = True  # Can run in real-time hook

    def __init__(self) -> None:
        """Initialize the detector."""
        pass

    def applies_to_platform(self, platform: Platform) -> bool:
        """Check if this detector applies to a platform."""
        if not self.platforms:
            return True  # Empty means all platforms
        return platform in self.platforms

    @abstractmethod
    async def detect(self, trace: Trace) -> DetectionResult:
        """Run detection on a complete trace.

        Args:
            trace: The trace to analyze

        Returns:
            DetectionResult with findings
        """
        ...

    async def detect_realtime(
        self,
        span: Span,
        context: dict,
    ) -> DetectionResult:
        """Run real-time detection on a single span.

        This is called during PreToolUse hooks for immediate detection.
        Override this for optimized real-time detection.

        Args:
            span: The current span being executed
            context: Session context (recent spans, stats, etc.)

        Returns:
            DetectionResult with findings
        """
        # Default: create a mini-trace and run full detection
        from pisama_core.traces.models import Trace, TraceMetadata

        mini_trace = Trace(
            trace_id=span.trace_id or "realtime",
            metadata=TraceMetadata(platform=span.platform),
        )
        mini_trace.add_span(span)

        # Add context spans if available
        for ctx_span in context.get("recent_spans", []):
            mini_trace.add_span(ctx_span)

        return await self.detect(mini_trace)

    def get_fix_recommendation(
        self,
        result: DetectionResult,
    ) -> Optional[FixRecommendation]:
        """Get a fix recommendation for a detection result.

        Override this to provide custom recommendations.

        Args:
            result: The detection result

        Returns:
            FixRecommendation or None
        """
        return result.recommendation

    async def run(self, trace: Trace) -> DetectionResult:
        """Run the detector with timing and error handling.

        Args:
            trace: The trace to analyze

        Returns:
            DetectionResult with findings
        """
        if not self.enabled:
            return DetectionResult.no_issue(self.name)

        start_time = time.perf_counter()

        try:
            result = await self.detect(trace)
            result.execution_time_ms = (time.perf_counter() - start_time) * 1000
            result.detector_version = self.version
            return result

        except Exception as e:
            return DetectionResult(
                detector_name=self.name,
                detector_version=self.version,
                detected=False,
                severity=0,
                summary=f"Detector error: {str(e)}",
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                metadata={"error": str(e)},
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, enabled={self.enabled})>"
