"""Detection orchestrator for running multiple detectors."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from pisama_core.detection.registry import DetectorRegistry
from pisama_core.detection.registry import registry as global_registry
from pisama_core.detection.result import DetectionResult
from pisama_core.traces.enums import Platform
from pisama_core.traces.models import Span, Trace
from pisama_core.utils._telemetry import record_first_run as _record_first_run


@dataclass
class AnalysisResult:
    """Result of analyzing a trace with multiple detectors."""

    trace_id: str
    platform: Platform

    # Results from all detectors
    detection_results: list[DetectionResult] = field(default_factory=list)

    # Aggregated metrics
    total_detectors_run: int = 0
    issues_detected: int = 0
    max_severity: int = 0
    total_execution_time_ms: float = 0.0

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_issues(self) -> bool:
        """Whether any issues were detected."""
        return self.issues_detected > 0

    @property
    def critical(self) -> bool:
        """Whether any critical issues (severity >= 60) were detected."""
        return self.max_severity >= 60

    def get_issues(self) -> list[DetectionResult]:
        """Get all results where issues were detected."""
        return [r for r in self.detection_results if r.detected]

    def get_by_severity(self, min_severity: int = 0) -> list[DetectionResult]:
        """Get results above a severity threshold."""
        return [r for r in self.detection_results if r.severity >= min_severity]

    def get_recommendations(self) -> list[dict[str, Any]]:
        """Get all fix recommendations from detected issues."""
        recommendations = []
        for result in self.get_issues():
            for rec in result.all_recommendations:
                recommendations.append(
                    {
                        "detector": result.detector_name,
                        "severity": result.severity,
                        "recommendation": rec.to_dict(),
                    }
                )
        return sorted(
            recommendations, key=lambda x: (-x["severity"], x["recommendation"]["priority"])
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "platform": str(self.platform),
            "detection_results": [r.to_dict() for r in self.detection_results],
            "total_detectors_run": self.total_detectors_run,
            "issues_detected": self.issues_detected,
            "max_severity": self.max_severity,
            "total_execution_time_ms": self.total_execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RealtimeResult:
    """Result of real-time analysis during a hook."""

    span_id: str
    should_block: bool = False
    severity: int = 0
    issues: list[str] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)

    # For blocking
    block_reason: Optional[str] = None

    # Timing
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "should_block": self.should_block,
            "severity": self.severity,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "block_reason": self.block_reason,
            "execution_time_ms": self.execution_time_ms,
        }


class DetectionOrchestrator:
    """Orchestrates running multiple detectors on traces.

    The orchestrator manages:
    - Running multiple detectors in parallel
    - Aggregating results
    - Real-time detection for hooks
    - Filtering by platform

    Example:
        orchestrator = DetectionOrchestrator()

        # Full analysis
        result = await orchestrator.analyze(trace)
        if result.has_issues:
            print(f"Found {result.issues_detected} issues")

        # Real-time analysis (for hooks)
        realtime = await orchestrator.analyze_realtime(span, context)
        if realtime.should_block:
            sys.exit(1)
    """

    def __init__(
        self,
        registry: Optional[DetectorRegistry] = None,
        severity_threshold: int = 40,
        block_threshold: int = 60,
        parallel: bool = True,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            registry: Detector registry to use (defaults to global)
            severity_threshold: Minimum severity to report
            block_threshold: Severity at which to recommend blocking
            parallel: Whether to run detectors in parallel
        """
        # DetectorRegistry implements __len__, so an intentionally empty
        # registry is falsy. Treat only None as "use the global registry";
        # otherwise callers cannot reliably isolate an orchestrator.
        self.registry = registry if registry is not None else global_registry
        self.severity_threshold = severity_threshold
        self.block_threshold = block_threshold
        self.parallel = parallel
        _record_first_run()

    async def analyze(self, trace: Trace) -> AnalysisResult:
        """Analyze a complete trace with all applicable detectors.

        Args:
            trace: The trace to analyze

        Returns:
            AnalysisResult with aggregated findings
        """
        platform = trace.metadata.platform
        detectors = self.registry.get_for_platform(platform)

        if not detectors:
            return AnalysisResult(
                trace_id=trace.trace_id,
                platform=platform,
            )

        # Run detectors
        if self.parallel:
            results = await asyncio.gather(
                *[d.run(trace) for d in detectors],
                return_exceptions=True,
            )
            # Filter out exceptions
            detection_results = [r for r in results if isinstance(r, DetectionResult)]
        else:
            detection_results = []
            for detector in detectors:
                result = await detector.run(trace)
                detection_results.append(result)

        # Keep successful no-issue results for detector accounting, but do not
        # report detected issues below the configured threshold.
        reported_results = [
            result
            for result in detection_results
            if not result.detected or result.severity >= self.severity_threshold
        ]

        # Aggregate
        return AnalysisResult(
            trace_id=trace.trace_id,
            platform=platform,
            detection_results=reported_results,
            total_detectors_run=len(detectors),
            issues_detected=sum(1 for r in reported_results if r.detected),
            max_severity=max((r.severity for r in reported_results), default=0),
            total_execution_time_ms=sum(r.execution_time_ms for r in detection_results),
        )

    async def analyze_realtime(
        self,
        span: Span,
        context: dict[str, Any],
    ) -> RealtimeResult:
        """Analyze a span in real-time for immediate detection.

        This is optimized for hook execution where latency matters.

        Args:
            span: The current span being executed
            context: Session context (recent spans, stats, etc.)

        Returns:
            RealtimeResult with immediate findings
        """
        import time

        start = time.perf_counter()

        platform = span.platform
        detectors = self.registry.get_realtime_capable(platform)

        if not detectors:
            return RealtimeResult(span_id=span.span_id)

        # Run real-time detection
        issues: list[str] = []
        recommendations: list[dict[str, Any]] = []
        max_severity = 0

        for detector in detectors:
            try:
                result = await detector.detect_realtime(span, context)
                if result.detected:
                    issues.append(result.summary)
                    max_severity = max(max_severity, result.severity)
                    for rec in result.all_recommendations:
                        recommendations.append(
                            {
                                "detector": result.detector_name,
                                "severity": result.severity,
                                **rec.to_dict(),
                            }
                        )
            except Exception:
                # Don't let detector errors block execution
                pass

        execution_time_ms = (time.perf_counter() - start) * 1000

        # Determine if we should block
        should_block = max_severity >= self.block_threshold
        block_reason = issues[0] if should_block and issues else None

        return RealtimeResult(
            span_id=span.span_id,
            should_block=should_block,
            severity=max_severity,
            issues=issues,
            recommendations=recommendations,
            block_reason=block_reason,
            execution_time_ms=execution_time_ms,
        )

    def get_detector_status(self) -> dict[str, Any]:
        """Get status of all registered detectors."""
        return {
            "total": self.registry.count,
            "enabled": self.registry.enabled_count,
            "detectors": [
                {
                    "name": d.name,
                    "enabled": d.enabled,
                    "realtime": d.realtime_capable,
                    "platforms": [str(p) for p in d.platforms] if d.platforms else ["all"],
                }
                for d in self.registry.get_all()
            ],
        }
