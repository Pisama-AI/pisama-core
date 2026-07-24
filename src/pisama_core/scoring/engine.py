"""Scoring engine for calculating severity."""

from typing import Any

from pisama_core.detection.result import DetectionResult
from pisama_core.scoring.thresholds import SeverityLevel, Thresholds


class ScoringEngine:
    """Engine for calculating aggregate severity scores.

    The scoring engine combines results from multiple detectors
    into a single severity score and priority ordering.
    """

    def __init__(
        self,
        alert_threshold: int = Thresholds.DEFAULT_ALERT_THRESHOLD,
        block_threshold: int = Thresholds.DEFAULT_BLOCK_THRESHOLD,
    ) -> None:
        self.alert_threshold = alert_threshold
        self.block_threshold = block_threshold

    def calculate_severity(self, results: list[DetectionResult]) -> int:
        """Calculate aggregate severity from multiple results.

        Uses max severity with additional weight for multiple issues.

        Args:
            results: Detection results to aggregate

        Returns:
            Aggregate severity score (0-100)
        """
        if not results:
            return 0

        detected_results = [r for r in results if r.detected]

        if not detected_results:
            return 0

        # Start with max severity
        max_severity = max(r.severity for r in detected_results)

        # Add weight for multiple issues (diminishing returns)
        issue_count = len(detected_results)
        if issue_count > 1:
            additional = min(15, (issue_count - 1) * 5)
            max_severity = min(100, max_severity + additional)

        return max_severity

    def calculate_confidence(self, results: list[DetectionResult]) -> float:
        """Calculate aggregate confidence from results.

        Args:
            results: Detection results

        Returns:
            Aggregate confidence (0.0-1.0)
        """
        if not results:
            return 1.0

        detected_results = [r for r in results if r.detected]

        if not detected_results:
            return 1.0

        # Weight confidence by severity
        total_weight = sum(r.severity for r in detected_results)
        if total_weight == 0:
            return sum(r.confidence for r in detected_results) / len(detected_results)

        weighted_confidence = (
            sum(r.confidence * r.severity for r in detected_results) / total_weight
        )

        return weighted_confidence

    def get_level(self, severity: int) -> SeverityLevel:
        """Get severity level for a score."""
        return Thresholds.get_level(severity)

    def should_alert(self, severity: int) -> bool:
        """Whether to alert at this severity."""
        return severity >= self.alert_threshold

    def should_block(self, severity: int) -> bool:
        """Whether to block at this severity."""
        return severity >= self.block_threshold

    def get_priority_order(self, results: list[DetectionResult]) -> list[DetectionResult]:
        """Get results ordered by priority.

        Args:
            results: Detection results

        Returns:
            Results sorted by severity (highest first)
        """
        detected = [r for r in results if r.detected]
        return sorted(detected, key=lambda r: (-r.severity, -r.confidence))

    def summarize(self, results: list[DetectionResult]) -> dict[str, Any]:
        """Create a summary of detection results.

        Args:
            results: Detection results

        Returns:
            Summary dictionary
        """
        severity = self.calculate_severity(results)
        confidence = self.calculate_confidence(results)
        detected = [r for r in results if r.detected]

        return {
            "severity": severity,
            "level": str(self.get_level(severity)),
            "confidence": confidence,
            "issues_found": len(detected),
            "total_detectors": len(results),
            "should_alert": self.should_alert(severity),
            "should_block": self.should_block(severity),
            "top_issues": [
                {"detector": r.detector_name, "severity": r.severity, "summary": r.summary}
                for r in self.get_priority_order(results)[:3]
            ],
        }
