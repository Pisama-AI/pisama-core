"""Tests for pisama_core.scoring module."""

from pisama_core.detection.result import DetectionResult
from pisama_core.scoring.engine import ScoringEngine
from pisama_core.scoring.thresholds import SeverityLevel, Thresholds


class TestSeverityLevel:
    """Tests for SeverityLevel enum."""

    def test_severity_level_values(self):
        """Test severity level values."""
        assert SeverityLevel.HEALTHY.value == "healthy"
        assert SeverityLevel.WARNING.value == "warning"
        assert SeverityLevel.PROBLEMATIC.value == "problematic"
        assert SeverityLevel.CRITICAL.value == "critical"
        assert SeverityLevel.SEVERE.value == "severe"


class TestThresholds:
    """Tests for Thresholds."""

    def test_threshold_ranges(self):
        """Test threshold range definitions."""
        assert Thresholds.HEALTHY == (0, 19)
        assert Thresholds.WARNING == (20, 39)
        assert Thresholds.PROBLEMATIC == (40, 59)
        assert Thresholds.CRITICAL == (60, 79)
        assert Thresholds.SEVERE == (80, 100)

    def test_default_thresholds(self):
        """Test default threshold values."""
        assert Thresholds.DEFAULT_ALERT_THRESHOLD == 40
        assert Thresholds.DEFAULT_BLOCK_THRESHOLD == 60
        assert Thresholds.DEFAULT_TERMINATE_THRESHOLD == 90

    def test_get_level_healthy(self):
        """Test getting severity level for healthy score."""
        assert Thresholds.get_level(0) == SeverityLevel.HEALTHY
        assert Thresholds.get_level(10) == SeverityLevel.HEALTHY
        assert Thresholds.get_level(19) == SeverityLevel.HEALTHY

    def test_get_level_warning(self):
        """Test getting severity level for warning score."""
        assert Thresholds.get_level(20) == SeverityLevel.WARNING
        assert Thresholds.get_level(30) == SeverityLevel.WARNING
        assert Thresholds.get_level(39) == SeverityLevel.WARNING

    def test_get_level_problematic(self):
        """Test getting severity level for problematic score."""
        assert Thresholds.get_level(40) == SeverityLevel.PROBLEMATIC
        assert Thresholds.get_level(50) == SeverityLevel.PROBLEMATIC
        assert Thresholds.get_level(59) == SeverityLevel.PROBLEMATIC

    def test_get_level_critical(self):
        """Test getting severity level for critical score."""
        assert Thresholds.get_level(60) == SeverityLevel.CRITICAL
        assert Thresholds.get_level(70) == SeverityLevel.CRITICAL
        assert Thresholds.get_level(79) == SeverityLevel.CRITICAL

    def test_get_level_severe(self):
        """Test getting severity level for severe score."""
        assert Thresholds.get_level(80) == SeverityLevel.SEVERE
        assert Thresholds.get_level(90) == SeverityLevel.SEVERE
        assert Thresholds.get_level(100) == SeverityLevel.SEVERE

    def test_should_block(self):
        """Test block threshold check."""
        assert Thresholds.should_block(59) is False
        assert Thresholds.should_block(60) is True
        assert Thresholds.should_block(100) is True

    def test_should_alert(self):
        """Test alert threshold check."""
        assert Thresholds.should_alert(39) is False
        assert Thresholds.should_alert(40) is True
        assert Thresholds.should_alert(50) is True

    def test_should_terminate(self):
        """Test terminate threshold check."""
        assert Thresholds.should_terminate(89) is False
        assert Thresholds.should_terminate(90) is True
        assert Thresholds.should_terminate(100) is True

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        assert Thresholds.should_block(50, threshold=50) is True
        assert Thresholds.should_block(49, threshold=50) is False
        assert Thresholds.should_alert(30, threshold=30) is True


class TestScoringEngine:
    """Tests for ScoringEngine."""

    def test_create_engine(self):
        """Test creating scoring engine."""
        engine = ScoringEngine()
        assert engine is not None
        assert engine.alert_threshold == 40
        assert engine.block_threshold == 60

    def test_create_engine_custom_thresholds(self):
        """Test creating engine with custom thresholds."""
        engine = ScoringEngine(alert_threshold=50, block_threshold=70)
        assert engine.alert_threshold == 50
        assert engine.block_threshold == 70

    def test_calculate_severity_empty(self):
        """Test severity calculation with no results."""
        engine = ScoringEngine()
        severity = engine.calculate_severity([])
        assert severity == 0

    def test_calculate_severity_single(self):
        """Test severity calculation with single result."""
        engine = ScoringEngine()
        result = DetectionResult.issue_found("test", severity=50, summary="test")
        severity = engine.calculate_severity([result])
        assert severity == 50

    def test_calculate_severity_multiple(self):
        """Test severity calculation with multiple results."""
        engine = ScoringEngine()
        results = [
            DetectionResult.issue_found("test1", severity=40, summary="issue 1"),
            DetectionResult.issue_found("test2", severity=60, summary="issue 2"),
            DetectionResult.no_issue("test3"),
        ]
        severity = engine.calculate_severity(results)
        # Should be max of detected issues (60) + bonus for multiple (5)
        assert severity == 65

    def test_calculate_severity_only_no_issues(self):
        """Test severity with only no-issue results."""
        engine = ScoringEngine()
        results = [
            DetectionResult.no_issue("test1"),
            DetectionResult.no_issue("test2"),
        ]
        severity = engine.calculate_severity(results)
        assert severity == 0

    def test_get_level(self):
        """Test getting severity level from score."""
        engine = ScoringEngine()

        assert engine.get_level(10) == SeverityLevel.HEALTHY
        assert engine.get_level(30) == SeverityLevel.WARNING
        assert engine.get_level(50) == SeverityLevel.PROBLEMATIC
        assert engine.get_level(70) == SeverityLevel.CRITICAL
        assert engine.get_level(90) == SeverityLevel.SEVERE

    def test_should_block(self):
        """Test should_block check."""
        engine = ScoringEngine()
        assert engine.should_block(65) is True
        assert engine.should_block(60) is True

    def test_should_not_block(self):
        """Test should_block returns False for low severity."""
        engine = ScoringEngine()
        assert engine.should_block(40) is False
        assert engine.should_block(59) is False

    def test_should_alert(self):
        """Test should_alert check."""
        engine = ScoringEngine()
        assert engine.should_alert(40) is True
        assert engine.should_alert(50) is True
        assert engine.should_alert(39) is False

    def test_get_priority_order(self):
        """Test getting priority ordered results."""
        engine = ScoringEngine()
        results = [
            DetectionResult.issue_found("low", severity=30, summary="low"),
            DetectionResult.issue_found("high", severity=80, summary="high"),
            DetectionResult.issue_found("medium", severity=50, summary="medium"),
            DetectionResult.no_issue("none"),
        ]
        ordered = engine.get_priority_order(results)
        assert len(ordered) == 3  # Only detected
        assert ordered[0].detector_name == "high"
        assert ordered[1].detector_name == "medium"
        assert ordered[2].detector_name == "low"

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        engine = ScoringEngine()
        results = [
            DetectionResult.issue_found("test", severity=50, summary="test"),
        ]
        # Set confidence on the result
        results[0].confidence = 0.9
        confidence = engine.calculate_confidence(results)
        assert 0 <= confidence <= 1

    def test_summarize(self):
        """Test result summarization."""
        engine = ScoringEngine()
        results = [
            DetectionResult.issue_found("test1", severity=70, summary="High issue"),
            DetectionResult.issue_found("test2", severity=40, summary="Medium issue"),
            DetectionResult.no_issue("test3"),
        ]
        summary = engine.summarize(results)

        assert "severity" in summary
        assert "level" in summary
        assert "confidence" in summary
        assert "issues_found" in summary
        assert summary["issues_found"] == 2
        assert summary["total_detectors"] == 3
        assert "should_alert" in summary
        assert "should_block" in summary
        assert "top_issues" in summary
