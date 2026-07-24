"""Tests for pisama_core.healing module."""

import pytest

from pisama_core.detection.result import DetectionResult, FixRecommendation, FixType
from pisama_core.healing.base import BaseFix
from pisama_core.healing.engine import HealingEngine
from pisama_core.healing.models import FixContext, FixResult, HealingPlan
from pisama_core.traces.enums import Platform


class TestFixContext:
    """Tests for FixContext model."""

    def test_create_context(self):
        """Test basic context creation."""
        context = FixContext(
            platform=Platform.CLAUDE_CODE,
            session_id="session-1",
        )
        assert context.session_id == "session-1"
        assert context.platform == Platform.CLAUDE_CODE

    def test_context_with_state(self):
        """Test context with current state."""
        context = FixContext(
            platform=Platform.CLAUDE_CODE,
            session_id="session-1",
            current_state={"tool_count": 10, "loop_detected": True},
        )
        assert context.current_state["tool_count"] == 10


class TestFixResult:
    """Tests for FixResult model."""

    def test_create_success_result(self):
        """Test creating successful fix result."""
        result = FixResult(
            success=True,
            fix_type="break_loop",
            message="Loop broken successfully",
        )
        assert result.success is True
        assert result.fix_type == "break_loop"

    def test_create_failure_result(self):
        """Test creating failed fix result."""
        result = FixResult(
            success=False,
            fix_type="break_loop",
            message="Failed to break loop",
            error="Permission denied",
        )
        assert result.success is False
        assert result.error == "Permission denied"


class TestHealingPlan:
    """Tests for HealingPlan model."""

    def test_create_plan(self):
        """Test creating healing plan."""
        primary = FixRecommendation(
            fix_type=FixType.BREAK_LOOP,
            instruction="Break the loop",
        )
        plan = HealingPlan(
            primary_fix=primary,
            fallback_fixes=[
                FixRecommendation(
                    fix_type=FixType.SWITCH_STRATEGY, instruction="Try different approach"
                ),
            ],
        )
        assert plan.primary_fix.fix_type == FixType.BREAK_LOOP
        assert len(plan.fallback_fixes) == 1

    def test_plan_to_dict(self):
        """Test plan serialization."""
        primary = FixRecommendation(
            fix_type=FixType.BREAK_LOOP,
            instruction="Break the loop",
        )
        plan = HealingPlan(primary_fix=primary)
        data = plan.to_dict()
        assert "primary_fix" in data
        assert data["primary_fix"]["fix_type"] == "break_loop"


class SimpleTestFix(BaseFix):
    """Simple fix for testing."""

    name = "test_fix"
    description = "A test fix"

    async def can_apply(self, context: FixContext) -> bool:
        """Check if fix can be applied."""
        return True

    async def apply(self, context: FixContext) -> FixResult:
        """Apply the fix."""
        return FixResult(
            success=True,
            fix_type=self.name,
            message="Test fix applied",
        )


class TestBaseFix:
    """Tests for BaseFix."""

    def test_fix_attributes(self):
        """Test fix has required attributes."""
        fix = SimpleTestFix()
        assert fix.name == "test_fix"
        assert fix.requires_approval is True
        assert fix.reversible is True

    @pytest.mark.asyncio
    async def test_apply_fix(self):
        """Test applying fix."""
        fix = SimpleTestFix()
        context = FixContext(platform=Platform.CLAUDE_CODE, session_id="sess-1")

        fix_result = await fix.apply(context)
        assert fix_result.success is True


class TestHealingEngine:
    """Tests for HealingEngine."""

    def test_create_engine(self):
        """Test creating healing engine."""
        engine = HealingEngine()
        assert engine is not None

    def test_analyze_detection_result(self):
        """Test analyzing detection result for healing plan."""
        engine = HealingEngine()
        result = DetectionResult.issue_found(
            detector_name="loop",
            severity=60,
            summary="Loop detected",
            fix_type=FixType.BREAK_LOOP,
            fix_instruction="Break the loop",
        )

        plan = engine.analyze(result)
        assert plan is not None
        assert plan.primary_fix is not None

    def test_analyze_no_issue_returns_escalate(self):
        """Test that no-issue result returns escalate plan."""
        engine = HealingEngine()
        result = DetectionResult.no_issue("test")

        plan = engine.analyze(result)
        # When no specific fix, should recommend escalate
        assert plan.primary_fix.fix_type == FixType.ESCALATE

    def test_register_fix(self):
        """Test registering a fix handler."""
        engine = HealingEngine()
        fix = SimpleTestFix()
        engine.register_fix(fix)

        assert engine.get_fix("test_fix") is fix

    @pytest.mark.asyncio
    async def test_heal_with_plan(self):
        """Test healing with a plan."""
        engine = HealingEngine()
        fix = SimpleTestFix()
        engine.register_fix(fix)

        primary = FixRecommendation(
            fix_type=FixType.BREAK_LOOP,
            instruction="Break the loop",
        )
        plan = HealingPlan(primary_fix=primary)
        context = FixContext(platform=Platform.CLAUDE_CODE, session_id="sess-1")

        # Note: This would try to find a fix for "break_loop" which may not be our test_fix
        # So we register our fix with the proper name
        engine._fixes["break_loop"] = fix

        fix_result = await engine.heal(plan, context)
        assert fix_result.success is True
