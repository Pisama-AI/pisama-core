"""Healing engine for coordinating fix application."""

import time
from typing import Any, Optional

from pisama_core.detection.result import DetectionResult, FixRecommendation, FixType
from pisama_core.healing.base import BaseFix
from pisama_core.healing.models import FixContext, FixResult, HealingPlan


class HealingEngine:
    """Orchestrates healing by analyzing issues and applying fixes.

    Example:
        engine = HealingEngine()

        # Analyze what fix is needed
        plan = engine.analyze(detection_result)

        # Apply the fix
        result = await engine.heal(plan, context)
    """

    def __init__(self) -> None:
        self._fixes: dict[str, BaseFix] = {}
        self._register_builtin_fixes()

    def _register_builtin_fixes(self) -> None:
        """Register built-in fixes."""
        from pisama_core.healing.fixes import (
            BreakLoopFix,
            EscalateFix,
            SwitchStrategyFix,
        )

        self.register_fix(BreakLoopFix())
        self.register_fix(SwitchStrategyFix())
        self.register_fix(EscalateFix())

    def register_fix(self, fix: BaseFix) -> None:
        """Register a fix implementation."""
        self._fixes[fix.name] = fix

    def get_fix(self, name: str) -> Optional[BaseFix]:
        """Get a fix by name."""
        return self._fixes.get(name)

    def analyze(self, result: DetectionResult) -> HealingPlan:
        """Analyze a detection result and create a healing plan.

        Args:
            result: The detection result

        Returns:
            A healing plan
        """
        if not result.detected or not result.recommendation:
            return HealingPlan(
                primary_fix=FixRecommendation(
                    fix_type=FixType.ESCALATE,
                    instruction="No specific fix available. Escalate to user.",
                ),
                requires_approval=True,
                detection_summary="No specific issue detected",
                severity=result.severity,
            )

        primary = result.recommendation
        fallbacks = result.alternative_recommendations

        # Determine approval requirement
        requires_approval = primary.requires_approval
        if result.severity >= 60:
            requires_approval = True  # Always approve critical fixes

        return HealingPlan(
            primary_fix=primary,
            fallback_fixes=fallbacks,
            requires_approval=requires_approval,
            estimated_impact=self._estimate_impact(primary.fix_type),
            detection_summary=result.summary,
            severity=result.severity,
        )

    async def heal(self, plan: HealingPlan, context: FixContext) -> FixResult:
        """Execute a healing plan.

        Args:
            plan: The healing plan
            context: The fix context

        Returns:
            Result of the healing attempt
        """
        fix_type = plan.primary_fix.fix_type
        fix = self.get_fix(str(fix_type))

        if not fix:
            return FixResult(
                success=False,
                fix_type=str(fix_type),
                message=f"No fix implementation for {fix_type}",
                error="Fix not found",
            )

        # Check if fix can be applied
        if not await fix.can_apply(context):
            # Try fallbacks
            for fallback in plan.fallback_fixes:
                fallback_fix = self.get_fix(str(fallback.fix_type))
                if fallback_fix and await fallback_fix.can_apply(context):
                    fix = fallback_fix
                    break
            else:
                return FixResult(
                    success=False,
                    fix_type=str(fix_type),
                    message="Cannot apply fix in current context",
                    error="Fix not applicable",
                )

        # Apply the fix
        start = time.perf_counter()
        result = await fix.apply(context)
        result.execution_time_ms = (time.perf_counter() - start) * 1000

        return result

    def _estimate_impact(self, fix_type: FixType) -> str:
        """Estimate the impact of a fix type."""
        high_impact = [FixType.TERMINATE, FixType.ROLLBACK]
        medium_impact = [FixType.RESET_CONTEXT, FixType.SWITCH_STRATEGY]

        if fix_type in high_impact:
            return "high"
        elif fix_type in medium_impact:
            return "medium"
        return "low"

    def get_available_fixes(self) -> list[dict[str, Any]]:
        """Get information about available fixes."""
        return [
            {
                "name": fix.name,
                "description": fix.description,
                "requires_approval": fix.requires_approval,
                "reversible": fix.reversible,
                "risk_level": fix.risk_level,
            }
            for fix in self._fixes.values()
        ]
