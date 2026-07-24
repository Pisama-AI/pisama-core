"""Switch strategy fix implementation."""

from pisama_core.healing.base import BaseFix
from pisama_core.healing.models import FixContext, FixResult


class SwitchStrategyFix(BaseFix):
    """Fix that suggests switching to a different strategy."""

    name = "switch_strategy"
    description = "Switch to a different approach for solving the problem"
    requires_approval = False
    reversible = True
    risk_level = "low"

    async def can_apply(self, context: FixContext) -> bool:
        """Can always apply switch_strategy."""
        return True

    async def apply(self, context: FixContext) -> FixResult:
        """Apply the switch strategy fix."""
        instruction = (
            "Your current approach may not be working optimally. "
            "Consider trying a different strategy:\n"
            "1. If searching, try different search terms or locations\n"
            "2. If executing, try a simpler or more direct approach\n"
            "3. If stuck, ask the user for more context\n"
        )

        if context.adapter:
            from pisama_core.injection.enforcement import EnforcementLevel

            context.adapter.inject_fix(
                directive=instruction,
                level=EnforcementLevel.SUGGEST,
            )

        return FixResult(
            success=True,
            fix_type=self.name,
            message="Strategy switch suggested",
            changes_made=["Suggested alternative approaches"],
        )

    def get_instruction(self, context=None) -> str:
        return (
            "Consider switching to a different strategy. Your current approach may not be optimal."
        )
