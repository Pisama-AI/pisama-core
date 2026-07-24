"""Break loop fix implementation."""

from pisama_core.healing.base import BaseFix
from pisama_core.healing.models import FixContext, FixResult


class BreakLoopFix(BaseFix):
    """Fix that breaks infinite loops and stuck patterns."""

    name = "break_loop"
    description = "Stop the current loop and try a different approach"
    requires_approval = False  # Can be auto-applied
    reversible = False
    risk_level = "low"

    async def can_apply(self, context: FixContext) -> bool:
        """Can always apply break_loop."""
        return True

    async def apply(self, context: FixContext) -> FixResult:
        """Apply the break loop fix."""
        instruction = (
            "STOP the current approach. You have been repeating the same action. "
            "Take a step back and try a completely different strategy, or ask "
            "the user for clarification on what they want to achieve."
        )

        # If adapter available, inject the instruction
        if context.adapter:
            from pisama_core.injection.enforcement import EnforcementLevel

            context.adapter.inject_fix(
                directive=instruction,
                level=EnforcementLevel.DIRECT,
            )

        return FixResult(
            success=True,
            fix_type=self.name,
            message="Loop break instruction issued",
            changes_made=["Issued instruction to stop loop"],
        )

    def get_instruction(self, context=None) -> str:
        return (
            "STOP the current loop. You have been repeating similar actions. "
            "Try a different approach or ask the user for guidance."
        )
