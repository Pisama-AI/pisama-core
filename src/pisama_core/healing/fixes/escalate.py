"""Escalate fix implementation."""

from pisama_core.healing.base import BaseFix
from pisama_core.healing.models import FixContext, FixResult


class EscalateFix(BaseFix):
    """Fix that escalates to user for manual intervention."""

    name = "escalate"
    description = "Escalate to user for guidance"
    requires_approval = False  # Escalation doesn't need approval
    reversible = True
    risk_level = "low"

    async def can_apply(self, context: FixContext) -> bool:
        """Can always escalate."""
        return True

    async def apply(self, context: FixContext) -> FixResult:
        """Apply the escalate fix."""
        instruction = (
            "I've encountered a situation that needs your input. "
            "Please provide guidance on how to proceed, or let me know "
            "if you'd like me to try a different approach."
        )

        if context.adapter:
            from pisama_core.injection.enforcement import EnforcementLevel

            context.adapter.inject_fix(
                directive=instruction,
                level=EnforcementLevel.DIRECT,
            )

        return FixResult(
            success=True,
            fix_type=self.name,
            message="Escalated to user",
            changes_made=["Requested user guidance"],
        )

    def get_instruction(self, context=None) -> str:
        return "This situation requires user input. Ask the user for guidance on how to proceed."
