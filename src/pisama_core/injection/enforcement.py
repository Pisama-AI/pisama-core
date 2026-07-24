"""Enforcement levels and engine for fix injection."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class EnforcementLevel(IntEnum):
    """Levels of enforcement for fix injection."""

    SUGGEST = 1  # Soft suggestion
    DIRECT = 2  # Direct instruction
    BLOCK = 3  # Block until compliance
    TERMINATE = 4  # Force terminate

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class EnforcementState:
    """Current enforcement state for a session."""

    level: EnforcementLevel = EnforcementLevel.SUGGEST
    violations: int = 0
    compliances: int = 0
    pending_directives: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)


class EnforcementEngine:
    """Engine for managing enforcement levels and escalation.

    The engine tracks compliance and escalates enforcement
    when directives are ignored.
    """

    def __init__(
        self,
        initial_level: EnforcementLevel = EnforcementLevel.SUGGEST,
        max_violations_before_escalation: int = 3,
    ) -> None:
        self.initial_level = initial_level
        self.max_violations = max_violations_before_escalation
        self._states: dict[str, EnforcementState] = {}

    def get_state(self, session_id: str) -> EnforcementState:
        """Get or create state for a session."""
        if session_id not in self._states:
            self._states[session_id] = EnforcementState(level=self.initial_level)
        return self._states[session_id]

    def get_level(self, severity: int, session_id: str) -> EnforcementLevel:
        """Get enforcement level based on severity and history.

        Args:
            severity: Current severity score
            session_id: Session identifier

        Returns:
            Appropriate enforcement level
        """
        state = self.get_state(session_id)

        # Base level from severity
        if severity >= 80:
            base_level = EnforcementLevel.BLOCK
        elif severity >= 60:
            base_level = EnforcementLevel.DIRECT
        elif severity >= 40:
            base_level = EnforcementLevel.SUGGEST
        else:
            return EnforcementLevel.SUGGEST

        # Escalate based on violations
        if state.violations >= self.max_violations:
            return EnforcementLevel(min(base_level + 1, EnforcementLevel.TERMINATE))

        return base_level

    def record_violation(self, session_id: str, tool_name: str = "") -> EnforcementLevel:
        """Record a violation (directive ignored).

        Args:
            session_id: Session identifier
            tool_name: Optional tool that violated

        Returns:
            New enforcement level
        """
        state = self.get_state(session_id)
        state.violations += 1

        if tool_name and tool_name not in state.blocked_tools:
            state.blocked_tools.append(tool_name)

        # Escalate if too many violations
        if state.violations >= self.max_violations:
            state.level = EnforcementLevel(min(state.level + 1, EnforcementLevel.TERMINATE))

        return state.level

    def record_compliance(self, session_id: str, directive_id: str) -> EnforcementLevel:
        """Record compliance with a directive.

        Args:
            session_id: Session identifier
            directive_id: ID of the directive that was followed

        Returns:
            New enforcement level (may de-escalate)
        """
        state = self.get_state(session_id)
        state.compliances += 1

        if directive_id in state.pending_directives:
            state.pending_directives.remove(directive_id)

        # De-escalate if no pending directives
        if not state.pending_directives and state.level > self.initial_level:
            state.level = EnforcementLevel(max(state.level - 1, self.initial_level))
            state.blocked_tools = []  # Clear blocks

        return state.level

    def add_directive(self, session_id: str, directive_id: str) -> None:
        """Add a pending directive."""
        state = self.get_state(session_id)
        if directive_id not in state.pending_directives:
            state.pending_directives.append(directive_id)

    def should_block(self, session_id: str, tool_name: str) -> tuple[bool, str]:
        """Check if a tool should be blocked.

        Args:
            session_id: Session identifier
            tool_name: Name of the tool

        Returns:
            Tuple of (should_block, reason)
        """
        state = self.get_state(session_id)

        if state.level >= EnforcementLevel.BLOCK:
            if state.level == EnforcementLevel.TERMINATE:
                return True, "Session terminated due to non-compliance"
            if tool_name in state.blocked_tools:
                return True, f"Tool '{tool_name}' blocked. Follow pending directives."
            if state.pending_directives:
                return True, "Pending directives must be addressed first"

        return False, ""

    def reset(self, session_id: str) -> None:
        """Reset state for a session."""
        if session_id in self._states:
            del self._states[session_id]

    def get_stats(self, session_id: str) -> dict[str, Any]:
        """Get enforcement statistics for a session."""
        state = self.get_state(session_id)
        return {
            "level": str(state.level),
            "violations": state.violations,
            "compliances": state.compliances,
            "pending_directives": len(state.pending_directives),
            "blocked_tools": state.blocked_tools,
        }
