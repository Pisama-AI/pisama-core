"""Base adapter interface for platform integrations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pisama_core.traces.enums import Platform
from pisama_core.traces.models import Span


class InjectionMethod(str, Enum):
    """Method used for fix injection."""

    STDERR = "stderr"  # Print to stderr
    RESOURCE = "resource"  # Update MCP resource
    STATE = "state"  # Modify agent state
    MESSAGE = "message"  # Inject message
    CALLBACK = "callback"  # Trigger callback
    WEBHOOK = "webhook"  # Send webhook


@dataclass
class InjectionResult:
    """Result of a fix injection attempt."""

    success: bool
    method: InjectionMethod
    message: Optional[str] = None
    directive_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # For blocking
    blocked: bool = False
    exit_code: Optional[int] = None

    # Error info
    error: Optional[str] = None


class PlatformAdapter(ABC):
    """Abstract base class for platform adapters.

    Each supported platform (Claude Code, LangGraph, AutoGen, CrewAI, n8n)
    must implement this interface to integrate with pisama-core.

    The adapter is responsible for:
    1. Converting platform-specific data to universal Span format
    2. Injecting fix directives into the platform
    3. Querying platform state
    4. Blocking actions when needed
    """

    @property
    @abstractmethod
    def platform_name(self) -> Platform:
        """The platform this adapter supports."""
        ...

    @property
    def platform_version(self) -> Optional[str]:
        """Version of the platform integration."""
        return None

    # ─────────────────────────────────────────────────────────────
    # TRACE CAPTURE
    # ─────────────────────────────────────────────────────────────

    @abstractmethod
    def capture_span(self, raw_data: Any) -> Span:
        """Convert platform-specific data to a universal Span.

        Args:
            raw_data: Platform-specific data (hook data, callback data, etc.)

        Returns:
            A Span representing the operation.
        """
        ...

    # ─────────────────────────────────────────────────────────────
    # FIX INJECTION
    # ─────────────────────────────────────────────────────────────

    @abstractmethod
    def inject_fix(
        self,
        directive: str,
        level: "EnforcementLevel",
        directive_id: Optional[str] = None,
    ) -> InjectionResult:
        """Inject a fix directive into the platform.

        Args:
            directive: The formatted fix directive text
            level: The enforcement level
            directive_id: Optional ID for tracking compliance

        Returns:
            Result of the injection attempt.
        """
        ...

    @abstractmethod
    def get_supported_injection_methods(self) -> list[InjectionMethod]:
        """Get the injection methods this platform supports."""
        ...

    # ─────────────────────────────────────────────────────────────
    # STATE & CONTEXT
    # ─────────────────────────────────────────────────────────────

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get the current platform state.

        Returns:
            Dictionary containing relevant state information.
        """
        ...

    def get_session_context(self) -> dict[str, Any]:
        """Get session context for analysis.

        Override this to provide additional context like recent
        tool calls, conversation history, etc.
        """
        return {}

    # ─────────────────────────────────────────────────────────────
    # BLOCKING
    # ─────────────────────────────────────────────────────────────

    @abstractmethod
    def can_block(self) -> bool:
        """Whether this platform supports blocking actions.

        Returns:
            True if the adapter can prevent actions from executing.
        """
        ...

    @abstractmethod
    def block_action(self, reason: str) -> bool:
        """Block the current action.

        Args:
            reason: Why the action is being blocked

        Returns:
            True if the block was successful.
        """
        ...

    # ─────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────

    def format_message(self, message: str, severity: int) -> str:
        """Format a message for this platform.

        Override to customize message formatting.
        """
        return message

    def supports_realtime(self) -> bool:
        """Whether this platform supports real-time detection.

        Returns:
            True if detection can happen before actions execute.
        """
        return self.can_block()


# Import here to avoid circular imports
from pisama_core.injection.enforcement import EnforcementLevel  # noqa: E402
