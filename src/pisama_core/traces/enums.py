"""Enumerations for trace models."""

from enum import Enum


class Platform(str, Enum):
    """Supported agent platforms."""

    CLAUDE_CODE = "claude_code"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    AUTOGEN = "autogen"
    CREWAI = "crewai"
    N8N = "n8n"
    OPENCLAW = "openclaw"
    DIFY = "dify"
    MOLTBOT = "moltbot"
    AUTORESEARCH = "autoresearch"
    MANAGED_AGENTS = "managed_agents"
    OPENAI = "openai"
    CODEX = "codex"
    BEDROCK = "bedrock"
    GEMINI = "gemini"
    GOOGLE_ADK = "google_adk"
    GENERIC = "generic"

    def __str__(self) -> str:
        return self.value


class SpanKind(str, Enum):
    """Type of span in the trace."""

    # Agent-level spans
    AGENT = "agent"  # An autonomous agent
    AGENT_TURN = "agent_turn"  # Single turn in agent conversation

    # Task-level spans
    TASK = "task"  # A discrete task
    WORKFLOW = "workflow"  # Multi-step workflow
    CHAIN = "chain"  # LangChain/LangGraph chain

    # Action-level spans
    TOOL = "tool"  # Tool/function call
    LLM = "llm"  # LLM inference call
    RETRIEVAL = "retrieval"  # RAG/retrieval operation

    # Communication spans
    MESSAGE = "message"  # Agent-to-agent message
    HANDOFF = "handoff"  # Task handoff between agents

    # System spans
    SYSTEM = "system"  # System-level operation
    HOOK = "hook"  # Hook execution

    # Interaction spans
    USER_INPUT = "user_input"  # User interaction/input
    USER_OUTPUT = "user_output"  # Output to user

    def __str__(self) -> str:
        return self.value


class SpanStatus(str, Enum):
    """Status of a span."""

    UNSET = "unset"  # Not yet determined
    IN_PROGRESS = "in_progress"  # Currently executing
    OK = "ok"  # Completed successfully
    ERROR = "error"  # Failed with error
    TIMEOUT = "timeout"  # Timed out
    CANCELLED = "cancelled"  # Was cancelled
    BLOCKED = "blocked"  # Blocked by intervention

    def __str__(self) -> str:
        return self.value

    @property
    def is_terminal(self) -> bool:
        """Whether this status is terminal (span is complete)."""
        return self not in (SpanStatus.UNSET, SpanStatus.IN_PROGRESS)

    @property
    def is_success(self) -> bool:
        """Whether this status indicates success."""
        return self == SpanStatus.OK

    @property
    def is_failure(self) -> bool:
        """Whether this status indicates failure."""
        return self in (SpanStatus.ERROR, SpanStatus.TIMEOUT, SpanStatus.BLOCKED)
