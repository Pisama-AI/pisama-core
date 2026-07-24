"""Conversation trace abstraction for multi-turn agent interactions.

This module provides a conversation-aware trace representation that supports:
- Multi-turn conversations from MAST-Data benchmarks
- User/agent/system/tool message exchanges
- Accumulated context tracking across turns
- Conversion to UniversalSpan for detection compatibility
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from pisama_core.ingestion.universal_trace import SpanStatus, SpanType, UniversalSpan


@dataclass
class ConversationTurnData:
    """Single turn in a conversation trace.

    Represents one message exchange in a multi-turn conversation,
    tracking the participant, content, and accumulated context.
    """

    # Turn identification
    turn_id: str
    turn_number: int

    # Participant info
    role: str  # user, agent, system, tool
    participant_id: str

    # Content
    content: str
    timestamp: Optional[datetime] = None

    # Tool-specific (for tool role)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None

    # Token tracking
    token_count: int = 0
    accumulated_tokens: int = 0

    # Computed fields
    content_hash: Optional[str] = None

    # Extra data
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]

        if not self.token_count and self.content:
            # Rough approximation: ~4 chars per token
            self.token_count = len(self.content) // 4

    def to_universal_span(self, trace_id: str) -> UniversalSpan:
        """Convert to UniversalSpan for detection compatibility."""
        # Determine span type based on role
        if self.role == "tool":
            span_type = SpanType.TOOL_CALL
        elif self.role == "agent":
            span_type = SpanType.AGENT
        elif self.role == "system":
            span_type = SpanType.CHAIN
        else:
            span_type = SpanType.UNKNOWN

        return UniversalSpan(
            id=self.turn_id,
            trace_id=trace_id,
            name=f"turn:{self.turn_number}:{self.role}",
            span_type=span_type,
            status=SpanStatus.OK,
            start_time=self.timestamp or datetime.now(timezone.utc),
            agent_id=self.participant_id,
            agent_name=self.participant_id,
            input_data={"role": self.role, "turn_number": self.turn_number},
            output_data={"content": self.content[:1000]},  # Truncate for span
            prompt=self.content if self.role == "user" else None,
            response=self.content if self.role == "agent" else None,
            tokens_total=self.token_count,
            tool_name=self.tool_calls[0].get("name") if self.tool_calls else None,
            tool_args=self.tool_calls[0].get("args") if self.tool_calls else None,
            source_format="conversation",
            metadata=self.extra,
        )


@dataclass
class ConversationTrace:
    """Full conversation with multi-turn support.

    Represents a complete multi-turn conversation between users and agents,
    supporting various conversation formats (MAST, OpenAI, Claude, etc.).
    """

    # Identification
    trace_id: str
    conversation_id: str
    framework: str

    # Turns
    turns: List[ConversationTurnData] = field(default_factory=list)

    # Aggregates
    total_turns: int = 0
    total_tokens: int = 0
    participants: List[str] = field(default_factory=list)

    # Source info
    source_format: str = "unknown"  # mast, openai, claude, generic

    # Extra data (annotations, ground truth, etc.)
    extra: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: ConversationTurnData) -> None:
        """Add turn and update accumulated context."""
        # Assign turn number if not set
        if turn.turn_number == 0:
            turn.turn_number = len(self.turns) + 1

        # Compute accumulated tokens
        prev_tokens = self.turns[-1].accumulated_tokens if self.turns else 0
        turn.accumulated_tokens = prev_tokens + turn.token_count

        self.turns.append(turn)
        self.total_turns = len(self.turns)
        self.total_tokens = turn.accumulated_tokens

        # Track participants
        if turn.participant_id not in self.participants:
            self.participants.append(turn.participant_id)

    def get_context_up_to_turn(self, turn_number: int) -> str:
        """Get accumulated context up to a specific turn.

        Args:
            turn_number: Turn number (1-indexed)

        Returns:
            Formatted conversation context
        """
        context_parts = []
        for t in self.turns[:turn_number]:
            context_parts.append(f"[{t.role}:{t.participant_id}]\n{t.content}")
        return "\n\n".join(context_parts)

    def get_turns_by_role(self, role: str) -> List[ConversationTurnData]:
        """Get all turns for a specific role."""
        return [t for t in self.turns if t.role == role]

    def get_agent_turns(self) -> List[ConversationTurnData]:
        """Get all agent response turns."""
        return self.get_turns_by_role("agent")

    def get_user_turns(self) -> List[ConversationTurnData]:
        """Get all user input turns."""
        return self.get_turns_by_role("user")

    def get_initial_task(self) -> Optional[str]:
        """Extract the initial task/prompt from the conversation."""
        for turn in self.turns[:3]:
            if turn.role in ("user", "system") and len(turn.content) > 20:
                return turn.content
        return None

    def iter_turn_pairs(self) -> Iterator[tuple[ConversationTurnData, ConversationTurnData]]:
        """Iterate over consecutive turn pairs for analysis."""
        for i in range(len(self.turns) - 1):
            yield self.turns[i], self.turns[i + 1]

    def to_universal_spans(self) -> List[UniversalSpan]:
        """Convert to UniversalSpan list for backward compatibility."""
        return [turn.to_universal_span(self.trace_id) for turn in self.turns]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "framework": self.framework,
            "total_turns": self.total_turns,
            "total_tokens": self.total_tokens,
            "participants": self.participants,
            "source_format": self.source_format,
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "turn_number": t.turn_number,
                    "role": t.role,
                    "participant_id": t.participant_id,
                    "content": t.content,
                    "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                    "tool_calls": t.tool_calls,
                    "tool_results": t.tool_results,
                    "token_count": t.token_count,
                    "accumulated_tokens": t.accumulated_tokens,
                    "content_hash": t.content_hash,
                    "extra": t.extra,
                }
                for t in self.turns
            ],
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTrace":
        """Create from dictionary."""
        trace = cls(
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            conversation_id=data.get("conversation_id", str(uuid.uuid4())),
            framework=data.get("framework", "unknown"),
            source_format=data.get("source_format", "unknown"),
            extra=data.get("extra", {}),
        )

        for turn_data in data.get("turns", []):
            timestamp = turn_data.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            turn = ConversationTurnData(
                turn_id=turn_data.get("turn_id", str(uuid.uuid4())),
                turn_number=turn_data.get("turn_number", 0),
                role=turn_data.get("role", "unknown"),
                participant_id=turn_data.get("participant_id", "unknown"),
                content=turn_data.get("content", ""),
                timestamp=timestamp,
                tool_calls=turn_data.get("tool_calls"),
                tool_results=turn_data.get("tool_results"),
                token_count=turn_data.get("token_count", 0),
                extra=turn_data.get("extra", {}),
            )
            trace.add_turn(turn)

        return trace
