"""Framework-agnostic trace abstractions for Pisama.

`UniversalTrace`/`UniversalSpan` and `ConversationTrace` were originally
backend-only modules. They live here now so the OSS detectors and the
standalone CLI can consume them without depending on the backend tree.

The backend's `app.ingestion.universal_trace` and
`app.ingestion.conversation_trace` modules re-export from here for
backwards compatibility.
"""

from pisama_core.ingestion.conversation_trace import (
    ConversationTrace,
    ConversationTurnData,
)
from pisama_core.ingestion.universal_trace import (
    SpanStatus,
    SpanType,
    UniversalSpan,
    UniversalTrace,
)

__all__ = [
    "SpanType",
    "SpanStatus",
    "UniversalSpan",
    "UniversalTrace",
    "ConversationTrace",
    "ConversationTurnData",
]
