"""Time utilities."""

from datetime import datetime, timezone


def now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def parse_iso_datetime(s: str) -> datetime:
    """Parse ISO 8601 timestamps consistently on Python 3.10 and newer."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
