"""Token Generator with Session Scoping.

Generates unique, session-scoped tokens to replace PII values.
Token format: [{TYPE}:{SESSION_PREFIX}:{RANDOM}]

Example:
    generator = TokenGenerator(session_id="abc123")
    token = generator.generate("EMAIL", "john@example.com")
    # "[EMAIL:abc1:a3f8c2d1]"
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypedDict


class TokenInfo(TypedDict):
    """Information stored about a generated token."""

    token: str
    pii_type: str
    original_value: str
    session_id: str
    created_at: str
    value_hash: str


@dataclass
class TokenGenerator:
    """Generates session-scoped tokens for PII replacement.

    Token Format: [{TYPE}:{SESSION_PREFIX}:{RANDOM}]

    - TYPE: The PII type (EMAIL, SSN, PHONE, etc.)
    - SESSION_PREFIX: First 4 chars of session ID (for session correlation)
    - RANDOM: 8-char random hex for uniqueness

    The generator maintains a local cache to ensure:
    - Same value in same session gets same token (consistency)
    - Collision detection and regeneration

    Example:
        generator = TokenGenerator(session_id="abc123def456")

        # Same value returns same token
        t1 = generator.generate("EMAIL", "john@example.com")
        t2 = generator.generate("EMAIL", "john@example.com")
        assert t1 == t2  # True

        # Different values get different tokens
        t3 = generator.generate("EMAIL", "jane@example.com")
        assert t1 != t3  # True
    """

    session_id: str
    _session_prefix: str = field(default="", init=False)
    _token_cache: dict[str, TokenInfo] = field(default_factory=dict, init=False)
    _value_to_token: dict[str, str] = field(default_factory=dict, init=False)
    _collision_retries: int = 10

    def __post_init__(self) -> None:
        """Initialize session prefix from session ID."""
        self._session_prefix = self._compute_session_prefix(self.session_id)

    def _compute_session_prefix(self, session_id: str) -> str:
        """Compute a 4-character session prefix.

        Uses first 4 chars of session ID, or hash if session ID is too short.

        Args:
            session_id: The full session ID.

        Returns:
            4-character prefix.
        """
        if len(session_id) >= 4:
            # Use first 4 alphanumeric chars
            prefix_chars = [c for c in session_id if c.isalnum()][:4]
            if len(prefix_chars) >= 4:
                return "".join(prefix_chars).lower()

        # Fallback: hash the session ID
        hash_hex = hashlib.sha256(session_id.encode()).hexdigest()
        return hash_hex[:4]

    def _compute_value_hash(self, pii_type: str, value: str) -> str:
        """Compute a hash of the PII type and value for cache lookup.

        Args:
            pii_type: The PII type.
            value: The original value.

        Returns:
            Hex digest of the hash.
        """
        combined = f"{pii_type}:{value}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _generate_random_suffix(self) -> str:
        """Generate an 8-character random hex suffix.

        Returns:
            8-character random hex string.
        """
        return secrets.token_hex(4)  # 4 bytes = 8 hex chars

    def generate(
        self,
        pii_type: str,
        original_value: str,
        *,
        force_new: bool = False,
    ) -> str:
        """Generate a token for a PII value.

        If the same value was previously tokenized in this session,
        returns the existing token for consistency.

        Args:
            pii_type: The type of PII (EMAIL, SSN, etc.)
            original_value: The original PII value to tokenize.
            force_new: If True, always generate a new token (for collision handling).

        Returns:
            The token string in format [{TYPE}:{SESSION_PREFIX}:{RANDOM}]
        """
        # PIIType subclasses str, but Enum.__str__ renders it as
        # "PIIType.EMAIL". Normalize enum-like values to their wire value so
        # generated tokens remain parseable and detokenizable.
        normalized_type = str(getattr(pii_type, "value", pii_type))

        # Check cache first (unless forcing new)
        value_hash = self._compute_value_hash(normalized_type, original_value)

        if not force_new and value_hash in self._value_to_token:
            return self._value_to_token[value_hash]

        # Generate new token with collision detection
        for attempt in range(self._collision_retries):
            random_suffix = self._generate_random_suffix()
            token = f"[{normalized_type}:{self._session_prefix}:{random_suffix}]"

            # Check for collision (same token already exists for different value)
            if token not in self._token_cache:
                # Store in caches
                token_info: TokenInfo = {
                    "token": token,
                    "pii_type": normalized_type,
                    "original_value": original_value,
                    "session_id": self.session_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "value_hash": value_hash,
                }
                self._token_cache[token] = token_info
                self._value_to_token[value_hash] = token
                return token

        # Exhausted retries - extremely unlikely
        raise RuntimeError(
            f"Failed to generate unique token after {self._collision_retries} attempts"
        )

    def get_token_info(self, token: str) -> TokenInfo | None:
        """Get information about a token.

        Args:
            token: The token to look up.

        Returns:
            TokenInfo if found, None otherwise.
        """
        return self._token_cache.get(token)

    def get_all_tokens(self) -> dict[str, TokenInfo]:
        """Get all tokens generated in this session.

        Returns:
            Dictionary of token -> TokenInfo.
        """
        return self._token_cache.copy()

    def get_token_count(self) -> int:
        """Get the number of tokens generated in this session.

        Returns:
            Number of tokens.
        """
        return len(self._token_cache)

    def clear_cache(self) -> None:
        """Clear the token cache.

        Warning: This will cause the same values to get different tokens
        if tokenized again in the same session.
        """
        self._token_cache.clear()
        self._value_to_token.clear()


class TokenParser:
    """Parses tokens to extract their components.

    Example:
        parser = TokenParser()
        components = parser.parse("[EMAIL:abc1:a3f8c2d1]")
        # {"pii_type": "EMAIL", "session_prefix": "abc1", "random": "a3f8c2d1"}
    """

    # Token pattern: [{TYPE}:{SESSION_PREFIX}:{RANDOM}]
    TOKEN_PATTERN = r"\[([A-Z_]+):([a-z0-9]{4}):([a-f0-9]{8})\]"

    def __init__(self) -> None:
        """Initialize the parser."""
        import re

        self._pattern = re.compile(self.TOKEN_PATTERN)

    def parse(self, token: str) -> dict[str, str] | None:
        """Parse a token into its components.

        Args:
            token: The token string to parse.

        Returns:
            Dictionary with pii_type, session_prefix, random, or None if invalid.
        """
        match = self._pattern.fullmatch(token)
        if not match:
            return None

        return {
            "pii_type": match.group(1),
            "session_prefix": match.group(2),
            "random": match.group(3),
        }

    def is_valid_token(self, token: str) -> bool:
        """Check if a string is a valid token format.

        Args:
            token: The string to check.

        Returns:
            True if valid token format.
        """
        return self._pattern.fullmatch(token) is not None

    def extract_tokens(self, text: str) -> list[str]:
        """Extract all tokens from a text string.

        Args:
            text: The text to search.

        Returns:
            List of token strings found.
        """
        return [match.group(0) for match in self._pattern.finditer(text)]

    def get_session_prefix(self, token: str) -> str | None:
        """Extract the session prefix from a token.

        Args:
            token: The token to parse.

        Returns:
            Session prefix or None if invalid.
        """
        parsed = self.parse(token)
        return parsed["session_prefix"] if parsed else None

    def get_pii_type(self, token: str) -> str | None:
        """Extract the PII type from a token.

        Args:
            token: The token to parse.

        Returns:
            PII type or None if invalid.
        """
        parsed = self.parse(token)
        return parsed["pii_type"] if parsed else None
