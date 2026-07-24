"""Unified Tokenizer - High-level API for PII Tokenization.

Combines detection, token generation, and vault storage into a
single easy-to-use interface for trace data tokenization.

Example:
    tokenizer = Tokenizer(session_id="abc123")

    # Tokenize a string
    result = tokenizer.tokenize_string("Contact john@example.com")
    # "Contact [EMAIL:abc1:a3f8c2d1]"

    # Tokenize trace data
    trace = {"input": "User SSN: 123-45-6789", "output": "Processed"}
    tokenized = tokenizer.tokenize_dict(trace)

    # Detokenize (for debugging with proper authorization)
    original = tokenizer.detokenize_string(result, reason="INC-123", principal="alice@example.com")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pisama_core.tokenization.detector import PIIDetector, PIIMatch, PIIPattern
from pisama_core.tokenization.generator import TokenGenerator, TokenParser
from pisama_core.tokenization.keychain import KeychainManager
from pisama_core.tokenization.vault import TokenVault


@dataclass
class TokenizationResult:
    """Result of tokenizing a string."""

    original: str
    tokenized: str
    tokens_created: int
    pii_detected: list[PIIMatch]
    processing_time_ms: float


@dataclass
class TokenizationStats:
    """Statistics from tokenization operations."""

    total_tokenized: int = 0
    tokens_by_type: dict[str, int] = field(default_factory=dict)
    fields_tokenized: int = 0
    errors: int = 0


class Tokenizer:
    """High-level tokenizer combining detection, generation, and storage.

    This is the main entry point for tokenizing trace data. It:
    1. Detects PII using configurable patterns
    2. Generates session-scoped tokens for each PII instance
    3. Stores token-to-value mappings in encrypted vault
    4. Supports detokenization for authorized debugging

    Example:
        # Initialize for a session
        tokenizer = Tokenizer(
            session_id="abc123",
            vault_path="~/.claude/pisama/vault.db",
        )

        # Tokenize trace data
        trace_data = {
            "tool": "Bash",
            "input": "echo 'User email: john@example.com'",
            "output": "User email: john@example.com",
        }

        tokenized = tokenizer.tokenize_dict(trace_data)
        # {
        #     "tool": "Bash",
        #     "input": "echo 'User email: [EMAIL:abc1:x]'",
        #     "output": "User email: [EMAIL:abc1:x]",
        # }

        # Later: detokenize for debugging (with audit log)
        original = tokenizer.detokenize_dict(tokenized, reason="Investigating bug")
    """

    DEFAULT_VAULT_PATH = Path.home() / ".claude" / "pisama" / "vault.db"

    def __init__(
        self,
        session_id: str,
        vault_path: str | Path | None = None,
        detector: PIIDetector | None = None,
        enabled: bool = True,
        fail_open: bool = True,
    ) -> None:
        """Initialize the tokenizer.

        Args:
            session_id: Unique session identifier for token scoping.
            vault_path: Path to the vault database. Defaults to ~/.claude/pisama/vault.db
            detector: Custom PIIDetector instance. Defaults to standard detector.
            enabled: Whether tokenization is enabled. If False, data passes through unchanged.
            fail_open: If True, failures result in pass-through. If False, failures raise errors.
        """
        self.session_id = session_id
        self.enabled = enabled
        self.fail_open = fail_open

        # Initialize components
        self._detector = detector or PIIDetector()
        self._generator = TokenGenerator(session_id=session_id)
        self._parser = TokenParser()

        # Vault and keychain (lazy initialization)
        self._vault_path = Path(vault_path or self.DEFAULT_VAULT_PATH)
        self._vault: TokenVault | None = None
        self._keychain: KeychainManager | None = None
        self._encryption_key: bytes | None = None

        # Stats
        self._stats = TokenizationStats()

    def _ensure_vault(self) -> tuple[TokenVault, bytes] | tuple[None, None]:
        """Ensure vault and encryption key are initialized.

        Returns:
            Tuple of (vault, key) or (None, None) if unavailable.
        """
        if self._vault is not None and self._encryption_key is not None:
            return self._vault, self._encryption_key

        try:
            # Initialize keychain
            if self._keychain is None:
                self._keychain = KeychainManager(allow_file_fallback=True)

            # Get or create encryption key
            self._encryption_key = self._keychain.get_or_create_key()

            # Initialize vault
            self._vault = TokenVault(self._vault_path)
            self._vault.initialize()

            return self._vault, self._encryption_key

        except Exception:
            if self.fail_open:
                return None, None
            raise

    def add_pattern(self, pattern: PIIPattern) -> None:
        """Add a custom PII detection pattern.

        Args:
            pattern: The pattern to add.
        """
        self._detector.add_pattern(pattern)

    def add_exclusion(self, value: str) -> None:
        """Add a value to exclude from detection.

        Args:
            value: The value to exclude.
        """
        self._detector.add_exclusion(value)

    def add_sensitive_field(self, field_name: str) -> None:
        """Mark a field name as sensitive (always tokenize its value).

        Args:
            field_name: The field name to mark.
        """
        self._detector.add_sensitive_field(field_name)

    def tokenize_string(self, text: str) -> str:
        """Tokenize PII in a string.

        Args:
            text: The text to tokenize.

        Returns:
            Text with PII replaced by tokens.
        """
        if not self.enabled:
            return text

        try:
            # Detect PII
            matches = self._detector.detect(text)

            if not matches:
                return text

            # Get vault and key
            vault, key = self._ensure_vault()

            # Replace PII with tokens (process in reverse order to preserve positions)
            result = text
            for match in reversed(matches):
                # Generate token
                token = self._generator.generate(match.pii_type, match.value)

                # Store in vault if available
                if vault and key:
                    vault.store(
                        match.pii_type,
                        token,
                        match.value,
                        self.session_id,
                        key,
                    )

                # Replace in text
                result = result[: match.start] + token + result[match.end :]

                # Update stats
                self._stats.total_tokenized += 1
                self._stats.tokens_by_type[match.pii_type] = (
                    self._stats.tokens_by_type.get(match.pii_type, 0) + 1
                )

            return result

        except Exception:
            self._stats.errors += 1
            if self.fail_open:
                return text
            raise

    def tokenize_dict(
        self,
        data: dict[str, Any],
        fields_to_tokenize: list[str] | None = None,
    ) -> dict[str, Any]:
        """Tokenize PII in a dictionary structure.

        Args:
            data: The dictionary to tokenize.
            fields_to_tokenize: Optional list of field names to tokenize.
                               If None, tokenizes all string fields.

        Returns:
            Dictionary with PII tokenized.
        """
        if not self.enabled:
            return data

        try:
            return self._tokenize_value(data, fields_to_tokenize)
        except Exception:
            self._stats.errors += 1
            if self.fail_open:
                return data
            raise

    def _tokenize_value(
        self,
        value: Any,
        fields_to_tokenize: list[str] | None,
        current_field: str = "",
    ) -> Any:
        """Recursively tokenize a value."""
        if isinstance(value, str):
            # Check if this field should be tokenized
            if fields_to_tokenize is None or current_field in fields_to_tokenize:
                return self.tokenize_string(value)
            return value

        elif isinstance(value, dict):
            return {k: self._tokenize_value(v, fields_to_tokenize, k) for k, v in value.items()}

        elif isinstance(value, list):
            return [self._tokenize_value(item, fields_to_tokenize, current_field) for item in value]

        else:
            return value

    def detokenize_string(
        self,
        text: str,
        reason: str,
        principal: str,
        ticket: str | None = None,
    ) -> str:
        """Detokenize a string (reveal original PII values).

        Args:
            text: Text containing tokens.
            reason: Justification for detokenization (logged for audit).
            principal: Authenticated user/service performing the detokenization
                (logged for audit; pass the request user's email/id from the
                API context, not an OS env var).
            ticket: Optional incident ticket reference.

        Returns:
            Text with tokens replaced by original values.
        """
        vault, key = self._ensure_vault()

        if vault is None or key is None:
            # No vault available - tokens cannot be resolved
            return text

        # Find all tokens in text
        token_pattern = re.compile(r"\[[A-Z_]+:[a-z0-9]{4}:[a-f0-9]{8}\]")
        matches = list(token_pattern.finditer(text))

        if not matches:
            return text

        # Log detokenization request (audit)
        self._log_detokenization(
            tokens=[m.group() for m in matches],
            reason=reason,
            principal=principal,
            ticket=ticket,
        )

        # Replace tokens with original values
        result = text
        for match in reversed(matches):
            token = match.group()
            original = vault.retrieve(token, key)

            if original:
                result = result[: match.start()] + original + result[match.end() :]

        return result

    def detokenize_dict(
        self,
        data: dict[str, Any],
        reason: str,
        principal: str,
        ticket: str | None = None,
    ) -> dict[str, Any]:
        """Detokenize a dictionary structure.

        Args:
            data: Dictionary containing tokens.
            reason: Justification for detokenization.
            principal: Authenticated user/service performing the detokenization.
            ticket: Optional incident ticket reference.

        Returns:
            Dictionary with tokens replaced by original values.
        """
        return self._detokenize_value(data, reason, principal, ticket)

    def _detokenize_value(
        self,
        value: Any,
        reason: str,
        principal: str,
        ticket: str | None,
    ) -> Any:
        """Recursively detokenize a value."""
        if isinstance(value, str):
            return self.detokenize_string(value, reason, principal, ticket)
        elif isinstance(value, dict):
            return {
                k: self._detokenize_value(v, reason, principal, ticket) for k, v in value.items()
            }
        elif isinstance(value, list):
            return [self._detokenize_value(item, reason, principal, ticket) for item in value]
        else:
            return value

    def _log_detokenization(
        self,
        tokens: list[str],
        reason: str,
        principal: str,
        ticket: str | None,
    ) -> None:
        """Log a detokenization request to the audit log."""
        audit_log_path = self._vault_path.parent / "audit_log.jsonl"

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "detokenize",
            "session_id": self.session_id,
            "principal": principal,
            "tokens_count": len(tokens),
            "reason": reason,
            "ticket": ticket,
        }

        try:
            audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Don't fail on audit log errors

    def contains_pii(self, text: str) -> bool:
        """Quick check if text contains any PII.

        Args:
            text: The text to check.

        Returns:
            True if PII is detected.
        """
        return self._detector.contains_pii(text)

    def get_stats(self) -> TokenizationStats:
        """Get tokenization statistics.

        Returns:
            TokenizationStats with counts.
        """
        return self._stats

    def get_vault_stats(self) -> dict | None:
        """Get vault statistics.

        Returns:
            Vault stats dictionary or None if vault unavailable.
        """
        vault, _ = self._ensure_vault()
        if vault:
            return vault.get_stats()
        return None

    def close(self) -> None:
        """Close the tokenizer and release resources."""
        if self._vault:
            self._vault.close()
            self._vault = None


# Convenience function for one-off tokenization
def tokenize_trace_data(
    data: dict[str, Any],
    session_id: str,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Convenience function for tokenizing trace data.

    Args:
        data: The trace data dictionary.
        session_id: The session ID for token scoping.
        fields: Optional list of field names to tokenize.

    Returns:
        Tokenized trace data.

    Example:
        trace = {"input": "Email: john@example.com", "output": "Done"}
        tokenized = tokenize_trace_data(trace, "session123")
    """
    tokenizer = Tokenizer(session_id=session_id)
    try:
        return tokenizer.tokenize_dict(data, fields)
    finally:
        tokenizer.close()
