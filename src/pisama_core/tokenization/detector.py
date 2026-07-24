"""PII Detection Module with Configurable Patterns.

Detects Personally Identifiable Information (PII) in trace data using
configurable regex patterns. Supports built-in patterns for common PII
types and custom patterns.

Example:
    detector = PIIDetector()
    matches = detector.detect("Contact john@example.com for SSN 123-45-6789")
    # [PIIMatch(type='EMAIL', value='john@example.com', ...),
    #  PIIMatch(type='SSN', value='123-45-6789', ...)]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Pattern


class PIIType(str, Enum):
    """Standard PII types with built-in detection patterns."""

    SSN = "SSN"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CC"
    API_KEY = "API_KEY"
    AWS_KEY = "AWS_KEY"
    JWT = "JWT"
    IP_ADDRESS = "IP"
    CUSTOM = "CUSTOM"


@dataclass
class PIIMatch:
    """A detected PII match in text."""

    pii_type: str
    value: str
    start: int
    end: int
    pattern_name: str

    def __repr__(self) -> str:
        return f"PIIMatch({self.pii_type}: '{self.value[:20]}...' at {self.start}-{self.end})"


@dataclass
class PIIPattern:
    """A pattern for detecting PII."""

    name: str
    pii_type: str
    pattern: str
    description: str = ""
    enabled: bool = True
    _compiled: Pattern[str] | None = field(default=None, repr=False, compare=False)

    @property
    def compiled(self) -> Pattern[str]:
        """Get compiled regex pattern with lazy initialization."""
        if self._compiled is None:
            self._compiled = re.compile(self.pattern)
        return self._compiled


# Built-in patterns matching the skill documentation
DEFAULT_PATTERNS: list[PIIPattern] = [
    PIIPattern(
        name="SSN",
        pii_type=PIIType.SSN,
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        description="US Social Security Number (123-45-6789)",
    ),
    PIIPattern(
        name="EMAIL",
        pii_type=PIIType.EMAIL,
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        description="Email addresses",
    ),
    PIIPattern(
        name="PHONE",
        pii_type=PIIType.PHONE,
        pattern=r"\b(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
        description="US/International phone numbers",
    ),
    PIIPattern(
        name="CREDIT_CARD",
        pii_type=PIIType.CREDIT_CARD,
        pattern=r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        description="Credit card numbers (with or without separators)",
    ),
    PIIPattern(
        name="API_KEY",
        pii_type=PIIType.API_KEY,
        pattern=r"\b(sk-|api_|key_|token_|secret_)[a-zA-Z0-9]{16,}\b",
        description="Common API key prefixes (sk-, api_, key_, token_, secret_)",
    ),
    PIIPattern(
        name="AWS_KEY",
        pii_type=PIIType.AWS_KEY,
        pattern=r"\bAKIA[0-9A-Z]{16}\b",
        description="AWS Access Key IDs",
    ),
    PIIPattern(
        name="JWT",
        pii_type=PIIType.JWT,
        pattern=r"\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b",
        description="JSON Web Tokens",
    ),
    PIIPattern(
        name="IP_ADDRESS",
        pii_type=PIIType.IP_ADDRESS,
        pattern=r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        description="IPv4 addresses",
    ),
]

# Default exclusions - known safe values that should not be tokenized
DEFAULT_EXCLUSIONS: list[str] = [
    "127.0.0.1",
    "0.0.0.0",
    "localhost",
    "example.com",
    "test@example.com",
    "user@example.com",
]


class PIIDetector:
    """Detects PII in text using configurable patterns.

    Supports:
    - Built-in patterns for common PII types (SSN, email, phone, etc.)
    - Custom patterns added at runtime
    - Exclusion patterns for known safe values
    - Field-name based detection (always tokenize certain field values)

    Example:
        detector = PIIDetector()

        # Add custom pattern
        detector.add_pattern(PIIPattern(
            name="EMPLOYEE_ID",
            pii_type="EMPLOYEE_ID",
            pattern=r"\\bEMP-\\d{6}\\b",
        ))

        # Detect PII
        matches = detector.detect("Contact EMP-123456 at john@acme.com")
    """

    def __init__(
        self,
        patterns: list[PIIPattern] | None = None,
        exclusions: list[str] | None = None,
        sensitive_fields: list[str] | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
            patterns: List of patterns to use. Defaults to DEFAULT_PATTERNS.
            exclusions: List of values to exclude from detection.
            sensitive_fields: Field names that should always have values tokenized.
        """
        self._patterns: dict[str, PIIPattern] = {}
        configured_exclusions = DEFAULT_EXCLUSIONS if exclusions is None else exclusions
        configured_sensitive_fields = (
            ["password", "secret", "api_key", "token", "credential", "ssn"]
            if sensitive_fields is None
            else sensitive_fields
        )
        self._exclusions: set[str] = set(configured_exclusions)
        self._sensitive_fields: set[str] = set(configured_sensitive_fields)

        # Load default or custom patterns
        for pattern in DEFAULT_PATTERNS if patterns is None else patterns:
            self._patterns[pattern.name] = pattern

    @property
    def patterns(self) -> dict[str, PIIPattern]:
        """Get all registered patterns."""
        return self._patterns.copy()

    @property
    def exclusions(self) -> set[str]:
        """Get all exclusion values."""
        return self._exclusions.copy()

    @property
    def sensitive_fields(self) -> set[str]:
        """Get all sensitive field names."""
        return self._sensitive_fields.copy()

    def add_pattern(self, pattern: PIIPattern) -> None:
        """Add a new PII detection pattern.

        Args:
            pattern: The pattern to add.
        """
        self._patterns[pattern.name] = pattern

    def remove_pattern(self, name: str) -> bool:
        """Remove a pattern by name.

        Args:
            name: The pattern name to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self._patterns:
            del self._patterns[name]
            return True
        return False

    def disable_pattern(self, name: str) -> bool:
        """Disable a pattern without removing it.

        Args:
            name: The pattern name to disable.

        Returns:
            True if disabled, False if not found.
        """
        if name in self._patterns:
            self._patterns[name].enabled = False
            return True
        return False

    def enable_pattern(self, name: str) -> bool:
        """Enable a previously disabled pattern.

        Args:
            name: The pattern name to enable.

        Returns:
            True if enabled, False if not found.
        """
        if name in self._patterns:
            self._patterns[name].enabled = True
            return True
        return False

    def add_exclusion(self, value: str) -> None:
        """Add a value to exclude from detection.

        Args:
            value: The value to exclude.
        """
        self._exclusions.add(value)

    def remove_exclusion(self, value: str) -> bool:
        """Remove an exclusion value.

        Args:
            value: The value to remove from exclusions.

        Returns:
            True if removed, False if not found.
        """
        if value in self._exclusions:
            self._exclusions.remove(value)
            return True
        return False

    def add_sensitive_field(self, field_name: str) -> None:
        """Add a field name that should always have values tokenized.

        Args:
            field_name: The field name to mark as sensitive.
        """
        self._sensitive_fields.add(field_name.lower())

    def is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field name is marked as sensitive.

        Args:
            field_name: The field name to check.

        Returns:
            True if the field is sensitive.
        """
        return field_name.lower() in self._sensitive_fields

    def _is_excluded(self, value: str) -> bool:
        """Check if a value should be excluded from tokenization.

        Args:
            value: The detected value.

        Returns:
            True if the value should be excluded.
        """
        # Direct match
        if value in self._exclusions:
            return True

        # Partial match for domain patterns (e.g., @mycompany.internal)
        for exclusion in self._exclusions:
            if exclusion.startswith("@") and value.endswith(exclusion):
                return True

        return False

    def detect(self, text: str) -> list[PIIMatch]:
        """Detect all PII in the given text.

        Args:
            text: The text to scan for PII.

        Returns:
            List of PIIMatch objects for each detected PII.
        """
        matches: list[PIIMatch] = []

        for pattern in self._patterns.values():
            if not pattern.enabled:
                continue

            for match in pattern.compiled.finditer(text):
                value = match.group()

                # Skip excluded values
                if self._is_excluded(value):
                    continue

                matches.append(
                    PIIMatch(
                        pii_type=pattern.pii_type,
                        value=value,
                        start=match.start(),
                        end=match.end(),
                        pattern_name=pattern.name,
                    )
                )

        # Sort by position and deduplicate overlapping matches
        matches.sort(key=lambda m: (m.start, -m.end))
        return self._deduplicate_overlapping(matches)

    def _deduplicate_overlapping(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """Remove overlapping matches, keeping the longer one.

        Args:
            matches: Sorted list of matches.

        Returns:
            Deduplicated list of matches.
        """
        if not matches:
            return matches

        result: list[PIIMatch] = []
        prev = matches[0]

        for current in matches[1:]:
            # If current overlaps with previous
            if current.start < prev.end:
                # Keep the longer match
                if (current.end - current.start) > (prev.end - prev.start):
                    prev = current
            else:
                result.append(prev)
                prev = current

        result.append(prev)
        return result

    def detect_in_dict(
        self,
        data: dict,
        path: str = "",
    ) -> list[tuple[str, list[PIIMatch]]]:
        """Recursively detect PII in a dictionary structure.

        Args:
            data: The dictionary to scan.
            path: Current path in the structure (for nested dicts).

        Returns:
            List of (path, matches) tuples for each field containing PII.
        """
        results: list[tuple[str, list[PIIMatch]]] = []

        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, str):
                # Check if this is a sensitive field (always tokenize)
                if self.is_sensitive_field(key):
                    # Create a synthetic match for the entire value
                    results.append(
                        (
                            current_path,
                            [
                                PIIMatch(
                                    pii_type="SENSITIVE_FIELD",
                                    value=value,
                                    start=0,
                                    end=len(value),
                                    pattern_name=f"field:{key}",
                                )
                            ],
                        )
                    )
                else:
                    # Normal pattern detection
                    matches = self.detect(value)
                    if matches:
                        results.append((current_path, matches))

            elif isinstance(value, dict):
                results.extend(self.detect_in_dict(value, current_path))

            elif isinstance(value, list):
                for i, item in enumerate(value):
                    item_path = f"{current_path}[{i}]"
                    if isinstance(item, str):
                        matches = self.detect(item)
                        if matches:
                            results.append((item_path, matches))
                    elif isinstance(item, dict):
                        results.extend(self.detect_in_dict(item, item_path))

        return results

    def contains_pii(self, text: str) -> bool:
        """Quick check if text contains any PII.

        Args:
            text: The text to check.

        Returns:
            True if any PII is detected.
        """
        for pattern in self._patterns.values():
            if not pattern.enabled:
                continue
            match = pattern.compiled.search(text)
            if match and not self._is_excluded(match.group()):
                return True
        return False

    def get_pattern_stats(self) -> dict[str, dict]:
        """Get statistics about patterns.

        Returns:
            Dictionary of pattern names to their stats.
        """
        return {
            name: {
                "pii_type": p.pii_type,
                "enabled": p.enabled,
                "description": p.description,
            }
            for name, p in self._patterns.items()
        }
