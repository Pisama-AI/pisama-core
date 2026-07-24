"""Pisama Tokenization Module - PII Protection for Traces.

This module provides PII tokenization for agent traces:
- PII detection with configurable patterns
- Session-scoped token generation
- Encrypted vault storage (AES-256-GCM)
- OS keychain integration for key management

Example:
    from pisama_core.tokenization import (
        PIIDetector,
        TokenGenerator,
        TokenVault,
        KeychainManager,
    )

    # Detect PII
    detector = PIIDetector()
    matches = detector.detect("Contact john@example.com")

    # Generate tokens
    generator = TokenGenerator(session_id="abc123")
    token = generator.generate("EMAIL", "john@example.com")

    # Store in vault
    keychain = KeychainManager()
    key = keychain.get_or_create_key()

    vault = TokenVault("~/.claude/pisama/vault.db")
    vault.initialize()
    vault.store("EMAIL", token, "john@example.com", "abc123", key)
"""

# Detection
from pisama_core.tokenization.detector import (
    DEFAULT_EXCLUSIONS,
    DEFAULT_PATTERNS,
    PIIDetector,
    PIIMatch,
    PIIPattern,
    PIIType,
)

# Token generation
from pisama_core.tokenization.generator import (
    TokenGenerator,
    TokenInfo,
    TokenParser,
)

# Keychain
from pisama_core.tokenization.keychain import (
    FileBackend,
    KeychainError,
    KeychainManager,
    KeychainResult,
    KeychainUnavailableError,
    LinuxSecretService,
    MacOSKeychain,
)

# Unified tokenizer
from pisama_core.tokenization.tokenizer import (
    TokenizationResult,
    TokenizationStats,
    Tokenizer,
    tokenize_trace_data,
)

# Vault storage
from pisama_core.tokenization.vault import (
    EncryptedValue,
    TokenRecord,
    TokenVault,
    derive_key_from_password,
)

__all__ = [
    # Detection
    "PIIDetector",
    "PIIMatch",
    "PIIPattern",
    "PIIType",
    "DEFAULT_PATTERNS",
    "DEFAULT_EXCLUSIONS",
    # Token generation
    "TokenGenerator",
    "TokenParser",
    "TokenInfo",
    # Vault storage
    "TokenVault",
    "TokenRecord",
    "EncryptedValue",
    "derive_key_from_password",
    # Keychain
    "KeychainManager",
    "KeychainResult",
    "KeychainError",
    "KeychainUnavailableError",
    "MacOSKeychain",
    "LinuxSecretService",
    "FileBackend",
    # Unified tokenizer
    "Tokenizer",
    "TokenizationResult",
    "TokenizationStats",
    "tokenize_trace_data",
]
