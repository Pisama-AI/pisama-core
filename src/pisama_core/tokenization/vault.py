"""SQLite Vault with AES-256-GCM Encryption.

Securely stores token-to-value mappings with at-rest encryption.
Uses AES-256-GCM for authenticated encryption with:
- Unique IV per record
- Authentication tag for integrity
- Key derived from OS keychain

Example:
    vault = TokenVault(db_path="~/.claude/pisama/vault.db")
    vault.initialize(encryption_key)

    # Store a token mapping
    vault.store("EMAIL", "[EMAIL:abc1:a3f8c2d1]", "john@example.com", "session123")

    # Retrieve original value
    value = vault.retrieve("[EMAIL:abc1:a3f8c2d1]", encryption_key)
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from pisama_core.utils._secure_files import ensure_owner_only_file

# Use cryptography library for AES-256-GCM
# This is a soft dependency - vault degrades gracefully without it
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class EncryptedValue(NamedTuple):
    """An encrypted value with IV and auth tag."""

    ciphertext: bytes
    iv: bytes
    tag: bytes


@dataclass
class TokenRecord:
    """A stored token record from the vault."""

    token: str
    pii_type: str
    session_id: str
    created_at: datetime
    encrypted_value: bytes
    iv: bytes
    value_hash: str


class TokenVault:
    """Secure SQLite vault for token-to-value mappings.

    Features:
    - AES-256-GCM encryption for values at rest
    - Unique IV per record
    - Value hashing for lookup without decryption
    - Session-based organization
    - GDPR-compliant deletion

    Schema:
        tokens:
            - id: INTEGER PRIMARY KEY
            - token: TEXT UNIQUE (the token string)
            - pii_type: TEXT (EMAIL, SSN, etc.)
            - encrypted_value: BLOB (AES-256-GCM encrypted)
            - iv: BLOB (12-byte initialization vector)
            - value_hash: TEXT (SHA-256 of original value for dedup)
            - session_id: TEXT (session this token belongs to)
            - created_at: TEXT (ISO timestamp)

    Example:
        vault = TokenVault("~/.claude/pisama/vault.db")
        vault.initialize(key)

        # Store and retrieve
        vault.store("EMAIL", "[EMAIL:abc1:x]", "john@example.com", "sess1", key)
        value = vault.retrieve("[EMAIL:abc1:x]", key)
    """

    # AES-256-GCM parameters
    KEY_SIZE = 32  # 256 bits
    IV_SIZE = 12  # 96 bits (GCM standard)
    TAG_SIZE = 16  # 128 bits (GCM standard)

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the vault.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path).expanduser()
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    def _ensure_dir(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._ensure_dir()
            # Create the vault with owner-only permissions before SQLite opens
            # it. A later chmod leaves a race window where encrypted records and
            # identifying metadata can be read by another local user.
            ensure_owner_only_file(self.db_path)
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialize(self) -> None:
        """Initialize the vault database schema.

        Creates the tokens table if it doesn't exist.
        """
        conn = self._get_connection()

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT UNIQUE NOT NULL,
                pii_type TEXT NOT NULL,
                encrypted_value BLOB NOT NULL,
                iv BLOB NOT NULL,
                value_hash TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tokens_session
                ON tokens(session_id);

            CREATE INDEX IF NOT EXISTS idx_tokens_type
                ON tokens(pii_type);

            CREATE INDEX IF NOT EXISTS idx_tokens_hash
                ON tokens(value_hash);

            CREATE INDEX IF NOT EXISTS idx_tokens_created
                ON tokens(created_at);
            """
        )

        conn.commit()
        self._initialized = True

    def _encrypt_value(self, value: str, key: bytes) -> EncryptedValue:
        """Encrypt a value using AES-256-GCM.

        Args:
            value: The plaintext value to encrypt.
            key: 32-byte encryption key.

        Returns:
            EncryptedValue with ciphertext, IV, and tag.

        Raises:
            RuntimeError: If cryptography library is not available.
        """
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError(
                "Encryption requires the 'cryptography' package. "
                "Install with: pip install cryptography"
            )

        # Generate random IV
        iv = os.urandom(self.IV_SIZE)

        # Encrypt with AES-256-GCM
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(iv, value.encode("utf-8"), None)

        # GCM appends the tag to the ciphertext
        # Split them for storage
        actual_ciphertext = ciphertext[: -self.TAG_SIZE]
        tag = ciphertext[-self.TAG_SIZE :]

        return EncryptedValue(
            ciphertext=actual_ciphertext,
            iv=iv,
            tag=tag,
        )

    def _decrypt_value(
        self,
        encrypted: bytes,
        iv: bytes,
        key: bytes,
    ) -> str:
        """Decrypt a value using AES-256-GCM.

        Args:
            encrypted: The ciphertext (without tag).
            iv: The initialization vector.
            key: 32-byte encryption key.

        Returns:
            The decrypted plaintext value.

        Raises:
            RuntimeError: If cryptography library is not available.
            ValueError: If decryption fails (wrong key or corrupted data).
        """
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError(
                "Decryption requires the 'cryptography' package. "
                "Install with: pip install cryptography"
            )

        # The stored format has ciphertext + tag concatenated
        aesgcm = AESGCM(key)

        try:
            plaintext = aesgcm.decrypt(iv, encrypted, None)
            return plaintext.decode("utf-8")
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}") from e

    def _hash_value(self, value: str) -> str:
        """Hash a value for lookup/deduplication.

        Args:
            value: The value to hash.

        Returns:
            Hex digest of SHA-256 hash.
        """
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def store(
        self,
        pii_type: str,
        token: str,
        original_value: str,
        session_id: str,
        key: bytes,
    ) -> bool:
        """Store a token-to-value mapping.

        Args:
            pii_type: The PII type (EMAIL, SSN, etc.)
            token: The generated token.
            original_value: The original PII value to encrypt.
            session_id: The session ID this token belongs to.
            key: 32-byte encryption key.

        Returns:
            True if stored successfully, False if token already exists.
        """
        if not self._initialized:
            self.initialize()

        # Encrypt the value
        encrypted = self._encrypt_value(original_value, key)

        # Combine ciphertext and tag for storage
        encrypted_blob = encrypted.ciphertext + encrypted.tag

        # Hash the value for deduplication lookup
        value_hash = self._hash_value(original_value)

        conn = self._get_connection()

        try:
            conn.execute(
                """
                INSERT INTO tokens
                    (token, pii_type, encrypted_value, iv, value_hash, session_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token,
                    pii_type,
                    encrypted_blob,
                    encrypted.iv,
                    value_hash,
                    session_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            return True

        except sqlite3.IntegrityError:
            # Token already exists
            return False

    def retrieve(self, token: str, key: bytes) -> str | None:
        """Retrieve the original value for a token.

        Args:
            token: The token to look up.
            key: 32-byte encryption key.

        Returns:
            The decrypted original value, or None if not found.
        """
        if not self._initialized:
            self.initialize()

        conn = self._get_connection()

        row = conn.execute(
            "SELECT encrypted_value, iv FROM tokens WHERE token = ?",
            (token,),
        ).fetchone()

        if row is None:
            return None

        encrypted_blob = row["encrypted_value"]
        iv = row["iv"]

        return self._decrypt_value(encrypted_blob, iv, key)

    def retrieve_batch(
        self,
        tokens: list[str],
        key: bytes,
    ) -> dict[str, str | None]:
        """Retrieve original values for multiple tokens.

        Args:
            tokens: List of tokens to look up.
            key: 32-byte encryption key.

        Returns:
            Dictionary of token -> value (or None if not found).
        """
        results: dict[str, str | None] = {}

        for token in tokens:
            results[token] = self.retrieve(token, key)

        return results

    def get_token_info(self, token: str) -> TokenRecord | None:
        """Get metadata about a token (without decrypting).

        Args:
            token: The token to look up.

        Returns:
            TokenRecord with metadata, or None if not found.
        """
        if not self._initialized:
            self.initialize()

        conn = self._get_connection()

        row = conn.execute(
            """
            SELECT token, pii_type, session_id, created_at,
                   encrypted_value, iv, value_hash
            FROM tokens WHERE token = ?
            """,
            (token,),
        ).fetchone()

        if row is None:
            return None

        return TokenRecord(
            token=row["token"],
            pii_type=row["pii_type"],
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            encrypted_value=row["encrypted_value"],
            iv=row["iv"],
            value_hash=row["value_hash"],
        )

    def find_by_value_hash(
        self,
        value_hash: str,
        session_id: str | None = None,
    ) -> list[str]:
        """Find tokens by value hash (for deduplication).

        Args:
            value_hash: SHA-256 hash of the original value.
            session_id: Optional session ID filter.

        Returns:
            List of matching tokens.
        """
        if not self._initialized:
            self.initialize()

        conn = self._get_connection()

        if session_id:
            rows = conn.execute(
                "SELECT token FROM tokens WHERE value_hash = ? AND session_id = ?",
                (value_hash, session_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT token FROM tokens WHERE value_hash = ?",
                (value_hash,),
            ).fetchall()

        return [row["token"] for row in rows]

    def list_session_tokens(self, session_id: str) -> list[str]:
        """List all tokens for a session.

        Args:
            session_id: The session ID.

        Returns:
            List of tokens in the session.
        """
        if not self._initialized:
            self.initialize()

        conn = self._get_connection()

        rows = conn.execute(
            "SELECT token FROM tokens WHERE session_id = ?",
            (session_id,),
        ).fetchall()

        return [row["token"] for row in rows]

    def delete_token(self, token: str) -> bool:
        """Delete a token from the vault.

        Args:
            token: The token to delete.

        Returns:
            True if deleted, False if not found.
        """
        if not self._initialized:
            self.initialize()

        conn = self._get_connection()

        cursor = conn.execute(
            "DELETE FROM tokens WHERE token = ?",
            (token,),
        )
        conn.commit()

        return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> int:
        """Delete all tokens for a session.

        Args:
            session_id: The session ID.

        Returns:
            Number of tokens deleted.
        """
        if not self._initialized:
            self.initialize()

        conn = self._get_connection()

        cursor = conn.execute(
            "DELETE FROM tokens WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()

        return cursor.rowcount

    def delete_by_value_hash(self, value_hash: str) -> int:
        """Delete all tokens for a specific value (GDPR erasure).

        Args:
            value_hash: SHA-256 hash of the value to delete.

        Returns:
            Number of tokens deleted.
        """
        if not self._initialized:
            self.initialize()

        conn = self._get_connection()

        cursor = conn.execute(
            "DELETE FROM tokens WHERE value_hash = ?",
            (value_hash,),
        )
        conn.commit()

        return cursor.rowcount

    def get_stats(self) -> dict:
        """Get vault statistics.

        Returns:
            Dictionary with counts and other stats.
        """
        if not self._initialized:
            self.initialize()

        conn = self._get_connection()

        total = conn.execute("SELECT COUNT(*) as c FROM tokens").fetchone()["c"]

        by_type = conn.execute(
            "SELECT pii_type, COUNT(*) as c FROM tokens GROUP BY pii_type"
        ).fetchall()

        sessions = conn.execute("SELECT COUNT(DISTINCT session_id) as c FROM tokens").fetchone()[
            "c"
        ]

        return {
            "total_tokens": total,
            "unique_sessions": sessions,
            "by_type": {row["pii_type"]: row["c"] for row in by_type},
            "db_path": str(self.db_path),
            "encryption_available": HAS_CRYPTOGRAPHY,
        }

    def vacuum(self) -> None:
        """Compact the database file."""
        if not self._initialized:
            return

        conn = self._get_connection()
        conn.execute("VACUUM")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "TokenVault":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()


def derive_key_from_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Derive a 256-bit key from a password using PBKDF2.

    Args:
        password: The password to derive from.
        salt: Optional salt (generated if not provided).

    Returns:
        Tuple of (key, salt).
    """
    if salt is None:
        salt = os.urandom(16)

    # Use PBKDF2 with SHA-256
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations=100000,  # OWASP recommended minimum
        dklen=32,  # 256 bits
    )

    return key, salt
