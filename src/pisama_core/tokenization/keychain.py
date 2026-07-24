"""OS Keychain Integration for Secure Key Storage.

Stores the vault encryption key in the OS keychain:
- macOS: Keychain Access (via security command)
- Linux: Secret Service API (via secretstorage or keyring)
- Fallback: File-based with warning

Example:
    keychain = KeychainManager()

    # Store key (first time setup)
    keychain.store_key(encryption_key)

    # Retrieve key
    key = keychain.get_key()
"""

from __future__ import annotations

import base64
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Service name for keychain storage
KEYCHAIN_SERVICE = "pisama-vault"
KEYCHAIN_ACCOUNT = "encryption-key"


class KeychainError(Exception):
    """Error interacting with the system keychain."""

    pass


class KeychainUnavailableError(KeychainError):
    """System keychain is not available."""

    pass


@dataclass
class KeychainResult:
    """Result of a keychain operation."""

    success: bool
    message: str
    backend: str


class KeychainBackend:
    """Abstract base for keychain backends."""

    name: str = "base"

    def is_available(self) -> bool:
        """Check if this backend is available."""
        return False

    def store_key(self, key: bytes) -> KeychainResult:
        """Store the encryption key."""
        raise NotImplementedError

    def get_key(self) -> bytes | None:
        """Retrieve the encryption key."""
        raise NotImplementedError

    def delete_key(self) -> KeychainResult:
        """Delete the encryption key."""
        raise NotImplementedError

    def key_exists(self) -> bool:
        """Check if a key is stored."""
        raise NotImplementedError


class MacOSKeychain(KeychainBackend):
    """macOS Keychain Access backend using security command."""

    name = "macos-keychain"

    def is_available(self) -> bool:
        """Check if running on macOS with security command."""
        if platform.system() != "Darwin":
            return False

        try:
            result = subprocess.run(
                ["which", "security"],
                capture_output=True,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def store_key(self, key: bytes) -> KeychainResult:
        """Store key in macOS Keychain."""
        # Encode key as base64 for storage
        encoded = base64.b64encode(key).decode("ascii")

        try:
            # First try to delete existing key (ignore errors)
            subprocess.run(
                [
                    "security",
                    "delete-generic-password",
                    "-s",
                    KEYCHAIN_SERVICE,
                    "-a",
                    KEYCHAIN_ACCOUNT,
                ],
                capture_output=True,
                check=False,
            )

            # Add new key
            subprocess.run(
                [
                    "security",
                    "add-generic-password",
                    "-s",
                    KEYCHAIN_SERVICE,
                    "-a",
                    KEYCHAIN_ACCOUNT,
                    "-w",
                    encoded,
                    "-U",  # Update if exists
                ],
                capture_output=True,
                check=True,
            )

            return KeychainResult(
                success=True,
                message="Key stored in macOS Keychain",
                backend=self.name,
            )

        except subprocess.CalledProcessError as e:
            return KeychainResult(
                success=False,
                message=f"Failed to store key: {e.stderr.decode()}",
                backend=self.name,
            )

    def get_key(self) -> bytes | None:
        """Retrieve key from macOS Keychain."""
        try:
            result = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s",
                    KEYCHAIN_SERVICE,
                    "-a",
                    KEYCHAIN_ACCOUNT,
                    "-w",  # Print password only
                ],
                capture_output=True,
                check=True,
            )

            encoded = result.stdout.decode("ascii").strip()
            return base64.b64decode(encoded)

        except subprocess.CalledProcessError:
            return None

    def delete_key(self) -> KeychainResult:
        """Delete key from macOS Keychain."""
        try:
            subprocess.run(
                [
                    "security",
                    "delete-generic-password",
                    "-s",
                    KEYCHAIN_SERVICE,
                    "-a",
                    KEYCHAIN_ACCOUNT,
                ],
                capture_output=True,
                check=True,
            )

            return KeychainResult(
                success=True,
                message="Key deleted from macOS Keychain",
                backend=self.name,
            )

        except subprocess.CalledProcessError as e:
            if b"could not be found" in e.stderr:
                return KeychainResult(
                    success=True,
                    message="Key not found (already deleted)",
                    backend=self.name,
                )
            return KeychainResult(
                success=False,
                message=f"Failed to delete key: {e.stderr.decode()}",
                backend=self.name,
            )

    def key_exists(self) -> bool:
        """Check if key exists in macOS Keychain."""
        return self.get_key() is not None


class LinuxSecretService(KeychainBackend):
    """Linux Secret Service backend using secretstorage library."""

    name = "linux-secret-service"

    def is_available(self) -> bool:
        """Check if running on Linux with secretstorage available."""
        if platform.system() != "Linux":
            return False

        try:
            import secretstorage  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_collection(self):
        """Get the default secret storage collection."""
        import secretstorage

        connection = secretstorage.dbus_init()
        collection = secretstorage.get_default_collection(connection)

        # Unlock if locked
        if collection.is_locked():
            collection.unlock()

        return collection, connection

    def store_key(self, key: bytes) -> KeychainResult:
        """Store key in Linux Secret Service."""
        try:
            collection, connection = self._get_collection()

            # Delete existing items first
            for item in collection.search_items(
                {"service": KEYCHAIN_SERVICE, "account": KEYCHAIN_ACCOUNT}
            ):
                item.delete()

            # Create new item
            collection.create_item(
                f"{KEYCHAIN_SERVICE} - {KEYCHAIN_ACCOUNT}",
                {"service": KEYCHAIN_SERVICE, "account": KEYCHAIN_ACCOUNT},
                key,
                replace=True,
            )

            return KeychainResult(
                success=True,
                message="Key stored in Linux Secret Service",
                backend=self.name,
            )

        except Exception as e:
            return KeychainResult(
                success=False,
                message=f"Failed to store key: {e}",
                backend=self.name,
            )

    def get_key(self) -> bytes | None:
        """Retrieve key from Linux Secret Service."""
        try:
            collection, connection = self._get_collection()

            items = list(
                collection.search_items({"service": KEYCHAIN_SERVICE, "account": KEYCHAIN_ACCOUNT})
            )

            if not items:
                return None

            return items[0].get_secret()

        except Exception:
            return None

    def delete_key(self) -> KeychainResult:
        """Delete key from Linux Secret Service."""
        try:
            collection, connection = self._get_collection()

            deleted = False
            for item in collection.search_items(
                {"service": KEYCHAIN_SERVICE, "account": KEYCHAIN_ACCOUNT}
            ):
                item.delete()
                deleted = True

            if deleted:
                return KeychainResult(
                    success=True,
                    message="Key deleted from Linux Secret Service",
                    backend=self.name,
                )
            else:
                return KeychainResult(
                    success=True,
                    message="Key not found (already deleted)",
                    backend=self.name,
                )

        except Exception as e:
            return KeychainResult(
                success=False,
                message=f"Failed to delete key: {e}",
                backend=self.name,
            )

    def key_exists(self) -> bool:
        """Check if key exists in Linux Secret Service."""
        return self.get_key() is not None


class FileBackend(KeychainBackend):
    """Fallback file-based storage — DEVELOPMENT ONLY.

    Stores the master key as plain Base64 with 0o600 perms. Anyone who
    can read the file (root, a compromised process running as the same
    user, a leaked dev backup) can decrypt the entire vault.

    This backend is ONLY suitable for local development without an OS
    keyring. Production deployments must use OSXKeychain, SecretService,
    or an explicit cloud KMS — never this fallback. Set
    ``PISAMA_ALLOW_INSECURE_FILE_KEYCHAIN=1`` to acknowledge the risk.
    """

    name = "file-fallback"

    def __init__(self, key_path: Path | None = None) -> None:
        """Initialize with optional custom path."""
        self.key_path = key_path or Path.home() / ".claude" / "pisama" / ".vault_key"

    def is_available(self) -> bool:
        """File backend is always available as fallback."""
        return True

    def store_key(self, key: bytes) -> KeychainResult:
        """Store key in file with restricted permissions."""
        if os.environ.get("PISAMA_ALLOW_INSECURE_FILE_KEYCHAIN") != "1":
            return KeychainResult(
                success=False,
                message=(
                    "Refusing to store master key in plaintext file — set "
                    "PISAMA_ALLOW_INSECURE_FILE_KEYCHAIN=1 to acknowledge "
                    "that this is dev-only and the vault is unencrypted "
                    "at rest. Production deployments must use a real "
                    "keyring backend (OSXKeychain, SecretService, KMS)."
                ),
                backend=self.name,
            )
        try:
            # Ensure directory exists
            self.key_path.parent.mkdir(parents=True, exist_ok=True)

            # Write key encoded as base64
            encoded = base64.b64encode(key).decode("ascii")
            self.key_path.write_text(encoded)

            # Restrict permissions (owner read/write only)
            os.chmod(self.key_path, 0o600)

            return KeychainResult(
                success=True,
                message=f"Key stored in file (INSECURE FALLBACK, dev-only): {self.key_path}",
                backend=self.name,
            )

        except Exception as e:
            return KeychainResult(
                success=False,
                message=f"Failed to store key: {e}",
                backend=self.name,
            )

    def get_key(self) -> bytes | None:
        """Retrieve key from file."""
        try:
            if not self.key_path.exists():
                return None

            encoded = self.key_path.read_text().strip()
            return base64.b64decode(encoded)

        except Exception:
            return None

    def delete_key(self) -> KeychainResult:
        """Delete key file."""
        try:
            if self.key_path.exists():
                self.key_path.unlink()
                return KeychainResult(
                    success=True,
                    message="Key file deleted",
                    backend=self.name,
                )
            else:
                return KeychainResult(
                    success=True,
                    message="Key file not found (already deleted)",
                    backend=self.name,
                )

        except Exception as e:
            return KeychainResult(
                success=False,
                message=f"Failed to delete key: {e}",
                backend=self.name,
            )

    def key_exists(self) -> bool:
        """Check if key file exists."""
        return self.key_path.exists()


class KeychainManager:
    """Manages encryption key storage across platforms.

    Automatically selects the best available backend:
    1. macOS Keychain (on macOS)
    2. Linux Secret Service (on Linux with secretstorage)
    3. File fallback (with security warning)

    Example:
        manager = KeychainManager()

        # First-time setup: generate and store key
        if not manager.key_exists():
            key = os.urandom(32)  # 256-bit key
            manager.store_key(key)

        # Retrieve key for use
        key = manager.get_key()
        if key is None:
            raise RuntimeError("No encryption key found")
    """

    def __init__(self, allow_file_fallback: bool = True) -> None:
        """Initialize the keychain manager.

        Args:
            allow_file_fallback: Whether to allow insecure file storage as fallback.
        """
        self._backends: list[KeychainBackend] = [
            MacOSKeychain(),
            LinuxSecretService(),
        ]

        if allow_file_fallback:
            self._backends.append(FileBackend())

        self._active_backend: KeychainBackend | None = None

    def _get_backend(self) -> KeychainBackend:
        """Get the best available backend."""
        if self._active_backend is not None:
            return self._active_backend

        for backend in self._backends:
            if backend.is_available():
                self._active_backend = backend
                return backend

        raise KeychainUnavailableError(
            "No keychain backend available. "
            "On Linux, install secretstorage: pip install secretstorage"
        )

    @property
    def backend_name(self) -> str:
        """Get the name of the active backend."""
        return self._get_backend().name

    def store_key(self, key: bytes) -> KeychainResult:
        """Store the encryption key.

        Args:
            key: 32-byte (256-bit) encryption key.

        Returns:
            KeychainResult with success status and message.
        """
        if len(key) != 32:
            return KeychainResult(
                success=False,
                message=f"Key must be 32 bytes, got {len(key)}",
                backend="validation",
            )

        return self._get_backend().store_key(key)

    def get_key(self) -> bytes | None:
        """Retrieve the encryption key.

        Returns:
            32-byte key or None if not found.
        """
        return self._get_backend().get_key()

    def delete_key(self) -> KeychainResult:
        """Delete the encryption key.

        Returns:
            KeychainResult with success status.
        """
        return self._get_backend().delete_key()

    def key_exists(self) -> bool:
        """Check if an encryption key is stored.

        Returns:
            True if key exists.
        """
        return self._get_backend().key_exists()

    def get_or_create_key(self) -> bytes:
        """Get existing key or create a new one.

        Returns:
            32-byte encryption key.

        Raises:
            KeychainError: If key cannot be stored.
        """
        key = self.get_key()
        if key is not None:
            return key

        # Generate new key
        key = os.urandom(32)

        result = self.store_key(key)
        if not result.success:
            raise KeychainError(f"Failed to store new key: {result.message}")

        return key

    def rotate_key(self) -> tuple[bytes, bytes]:
        """Rotate the encryption key.

        Returns:
            Tuple of (old_key, new_key).

        Raises:
            KeychainError: If key rotation fails.
        """
        old_key = self.get_key()
        if old_key is None:
            raise KeychainError("No existing key to rotate")

        # Generate new key
        new_key = os.urandom(32)

        # Store new key
        result = self.store_key(new_key)
        if not result.success:
            raise KeychainError(f"Failed to store rotated key: {result.message}")

        return old_key, new_key

    def get_status(self) -> dict:
        """Get keychain status information.

        Returns:
            Dictionary with backend info and key status.
        """
        try:
            backend = self._get_backend()
            return {
                "available": True,
                "backend": backend.name,
                "key_exists": backend.key_exists(),
                "is_secure": backend.name != "file-fallback",
            }
        except KeychainUnavailableError:
            return {
                "available": False,
                "backend": None,
                "key_exists": False,
                "is_secure": False,
            }
