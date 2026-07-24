"""Owner-only local file primitives for security-sensitive data."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO, cast

OWNER_ONLY_MODE = 0o600


def ensure_owner_only_file(path: Path) -> None:
    """Create a file without a permissive-mode race and restrict existing files."""
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, OWNER_ONLY_MODE)
    os.close(fd)
    os.chmod(path, OWNER_ONLY_MODE)


@contextmanager
def owner_only_text_file(path: Path, *, append: bool) -> Iterator[TextIO]:
    """Open a text file for an owner-only append or truncating write."""
    operation = os.O_APPEND if append else os.O_TRUNC
    fd = os.open(path, os.O_CREAT | operation | os.O_WRONLY, OWNER_ONLY_MODE)
    try:
        try:
            os.fchmod(fd, OWNER_ONLY_MODE)
        except (AttributeError, OSError):
            os.chmod(path, OWNER_ONLY_MODE)
        mode = "a" if append else "w"
        with cast(TextIO, os.fdopen(fd, mode)) as file:
            fd = -1
            yield file
    finally:
        if fd >= 0:
            os.close(fd)
