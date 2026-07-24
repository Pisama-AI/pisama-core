"""Configuration loading and saving."""

import json
from pathlib import Path
from typing import Optional

from pisama_core.config.models import PisamaConfig


def load_config(path: Optional[Path] = None) -> PisamaConfig:
    """Load configuration from a file.

    Args:
        path: Path to config file. Defaults to ~/.pisama/config.json

    Returns:
        Loaded configuration, or defaults if file doesn't exist
    """
    if path is None:
        path = Path.home() / ".pisama" / "config.json"

    if not path.exists():
        return PisamaConfig()

    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return PisamaConfig()
        return PisamaConfig.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return PisamaConfig()


def save_config(config: PisamaConfig, path: Optional[Path] = None) -> None:
    """Save configuration to a file.

    Args:
        config: Configuration to save
        path: Path to save to. Defaults to ~/.pisama/config.json
    """
    if path is None:
        path = Path.home() / ".pisama" / "config.json"

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
