"""Tests for pisama_core.config module."""

import json
import tempfile
from pathlib import Path

from pisama_core.config.loader import load_config
from pisama_core.config.models import DetectionConfig, HealingConfig, PisamaConfig


class TestDetectionConfig:
    """Tests for DetectionConfig model."""

    def test_create_default(self):
        """Test default detection config."""
        config = DetectionConfig()
        assert config.enabled is True
        assert config.severity_threshold == 40

    def test_create_with_values(self):
        """Test detection config with custom values."""
        config = DetectionConfig(
            enabled=True,
            severity_threshold=50,
            enabled_detectors=["loop", "hallucination"],
        )
        assert config.severity_threshold == 50
        assert "loop" in config.enabled_detectors


class TestHealingConfig:
    """Tests for HealingConfig model."""

    def test_create_default(self):
        """Test default healing config."""
        config = HealingConfig()
        assert config.enabled is True
        assert config.mode == "manual"

    def test_create_with_values(self):
        """Test healing config with custom values."""
        config = HealingConfig(
            enabled=True,
            mode="auto",
            auto_fix_types=["break_loop", "switch_strategy"],
            blocked_fix_types=["terminate"],
            max_auto_fixes=5,
        )
        assert config.mode == "auto"
        assert "break_loop" in config.auto_fix_types
        assert config.max_auto_fixes == 5


class TestPisamaConfig:
    """Tests for PisamaConfig model."""

    def test_create_default(self):
        """Test default config."""
        config = PisamaConfig()
        assert config.detection is not None
        assert config.healing is not None

    def test_create_with_values(self):
        """Test config with custom values."""
        config = PisamaConfig(
            detection=DetectionConfig(severity_threshold=60),
            healing=HealingConfig(mode="auto"),
        )
        assert config.detection.severity_threshold == 60
        assert config.healing.mode == "auto"

    def test_to_dict(self):
        """Test config serialization."""
        config = PisamaConfig()
        data = config.to_dict()
        assert "detection" in data
        assert "healing" in data

    def test_from_dict(self):
        """Test config deserialization."""
        data = {
            "detection": {
                "enabled": True,
                "severity_threshold": 50,
            },
            "healing": {
                "enabled": True,
                "mode": "report",
            },
        }
        config = PisamaConfig.from_dict(data)
        assert config.detection.severity_threshold == 50
        assert config.healing.mode == "report"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config(self):
        """Test loading default config when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config = load_config(config_path)
            assert config is not None
            assert config.detection.enabled is True

    def test_load_from_file(self):
        """Test loading config from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            # Write config file
            config_data = {
                "detection": {
                    "enabled": True,
                    "severity_threshold": 55,
                },
                "healing": {
                    "enabled": True,
                    "mode": "auto",
                    "auto_fix_types": ["break_loop"],
                },
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            # Load config
            config = load_config(config_path)
            assert config.detection.severity_threshold == 55
            assert config.healing.mode == "auto"

    def test_load_partial_config(self):
        """Test loading config with only some fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            # Write partial config
            config_data = {
                "healing": {
                    "mode": "report",
                },
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            # Load config - should use defaults for missing fields
            config = load_config(config_path)
            assert config.detection is not None  # Default
            assert config.healing.mode == "report"

    def test_load_invalid_json(self):
        """Test loading invalid JSON falls back to defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            # Write invalid JSON
            with open(config_path, "w") as f:
                f.write("{ invalid json }")

            # Should return default config
            config = load_config(config_path)
            assert config is not None
            assert config.detection.enabled is True
