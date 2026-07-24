"""Detector registry for managing available detectors."""

from typing import Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.traces.enums import Platform


class DetectorRegistry:
    """Registry for managing detectors.

    The registry allows:
    - Registering detectors by name
    - Getting detectors for specific platforms
    - Enabling/disabling detectors
    - Listing all available detectors

    Example:
        from pisama_core.detection import registry

        # Register a detector
        registry.register(MyDetector())

        # Get detectors for a platform
        detectors = registry.get_for_platform(Platform.CLAUDE_CODE)

        # Run all detectors
        for detector in registry.get_enabled():
            result = await detector.run(trace)
    """

    def __init__(self) -> None:
        self._detectors: dict[str, BaseDetector] = {}

    def register(self, detector: BaseDetector) -> None:
        """Register a detector.

        Args:
            detector: The detector instance to register
        """
        self._detectors[detector.name] = detector

    def unregister(self, name: str) -> Optional[BaseDetector]:
        """Unregister a detector by name.

        Args:
            name: Name of the detector to remove

        Returns:
            The removed detector, or None if not found
        """
        return self._detectors.pop(name, None)

    def get(self, name: str) -> Optional[BaseDetector]:
        """Get a detector by name.

        Args:
            name: Name of the detector

        Returns:
            The detector, or None if not found
        """
        return self._detectors.get(name)

    def get_all(self) -> list[BaseDetector]:
        """Get all registered detectors."""
        return list(self._detectors.values())

    def get_enabled(self) -> list[BaseDetector]:
        """Get all enabled detectors."""
        return [d for d in self._detectors.values() if d.enabled]

    def get_for_platform(self, platform: Platform) -> list[BaseDetector]:
        """Get all detectors that apply to a platform.

        Args:
            platform: The platform to filter by

        Returns:
            List of applicable detectors
        """
        return [
            d for d in self._detectors.values() if d.enabled and d.applies_to_platform(platform)
        ]

    def get_realtime_capable(self, platform: Platform) -> list[BaseDetector]:
        """Get detectors capable of real-time detection for a platform.

        Args:
            platform: The platform to filter by

        Returns:
            List of real-time capable detectors
        """
        return [d for d in self.get_for_platform(platform) if d.realtime_capable]

    def enable(self, name: str) -> bool:
        """Enable a detector by name.

        Args:
            name: Name of the detector

        Returns:
            True if detector was found and enabled
        """
        detector = self.get(name)
        if detector:
            detector.enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a detector by name.

        Args:
            name: Name of the detector

        Returns:
            True if detector was found and disabled
        """
        detector = self.get(name)
        if detector:
            detector.enabled = False
            return True
        return False

    def enable_all(self) -> None:
        """Enable all detectors."""
        for detector in self._detectors.values():
            detector.enabled = True

    def disable_all(self) -> None:
        """Disable all detectors."""
        for detector in self._detectors.values():
            detector.enabled = False

    @property
    def count(self) -> int:
        """Number of registered detectors."""
        return len(self._detectors)

    @property
    def enabled_count(self) -> int:
        """Number of enabled detectors."""
        return len(self.get_enabled())

    def __len__(self) -> int:
        return self.count

    def __contains__(self, name: str) -> bool:
        return name in self._detectors

    def __repr__(self) -> str:
        return f"<DetectorRegistry(count={self.count}, enabled={self.enabled_count})>"


# Global registry instance
registry = DetectorRegistry()
