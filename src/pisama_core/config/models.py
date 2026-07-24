"""Configuration models for Pisama."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionConfig:
    """Configuration for detection."""

    enabled: bool = True
    enabled_detectors: list[str] = field(default_factory=list)  # Empty = all
    severity_threshold: int = 40
    realtime_enabled: bool = True
    parallel: bool = True


@dataclass
class HealingConfig:
    """Configuration for healing."""

    enabled: bool = True
    mode: str = "manual"  # report, manual, auto
    auto_fix_types: list[str] = field(
        default_factory=lambda: [
            "break_loop",
            "switch_strategy",
            "escalate",
        ]
    )
    blocked_fix_types: list[str] = field(
        default_factory=lambda: [
            "terminate",
            "rollback",
        ]
    )
    max_auto_fixes: int = 10
    cooldown_seconds: int = 30


@dataclass
class AuditConfig:
    """Configuration for audit logging."""

    enabled: bool = True
    log_dir: str = "~/.pisama/audit"
    log_file: str = "audit_log.jsonl"
    max_log_size_mb: int = 100


@dataclass
class InjectionConfig:
    """Configuration for fix injection."""

    enabled: bool = True
    default_level: str = "suggest"
    block_threshold: int = 60
    max_violations_before_escalation: int = 3


@dataclass
class PisamaConfig:
    """Complete Pisama configuration."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    healing: HealingConfig = field(default_factory=HealingConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)

    # Ignored patterns
    ignored_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection": {
                "enabled": self.detection.enabled,
                "enabled_detectors": self.detection.enabled_detectors,
                "severity_threshold": self.detection.severity_threshold,
                "realtime_enabled": self.detection.realtime_enabled,
                "parallel": self.detection.parallel,
            },
            "healing": {
                "enabled": self.healing.enabled,
                "mode": self.healing.mode,
                "auto_fix_types": self.healing.auto_fix_types,
                "blocked_fix_types": self.healing.blocked_fix_types,
                "max_auto_fixes": self.healing.max_auto_fixes,
                "cooldown_seconds": self.healing.cooldown_seconds,
            },
            "audit": {
                "enabled": self.audit.enabled,
                "log_dir": self.audit.log_dir,
                "log_file": self.audit.log_file,
                "max_log_size_mb": self.audit.max_log_size_mb,
            },
            "injection": {
                "enabled": self.injection.enabled,
                "default_level": self.injection.default_level,
                "block_threshold": self.injection.block_threshold,
                "max_violations_before_escalation": self.injection.max_violations_before_escalation,
            },
            "ignored_patterns": self.ignored_patterns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PisamaConfig":
        detection_data = data.get("detection", {})
        healing_data = data.get("healing", {})
        audit_data = data.get("audit", {})
        injection_data = data.get("injection", {})

        return cls(
            detection=DetectionConfig(
                enabled=detection_data.get("enabled", True),
                enabled_detectors=detection_data.get("enabled_detectors", []),
                severity_threshold=detection_data.get("severity_threshold", 40),
                realtime_enabled=detection_data.get("realtime_enabled", True),
                parallel=detection_data.get("parallel", True),
            ),
            healing=HealingConfig(
                enabled=healing_data.get("enabled", True),
                mode=healing_data.get("mode", "manual"),
                auto_fix_types=healing_data.get(
                    "auto_fix_types", ["break_loop", "switch_strategy", "escalate"]
                ),
                blocked_fix_types=healing_data.get("blocked_fix_types", ["terminate", "rollback"]),
                max_auto_fixes=healing_data.get("max_auto_fixes", 10),
                cooldown_seconds=healing_data.get("cooldown_seconds", 30),
            ),
            audit=AuditConfig(
                enabled=audit_data.get("enabled", True),
                log_dir=audit_data.get("log_dir", "~/.pisama/audit"),
                log_file=audit_data.get("log_file", "audit_log.jsonl"),
                max_log_size_mb=audit_data.get("max_log_size_mb", 100),
            ),
            injection=InjectionConfig(
                enabled=injection_data.get("enabled", True),
                default_level=injection_data.get("default_level", "suggest"),
                block_threshold=injection_data.get("block_threshold", 60),
                max_violations_before_escalation=injection_data.get(
                    "max_violations_before_escalation", 3
                ),
            ),
            ignored_patterns=data.get("ignored_patterns", []),
        )
