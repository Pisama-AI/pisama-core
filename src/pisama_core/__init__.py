"""Pisama Core - Detection, Scoring, and Healing Engine for Agent Forensics.

This package provides the shared core functionality for the Pisama platform,
supporting multiple agent frameworks including Claude Code, LangGraph, AutoGen,
CrewAI, and n8n.

Example:
    from pisama_core import DetectionOrchestrator, ScoringEngine, Trace

    orchestrator = DetectionOrchestrator()
    scoring = ScoringEngine()

    result = await orchestrator.analyze(trace)
    severity = scoring.calculate_severity([result])
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("pisama-core")
except PackageNotFoundError:  # editable install without metadata
    __version__ = "0.0.0+unknown"

# Traces
# Adapters
from pisama_core.adapters.base import InjectionResult, PlatformAdapter

# Audit
from pisama_core.audit.logger import AuditLogger
from pisama_core.audit.models import AuditEvent, AuditEventType
from pisama_core.config.loader import load_config

# Config
from pisama_core.config.models import DetectionConfig, HealingConfig, PisamaConfig

# Auto-register built-in detectors so the documented quickstart works.
# See pisama_core.detection.detectors.__init__ for the registration list.
from pisama_core.detection import detectors as _detectors  # noqa: F401

# Detection
from pisama_core.detection.base import BaseDetector
from pisama_core.detection.diagnosis import DiagnosisRecord
from pisama_core.detection.orchestrator import DetectionOrchestrator
from pisama_core.detection.registry import DetectorRegistry, registry
from pisama_core.detection.result import DetectionResult, Evidence, FixRecommendation
from pisama_core.healing.base import BaseFix
from pisama_core.healing.engine import HealingEngine

# Healing
from pisama_core.healing.models import FixContext, FixResult, HealingPlan
from pisama_core.injection.enforcement import EnforcementEngine, EnforcementLevel

# Injection
from pisama_core.injection.protocol import FixInjectionProtocol

# Scoring
from pisama_core.scoring.engine import ScoringEngine
from pisama_core.scoring.thresholds import SeverityLevel, Thresholds

# Tokenization
from pisama_core.tokenization import (
    KeychainManager,
    PIIDetector,
    PIIMatch,
    PIIPattern,
    PIIType,
    TokenGenerator,
    Tokenizer,
    TokenParser,
    TokenVault,
    tokenize_trace_data,
)
from pisama_core.traces.enums import Platform, SpanKind, SpanStatus
from pisama_core.traces.models import Event, Span, Trace, TraceMetadata

# Telemetry
from pisama_core.utils._telemetry import disable_telemetry, enable_telemetry

__all__ = [
    # Version
    "__version__",
    # Traces
    "Event",
    "Span",
    "Trace",
    "TraceMetadata",
    "Platform",
    "SpanKind",
    "SpanStatus",
    # Detection
    "BaseDetector",
    "DetectionResult",
    "Evidence",
    "FixRecommendation",
    "DetectorRegistry",
    "registry",
    "DetectionOrchestrator",
    "DiagnosisRecord",
    # Scoring
    "ScoringEngine",
    "Thresholds",
    "SeverityLevel",
    # Healing
    "FixContext",
    "FixResult",
    "HealingPlan",
    "BaseFix",
    "HealingEngine",
    # Injection
    "FixInjectionProtocol",
    "EnforcementLevel",
    "EnforcementEngine",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    # Config
    "PisamaConfig",
    "DetectionConfig",
    "HealingConfig",
    "load_config",
    # Adapters
    "PlatformAdapter",
    "InjectionResult",
    # Telemetry
    "disable_telemetry",
    "enable_telemetry",
    # Tokenization
    "PIIDetector",
    "PIIMatch",
    "PIIPattern",
    "PIIType",
    "TokenGenerator",
    "TokenParser",
    "TokenVault",
    "KeychainManager",
    "Tokenizer",
    "tokenize_trace_data",
]
