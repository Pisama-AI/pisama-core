"""Detection engine for Pisama.

Provides 20+ detectors for identifying agent failure patterns.
"""

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.diagnosis import (
    CausalCandidateRecord,
    CoverageRecord,
    DiagnosisEvidence,
    DiagnosisRecord,
    InterventionRecord,
)
from pisama_core.detection.orchestrator import DetectionOrchestrator
from pisama_core.detection.registry import DetectorRegistry, registry
from pisama_core.detection.result import DetectionResult, Evidence, FixRecommendation

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "Evidence",
    "FixRecommendation",
    "DetectorRegistry",
    "registry",
    "DetectionOrchestrator",
    "CausalCandidateRecord",
    "CoverageRecord",
    "DiagnosisEvidence",
    "DiagnosisRecord",
    "InterventionRecord",
]
