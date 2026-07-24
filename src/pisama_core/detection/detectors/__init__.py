"""Built-in detectors for Pisama.

This module auto-registers all built-in detectors with the global registry.
"""

from pisama_core.detection.detectors.approval import ApprovalBypassDetector
from pisama_core.detection.detectors.citation import CitationDetector
from pisama_core.detection.detectors.communication import CommunicationDetector
from pisama_core.detection.detectors.completion import CompletionDetector
from pisama_core.detection.detectors.context import ContextDetector
from pisama_core.detection.detectors.convergence import ConvergenceDetector
from pisama_core.detection.detectors.coordination import CoordinationDetector
from pisama_core.detection.detectors.corruption import CorruptionDetector
from pisama_core.detection.detectors.cost import CostDetector
from pisama_core.detection.detectors.critic import CriticQualityDetector
from pisama_core.detection.detectors.decomposition import DecompositionDetector
from pisama_core.detection.detectors.derailment import DerailmentDetector
from pisama_core.detection.detectors.entity_confusion import EntityConfusionDetector
from pisama_core.detection.detectors.escalation import EscalationLoopDetector
from pisama_core.detection.detectors.exploration_safety import ExplorationSafetyDetector
from pisama_core.detection.detectors.hallucination import HallucinationDetector
from pisama_core.detection.detectors.injection import InjectionDetector

# Import detectors to trigger registration
from pisama_core.detection.detectors.loop import LoopDetector
from pisama_core.detection.detectors.mcp_protocol import MCPProtocolDetector
from pisama_core.detection.detectors.memory_staleness import MemoryStalenessDetector
from pisama_core.detection.detectors.model_selection import ModelSelectionDetector
from pisama_core.detection.detectors.overflow import OverflowDetector
from pisama_core.detection.detectors.parallel_consistency import ParallelConsistencyDetector
from pisama_core.detection.detectors.persona import PersonaDetector
from pisama_core.detection.detectors.propagation import ErrorPropagationDetector
from pisama_core.detection.detectors.reasoning import ReasoningConsistencyDetector
from pisama_core.detection.detectors.repetition import RepetitionDetector
from pisama_core.detection.detectors.routing import RoutingDetector
from pisama_core.detection.detectors.specification import SpecificationDetector
from pisama_core.detection.detectors.starvation import TaskStarvationDetector
from pisama_core.detection.detectors.withholding import WithholdingDetector
from pisama_core.detection.detectors.workflow import WorkflowDetector
from pisama_core.detection.registry import registry

# Register all built-in detectors
_BUILTIN_DETECTORS = [
    LoopDetector(),
    RepetitionDetector(),
    CoordinationDetector(),
    HallucinationDetector(),
    CostDetector(),
    DerailmentDetector(),
    ContextDetector(),
    CommunicationDetector(),
    SpecificationDetector(),
    InjectionDetector(),
    CorruptionDetector(),
    PersonaDetector(),
    OverflowDetector(),
    DecompositionDetector(),
    WorkflowDetector(),
    WithholdingDetector(),
    CompletionDetector(),
    ConvergenceDetector(),
    ParallelConsistencyDetector(),
    MemoryStalenessDetector(),
    ApprovalBypassDetector(),
    ModelSelectionDetector(),
    MCPProtocolDetector(),
    RoutingDetector(),
    ErrorPropagationDetector(),
    CriticQualityDetector(),
    EscalationLoopDetector(),
    CitationDetector(),
    ReasoningConsistencyDetector(),
    EntityConfusionDetector(),
    TaskStarvationDetector(),
    ExplorationSafetyDetector(),
]

for detector in _BUILTIN_DETECTORS:
    registry.register(detector)

__all__ = [
    "LoopDetector",
    "RepetitionDetector",
    "CoordinationDetector",
    "HallucinationDetector",
    "CostDetector",
    "DerailmentDetector",
    "ContextDetector",
    "CommunicationDetector",
    "SpecificationDetector",
    "InjectionDetector",
    "CorruptionDetector",
    "PersonaDetector",
    "OverflowDetector",
    "DecompositionDetector",
    "WorkflowDetector",
    "WithholdingDetector",
    "CompletionDetector",
    "ConvergenceDetector",
    "ParallelConsistencyDetector",
    "MemoryStalenessDetector",
    "ApprovalBypassDetector",
    "ModelSelectionDetector",
    "MCPProtocolDetector",
    "RoutingDetector",
    "ErrorPropagationDetector",
    "CriticQualityDetector",
    "EscalationLoopDetector",
    "CitationDetector",
    "ReasoningConsistencyDetector",
    "EntityConfusionDetector",
    "TaskStarvationDetector",
    "ExplorationSafetyDetector",
]
