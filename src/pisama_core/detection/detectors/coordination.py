"""Coordination detector for multi-agent systems."""

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import Platform, SpanKind
from pisama_core.traces.models import Trace


class CoordinationDetector(BaseDetector):
    """Detects coordination failures in multi-agent systems.

    Identifies:
    - Deadlocks between agents
    - Message storms
    - Role boundary violations
    - Circular dependencies
    """

    name = "coordination"
    description = "Detects multi-agent coordination failures"
    version = "1.0.0"
    platforms = [
        Platform.LANGGRAPH,
        Platform.AUTOGEN,
        Platform.CREWAI,
        Platform.OPENCLAW,
        Platform.DIFY,
    ]
    severity_range = (30, 80)
    realtime_capable = False  # Needs full trace context

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect coordination issues."""
        # Get agent and message spans
        agent_spans = trace.get_spans_by_kind(SpanKind.AGENT)
        message_spans = trace.get_spans_by_kind(SpanKind.MESSAGE)

        if len(agent_spans) < 2:
            return DetectionResult.no_issue(self.name)

        issues = []
        severity = 0

        # Check for message storms
        if len(message_spans) > 20:
            msg_ratio = len(message_spans) / len(trace.spans)
            if msg_ratio > 0.5:
                severity += 30
                issues.append(f"Message storm detected ({len(message_spans)} messages)")

        # Check for balanced participation
        agent_names = [s.name for s in agent_spans]
        unique_agents = set(agent_names)
        if len(unique_agents) >= 2:
            from collections import Counter

            counts = Counter(agent_names)
            max_count = max(counts.values())
            min_count = min(counts.values())

            if max_count > min_count * 3:
                severity += 25
                dominant = counts.most_common(1)[0]
                issues.append(f"Agent '{dominant[0]}' dominates conversation ({dominant[1]} turns)")

        # Check for handoff loops
        handoffs = trace.get_spans_by_kind(SpanKind.HANDOFF)
        if len(handoffs) > 10:
            severity += 30
            issues.append(f"Excessive handoffs ({len(handoffs)}) - possible coordination loop")

        # OpenClaw-specific: detect session spawn storms
        if trace.metadata.platform == Platform.OPENCLAW:
            spawn_spans = [s for s in handoffs if "spawn" in s.name.lower()]
            if len(spawn_spans) > 5:
                severity += 35
                issues.append(
                    f"Session spawn storm ({len(spawn_spans)} spawns) - agents spawning without converging"
                )

            # Detect circular routing via sessions_send
            send_spans = [s for s in handoffs if "send" in s.name.lower()]
            if len(send_spans) >= 3:
                senders = [s.attributes.get("source_agent", s.name) for s in send_spans]
                receivers = [s.attributes.get("target_agent", "") for s in send_spans]
                pairs = list(zip(senders, receivers))
                for i, (s1, r1) in enumerate(pairs):
                    for s2, r2 in pairs[i + 1 :]:
                        if s1 == r2 and s2 == r1:
                            severity += 40
                            issues.append(f"Circular routing: {s1} <-> {s2}")
                            break

        if not issues:
            return DetectionResult.no_issue(self.name)

        return DetectionResult.issue_found(
            detector_name=self.name,
            severity=min(80, severity),
            summary=issues[0],
            fix_type=FixType.ESCALATE,
            fix_instruction="Multi-agent coordination issue detected. Review agent interactions.",
        )
