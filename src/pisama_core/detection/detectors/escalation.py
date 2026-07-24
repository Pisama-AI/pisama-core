"""Escalation loop detector for multi-agent handoff cycles."""

from collections import Counter

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Trace


class EscalationLoopDetector(BaseDetector):
    """Detects stale escalation loops in multi-agent handoff chains.

    Identifies:
    - Round-trip handoff cycles (A->B and B->A repeating with stale output)
    - Approval shopping (same source escalating to 3+ targets with no resolution)
    - Excessive handoffs indicating an unresolved escalation spiral
    """

    name = "escalation_loop"
    description = "Detects stale escalation loops and approval shopping in agent handoff chains"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (30, 90)
    realtime_capable = False

    max_round_trips = 2
    min_approval_targets = 3

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect escalation loop patterns in handoff spans."""
        handoffs = trace.get_spans_by_kind(SpanKind.HANDOFF)

        if len(handoffs) < 3:
            return DetectionResult.no_issue(self.name)

        issues: list[str] = []
        severity = 0

        # Build directed pair counts
        sorted_handoffs = sorted(handoffs, key=lambda s: s.start_time)
        pairs: list[tuple[str, str]] = []
        source_to_targets: dict[str, set] = {}
        outputs: list[str] = []

        for span in sorted_handoffs:
            src = span.attributes.get("source_agent", "")
            tgt = span.attributes.get("target_agent", "")
            if src and tgt:
                pairs.append((src, tgt))
                source_to_targets.setdefault(src, set()).add(tgt)
            out = ""
            if span.output_data:
                out = str(span.output_data.get("output", ""))
            outputs.append(out)

        pair_counts = Counter(pairs)

        # Check for round-trip loops: A->B and B->A both exceed threshold.
        # Use seen_pairs to avoid double-counting (A,B) and (B,A) as separate loops.
        seen_pairs: set[frozenset] = set()
        for (src, tgt), count_fwd in pair_counts.items():
            key = frozenset([src, tgt])
            if key in seen_pairs:
                continue
            count_rev = pair_counts.get((tgt, src), 0)
            if count_rev > 0:
                round_trips = min(count_fwd, count_rev)
                if round_trips > self.max_round_trips:
                    seen_pairs.add(key)
                    severity += 40
                    issues.append(
                        f"Escalation loop: {src} <-> {tgt} cycled {round_trips}x "
                        f"({count_fwd} fwd, {count_rev} rev)"
                    )
                    # Bonus severity for stale (near-identical) outputs
                    if len(outputs) >= 2 and _jaccard(outputs[0], outputs[-1]) >= 0.70:
                        severity += 20
                        issues.append("Stale outputs across loop iterations (Jaccard >= 0.70)")

        # Check approval shopping: same source -> 3+ different targets
        for src, targets in source_to_targets.items():
            if len(targets) >= self.min_approval_targets:
                severity += 30
                issues.append(
                    f"Approval shopping: {src} escalated to {len(targets)} different targets"
                )

        if not issues:
            return DetectionResult.no_issue(self.name)

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=min(90, severity),
            summary=issues[0],
            fix_type=FixType.ESCALATE,
            fix_instruction=(
                "Break the escalation loop. Assign a definitive decision-maker or "
                "terminate the handoff chain and surface the unresolved issue to a human."
            ),
        )

        for issue in issues:
            result.add_evidence(
                description=issue,
                span_ids=[s.span_id for s in sorted_handoffs],
                data={"pair_counts": dict(pair_counts)},
            )

        return result


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    # Empty outputs carry no information — treat as dissimilar, not identical.
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union)
