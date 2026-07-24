"""Routing detector for identifying misrouted inputs to wrong specialist agents."""

import re
from collections import Counter
from typing import Any

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Span, Trace

# Common stop words to exclude from keyword extraction
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "must",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "like",
        "through",
        "after",
        "over",
        "between",
        "out",
        "against",
        "during",
        "without",
        "before",
        "under",
        "around",
        "among",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "because",
        "if",
        "when",
        "where",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
        "please",
        "help",
        "want",
        "get",
        "make",
        "use",
        "also",
        "then",
    }
)


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text.

    Focuses on nouns, technical terms, and domain-specific words
    by filtering out stop words and very short tokens.
    """
    if not text:
        return set()
    # Lowercase and split on non-alphanumeric
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    # Keep tokens that are 3+ chars and not stop words
    return {t for t in tokens if len(t) >= 3 and t not in _STOP_WORDS}


def _keyword_overlap(set_a: set[str], set_b: set[str]) -> float:
    """Compute keyword overlap ratio.

    Returns the fraction of set_a keywords found in set_b.
    Returns 1.0 if set_a is empty (nothing to match against).
    """
    if not set_a:
        return 1.0
    if not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a)


class RoutingDetector(BaseDetector):
    """Detects inputs sent to the wrong specialist agent or route.

    This detector identifies:
    - Low keyword overlap between input topic and handler description/domain
    - Route bouncing: multiple consecutive route changes for the same input
    """

    name = "routing"
    description = "Detects misrouted inputs sent to wrong specialist agents"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (25, 70)
    realtime_capable = False

    # Configuration
    min_overlap_threshold = 0.20
    max_route_bounces = 2

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect routing issues in a trace."""
        agent_spans = trace.get_spans_by_kind(SpanKind.AGENT)
        handoff_spans = trace.get_spans_by_kind(SpanKind.HANDOFF)

        if not agent_spans:
            return DetectionResult.no_issue(self.name)

        sorted_agents = sorted(agent_spans, key=lambda s: s.start_time)

        issues: list[str] = []
        severity = 0
        evidence_data: dict[str, Any] = {}

        # --- Check 1: Input-to-handler keyword overlap ---
        mismatch_details = self._check_input_handler_overlap(sorted_agents)
        if mismatch_details:
            severity += mismatch_details["severity"]
            issues.append(mismatch_details["summary"])
            evidence_data["mismatches"] = mismatch_details["mismatches"]

        # --- Check 2: Route bouncing ---
        bounce_details = self._check_route_bouncing(sorted_agents, handoff_spans)
        if bounce_details:
            severity += bounce_details["severity"]
            issues.append(bounce_details["summary"])
            evidence_data["bounces"] = bounce_details["bounces"]

        if not issues:
            return DetectionResult.no_issue(self.name)

        severity = max(self.severity_range[0], min(self.severity_range[1], severity))

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=issues[0],
            fix_type=FixType.SWITCH_STRATEGY,
            fix_instruction=(
                "Input appears misrouted. Re-evaluate the routing decision and "
                "direct to the specialist whose domain matches the input topic."
            ),
        )

        for issue in issues:
            result.add_evidence(
                description=issue,
                span_ids=[s.span_id for s in sorted_agents[:10]],
                data=evidence_data,
            )

        return result

    def _check_input_handler_overlap(self, agent_spans: list[Span]) -> dict[str, Any] | None:
        """Check whether input keywords match handler description/domain."""
        mismatches: list[dict[str, Any]] = []

        for span in agent_spans:
            input_text = self._extract_input_text(span)
            if not input_text:
                continue

            input_keywords = _extract_keywords(input_text)
            if len(input_keywords) < 2:
                # Too little input to make a meaningful comparison
                continue

            # Build handler keywords from name, description, domain attributes
            handler_text = self._build_handler_text(span)
            handler_keywords = _extract_keywords(handler_text)
            if not handler_keywords:
                continue

            overlap = _keyword_overlap(input_keywords, handler_keywords)
            if overlap < self.min_overlap_threshold:
                mismatches.append(
                    {
                        "span_id": span.span_id,
                        "handler": span.name,
                        "overlap": round(overlap, 3),
                        "input_keywords": sorted(input_keywords)[:10],
                        "handler_keywords": sorted(handler_keywords)[:10],
                    }
                )

        if not mismatches:
            return None

        worst = min(mismatches, key=lambda m: m["overlap"])
        sev = 25 + int((1 - worst["overlap"]) * 30)  # 25-55 range

        return {
            "severity": sev,
            "summary": (
                f"Input misrouted to '{worst['handler']}' (topic overlap {worst['overlap']:.0%})"
            ),
            "mismatches": mismatches,
        }

    def _check_route_bouncing(
        self,
        agent_spans: list[Span],
        handoff_spans: list[Span],
    ) -> dict[str, Any] | None:
        """Check for multiple consecutive route changes (bouncing)."""
        # Use handoffs if available, otherwise look at agent sequence
        if handoff_spans:
            sorted_handoffs = sorted(handoff_spans, key=lambda s: s.start_time)
            route_sequence = [s.attributes.get("target_agent", s.name) for s in sorted_handoffs]
        else:
            route_sequence = [s.name for s in agent_spans]

        if len(route_sequence) < 3:
            return None

        # Find consecutive route changes (no route repeats in sequence)
        bounces: list[dict[str, Any]] = []
        consecutive_changes = 0
        for i in range(1, len(route_sequence)):
            if route_sequence[i] != route_sequence[i - 1]:
                consecutive_changes += 1
            else:
                if consecutive_changes > self.max_route_bounces:
                    bounces.append(
                        {
                            "start_index": i - consecutive_changes,
                            "end_index": i - 1,
                            "changes": consecutive_changes,
                            "routes": route_sequence[i - consecutive_changes : i],
                        }
                    )
                consecutive_changes = 0

        # Check the trailing window
        if consecutive_changes > self.max_route_bounces:
            bounces.append(
                {
                    "start_index": len(route_sequence) - consecutive_changes - 1,
                    "end_index": len(route_sequence) - 1,
                    "changes": consecutive_changes,
                    "routes": route_sequence[-(consecutive_changes + 1) :],
                }
            )

        # Also detect A->B->A->B pattern (ping-pong between two routes)
        pair_counts: Counter[tuple[str, str]] = Counter()
        for i in range(1, len(route_sequence)):
            if route_sequence[i] != route_sequence[i - 1]:
                pair = (route_sequence[i - 1], route_sequence[i])
                pair_counts[pair] += 1

        for (src, dst), count in pair_counts.items():
            reverse_count = pair_counts.get((dst, src), 0)
            round_trips = min(count, reverse_count)
            if round_trips >= 2:
                bounces.append(
                    {
                        "type": "ping_pong",
                        "agents": [src, dst],
                        "round_trips": round_trips,
                    }
                )

        if not bounces:
            return None

        max_changes = max(b.get("changes", b.get("round_trips", 0)) for b in bounces)
        sev = 25 + min(max_changes * 10, 40)

        return {
            "severity": sev,
            "summary": f"Route bouncing detected ({max_changes} consecutive route changes)",
            "bounces": bounces,
        }

    @staticmethod
    def _extract_input_text(span: Span) -> str:
        """Extract input text from a span."""
        parts: list[str] = []
        if span.input_data:
            for key in ("input", "query", "text", "message", "prompt", "content"):
                val = span.input_data.get(key)
                if isinstance(val, str):
                    parts.append(val)
        return " ".join(parts)

    @staticmethod
    def _build_handler_text(span: Span) -> str:
        """Build a text representation of the handler's domain from span metadata."""
        parts: list[str] = [span.name]
        if span.attributes:
            for key in (
                "description",
                "domain",
                "role",
                "system_prompt",
                "specialization",
                "capabilities",
            ):
                val = span.attributes.get(key)
                if isinstance(val, str):
                    parts.append(val)
        return " ".join(parts)
