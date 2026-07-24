"""Parallel consistency detector for contradictory results from parallel branches."""

import re
from collections import defaultdict
from typing import Any

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.models import Span, Trace


class ParallelConsistencyDetector(BaseDetector):
    """Detects contradictory results from parallel branches merged without reconciliation.

    This detector identifies:
    - Parallel span groups (shared parent, overlapping time ranges)
    - Contradictory facts across parallel outputs (numbers, entities, booleans)
    - Missing reconciliation in downstream merge spans
    """

    name = "parallel_consistency"
    description = (
        "Detects contradictory results from parallel branches merged without reconciliation"
    )
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (30, 75)
    realtime_capable = False

    # Patterns for extracting facts
    _NUMBER_PATTERN = re.compile(
        r"(?P<entity>[\w\s]{2,30})\s*(?:is|was|are|were|=|:)\s*"
        r"(?P<value>\$?[\d,]+\.?\d*%?)",
        re.IGNORECASE,
    )
    _BOOLEAN_PATTERN = re.compile(
        r"(?P<entity>[\w\s]{2,30})\s*(?:is|was|are|were)\s+"
        r"(?P<value>(?:not\s+)?(?:true|false|yes|no|correct|incorrect|valid|invalid|"
        r"available|unavailable|enabled|disabled|active|inactive|approved|rejected|"
        r"confirmed|denied|successful|failed|compliant|non-compliant))",
        re.IGNORECASE,
    )
    _RECONCILIATION_MARKERS = [
        "however",
        "conflicting",
        "on the other hand",
        "contradiction",
        "disagree",
        "inconsistent",
        "reconcil",
        "mismatch",
        "discrepan",
        "differ",
        "conflict",
        "whereas",
        "but the other",
        "in contrast",
    ]

    _TEXT_PATTERN = re.compile(
        r"(?P<entity>(?:[a-zA-Z][\w]*(?:\s+[a-zA-Z][\w]*){0,5}))\s+"
        r"(?:is|was|are|were)\s+"
        r"(?P<value>[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{0,}){0,3})"
        r"(?=[\s,\.\!\?;]|$)",
    )

    # Boolean negation pairs
    _BOOLEAN_NEGATIONS: dict[str, str] = {
        "true": "false",
        "false": "true",
        "yes": "no",
        "no": "yes",
        "correct": "incorrect",
        "incorrect": "correct",
        "valid": "invalid",
        "invalid": "valid",
        "available": "unavailable",
        "unavailable": "available",
        "enabled": "disabled",
        "disabled": "enabled",
        "active": "inactive",
        "inactive": "active",
        "approved": "rejected",
        "rejected": "approved",
        "confirmed": "denied",
        "denied": "confirmed",
        "successful": "failed",
        "failed": "successful",
        "compliant": "non-compliant",
        "non-compliant": "compliant",
    }

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect contradictory parallel branch outputs."""
        if len(trace.spans) < 3:
            return DetectionResult.no_issue(self.name)

        # Find parallel span groups
        parallel_groups = self._find_parallel_groups(trace)
        if not parallel_groups:
            return DetectionResult.no_issue(self.name)

        issues: list[str] = []
        severity = 0
        evidence_spans: list[str] = []

        for parent_id, group in parallel_groups.items():
            # Extract facts from each parallel branch
            branch_facts = []
            for span in group:
                facts = self._extract_facts(span)
                if facts:
                    branch_facts.append((span, facts))

            if len(branch_facts) < 2:
                continue

            # Find contradictions between branches
            contradictions = self._find_contradictions(branch_facts)
            if not contradictions:
                continue

            # Check if downstream spans reconcile the contradictions
            downstream = self._find_downstream_spans(trace, parent_id, group)
            reconciled = self._check_reconciliation(downstream)

            if not reconciled:
                for contradiction in contradictions:
                    issues.append(contradiction["description"])
                    evidence_spans.extend(contradiction["span_ids"])
                    severity += contradiction["severity_contribution"]

        if not issues:
            return DetectionResult.no_issue(self.name)

        severity = max(self.severity_range[0], min(self.severity_range[1], severity))

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=issues[0]
            if len(issues) == 1
            else f"{len(issues)} contradictions in parallel branches",
            fix_type=FixType.ESCALATE,
            fix_instruction=(
                "Parallel branches produced contradictory results that were merged "
                "without reconciliation. Add a reconciliation step to resolve conflicts "
                "before using downstream."
            ),
        )

        for issue in issues:
            result.add_evidence(
                description=issue,
                span_ids=evidence_spans,
            )

        return result

    def _find_parallel_groups(self, trace: Trace) -> dict[str, list[Span]]:
        """Find groups of spans that share a parent and overlap in time."""
        children_by_parent: dict[str, list[Span]] = defaultdict(list)

        for span in trace.spans:
            if span.parent_id is not None:
                children_by_parent[span.parent_id].append(span)

        parallel_groups: dict[str, list[Span]] = {}
        for parent_id, children in children_by_parent.items():
            if len(children) < 2:
                continue

            # Check for time overlap among siblings
            sorted_children = sorted(children, key=lambda s: s.start_time)
            overlapping: list[Span] = []

            for i, span_a in enumerate(sorted_children):
                for span_b in sorted_children[i + 1 :]:
                    end_a = span_a.end_time or span_a.start_time
                    if span_b.start_time < end_a:
                        if span_a not in overlapping:
                            overlapping.append(span_a)
                        if span_b not in overlapping:
                            overlapping.append(span_b)

            if len(overlapping) >= 2:
                parallel_groups[parent_id] = overlapping

        return parallel_groups

    def _extract_facts(self, span: Span) -> list[dict[str, str]]:
        """Extract key facts (numbers, booleans) from a span's output."""
        text = self._get_span_text(span)
        if not text:
            return []

        facts: list[dict[str, str]] = []

        # Extract number facts
        for match in self._NUMBER_PATTERN.finditer(text):
            entity = match.group("entity").strip().lower()
            value = match.group("value").strip()
            facts.append({"entity": entity, "value": value, "type": "number"})

        # Extract boolean facts
        for match in self._BOOLEAN_PATTERN.finditer(text):
            entity = match.group("entity").strip().lower()
            value = match.group("value").strip().lower()
            facts.append({"entity": entity, "value": value, "type": "boolean"})

        # Extract text facts (proper-noun values like names, locations)
        seen_entities: set[str] = set()
        for match in self._TEXT_PATTERN.finditer(text):
            entity = match.group("entity").strip().lower()
            value = match.group("value").strip()
            if entity in seen_entities:
                continue
            if len(entity) < 4:
                continue
            if (
                value.lower() in self._BOOLEAN_NEGATIONS
                or value.lower().replace("not ", "") in self._BOOLEAN_NEGATIONS
            ):
                continue
            seen_entities.add(entity)
            facts.append({"entity": entity, "value": value, "type": "text"})

        return facts

    def _get_span_text(self, span: Span) -> str:
        """Extract text content from a span's output_data."""
        parts: list[str] = []

        if span.output_data:
            for key in ("text", "content", "result", "output", "response", "message"):
                val = span.output_data.get(key)
                if isinstance(val, str):
                    parts.append(val)

            # If no known key, try all string values
            if not parts:
                for val in span.output_data.values():
                    if isinstance(val, str) and len(val) > 10:
                        parts.append(val)

        return " ".join(parts)

    def _find_contradictions(
        self, branch_facts: list[tuple[Span, list[dict[str, str]]]]
    ) -> list[dict[str, Any]]:
        """Find contradictory facts across parallel branches."""
        contradictions: list[dict[str, Any]] = []

        for i, (span_a, facts_a) in enumerate(branch_facts):
            for span_b, facts_b in branch_facts[i + 1 :]:
                for fact_a in facts_a:
                    for fact_b in facts_b:
                        if fact_a["entity"] != fact_b["entity"]:
                            continue
                        if fact_a["value"] == fact_b["value"]:
                            continue

                        is_contradiction = False
                        severity_contribution = 0

                        if fact_a["type"] == "number" and fact_b["type"] == "number":
                            is_contradiction = True
                            severity_contribution = 25

                        elif fact_a["type"] == "boolean" and fact_b["type"] == "boolean":
                            val_a = fact_a["value"].replace("not ", "")
                            val_b = fact_b["value"].replace("not ", "")
                            negation_a = self._BOOLEAN_NEGATIONS.get(val_a, "")
                            is_negated = (
                                val_a == negation_a
                                or val_b == self._BOOLEAN_NEGATIONS.get(val_a, "")
                                or fact_a["value"].startswith("not ")
                                != fact_b["value"].startswith("not ")
                            )
                            if is_negated or val_a != val_b:
                                is_contradiction = True
                                severity_contribution = 30

                        elif fact_a["type"] == "text" and fact_b["type"] == "text":
                            if fact_a["value"].lower() != fact_b["value"].lower():
                                is_contradiction = True
                                severity_contribution = 20

                        if is_contradiction:
                            contradictions.append(
                                {
                                    "description": (
                                        f"Parallel branches contradict on '{fact_a['entity']}': "
                                        f"'{fact_a['value']}' vs '{fact_b['value']}'"
                                    ),
                                    "span_ids": [span_a.span_id, span_b.span_id],
                                    "severity_contribution": severity_contribution,
                                }
                            )

        return contradictions

    def _find_downstream_spans(
        self, trace: Trace, parent_id: str, parallel_spans: list[Span]
    ) -> list[Span]:
        """Find spans that come after the parallel group (potential merge points)."""
        if not parallel_spans:
            return []

        latest_end = max(
            (s.end_time or s.start_time for s in parallel_spans),
        )

        downstream: list[Span] = []
        for span in trace.spans:
            if span in parallel_spans:
                continue
            # Downstream: same parent or child of parent, starts after parallel group
            if span.start_time >= latest_end:
                downstream.append(span)

        return sorted(downstream, key=lambda s: s.start_time)[:10]

    def _check_reconciliation(self, downstream_spans: list[Span]) -> bool:
        """Check if any downstream span contains reconciliation language."""
        for span in downstream_spans:
            text = self._get_span_text(span)
            if not text:
                # Also check span name
                text = span.name.lower()
            else:
                text = text.lower()

            for marker in self._RECONCILIATION_MARKERS:
                if marker in text:
                    return True

        return False
