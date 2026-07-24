"""Context neglect detector for identifying when agents ignore upstream context.

F7: Context Neglect Detection (MAST Taxonomy)

Detects when an agent ignores or fails to use upstream context
provided by previous agents or steps in the workflow.

Detection Methods:
1. Key element extraction (numbers, dates, names, URLs, emails, keywords)
2. Weighted utilization scoring
3. Context reference phrase detection
4. Adaptation phrase detection
5. Critical context marker detection with topic extraction
6. Task-addressed heuristic for leniency

Ported from backend/app/detection/context.py (v1.2+).
"""

import logging
import re
from typing import Any, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import Platform
from pisama_core.traces.models import Trace

logger = logging.getLogger(__name__)

# --- Constants ported from backend ---

# v1.2: Markers indicating critical context that should not be ignored
CRITICAL_CONTEXT_MARKERS: list[str] = [
    "critical:",
    "important:",
    "critical -",
    "important -",
    "must handle",
    "must address",
    "must review",
    "must consider",
    "required:",
    "mandatory:",
    "do not ignore",
    "don't ignore",
]

# Phrases that indicate the output is building on/referencing prior context
CONTEXT_REFERENCE_PHRASES: list[str] = [
    "based on",
    "building on",
    "as discussed",
    "as mentioned",
    "from the previous",
    "from earlier",
    "continuing from",
    "following up on",
    "as per",
    "according to",
    "incorporating",
    "using the",
    "reviewing the",
    "analyzed the",
    "examined the",
    "looked at the",
    "the existing",
    "the current",
    "the original",
    "our previous",
    "the previous",
    "previous analysis",
    "previous research",
    "previous work",
    "prior work",
    "reflecting on",
    "building upon",
    "extending",
    "referenced",
    "referring to",
    "as noted",
]

# Phrases that indicate legitimate adaptation of context
ADAPTATION_PHRASES: list[str] = [
    "reformatted",
    "restructured",
    "reorganized",
    "updated format",
    "different approach",
    "alternative method",
    "new methodology",
    "refactored",
    "rewritten",
    "rewrote",
    "reimplemented",
    "improved",
    "enhanced",
    "optimized",
    "streamlined",
    "modernized",
    "simplified",
    "clarified",
    "pivot",
    "pivoting",
    "adjusted",
    "modified approach",
    "adapted",
    "evolved",
    "iterated",
]

_STOPWORDS = frozenset(
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
        "must",
        "shall",
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
        "and",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "not",
        "it",
    }
)


class ContextDetector(BaseDetector):
    """Detects context neglect -- when an agent ignores upstream context.

    Analyzes whether key information from the provided context
    is reflected in the agent's output, using weighted element
    matching, context reference detection, adaptation recognition,
    and critical marker enforcement.

    Span convention:
        Each span should carry ``input_data`` with a ``"context"`` key and
        ``output_data`` with a ``"content"`` key. An optional ``"task"`` key
        in ``input_data`` enables the task-addressed heuristic. For adjacent
        spans, the previous span's output is automatically prepended to the
        current span's context.
    """

    name = "context"
    description = "Detects context neglect in agent responses"
    version = "1.3.0"
    platforms: list[Platform] = []  # All platforms
    severity_range = (0, 100)
    realtime_capable = False

    # Default thresholds
    utilization_threshold: float = 0.65
    min_context_length: int = 50

    # --- Internal helpers (ported faithfully from backend) ---

    @staticmethod
    def _extract_key_elements(text: str) -> dict[str, set[str]]:
        """Extract structured elements from *text*."""
        elements: dict[str, set[str]] = {
            "numbers": set(),
            "dates": set(),
            "names": set(),
            "urls": set(),
            "emails": set(),
            "keywords": set(),
        }

        numbers = re.findall(r"\b\d+(?:\.\d+)?(?:%|k|m|b)?\b", text.lower())
        elements["numbers"] = set(numbers)

        dates = re.findall(
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
            r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,?\s+\d{4})?)\b",
            text.lower(),
        )
        elements["dates"] = set(dates)

        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        elements["names"] = set(capitalized)

        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
        elements["urls"] = set(urls)

        emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
        elements["emails"] = set(emails)

        words = text.lower().split()
        keywords = {w for w in words if len(w) > 4 and w not in _STOPWORDS}
        elements["keywords"] = keywords

        return elements

    @staticmethod
    def _compute_utilization(
        context_elements: dict[str, set[str]],
        output_elements: dict[str, set[str]],
    ) -> tuple[float, list[str]]:
        """Compute weighted utilization score and list of missing elements."""
        total_weight = 0.0
        utilized_weight = 0.0
        missing: list[str] = []

        weights: dict[str, float] = {
            "numbers": 3.0,
            "dates": 3.0,
            "names": 2.5,
            "urls": 2.0,
            "emails": 2.0,
            "keywords": 1.0,
        }

        for element_type, weight in weights.items():
            context_set = context_elements.get(element_type, set())
            output_set = output_elements.get(element_type, set())

            for item in context_set:
                total_weight += weight
                if any(item.lower() in o.lower() or o.lower() in item.lower() for o in output_set):
                    utilized_weight += weight
                else:
                    if element_type in ["numbers", "dates", "names"]:
                        missing.append(f"{element_type}: {item}")

        if total_weight == 0:
            return 1.0, []

        return utilized_weight / total_weight, missing[:10]

    @staticmethod
    def _check_context_reference(output: str) -> bool:
        """v1.1: Check if output explicitly references prior context."""
        output_lower = output.lower()
        for phrase in CONTEXT_REFERENCE_PHRASES:
            if phrase in output_lower:
                return True
        return False

    @staticmethod
    def _check_adaptation(output: str) -> bool:
        """v1.1: Check if output indicates legitimate adaptation of context."""
        output_lower = output.lower()
        for phrase in ADAPTATION_PHRASES:
            if phrase in output_lower:
                return True
        return False

    @staticmethod
    def _has_critical_context(context: str) -> bool:
        """v1.2: Check if context contains critical markers."""
        context_lower = context.lower()
        for marker in CRITICAL_CONTEXT_MARKERS:
            if marker in context_lower:
                return True
        return False

    @staticmethod
    def _extract_critical_topics(context: str) -> set[str]:
        """v1.2: Extract key topic words from critical context sections."""
        context_lower = context.lower()
        critical_topics: set[str] = set()

        sections = re.split(r"[.!?]", context_lower)
        for section in sections:
            has_marker = any(marker in section for marker in CRITICAL_CONTEXT_MARKERS)
            if has_marker:
                hyphenated = re.findall(r"\b[a-z]+-[a-z]+(?:-[a-z]+)*\b", section)
                critical_topics.update(hyphenated)

                compounds = re.findall(
                    r"\b[a-z]+(?:manager|handler|controller|service|config|policy|hook|store)\b",
                    section,
                )
                critical_topics.update(compounds)

        return critical_topics

    @staticmethod
    def _check_critical_topics_addressed(
        critical_topics: set[str],
        output: str,
    ) -> tuple[bool, set[str]]:
        """v1.2: Check if critical topics from context are addressed in output.

        Returns (addressed, missing_topics).
        """
        if not critical_topics:
            return True, set()

        output_lower = output.lower()
        addressed: set[str] = set()
        missing: set[str] = set()

        for topic in critical_topics:
            if topic in output_lower:
                addressed.add(topic)
                continue

            topic_parts = [p for p in topic.split("-") if len(p) > 4]
            if len(topic_parts) >= 2:
                parts_matched = sum(1 for part in topic_parts if part in output_lower)
                if parts_matched >= 2:
                    addressed.add(topic)
                    continue

            if len(topic) > 10 and topic in output_lower:
                addressed.add(topic)
                continue

            missing.add(topic)

        if not critical_topics:
            return True, set()

        coverage = len(addressed) / len(critical_topics)
        return coverage >= 0.5, missing

    @staticmethod
    def _is_task_addressed(task: Optional[str], output: str) -> bool:
        """v1.1: Check if output addresses the core task request."""
        if not task:
            return False

        task_lower = task.lower()
        output_lower = output.lower()

        action_patterns: list[tuple[str, list[str]]] = [
            ("update", ["updated", "updating", "incorporated", "added new"]),
            ("continue", ["continued", "continuing", "building on", "following up"]),
            ("improve", ["improved", "improving", "enhanced", "optimized"]),
            ("fix", ["fixed", "fixing", "resolved", "corrected", "patched"]),
            ("analyze", ["analyzed", "analysis", "examined", "reviewed"]),
            ("report", ["reported", "report", "findings", "documented"]),
            ("review", ["reviewed", "review", "examined", "checked"]),
            ("document", ["documented", "documentation", "docs"]),
        ]
        for action, indicators in action_patterns:
            if action in task_lower:
                if any(ind in output_lower for ind in indicators):
                    return True

        task_words = set(task_lower.split())
        output_words = set(output_lower.split())
        stopwords = {"the", "a", "an", "to", "for", "with", "from", "and", "or", "in", "on"}
        task_keywords = {w for w in task_words if len(w) > 3 and w not in stopwords}

        if task_keywords:
            overlap = task_keywords & output_words
            if len(overlap) >= len(task_keywords) * 0.5:
                return True

        return False

    # --- Core single-pair detection ---

    def _detect_single(
        self,
        context: str,
        output: str,
        task: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Run context neglect detection on a single context/output pair.

        Returns a dict with detection results if neglect found, else None.
        """
        if len(context) < self.min_context_length:
            return None

        context_elements = self._extract_key_elements(context)
        output_elements = self._extract_key_elements(output)

        utilization, missing = self._compute_utilization(context_elements, output_elements)

        context_referenced = self._check_context_reference(output)
        adaptation_detected = self._check_adaptation(output)
        task_addressed = self._is_task_addressed(task, output) if task else False

        has_critical_context = self._has_critical_context(context)

        critical_topics: set[str] = set()
        critical_topics_addressed = True
        critical_topics_missing: set[str] = set()
        if has_critical_context:
            critical_topics = self._extract_critical_topics(context)
            critical_topics_addressed, critical_topics_missing = (
                self._check_critical_topics_addressed(critical_topics, output)
            )

        task_utilization_threshold = 0.50 if has_critical_context else 0.35

        # Detection logic (mirrors backend exactly)
        if has_critical_context and not critical_topics_addressed:
            detected = True
            missing.extend([f"critical: {t}" for t in list(critical_topics_missing)[:5]])
        elif utilization >= self.utilization_threshold:
            detected = False
        elif context_referenced and utilization >= 0.25:
            detected = False
        elif adaptation_detected and task_addressed and utilization >= 0.30:
            detected = False
        elif task_addressed and utilization >= task_utilization_threshold:
            detected = False
        else:
            detected = True

        if not detected:
            return None

        # Severity
        if utilization < 0.1:
            severity = 80
        elif utilization < 0.2:
            severity = 55
        else:
            severity = 30

        confidence = 1 - utilization

        agent_prefix = f"Agent '{agent_name}'" if agent_name else "Agent"
        explanation = (
            f"{agent_prefix} failed to utilize upstream context. "
            f"Context utilization: {utilization:.1%} (threshold: {self.utilization_threshold:.1%}). "
            f"Missing key elements: {', '.join(missing[:5]) if missing else 'general context'}."
        )

        return {
            "detected": True,
            "severity": severity,
            "confidence": confidence,
            "summary": explanation,
            "evidence": {
                "context_utilization": round(utilization, 4),
                "missing_elements": missing[:10],
                "context_referenced": context_referenced,
                "adaptation_detected": adaptation_detected,
                "task_addressed": task_addressed,
                "has_critical_context": has_critical_context,
                "critical_topics": list(critical_topics),
                "critical_topics_missing": list(critical_topics_missing),
            },
        }

    # --- Trace-level detect (BaseDetector interface) ---

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect context neglect across all spans in a trace.

        For each span, uses ``input_data.context`` (plus the previous span's
        output if available) and ``output_data.content``. Returns the
        highest-severity finding.
        """
        worst: Optional[dict[str, Any]] = None

        sorted_spans = sorted(trace.spans, key=lambda s: s.start_time)

        for i, span in enumerate(sorted_spans):
            context = (span.input_data or {}).get("context", "")
            output = (span.output_data or {}).get("content", "")
            task = (span.input_data or {}).get("task", "")
            agent_name = span.name

            # Prepend previous span's output as additional context
            if i > 0:
                prev_output = (sorted_spans[i - 1].output_data or {}).get("content", "")
                if prev_output:
                    context = f"{context}\n{prev_output}" if context else prev_output

            if not context or not output:
                continue

            finding = self._detect_single(
                context=context,
                output=output,
                task=task or None,
                agent_name=agent_name,
            )
            if finding and (worst is None or finding["severity"] > worst["severity"]):
                worst = finding

        if worst is None:
            return DetectionResult.no_issue(self.name)

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=worst["severity"],
            summary=worst["summary"],
            fix_type=FixType.RESET_CONTEXT,
            fix_instruction=(
                "Ensure the agent's prompt explicitly references the context. "
                "Add instructions like: 'Use the following context to inform your response: [CONTEXT]. "
                "Make sure to reference specific details from the context.'"
            ),
        )
        result.confidence = worst["confidence"]
        result.add_evidence(
            description=worst["summary"],
            data=worst["evidence"],
        )
        return result
