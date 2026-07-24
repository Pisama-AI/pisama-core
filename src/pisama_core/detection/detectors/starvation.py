"""Task starvation detector for identifying planned tasks that never execute.

Detects:
- Tasks listed in plans/decompositions that have no matching execution span
- Queued work items that are never processed
- Planned steps that are silently dropped

Version History:
- v1.0: Initial implementation
"""

import re
from typing import Any

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Span, Trace

# Patterns for extracting planned tasks from text
_PLAN_ITEM_PATTERNS = [
    r"(?:^|\n)\s*\d+[.)]\s*(.+)",  # "1. Do something" or "1) Do something"
    r"(?:^|\n)\s*[-•*]\s+(.+)",  # "- Do something" or "• Do something"
    r"(?:^|\n)\s*(?:step|task)\s*\d*[:.]\s*(.+)",  # "Step 1: Do something"
    r"(?:^|\n)\s*(?:action\s+item)[:.]\s*(.+)",  # "Action item: Do something"
]

# Patterns that indicate a span contains a plan or task list
_PLAN_CONTEXT_PATTERNS = [
    r"\b(?:plan|steps|tasks|todo|action\s+items|agenda|roadmap)\b",
    r"\b(?:here(?:\'s|\s+is)\s+(?:the|my|our)\s+plan)\b",
    r"\b(?:i\s+will|we\s+(?:will|need\s+to)|let\s+me)\b",
    r"\b(?:first|then|next|after\s+that|finally)\b.*\b(?:then|next|after|finally)\b",
]

# Generic filler words in task lists — present in almost every planned/executed
# description, so matching on them alone is not informative. Excluded from the
# keyword set used for fuzzy matching.
_GENERIC_TASK_WORDS = {
    "task",
    "tasks",
    "step",
    "steps",
    "action",
    "actions",
    "item",
    "items",
    "work",
    "job",
    "jobs",
    "subtask",
    "subtasks",
    "complete",
    "completed",
    "completing",
    "done",
    "finish",
    "finished",
    "process",
    "processed",
    "processing",
}

# Words to strip from task descriptions for keyword matching
_TASK_STOP_WORDS = {
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
    "to",
    "in",
    "on",
    "at",
    "by",
    "of",
    "for",
    "with",
    "from",
    "and",
    "or",
    "but",
    "not",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "we",
    "i",
    "you",
    "they",
    "he",
    "she",
    "will",
    "would",
    "should",
    "could",
    "can",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "may",
    "might",
    "must",
    "shall",
    "need",
    "let",
    "me",
    "my",
    "our",
    "your",
}


class TaskStarvationDetector(BaseDetector):
    """Detects planned or queued tasks that never execute.

    Analyzes trace spans to find:
    - Task lists in plans/decompositions without corresponding execution spans
    - Numbered steps that are skipped during execution
    - Work items that appear to be queued but never processed
    """

    name = "task_starvation"
    description = "Detects planned tasks that never execute"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (20, 60)
    realtime_capable = False

    # Configuration
    keyword_overlap_threshold = 0.5  # Minimum keyword overlap for fuzzy matching
    min_planned_tasks = 2  # Need at least this many planned tasks to analyze
    min_starvation_ratio = 0.2  # At least 20% of tasks must be starved to flag

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect task starvation in a trace."""
        if not trace.spans:
            return DetectionResult.no_issue(self.name)

        sorted_spans = sorted(trace.spans, key=lambda s: s.start_time)

        # Extract planned tasks from plan/decomposition spans
        planned_tasks = self._extract_planned_tasks(sorted_spans)

        if len(planned_tasks) < self.min_planned_tasks:
            return DetectionResult.no_issue(self.name)

        # Extract executed tasks from execution spans, excluding the spans
        # we already parsed as planning spans (otherwise the planner's own
        # span trivially "executes" every task it listed).
        planning_span_ids = {
            source_span_id
            for task in planned_tasks
            if isinstance(source_span_id := task.get("source_span_id"), str)
        }
        executed_tasks = self._extract_executed_tasks(
            sorted_spans, exclude_span_ids=planning_span_ids
        )

        # Compare planned vs executed
        starved = self._find_starved_tasks(planned_tasks, executed_tasks)

        if not starved:
            return DetectionResult.no_issue(self.name)

        starvation_ratio = len(starved) / len(planned_tasks)
        if starvation_ratio < self.min_starvation_ratio:
            return DetectionResult.no_issue(self.name)

        # Negative-signal gate: if the agent executed at least as many
        # distinct tasks as it planned, unmatched planned items are almost
        # always paraphrase mismatches (e.g., planned "email_campaign"
        # executed as "newsletter_blast") rather than silently dropped
        # work. True starvation requires the executed count to actually
        # fall short of the planned count.
        if len(executed_tasks) >= len(planned_tasks):
            return DetectionResult.no_issue(self.name)

        # Calculate severity based on number and ratio of starved tasks
        severity = self.severity_range[0]
        if starvation_ratio > 0.5:
            severity += 25
        elif starvation_ratio > 0.3:
            severity += 15

        if len(starved) >= 3:
            severity += 10

        severity = min(self.severity_range[1], severity)

        starved_descriptions = [t["description"][:60] for t in starved[:5]]
        summary = (
            f"{len(starved)} of {len(planned_tasks)} planned tasks never executed: "
            f"{'; '.join(starved_descriptions)}"
        )

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=summary,
            fix_type=FixType.SWITCH_STRATEGY,
            fix_instruction=(
                "Review the planned task list and ensure all steps are executed. "
                "If tasks are intentionally skipped, document the reason."
            ),
        )

        result.add_evidence(
            description=f"{len(starved)} tasks were planned but never executed",
            span_ids=[t.get("source_span_id", "") for t in starved if t.get("source_span_id")],
            data={
                "planned_count": len(planned_tasks),
                "executed_count": len(executed_tasks),
                "starved_count": len(starved),
                "starvation_ratio": round(starvation_ratio, 2),
                "starved_tasks": [t["description"][:100] for t in starved],
            },
        )

        return result

    def _extract_planned_tasks(self, spans: list[Span]) -> list[dict[str, Any]]:
        """Extract planned tasks from spans that contain plans or task lists."""
        planned: list[dict[str, Any]] = []
        seen_descriptions: set[str] = set()

        for span in spans:
            # Check if this span likely contains a plan
            text = self._get_plan_text(span)
            if not text:
                continue

            text_lower = text.lower()
            has_plan_context = any(
                re.search(pattern, text_lower) for pattern in _PLAN_CONTEXT_PATTERNS
            )

            # Also check span kind/name for plan indicators
            is_planning_span = (
                span.kind == SpanKind.TASK
                or "plan" in span.name.lower()
                or "decompos" in span.name.lower()
                or "task" in span.name.lower()
            )

            if not has_plan_context and not is_planning_span:
                continue

            # Extract individual task items
            items = self._extract_task_items(text)
            for item in items:
                normalized = item.strip().lower()
                if normalized and normalized not in seen_descriptions:
                    seen_descriptions.add(normalized)
                    planned.append(
                        {
                            "description": item.strip(),
                            "keywords": self._extract_keywords(item),
                            "source_span_id": span.span_id,
                        }
                    )

        return planned

    def _extract_executed_tasks(
        self,
        spans: list[Span],
        exclude_span_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract executed tasks from execution-bearing spans.

        Includes TASK spans (discrete executed work), TOOL/AGENT/AGENT_TURN
        spans, plus WORKFLOW/CHAIN/RETRIEVAL which also represent actual work.
        Spans in ``exclude_span_ids`` are skipped — used to exclude the
        planning span itself from the executed list.
        """
        executed: list[dict[str, Any]] = []
        exclude_span_ids = exclude_span_ids or set()

        execution_kinds = (
            SpanKind.TOOL,
            SpanKind.AGENT,
            SpanKind.AGENT_TURN,
            SpanKind.TASK,
            SpanKind.WORKFLOW,
            SpanKind.CHAIN,
            SpanKind.RETRIEVAL,
        )

        for span in spans:
            if span.kind not in execution_kinds:
                continue
            if span.span_id in exclude_span_ids:
                continue

            # Build a description from span name, input, and output.
            # Including output data is critical: execution spans often record
            # what was done in their output (e.g., "Completed: fetch_data").
            description_parts = [span.name]
            if span.input_data:
                for key in ("task", "query", "prompt", "command", "action", "content", "input"):
                    val = span.input_data.get(key, "")
                    if isinstance(val, str) and val:
                        description_parts.append(val)
                        break
            if span.output_data:
                for key in ("output", "content", "text", "response", "result"):
                    val = span.output_data.get(key, "")
                    if isinstance(val, str) and val:
                        description_parts.append(val)
                        break

            description = " ".join(description_parts)
            keywords = self._extract_keywords(description)

            # Also keep the raw lowercased name/description for substring matching
            # so short/digit-bearing task identifiers (e.g., "task_0") match even
            # when keyword extraction strips them to a common stem.
            executed.append(
                {
                    "description": description,
                    "keywords": keywords,
                    "raw_text": description.lower(),
                    "span_id": span.span_id,
                }
            )

        return executed

    def _find_starved_tasks(
        self,
        planned: list[dict[str, Any]],
        executed: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Find planned tasks that have no matching execution span.

        Matches via (a) keyword-overlap fuzzy match and (b) substring fallback
        on the raw lowercased task description. The substring fallback is
        essential for identifier-style task names (e.g., "task_0",
        "fetch_data") where keyword extraction strips digits/underscores and
        collapses distinct tasks into a single stem.
        """
        starved: list[dict[str, Any]] = []

        for task in planned:
            task_desc = task["description"].strip().lower()
            task_kw = task["keywords"]

            matched = False
            # (b) Substring match: if the full planned task description
            # appears verbatim in any execution span's text, it's executed.
            if task_desc:
                for exec_task in executed:
                    raw = exec_task.get("raw_text", "")
                    if raw and task_desc in raw:
                        matched = True
                        break

            # (a) Keyword overlap fallback for natural-language tasks
            if not matched and task_kw:
                for exec_task in executed:
                    exec_kw = exec_task["keywords"]
                    if not exec_kw:
                        continue
                    overlap = len(task_kw & exec_kw)
                    min_size = min(len(task_kw), len(exec_kw))
                    if min_size > 0 and overlap / min_size >= self.keyword_overlap_threshold:
                        matched = True
                        break

            # Skip tasks with no usable signal (no description, no keywords)
            if not task_desc and not task_kw:
                continue

            if not matched:
                starved.append(task)

        return starved

    @staticmethod
    def _extract_task_items(text: str) -> list[str]:
        """Extract individual task items from text."""
        items: list[str] = []

        for pattern in _PLAN_ITEM_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if len(matches) >= 2:
                items.extend(m.strip() for m in matches if m.strip())
                break

        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for item in items:
            key = item.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique

    @staticmethod
    def _stem(word: str) -> str:
        """Crude Porter-lite stemmer sufficient for task-name fuzzy matching.

        Handles common English suffixes so ``vulnerability`` and
        ``vulnerabilities``, or ``assess`` and ``assessment``, collapse
        to a common form. Intentionally minimal — we only need enough
        normalisation to match paraphrased task identifiers.
        """
        for suffix in (
            "ization",
            "ational",
            "ment",
            "ness",
            "tion",
            "sion",
            "ies",
            "ing",
            "ed",
            "es",
            "ly",
            "er",
            "al",
            "s",
        ):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[: -len(suffix)]
        return word

    @classmethod
    def _extract_keywords(cls, text: str) -> set[str]:
        """Extract meaningful keywords from text for fuzzy matching.

        Words are stemmed so paraphrased task names (e.g.,
        ``vulnerability_assessment`` vs ``assess_vulnerabilities``) can
        overlap. Generic task-list words like ``task`` / ``step`` /
        ``action`` are excluded from the keyword set so their presence
        alone cannot produce a false fuzzy match.
        """
        words = re.findall(r"[a-z]+", text.lower())
        result: set[str] = set()
        for w in words:
            if len(w) <= 2 or w in _TASK_STOP_WORDS or w in _GENERIC_TASK_WORDS:
                continue
            result.add(cls._stem(w))
        return result

    @staticmethod
    def _get_plan_text(span: Span) -> str:
        """Get text from a span that might contain a plan."""
        parts: list[str] = []

        # Check output_data first (plans are usually in outputs)
        if span.output_data:
            for key in ("content", "text", "response", "plan", "decomposition", "output"):
                val = span.output_data.get(key, "")
                if isinstance(val, str) and val:
                    parts.append(val)
                    break
            # Also check for list values
            for key in ("tasks", "steps", "plan"):
                val = span.output_data.get(key)
                if isinstance(val, list):
                    parts.append("\n".join(f"- {item}" for item in val if isinstance(item, str)))

        # Check input_data for task descriptions
        if span.input_data:
            for key in ("content", "text", "prompt", "input"):
                val = span.input_data.get(key, "")
                if isinstance(val, str) and val:
                    parts.append(val)
                    break

        return " ".join(parts)
