"""Withholding detector for identifying information withholding by agents.

Detects F8: Information Withholding (MAST Taxonomy):
- Agent discovers important information but doesn't pass it on
- Agent summarizes away critical details
- Agent selectively omits negative findings
- Agent's output is significantly less informative than its internal state

Version History:
- v1.0: Initial pisama-core port from backend v1.6
"""

import re
from typing import Any, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Trace

# Reasoning trace markers -- if internal_state shows deliberate reasoning
# process, the agent is meant to expose its thinking, not forward it verbatim.
REASONING_MARKERS = [
    r"\b(?:thinking|plan|approach|reasoning|strategy|analysis)\s*:",
    r"\b(?:step \d|first|then|next|finally)\b.*\b(?:step \d|then|next|finally)\b",
    r"\blet me (?:think|consider|analyze|plan)\b",
]


class WithholdingDetector(BaseDetector):
    """Detects information withholding -- agent hides info from peers.

    Compares agent's internal findings with what it communicates to
    downstream agents or final output.
    """

    name = "withholding"
    description = "Detects information withholding by agents"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (10, 80)
    realtime_capable = False

    # Configuration
    critical_retention_threshold: float = 0.8
    detail_retention_threshold: float = 0.6

    # Critical information patterns that should be passed on
    CRITICAL_PATTERNS = [
        (r"\b(error|failure|exception|bug|issue|problem)\b", "error_condition"),
        (r"\b(warning|caution|alert|risk|danger)\b", "warning"),
        (r"\b(critical|urgent|important|essential|crucial)\b", "priority_marker"),
        (r"\b(security|vulnerability|exploit|breach)\b", "security_issue"),
        (r"\b(deadline|due date|expires?|timeout)\b", "time_constraint"),
        (r"\b(cost|price|fee|charge|expense)\s*[:=]?\s*\$?\d+", "financial_info"),
        (r"\b(blocked|blocker|impediment|obstacle)\b", "blocker"),
        (r"\b(deprecated|obsolete|outdated|legacy)\b", "deprecation"),
        (r"\bnot\s+(?:working|functional|available|supported)\b", "unavailability"),
        (r"\b(failed|unsuccessful|unable|cannot)\b", "failure_indicator"),
    ]

    # Importance-weighted critical patterns
    CRITICAL_PATTERNS_WEIGHTED = [
        (r"\b(error|failure|exception|bug)\b", "error_condition", 1.0),
        (r"\b(security|vulnerability|exploit|breach)\b", "security_issue", 1.0),
        (r"\b(blocked|blocker|impediment)\b", "blocker", 0.9),
        (r"\b(issue|problem)\b", "issue", 0.8),
        (r"\b(warning|caution|alert|risk|danger)\b", "warning", 0.7),
        (r"\b(critical|urgent|important)\b", "priority_marker", 0.6),
        (r"\b(deadline|due date|expires?|timeout)\b", "time_constraint", 0.5),
        (r"\b(cost|price|fee|charge|expense)\s*[:=]?\s*\$?\d+", "financial_info", 0.5),
        (r"\b(deprecated|obsolete|outdated|legacy)\b", "deprecation", 0.4),
        (r"\bnot\s+(?:working|functional|available|supported)\b", "unavailability", 0.6),
        (r"\b(failed|unsuccessful|unable|cannot)\b", "failure_indicator", 0.7),
    ]

    # Negative finding patterns
    NEGATIVE_FINDING_PATTERNS = [
        r"\b(no|not|none|neither|never)\s+\w+",
        r"\b(missing|absent|lacking|without)\b",
        r"\b(incorrect|wrong|invalid|bad)\b",
        r"\b(rejected|denied|refused|declined)\b",
        r"\b(insufficient|inadequate|incomplete)\b",
    ]

    # Summary indicators
    SUMMARY_PATTERNS = [
        r"\b(in summary|to summarize|in brief|briefly)\b",
        r"\b(the main point|key takeaway|bottom line)\b",
        r"\b(essentially|basically|simply put)\b",
        r"\b(overall|in general|generally speaking)\b",
    ]

    # Tasks that explicitly request summary/list format
    SUMMARY_TASK_PATTERNS = [
        r"\bsummar(?:y|ize|ise)\b",
        r"\bbrief(?:ly)?\b",
        r"\boverview\b",
        r"\bkey\s+(?:points?|findings?|takeaways?)\b",
        r"\bhighlights?\b",
        r"\blist\b",
    ]

    # Expanded patterns for condensed output tasks
    CONDENSED_OUTPUT_TASKS = [
        r"\b(?:get|give|provide)\s+(?:me\s+)?(?:the\s+)?(?:main|key|important|top)\b",
        r"\b(?:executive\s+)?summar(?:y|ize|ise)\b",
        r"\b(?:brief|short|concise|quick)\s+(?:overview|summary|report)\b",
        r"\btl;?dr\b",
        r"\bhighlights?\b",
        r"\b(?:bullet\s+)?(?:points?|list)\b",
        r"\bconclusion\b",
        r"\b(?:bottom\s+line|takeaway)\b",
        r"\b(?:in\s+)?one\s+(?:sentence|paragraph|word)\b",
        r"\babstract\b",
        r"\b(?:max(?:imum)?|under|less\s+than)\s+\d+\s*(?:word|char|line)",
        r"\b(?:keep\s+it\s+)?(?:short|brief|concise)\b",
    ]

    # Roles expected to summarize
    SUMMARIZING_ROLES = [
        "manager",
        "coordinator",
        "reporter",
        "summarizer",
        "summary",
        "lead",
        "pm",
        "director",
        "executive",
        "supervisor",
    ]

    # Patterns indicating full information is accessible
    FULL_ACCESS_PATTERNS = [
        r"\bsee\s+(?:full|complete|detailed)\b",
        r"\bfull\s+(?:details?|report|analysis)\s+(?:at|in|available)\b",
        r"\bmore\s+(?:details?|information)\s+(?:at|in|available)\b",
        r"\blink(?:s|ed)?\s+(?:to|below|above)\b",
        r"\b(?:attached|appendix|supplement)\b",
        r"\brefer(?:ence)?\s+to\b",
        r"\bfor\s+(?:more|additional|complete)\s+(?:details?|information)\b",
    ]

    # Patterns indicating organized/structured output
    STRUCTURED_OUTPUT_PATTERNS = [
        r"(?:^|\n)\s*[-•*]\s+",
        r"(?:^|\n)\s*\d+[.)]\s+",
        r"(?:^|\n)\s*#{1,6}\s+",
        r'"[^"]+":\s*[{\[]',
        r"\n\s+\w+:\s+",
    ]

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect information withholding in a trace.

        Looks for agent spans with internal state and output data,
        comparing what was discovered vs what was communicated.
        Also supports direct golden-dataset-style input via trace metadata.
        """
        # Check for golden-dataset-style input
        internal_state = trace.metadata.custom.get("internal_state", "")
        agent_output = trace.metadata.custom.get("agent_output", "")

        if internal_state and agent_output:
            task_context = trace.metadata.custom.get("task_context")
            agent_role = trace.metadata.custom.get("agent_role")
            return self._detect_from_text(internal_state, agent_output, task_context, agent_role)

        # Extract from trace spans
        agent_spans = trace.get_spans_by_kind(SpanKind.AGENT)
        if not agent_spans:
            agent_spans = trace.get_spans_by_kind(SpanKind.AGENT_TURN)

        if not agent_spans:
            return DetectionResult.no_issue(self.name)

        all_issues: list[str] = []
        max_severity = 0

        for span in agent_spans:
            internal = ""
            output = ""

            if span.input_data:
                internal = str(
                    span.input_data.get("internal_state", span.input_data.get("context", ""))
                )
            if span.output_data:
                output = str(span.output_data.get("content", span.output_data.get("response", "")))

            # Also check events for reasoning traces
            for event in span.events:
                if event.name in ("reasoning", "thinking", "internal"):
                    internal += " " + str(event.attributes.get("content", ""))

            if not internal or not output:
                continue

            inner_result = self._detect_from_text(internal, output)
            if inner_result.detected:
                all_issues.append(inner_result.summary)
                max_severity = max(max_severity, inner_result.severity)

        if not all_issues:
            return DetectionResult.no_issue(self.name)

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=max_severity,
            summary=all_issues[0],
            fix_type=FixType.ESCALATE,
            fix_instruction="Ensure agent passes all critical findings to downstream agents.",
        )
        for issue in all_issues:
            result.add_evidence(
                description=issue,
                span_ids=[s.span_id for s in agent_spans],
            )
        return result

    def _detect_from_text(
        self,
        internal_state: str,
        agent_output: str,
        task_context: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> DetectionResult:
        """Core detection logic operating on text inputs.

        Faithfully ports the backend InformationWithholdingDetector.detect() method.
        """
        if not internal_state or not agent_output:
            return DetectionResult.no_issue(self.name)

        issues: list[dict[str, Any]] = []

        # False positive indicator checks
        is_summary_task = self._is_summary_task(task_context or internal_state)
        has_full_access = self._has_full_access_indicators(agent_output)
        is_structured = self._is_structured_output(agent_output)
        is_condensed_expected = self._is_condensed_output_expected(
            task_context or "", internal_state
        )
        is_summarizing_role = self._is_summarizing_role(agent_role)

        # More lenient thresholds for summarizing contexts
        effective_critical_threshold = self.critical_retention_threshold
        effective_detail_threshold = self.detail_retention_threshold

        if is_summarizing_role or is_condensed_expected:
            effective_critical_threshold = 0.6
            effective_detail_threshold = 0.4

        # Reasoning trace detection
        reasoning_marker_count = sum(
            1 for pattern in REASONING_MARKERS if re.search(pattern, internal_state, re.IGNORECASE)
        )
        if reasoning_marker_count >= 2:
            effective_critical_threshold *= 0.7

        # Extract critical items from internal state
        internal_critical = self._extract_critical_items(internal_state)
        found, retained, missing_critical = self._check_item_retention(
            internal_critical, agent_output
        )

        # Check critical item retention
        retention_ratio = retained / found if found > 0 else 1.0

        # Detect critical omissions
        if retention_ratio < effective_critical_threshold and missing_critical:
            for missed in missing_critical[:5]:
                issues.append(
                    {
                        "type": "critical_omission",
                        "severity": "severe",
                        "info": missed,
                        "description": f"Critical information not passed on: {missed}",
                    }
                )

        # Check for negative finding suppression
        internal_negatives = self._extract_negative_findings(internal_state)
        output_negatives = self._extract_negative_findings(agent_output)

        if len(internal_negatives) > len(output_negatives) + 2:
            suppressed_count = len(internal_negatives) - len(output_negatives)
            issues.append(
                {
                    "type": "negative_suppression",
                    "severity": "moderate",
                    "info": f"{suppressed_count} negative findings",
                    "description": f"Agent suppressed {suppressed_count} negative findings from output",
                }
            )

        # Check entity retention
        internal_entities = self._extract_entities(internal_state)
        output_entities = self._extract_entities(agent_output)

        if internal_entities:
            entity_retention = len(internal_entities & output_entities) / len(internal_entities)
            if entity_retention < effective_detail_threshold:
                lost_entities = internal_entities - output_entities
                issues.append(
                    {
                        "type": "detail_loss",
                        "severity": "minor",
                        "info": f"{len(lost_entities)} entities/details",
                        "description": f"Lost {len(lost_entities)} specific details in output",
                    }
                )

        # Check for over-summarization
        internal_density = self._calculate_information_density(internal_state)
        output_density = self._calculate_information_density(agent_output)

        if internal_density > 0 and output_density < internal_density * 0.5:
            if re.search(r"|".join(self.SUMMARY_PATTERNS), agent_output, re.IGNORECASE):
                issues.append(
                    {
                        "type": "context_stripping",
                        "severity": "minor",
                        "info": "excessive summarization",
                        "description": "Output significantly less detailed than internal state",
                    }
                )

        # Reduce false positives based on task type and output characteristics
        if issues:
            if is_summary_task or is_condensed_expected or is_summarizing_role:
                issues = [
                    i for i in issues if i["type"] not in ("context_stripping", "detail_loss")
                ]

            if has_full_access:
                issues = [i for i in issues if i["severity"] in ("severe", "critical")]

            if is_structured:
                issues = [i for i in issues if i["type"] != "detail_loss"]

        if not issues:
            result = DetectionResult.no_issue(self.name)
            result.metadata = {
                "information_retention_ratio": retention_ratio,
                "critical_items_found": found,
                "critical_items_reported": retained,
            }
            return result

        # Calculate severity
        severity_map = {"critical": 80, "severe": 65, "moderate": 45, "minor": 25}
        max_sev = max(severity_map.get(i["severity"], 25) for i in issues)

        # Confidence
        confidence = min(0.95, 0.6 + (len(issues) * 0.1))

        issue_types = set(i["type"] for i in issues)
        summary = (
            f"Detected {len(issues)} withholding issue(s): {', '.join(issue_types)}. "
            f"Critical info retention: {retention_ratio:.1%}"
        )

        # Suggest fix
        if any(i["type"] == "critical_omission" for i in issues):
            fix_instruction = "Ensure agent passes all critical findings to downstream agents"
        elif any(i["type"] == "negative_suppression" for i in issues):
            fix_instruction = "Configure agent to report negative findings, not just positive ones"
        else:
            fix_instruction = "Review agent's summarization settings to preserve more detail"

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=max_sev,
            summary=summary,
            fix_type=FixType.ESCALATE,
            fix_instruction=fix_instruction,
        )
        result.confidence = confidence
        result.metadata = {
            "information_retention_ratio": retention_ratio,
            "critical_items_found": found,
            "critical_items_reported": retained,
            "issue_types": list(issue_types),
        }
        for issue in issues:
            result.add_evidence(description=issue["description"])
        return result

    # ------------------------------------------------------------------
    # Extraction and analysis helpers
    # ------------------------------------------------------------------

    def _extract_critical_items(self, text: str) -> list[tuple[str, str, str]]:
        """Extract critical information items from text."""
        items: list[tuple[str, str, str]] = []
        for pattern, item_type in self.CRITICAL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                items.append((match.group(), item_type, context))
        return items

    def _extract_negative_findings(self, text: str) -> list[str]:
        """Extract negative findings from text."""
        findings: list[str] = []
        for pattern in self.NEGATIVE_FINDING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                findings.append(context)
        return findings

    def _is_summary_task(self, task: str) -> bool:
        """Check if task explicitly requests summary/list format."""
        if not task:
            return False
        for pattern in self.SUMMARY_TASK_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                return True
        return False

    def _is_condensed_output_expected(self, task: str, context: str = "") -> bool:
        """Determine if task expects condensed/summarized output."""
        combined = f"{task} {context}".lower()
        for pattern in self.CONDENSED_OUTPUT_TASKS:
            if re.search(pattern, combined, re.IGNORECASE):
                return True
        return False

    def _is_summarizing_role(self, role: Optional[str]) -> bool:
        """Check if agent role is expected to summarize."""
        if not role:
            return False
        role_lower = role.lower()
        return any(r in role_lower for r in self.SUMMARIZING_ROLES)

    def _has_full_access_indicators(self, text: str) -> bool:
        """Check if output provides access to full information."""
        for pattern in self.FULL_ACCESS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _is_structured_output(self, text: str) -> bool:
        """Check if output is well-structured (organized, not withheld)."""
        structure_count = 0
        for pattern in self.STRUCTURED_OUTPUT_PATTERNS:
            if re.search(pattern, text, re.MULTILINE):
                structure_count += 1
        return structure_count >= 2

    def _calculate_information_density(self, text: str) -> float:
        """Calculate information density of text."""
        if not text:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        stopwords = {
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
            "can",
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
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "what",
            "which",
        }

        substantive = sum(1 for w in words if w.lower() not in stopwords and len(w) > 2)
        return substantive / len(words)

    def _extract_entities(self, text: str) -> set[str]:
        """Extract named entities and key terms."""
        entities: set[str] = set()

        # Numbers with units
        numbers = re.findall(
            r"\b\d+(?:\.\d+)?(?:\s*(?:%|dollars?|USD|EUR|GB|MB|KB|ms|seconds?|minutes?|hours?|days?))?",
            text,
        )
        entities.update(numbers)

        # Capitalized terms
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        entities.update(caps)

        # Technical terms
        tech = re.findall(
            r"\b(?:API|URL|HTTP|JSON|XML|SQL|SDK|REST|OAuth|JWT|AWS|GCP|Azure)\b",
            text,
            re.IGNORECASE,
        )
        entities.update(e.upper() for e in tech)

        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        entities.update(quoted)

        return entities

    def _check_item_retention(
        self,
        internal_items: list[tuple[str, str, str]],
        output_text: str,
    ) -> tuple[int, int, list[str]]:
        """Check how many critical items are retained in output."""
        found = len(internal_items)
        retained_count = 0
        missing: list[str] = []

        output_lower = output_text.lower()

        for item, item_type, context in internal_items:
            item_found = item.lower() in output_lower
            context_words = set(context.lower().split())
            context_overlap = len(context_words & set(output_lower.split())) / max(
                len(context_words), 1
            )

            if item_found or context_overlap > 0.5:
                retained_count += 1
            else:
                missing.append(f"{item_type}: {item}")

        return found, retained_count, missing
