"""Derailment detector for identifying task focus deviation and goal drift.

F6: Task Derailment Detection (MAST Taxonomy)

Detects when an agent goes off-topic or deviates from the assigned task.
One of the most common failures (7.4% in MAST-Data).

Detection Methods:
1. Semantic similarity between task and output (Jaccard fallback)
2. Topic drift via keyword novelty
3. Task coverage verification
4. Task substitution detection (authentication vs authorization)
5. Research focus mismatch (pricing vs features)
6. Content-type matching for writing tasks
7. Framework-specific benign pattern recognition

Ported from backend/app/detection/derailment.py (v1.5+).
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

# v1.3: Semantic clusters for related task concepts
TASK_CLUSTERS: dict[str, list[str]] = {
    "authentication": [
        "authenticate",
        "login",
        "password",
        "credential",
        "identity",
        "sign in",
        "sign-in",
        "logged in",
    ],
    "authorization": [
        "authorize",
        "permission",
        "access control",
        "role",
        "rbac",
        "acl",
        "privilege",
        "allowed",
    ],
    "encrypt": ["encrypt", "encryption", "cipher", "encode"],
    "decrypt": ["decrypt", "decryption", "decipher", "decode"],
    "upload": ["upload", "uploading", "send file", "put file"],
    "download": ["download", "downloading", "get file", "fetch file"],
    "create": ["create", "creating", "add new", "insert", "generate"],
    "delete": ["delete", "deleting", "remove", "drop", "destroy"],
    "frontend": [
        "frontend",
        "front-end",
        "ui",
        "user interface",
        "client-side",
        "react",
        "vue",
        "angular",
    ],
    "backend": ["backend", "back-end", "server-side", "api", "database", "server"],
    "unit test": ["unit test", "unit testing", "isolated test", "mock"],
    "integration test": ["integration test", "integration testing", "e2e", "end-to-end"],
}

SUBSTITUTION_PAIRS: list[tuple[str, str]] = [
    ("authentication", "authorization"),
    ("authorization", "authentication"),
    ("encrypt", "decrypt"),
    ("decrypt", "encrypt"),
    ("upload", "download"),
    ("download", "upload"),
    ("create", "delete"),
    ("delete", "create"),
    ("frontend", "backend"),
    ("backend", "frontend"),
    ("unit test", "integration test"),
    ("integration test", "unit test"),
]

# v1.4: Research/analysis focus terms
RESEARCH_FOCUS_CLUSTERS: dict[str, list[str]] = {
    "pricing": [
        "price",
        "pricing",
        "cost",
        "rate",
        "fee",
        "subscription",
        "tier",
        "plan",
        "dollar",
        "$",
        "budget",
        "expense",
    ],
    "features": [
        "feature",
        "functionality",
        "capability",
        "function",
        "tool",
        "option",
        "integration",
    ],
    "market": ["market", "positioning", "segment", "target", "audience", "demographic"],
    "technical": [
        "technical",
        "architecture",
        "stack",
        "technology",
        "infrastructure",
        "performance",
    ],
    "competitor": ["competitor", "competition", "rival", "alternative", "versus", "vs"],
    "strategy": ["strategy", "strategic", "approach", "roadmap", "vision", "plan"],
}

# v1.4: Content-type patterns for writing tasks
CONTENT_TYPE_PATTERNS: dict[str, list[str]] = {
    "documentation": [
        "#",
        "##",
        "###",
        "overview",
        "guide",
        "reference",
        "api",
        "endpoint",
        "parameter",
        "return",
        "example",
        "usage",
        "method",
        "function",
    ],
    "report": [
        "summary",
        "findings",
        "analysis",
        "conclusion",
        "recommendation",
        "result",
        "data",
        "metric",
    ],
    "code": [
        "def ",
        "function ",
        "class ",
        "import ",
        "const ",
        "let ",
        "var ",
        "return ",
        "if ",
        "for ",
        "while ",
    ],
}

# v1.5: Framework-specific benign patterns
FRAMEWORK_BENIGN_PATTERNS: dict[str, list[str]] = {
    "AG2": [
        r"\b(?:tool_code|tool_result|human_input)\b",
        r"\b(?:terminate|TERMINATE)\b",
        r"\b(?:groupchat|speaker_selection)\b",
        r"\b(?:UserProxy|AssistantAgent|ConversableAgent)\b",
    ],
    "MetaGPT": [
        r"\b(?:ProductManager|Architect|ProjectManager|Engineer|QAEngineer)\b",
        r"\b(?:WriteDesign|WritePRD|WriteCode|WriteTest)\b",
        r"\b(?:Message|ActionOutput|RoleReactMode)\b",
        r"\b(?:memory|working_memory|recovered)\b",
    ],
    "ChatDev": [
        r"\b(?:CEO|CTO|CPO|Programmer|Reviewer|Designer|Tester)\b",
        r"\b(?:PhaseType|ChatChain|RolePlay)\b",
        r"\b(?:software_info|chat_env)\b",
    ],
    "Camel": [
        r"\b(?:AI_USER|AI_ASSISTANT|TASK_SPECIFIER)\b",
        r"\b(?:role_playing|inception_prompt)\b",
        r"\b(?:SystemMessage|ChatMessage)\b",
    ],
    "LangGraph": [
        r"\b(?:StateGraph|MessageGraph|CompiledGraph)\b",
        r"\b(?:add_node|add_edge|add_conditional_edges)\b",
        r"\b(?:START|END|should_continue)\b",
    ],
    "CrewAI": [
        r"\b(?:Agent|Task|Crew|Process)\b",
        r"\b(?:backstory|goal|expected_output)\b",
        r"\b(?:sequential|hierarchical)\b",
    ],
}

# v1.5: Common agent coordination patterns (benign)
COORDINATION_PATTERNS: list[str] = [
    r"\b(?:delegating to|handing off to|passing to|coordinating with)\b",
    r"\b(?:agent \d+|step \d+|phase \d+)\b",
    r"\b(?:my role|my task|assigned to me)\b",
    r"\b(?:waiting for|blocked on|depends on)\b",
    r"\b(?:reporting back|returning result|providing feedback)\b",
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
        "and",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "not",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "can",
        "don",
        "now",
        "it",
        "its",
        "this",
        "that",
    }
)


class DerailmentDetector(BaseDetector):
    """Detects task derailment -- when an agent goes off-topic or deviates from the assigned task.

    Uses semantic similarity (Jaccard fallback), topic drift, task coverage,
    task substitution detection, research focus mismatch, and content-type
    matching to identify goal drift.

    Span convention:
        Each span is expected to carry ``input_data`` with a ``"task"`` key and
        ``output_data`` with a ``"content"`` key. An optional ``"context"`` key
        in ``input_data`` enriches the analysis. The detector evaluates every
        matching span in the trace independently.
    """

    name = "derailment"
    description = "Detects task focus deviation and goal drift"
    version = "1.7.0"
    platforms: list[Platform] = []  # All platforms
    severity_range = (0, 100)
    realtime_capable = False

    # Default thresholds
    similarity_threshold: float = 0.3
    drift_threshold: float = 0.5
    min_output_length: int = 20
    confidence_scaling: float = 1.0
    task_coverage_threshold: float = 0.6

    # --- Internal helpers (ported faithfully from backend) ---

    @staticmethod
    def _extract_key_terms(text: str) -> set[str]:
        """Extract key terms from *text*, filtering stopwords."""
        words = text.lower().split()
        return {w for w in words if len(w) > 2 and w not in _STOPWORDS}

    @staticmethod
    def _compute_similarity_jaccard(text1: str, text2: str) -> float:
        """Jaccard similarity between two texts (word-level)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def _compute_topic_drift(
        self,
        task: str,
        output: str,
        context: Optional[str] = None,
    ) -> tuple[float, float]:
        """Compute topic drift and task coverage.

        Returns:
            (drift_score, task_coverage)
        """
        task_terms = self._extract_key_terms(task)
        output_terms = self._extract_key_terms(output)

        if not task_terms:
            return 0.0, 1.0

        overlap = task_terms & output_terms
        coverage = len(overlap) / len(task_terms)

        if context:
            context_terms = self._extract_key_terms(context)
            new_terms = output_terms - task_terms - context_terms
        else:
            new_terms = output_terms - task_terms

        novelty_ratio = len(new_terms) / max(len(output_terms), 1)
        drift_score = (1 - coverage) * 0.6 + novelty_ratio * 0.4
        return min(drift_score, 1.0), coverage

    @staticmethod
    def _count_cluster_matches(text: str, concept: str) -> int:
        """Count how many terms from a concept's cluster appear in *text*."""
        if concept not in TASK_CLUSTERS:
            return 1 if concept in text else 0

        count = 0
        text_lower = text.lower()
        for term in TASK_CLUSTERS[concept]:
            if term in text_lower:
                count += 1
        if concept in text_lower:
            count += 1
        return count

    @staticmethod
    def _count_focus_matches(text: str, focus: str) -> int:
        """v1.4: Count matches for a research focus cluster."""
        if focus not in RESEARCH_FOCUS_CLUSTERS:
            return 1 if focus in text.lower() else 0

        count = 0
        text_lower = text.lower()
        for term in RESEARCH_FOCUS_CLUSTERS[focus]:
            if term in text_lower:
                count += 1
        return count

    def _detect_research_focus_mismatch(
        self,
        task: str,
        output: str,
    ) -> tuple[bool, Optional[str]]:
        """v1.4: Detect when research/analysis focuses on wrong aspect."""
        task_lower = task.lower()

        research_indicators = [
            "research",
            "analyze",
            "analysis",
            "study",
            "investigate",
            "examine",
            "evaluate",
        ]
        if not any(ind in task_lower for ind in research_indicators):
            return False, None

        requested_focus: Optional[str] = None
        for focus in RESEARCH_FOCUS_CLUSTERS:
            if focus in task_lower:
                requested_focus = focus
                break

        if not requested_focus:
            return False, None

        output_lower = output.lower()
        focus_counts: dict[str, int] = {}
        for focus in RESEARCH_FOCUS_CLUSTERS:
            focus_counts[focus] = self._count_focus_matches(output_lower, focus)

        requested_count = focus_counts.get(requested_focus, 0)

        for other_focus, other_count in focus_counts.items():
            if (
                other_focus != requested_focus
                and other_count >= 2
                and other_count > requested_count * 1.5
            ):
                return True, (
                    f"Task asks for '{requested_focus}' research but output focuses on "
                    f"'{other_focus}' ({other_count} matches vs {requested_count} for {requested_focus})"
                )

        return False, None

    @staticmethod
    def _matches_content_type(output: str, content_type: str) -> bool:
        """v1.4: Check if output matches the expected content type."""
        if content_type not in CONTENT_TYPE_PATTERNS:
            return False

        patterns = CONTENT_TYPE_PATTERNS[content_type]
        output_lower = output.lower()
        matches = sum(1 for p in patterns if p.lower() in output_lower)
        return matches >= 3

    @staticmethod
    def _is_writing_task(task: str) -> tuple[bool, Optional[str]]:
        """v1.4: Check if task is a writing task and identify expected content type."""
        task_lower = task.lower()
        writing_indicators: dict[str, list[str]] = {
            "documentation": [
                "write documentation",
                "create documentation",
                "document",
                "api documentation",
                "write docs",
            ],
            "report": ["write report", "create report", "prepare report", "generate report"],
            "code": ["write code", "create code", "implement", "develop", "build"],
        }
        for content_type, indicators in writing_indicators.items():
            if any(ind in task_lower for ind in indicators):
                return True, content_type
        return False, None

    def _detect_task_substitution(
        self,
        task: str,
        output: str,
    ) -> tuple[bool, Optional[str]]:
        """Detect if the output addresses a different but related task.

        v1.3: Uses semantic clusters to detect when output focuses on
        a related but different concept than what the task requested.
        """
        task_lower = task.lower()
        output_lower = output.lower()

        for task_concept, wrong_concept in SUBSTITUTION_PAIRS:
            if task_concept not in task_lower:
                continue

            task_concept_count = self._count_cluster_matches(output_lower, task_concept)
            wrong_concept_count = self._count_cluster_matches(output_lower, wrong_concept)

            if wrong_concept_count >= 2 and wrong_concept_count > task_concept_count:
                return True, (
                    f"Task asks for '{task_concept}' but output focuses on "
                    f"'{wrong_concept}' ({wrong_concept_count} matches vs {task_concept_count})"
                )

        return False, None

    def _is_task_addressed(
        self,
        task: str,
        output: str,
        coverage: float,
    ) -> bool:
        """Check if the core task is addressed in the output.

        v1.1+: Reduces over-detection on outputs that complete the task
        but include helpful additional information.
        """
        if coverage >= self.task_coverage_threshold:
            return True

        task_lower = task.lower()
        output_lower = output.lower()

        # v1.7: Task echo detection
        task_prefix = task_lower[: min(80, len(task_lower))].strip()
        if task_prefix and task_prefix in output_lower[: len(task_prefix) + 50]:
            return True

        # v1.7: Patch/diff output
        if "diff --git" in output or ("applied fix" in output_lower and "---" in output):
            return True

        # v1.7: Answer output
        output_start = output_lower[:40].strip()
        if output_start.startswith(
            ("answer:", "to answer", "the answer", "i followed these steps")
        ):
            return True

        # Action verb patterns
        action_patterns: list[tuple[str, list[str]]] = [
            ("analyze", ["analysis", "analyzed", "analyzing", "findings", "results", "examined"]),
            ("debug", ["fixed", "bug", "issue", "resolved", "debugging", "error", "problem"]),
            ("fix", ["fixed", "resolved", "corrected", "patched", "repaired"]),
            (
                "review",
                [
                    "review",
                    "reviewed",
                    "reviewing",
                    "found",
                    "identified",
                    "issue",
                    "bug",
                    "problem",
                    "style",
                    "suggestion",
                    "improvement",
                    "code",
                ],
            ),
            ("write", ["here", "following", "created", "written", "wrote"]),
            ("create", ["created", "here", "following", "built", "made"]),
            ("test", ["tested", "testing", "test", "passed", "failed", "verified", "checked"]),
            ("optimize", ["optimized", "improved", "faster", "performance", "efficient"]),
            ("document", ["documentation", "documented", "docs", "description"]),
            ("implement", ["implemented", "implementation", "built", "created", "added"]),
            ("check", ["checked", "verified", "confirmed", "validated", "found"]),
            ("validate", ["validated", "verified", "confirmed", "checked"]),
        ]
        for action, indicators in action_patterns:
            if action in task_lower:
                if any(ind in output_lower for ind in indicators):
                    return True

        # Focus terms
        focus_terms: dict[str, list[str]] = {
            "bug": ["bug", "issue", "error", "problem", "defect", "fix"],
            "security": ["security", "vulnerability", "secure", "auth", "permission"],
            "performance": ["performance", "speed", "fast", "slow", "optimize", "efficient"],
            "style": ["style", "format", "convention", "naming", "readable"],
            "quality": ["quality", "clean", "maintainable", "readable", "best practice"],
        }
        for focus, related_terms in focus_terms.items():
            if focus in task_lower:
                if any(term in output_lower for term in related_terms):
                    return True

        return False

    @staticmethod
    def _has_framework_benign_pattern(text: str, framework: Optional[str] = None) -> bool:
        """v1.5: Check if text contains framework-specific benign patterns."""
        if framework and framework in FRAMEWORK_BENIGN_PATTERNS:
            for pattern in FRAMEWORK_BENIGN_PATTERNS[framework]:
                if re.search(pattern, text, re.IGNORECASE):
                    return True

        for pattern in COORDINATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _calibrate_confidence(
        self,
        similarity: float,
        drift_score: float,
        severity_label: str,
        output_length: int,
    ) -> float:
        """Calibrate confidence based on evidence strength."""
        severity_weight = {
            "none": 0.0,
            "minor": 0.6,
            "moderate": 0.75,
            "severe": 0.9,
        }.get(severity_label, 0.5)

        length_factor = min(1.0, output_length / 200)
        signal_strength = (drift_score + (1 - similarity)) / 2
        base_confidence = severity_weight * 0.4 + signal_strength * 0.4 + length_factor * 0.2
        return round(min(0.99, base_confidence * self.confidence_scaling), 4)

    # --- Core detect_single: runs detection for one (task, output) pair ---

    def _detect_single(
        self,
        task: str,
        output: str,
        context: Optional[str] = None,
        agent_name: Optional[str] = None,
        framework: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Run derailment detection on a single task/output pair.

        Returns a dict with detection results if an issue is found, else None.
        """
        if len(output) < self.min_output_length:
            return None

        similarity = self._compute_similarity_jaccard(task, output)
        drift_score, task_coverage = self._compute_topic_drift(task, output, context)

        # v1.4: Content-type matching for writing tasks
        is_writing, expected_content_type = self._is_writing_task(task)
        if is_writing and expected_content_type:
            if self._matches_content_type(output, expected_content_type):
                return None

        # v1.4: Research focus mismatch
        focus_mismatch, focus_desc = self._detect_research_focus_mismatch(task, output)

        # v1.2: Task substitution
        substitution_detected, substitution_desc = self._detect_task_substitution(task, output)

        # v1.1: Task addressed check
        task_addressed = self._is_task_addressed(task, output, task_coverage)

        # v1.5: Framework-specific benign patterns
        has_framework_benign = self._has_framework_benign_pattern(output, framework)

        # Detection logic
        if substitution_detected:
            detected = True
        elif focus_mismatch:
            detected = True
        elif has_framework_benign and task_addressed:
            detected = False
        elif task_addressed:
            detected = similarity < 0.1
        else:
            detected = similarity < self.similarity_threshold or drift_score > self.drift_threshold

        if not detected:
            return None

        # Determine severity
        if drift_score > 0.8 or similarity < 0.1:
            severity_label = "severe"
        elif drift_score > 0.6 or similarity < 0.2:
            severity_label = "moderate"
        else:
            severity_label = "minor"

        confidence = self._calibrate_confidence(
            similarity=similarity,
            drift_score=drift_score,
            severity_label=severity_label,
            output_length=len(output),
        )

        # v1.6: Exploration tolerance
        if task and output:
            last_portion = output[int(len(output) * 0.7) :]
            task_words = task.lower().split()[:10]
            if task_words:
                final_coverage = sum(
                    1 for word in task_words if word in last_portion.lower()
                ) / len(task_words)
                if final_coverage > 0.3:
                    confidence *= 0.6

        # Map severity label to numeric
        severity_map = {"minor": 30, "moderate": 55, "severe": 85}
        severity = severity_map.get(severity_label, 40)

        agent_prefix = f"Agent '{agent_name}'" if agent_name else "Agent"
        explanation = (
            f"{agent_prefix} deviated from the assigned task. "
            f"Task-output similarity: {similarity:.2f} (threshold: {self.similarity_threshold}). "
            f"Topic drift score: {drift_score:.2f} (threshold: {self.drift_threshold}). "
            f"Task coverage: {task_coverage:.2f}."
        )

        return {
            "detected": True,
            "severity": severity,
            "confidence": confidence,
            "summary": explanation,
            "evidence": {
                "similarity": round(similarity, 4),
                "drift_score": round(drift_score, 4),
                "task_coverage": round(task_coverage, 4),
                "task_addressed": task_addressed,
                "substitution_detected": substitution_detected,
                "substitution_description": substitution_desc,
                "focus_mismatch_detected": focus_mismatch,
                "focus_mismatch_description": focus_desc,
                "framework_benign_pattern": has_framework_benign,
                "framework": framework,
                "output_length": len(output),
            },
        }

    # --- Trace-level detect (BaseDetector interface) ---

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect task derailment across all spans in a trace.

        Examines each span for ``input_data.task`` and ``output_data.content``.
        Returns the highest-severity finding.
        """
        framework = trace.metadata.tags.get("framework") if trace.metadata else None
        worst: Optional[dict[str, Any]] = None

        for span in trace.spans:
            task = (span.input_data or {}).get("task", "")
            output = (span.output_data or {}).get("content", "")
            context = (span.input_data or {}).get("context", "")
            agent_name = span.name

            if not task or not output:
                continue

            finding = self._detect_single(
                task=task,
                output=output,
                context=context or None,
                agent_name=agent_name,
                framework=framework,
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
                "Add explicit task reminders in the prompt. Consider using: "
                "'Stay focused on the following task: [TASK]. Do not address unrelated topics.'"
            ),
        )
        result.confidence = worst["confidence"]
        result.add_evidence(
            description=worst["summary"],
            data=worst["evidence"],
        )
        return result
