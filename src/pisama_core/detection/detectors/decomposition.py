"""Decomposition detector for identifying task breakdown failures.

Detects F2: Poor Task Decomposition (MAST Taxonomy):
- Subtasks that are impossible or ill-defined
- Missing dependencies between subtasks
- Circular dependencies
- Subtasks that duplicate work
- Subtasks that are too large or too granular
- Subtasks that are too vague/non-actionable
- Steps unrelated to the task
- Steps in wrong logical order
- Task requirements not covered by any step

Version History:
- v1.0: Initial pisama-core port from backend v1.9
"""

import re
from dataclasses import dataclass
from typing import Any, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Trace

# Words that indicate vague/non-actionable steps
VAGUE_INDICATORS = [
    "etc",
    "various",
    "miscellaneous",
    "general",
    "overall",
    "appropriate",
    "as needed",
    "if necessary",
    "possibly",
    "might",
    "maybe",
    "could potentially",
    "consider",
    "high-level",
    "broadly",
    "generally speaking",
    "explore options",
    "look into",
    "think about",
    "strategy",
    "approach",
    "framework",
]

# Words that indicate a step is too complex/broad
COMPLEXITY_INDICATORS = [
    "entire",
    "complete",
    "full",
    "all",
    "whole",
    "comprehensive",
    "end-to-end",
    "everything",
    "system",
    "platform",
    "infrastructure",
    "architecture",
    "refactor",
    "redesign",
    "rebuild",
    "rewrite",
    "migrate",
    "transform",
    "overhaul",
]

# Words that indicate a complex task requiring more decomposition
COMPLEX_TASK_INDICATORS = [
    "system",
    "platform",
    "application",
    "service",
    "authentication",
    "authorization",
    "database",
    "migration",
    "refactor",
    "integration",
    "infrastructure",
    "deployment",
    "pipeline",
]

# Words that indicate a simple task that doesn't require decomposition
SIMPLE_TASK_INDICATORS = [
    "simple",
    "basic",
    "quick",
    "small",
    "minor",
    "single",
    "one",
    "just",
    "only",
    "add",
    "fix",
    "update",
    "change",
    "tweak",
    "button",
    "field",
    "input",
    "label",
    "text",
    "feature",
    "function",
    "method",
    "endpoint",
]

# Patterns indicating direct implementation (not decomposition)
DIRECT_IMPLEMENTATION_PATTERNS = [
    r"\b(?:dive\s+right\s+in|straightforward|directly|right\s+away)\b",
    r"\b(?:simple|easy|quick)\s+(?:approach|implementation|solution)\b",
    r"\b(?:here\'s|here\s+is)\s+(?:the|my|a)\s+(?:code|implementation|solution)\b",
    r"\b(?:i\'ll|let\s+me)\s+(?:just|simply|directly)\b",
    r"\b(?:no\s+need\s+(?:to|for)|without\s+(?:the\s+)?need)\b",
    r"\bstraightforward\s+approach\b",
]


@dataclass
class Subtask:
    """A parsed subtask from a decomposition."""

    id: str
    description: str
    dependencies: list[str]
    assigned_agent: Optional[str] = None
    estimated_complexity: Optional[str] = None


class DecompositionDetector(BaseDetector):
    """Detects poor task decomposition -- subtasks ill-defined or impossible.

    Analyzes task breakdown for logical issues, dependency problems,
    coverage gaps, vague steps, overly complex steps, irrelevant steps,
    wrong ordering, and missing requirements.
    """

    name = "decomposition"
    description = "Detects task breakdown failures in agent decomposition"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (10, 90)
    realtime_capable = False

    # Configuration
    min_subtasks: int = 2
    max_subtasks: int = 15
    check_dependencies: bool = True

    # Common articles/prepositions that shouldn't be treated as nouns
    _DEP_STOP_WORDS = frozenset(
        {
            "a",
            "an",
            "the",
            "it",
            "its",
            "to",
            "in",
            "on",
            "at",
            "by",
            "of",
            "for",
        }
    )

    # Common ordering dependency pairs
    _ORDERING_DEPS = [
        ({"build", "compile", "package"}, {"deploy", "release", "ship", "publish"}),
        (
            {"validate", "verify", "check", "lint", "test"},
            {"deploy", "release", "ship", "publish", "merge"},
        ),
        (
            {"create account", "register", "sign up", "create user"},
            {"send email", "send verification", "verification email", "welcome email"},
        ),
        (
            {"test", "run tests", "unit test", "integration test"},
            {"deploy", "release", "ship", "publish"},
        ),
        (
            {"design", "plan", "architect", "define schema"},
            {"implement", "build", "code", "develop"},
        ),
    ]

    # Words/phrases universally relevant to any software task
    _GENERIC_DEV_WORDS = {
        "create",
        "setup",
        "configure",
        "deploy",
        "test",
        "build",
        "implement",
        "update",
        "migrate",
        "integrate",
        "monitor",
        "validate",
        "write",
        "design",
        "plan",
        "review",
        "define",
        "handle",
        "process",
        "refactor",
        "optimize",
        "document",
        "install",
        "check",
        "verify",
        "authenticate",
        "authorize",
        "debug",
        "fix",
        "add",
        "remove",
        "parse",
        "send",
        "receive",
        "fetch",
        "store",
        "query",
        "run",
        "execute",
        "package",
        "lint",
        "compile",
        "tests",
        "testing",
        "schema",
        "database",
        "endpoint",
        "endpoints",
        "form",
        "input",
        "validation",
        "security",
        "logging",
        "error",
        "documentation",
        "middleware",
        "server",
        "client",
        "model",
        "models",
        "controller",
        "service",
        "module",
        "config",
        "configuration",
        "pipeline",
        "workflow",
        "api",
        "data",
        "table",
        "tables",
    }

    # Stop words for missing requirement detection
    _REQ_STOP_WORDS = frozenset(
        {
            "build",
            "create",
            "implement",
            "setup",
            "develop",
            "design",
            "write",
            "deploy",
            "configure",
            "install",
            "update",
            "upgrade",
            "using",
            "existing",
            "based",
            "between",
            "across",
            "current",
            "system",
            "application",
            "feature",
            "service",
            "module",
            "component",
            "project",
            "platform",
            "solution",
            "architecture",
            "infrastructure",
            "should",
            "would",
            "could",
            "about",
            "their",
            "which",
            "where",
        }
    )

    _REQ_STEM_SUFFIXES = (
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ing",
        "ed",
        "er",
        "es",
        "ly",
        "al",
        "ous",
        "ive",
        "able",
        "ible",
    )

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect decomposition issues in a trace.

        Looks for planning/task spans and analyzes their decomposition quality.
        Also supports direct dict-based input via trace metadata.
        """
        # Check if trace carries golden-dataset-style data in metadata
        task_description = trace.metadata.custom.get("task_description", "")
        decomposition_text = trace.metadata.custom.get("decomposition", "")

        if task_description and decomposition_text:
            return self._detect_from_text(task_description, decomposition_text)

        # Otherwise extract from trace spans
        planning_spans = [
            s
            for s in trace.spans
            if s.kind == SpanKind.TASK or "plan" in s.name.lower() or "decompos" in s.name.lower()
        ]

        if not planning_spans:
            return DetectionResult.no_issue(self.name)

        all_issues: list[str] = []
        max_severity = 0
        evidence_data: dict[str, Any] = {}

        for span in planning_spans:
            task = ""
            output = ""
            if span.input_data:
                task = span.input_data.get("task", span.input_data.get("task_description", ""))
            if span.output_data:
                output = span.output_data.get("content", span.output_data.get("decomposition", ""))

            if not task and not output:
                continue

            inner_result = self._detect_from_text(task or "unknown task", output or "")
            if inner_result.detected:
                all_issues.append(inner_result.summary)
                max_severity = max(max_severity, inner_result.severity)
                evidence_data[span.span_id] = inner_result.metadata

        if not all_issues:
            return DetectionResult.no_issue(self.name)

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=max_severity,
            summary=all_issues[0],
            fix_type=FixType.SWITCH_STRATEGY,
            fix_instruction="Revise the task decomposition to address structural issues.",
        )
        for issue in all_issues:
            result.add_evidence(
                description=issue,
                span_ids=[s.span_id for s in planning_spans],
                data=evidence_data,
            )
        return result

    def _detect_from_text(
        self,
        task_description: str,
        decomposition: str,
    ) -> DetectionResult:
        """Core detection logic operating on text inputs.

        Faithfully ports the backend TaskDecompositionDetector.detect() method.
        """
        # Handle list-format decomposition
        if isinstance(decomposition, list):
            decomposition = "\n".join(str(item) for item in decomposition)

        # Recognize SWE-bench / patch-plan format
        decomp_lower = decomposition.lower()
        has_fix_ref = any(
            kw in decomp_lower for kw in ["fix:", "patch:", "apply patch", "apply fix"]
        )
        has_test_ref = any(
            kw in decomp_lower for kw in ["tests to pass", "test_", "tests/test_", "::test_"]
        )
        if has_fix_ref and has_test_ref:
            return DetectionResult.no_issue(self.name)

        # Check for direct implementation of simple tasks
        is_simple = self._is_simple_task(task_description)
        is_direct = self._is_direct_implementation(decomposition)

        if is_simple and is_direct:
            return DetectionResult.no_issue(self.name)

        subtasks = self._parse_subtasks(decomposition)

        # If no subtasks found, check task complexity
        if not subtasks:
            if self._is_complex_task(task_description):
                return DetectionResult.issue_found(
                    detector_name=self.name,
                    severity=50,
                    summary=f"Complex task '{task_description[:50]}...' lacks structured decomposition",
                    fix_type=FixType.SWITCH_STRATEGY,
                    fix_instruction="Provide a clear numbered or bulleted list of subtasks",
                )
            return DetectionResult.no_issue(self.name)

        # Simple task with direct approach and minimal decomposition
        if is_simple and is_direct and len(subtasks) <= 3:
            return DetectionResult.no_issue(self.name)

        issues: list[str] = []
        problematic: list[str] = []
        vague_count = 0
        complex_count = 0

        # Task-aware minimum subtasks
        min_required = self._get_min_subtasks_for_task(task_description)
        if len(subtasks) < min_required:
            issues.append("wrong_granularity")
            problematic.append("too_few_subtasks")
        elif len(subtasks) > self.max_subtasks:
            issues.append("wrong_granularity")
            problematic.append("too_many_subtasks")

        impossible = self._detect_impossible_subtasks(subtasks)
        if impossible:
            issues.append("impossible_subtask")
            problematic.extend(impossible)

        circular = self._detect_circular_dependencies(subtasks)
        if circular:
            issues.append("circular_dependency")
            for c1, c2 in circular:
                problematic.extend([c1, c2])

        duplicates = self._detect_duplicate_work(subtasks)
        if duplicates:
            issues.append("duplicate_work")
            for d1, d2 in duplicates:
                problematic.extend([d1, d2])

        if self.check_dependencies:
            missing_deps = self._detect_missing_dependencies(subtasks)
            if missing_deps:
                issues.append("missing_dependency")
                problematic.extend(missing_deps)

        vague = self._detect_vague_subtasks(subtasks)
        if vague:
            issues.append("vague_subtask")
            problematic.extend(vague)
            vague_count = len(vague)

        complex_subtasks = self._detect_overly_complex_subtasks(subtasks)
        if complex_subtasks:
            issues.append("overly_complex")
            problematic.extend(complex_subtasks)
            complex_count = len(complex_subtasks)

        irrelevant = self._detect_irrelevant_steps(subtasks, task_description)
        if irrelevant:
            issues.append("irrelevant_step")
            problematic.extend(irrelevant)

        wrong_order = self._detect_ordering_issues(subtasks)
        if wrong_order:
            issues.append("wrong_order")
            problematic.extend(wrong_order)

        missing_reqs = self._detect_missing_requirements(subtasks, task_description)
        if missing_reqs:
            issues.append("missing_requirement")
            problematic.extend(missing_reqs)

        if not issues:
            return DetectionResult.no_issue(self.name)

        # Determine severity
        if "circular_dependency" in issues or "impossible_subtask" in issues:
            severity = 80
        elif len(issues) >= 2:
            severity = 55
        else:
            severity = 30

        # Weighted confidence
        _issue_weights = {
            "circular_dependency": 0.50,
            "impossible_subtask": 0.40,
            "irrelevant_step": 0.35,
            "wrong_order": 0.30,
            "overly_complex": 0.25,
            "missing_dependency": 0.25,
            "duplicate_work": 0.25,
            "missing_requirement": 0.25,
            "wrong_granularity": 0.15,
            "missing_subtask": 0.15,
            "vague_subtask": 0.10,
        }
        weighted_sum = sum(_issue_weights.get(i, 0.15) for i in issues)
        confidence = min(0.95, 0.15 + weighted_sum)

        unique_problematic = list(set(problematic))[:5]
        summary = (
            f"Task decomposition has {len(issues)} issues: {', '.join(issues)}. "
            f"Affected: {', '.join(unique_problematic)}"
        )

        # Build fix instruction
        fixes = []
        if "circular_dependency" in issues:
            fixes.append("Break circular dependencies by reordering subtasks")
        if "impossible_subtask" in issues:
            fixes.append("Redefine impossible subtasks with achievable scope")
        if "duplicate_work" in issues:
            fixes.append("Merge duplicate subtasks")
        if "missing_dependency" in issues:
            fixes.append("Add missing dependencies between subtasks")
        if "vague_subtask" in issues:
            fixes.append("Replace vague steps with specific, actionable tasks")
        if "overly_complex" in issues:
            fixes.append("Break down overly complex steps into smaller subtasks")
        if "wrong_granularity" in issues and "too_few_subtasks" in problematic:
            fixes.append(f"Add more subtasks (minimum {min_required} recommended)")
        if "irrelevant_step" in issues:
            fixes.append("Remove steps unrelated to the task")
        if "wrong_order" in issues:
            fixes.append("Reorder steps so prerequisites come before dependent steps")
        if "missing_requirement" in issues:
            fixes.append("Add steps covering all stated task requirements")

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=summary,
            fix_type=FixType.SWITCH_STRATEGY,
            fix_instruction="; ".join(fixes) if fixes else "Revise the task decomposition",
        )
        result.confidence = confidence
        result.metadata = {
            "issues": issues,
            "subtask_count": len(subtasks),
            "problematic_subtasks": list(set(problematic)),
            "vague_count": vague_count,
            "complex_count": complex_count,
        }
        return result

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_subtasks(self, decomposition: str) -> list[Subtask]:
        """Parse structured subtasks from decomposition text."""
        subtasks: list[Subtask] = []

        patterns = [
            (r"\d+[.)]\s*", r"\d+[.)]"),
            (r"[-•*]\s+", r"[-•*]\s"),
            (r"(?:step|task|subtask)\s*\d*[:.]\s*", r"(?:step|task|subtask)\s*\d*[:.]"),
            (
                r"(?:phase|part|stage)\s*[\dA-Za-z]*[:.]\s*",
                r"(?:phase|part|stage)\s*[\dA-Za-z]*[:.]",
            ),
            (
                r"(?:first|second|third|then|next|finally|lastly)[,:]?\s+",
                r"(?:first|second|third|then|next|finally|lastly)[,:]?\s",
            ),
        ]

        items: list[str] = []
        for prefix_pat, lookahead_pat in patterns:
            nl_matches = re.findall(
                r"(?:^|\n)\s*" + prefix_pat + r"([^\n]+)",
                decomposition,
                re.IGNORECASE,
            )
            if len(nl_matches) >= 2:
                items = nl_matches
                break
            sl_matches = re.findall(
                prefix_pat + r"(.*?)(?=\s*" + lookahead_pat + r"|$)",
                decomposition,
                re.IGNORECASE,
            )
            sl_matches = [m.strip() for m in sl_matches if m.strip()]
            if len(sl_matches) >= 2:
                items = sl_matches
                break

        if not items:
            colon_pattern = r"(?:^|\n)\s*([A-Z][^:]+):\s*([^\n]+)"
            colon_matches = re.findall(colon_pattern, decomposition)
            if colon_matches:
                items = [f"{label}: {desc}" for label, desc in colon_matches]

        for i, item in enumerate(items):
            deps: list[str] = []
            dep_patterns = [
                r"(?:after|following|requires?|depends?\s+on)\s+(?:step|task)?\s*(\d+)",
                r"(?:once|when)\s+(?:step|task)?\s*(\d+)\s+(?:is\s+)?(?:complete|done)",
            ]
            for pattern in dep_patterns:
                dep_matches = re.findall(pattern, item.lower())
                deps.extend([f"task_{int(d) - 1}" for d in dep_matches if int(d) <= i])

            subtasks.append(
                Subtask(
                    id=f"task_{i}",
                    description=item.strip(),
                    dependencies=deps,
                )
            )

        return subtasks

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _detect_impossible_subtasks(self, subtasks: list[Subtask]) -> list[str]:
        """Detect subtasks with impossible/undefined requirements."""
        impossible_indicators = [
            "impossible",
            "cannot",
            "unable",
            "no way",
            "infeasible",
            "undefined",
            "unknown",
            "unclear",
            "ambiguous",
            "without access",
            "no information",
            "missing",
        ]
        problematic = []
        for subtask in subtasks:
            desc_lower = subtask.description.lower()
            for indicator in impossible_indicators:
                if indicator in desc_lower:
                    problematic.append(subtask.id)
                    break
        return problematic

    def _detect_circular_dependencies(self, subtasks: list[Subtask]) -> list[tuple[str, str]]:
        """Detect circular dependencies between subtasks."""
        circular: list[tuple[str, str]] = []
        dep_map = {st.id: set(st.dependencies) for st in subtasks}
        for task_id, deps in dep_map.items():
            for dep in deps:
                if dep in dep_map and task_id in dep_map[dep]:
                    if (dep, task_id) not in circular:
                        circular.append((task_id, dep))
        return circular

    def _detect_duplicate_work(self, subtasks: list[Subtask]) -> list[tuple[str, str]]:
        """Detect subtasks that duplicate work (>70% word overlap)."""
        duplicates: list[tuple[str, str]] = []
        for i, st1 in enumerate(subtasks):
            words1 = set(st1.description.lower().split())
            for j, st2 in enumerate(subtasks[i + 1 :], i + 1):
                words2 = set(st2.description.lower().split())
                if not words1 or not words2:
                    continue
                overlap = len(words1 & words2) / min(len(words1), len(words2))
                if overlap > 0.7:
                    duplicates.append((st1.id, st2.id))
        return duplicates

    def _detect_missing_dependencies(self, subtasks: list[Subtask]) -> list[str]:
        """Detect missing explicit dependencies between subtasks.

        Filters stopwords from noun extraction and respects sequential
        ordering (if producer step comes before consumer step, implicit dep is met).
        """
        missing: list[str] = []
        output_indicators = ["create", "generate", "produce", "build", "write"]
        input_indicators = ["use", "read", "process", "analyze", "with"]

        id_to_idx = {st.id: i for i, st in enumerate(subtasks)}
        outputs: dict[str, tuple[str, int]] = {}
        for st in subtasks:
            desc_lower = st.description.lower()
            for indicator in output_indicators:
                if indicator in desc_lower:
                    words = desc_lower.split()
                    idx = words.index(indicator) if indicator in words else -1
                    if idx >= 0 and idx + 1 < len(words):
                        noun = words[idx + 1]
                        if noun not in self._DEP_STOP_WORDS and len(noun) > 1:
                            outputs[noun] = (st.id, id_to_idx.get(st.id, -1))

        for st in subtasks:
            desc_lower = st.description.lower()
            consumer_idx = id_to_idx.get(st.id, -1)
            for indicator in input_indicators:
                if indicator in desc_lower:
                    words = desc_lower.split()
                    idx = words.index(indicator) if indicator in words else -1
                    if idx >= 0 and idx + 1 < len(words):
                        needed = words[idx + 1]
                        if needed in outputs:
                            producer_id, producer_idx = outputs[needed]
                            if producer_id == st.id:
                                continue
                            # Sequential ordering satisfies implicit dependency
                            if producer_idx < consumer_idx:
                                continue
                            if producer_id not in st.dependencies:
                                missing.append(st.id)
        return missing

    def _detect_vague_subtasks(self, subtasks: list[Subtask]) -> list[str]:
        """Detect subtasks that are too vague or non-actionable."""
        vague: list[str] = []
        for subtask in subtasks:
            desc_lower = subtask.description.lower()
            vague_count = sum(1 for ind in VAGUE_INDICATORS if ind in desc_lower)
            action_verbs = [
                "create",
                "build",
                "implement",
                "write",
                "configure",
                "set up",
                "install",
                "deploy",
                "test",
                "validate",
                "define",
                "design",
                "develop",
                "add",
                "remove",
                "update",
                "modify",
                "fix",
                "integrate",
                "connect",
                "display",
                "show",
                "render",
                "format",
                "parse",
                "fetch",
                "load",
                "save",
                "store",
                "delete",
                "call",
                "invoke",
                "execute",
                "run",
                "process",
                "handle",
                "index",
                "containerize",
                "evaluate",
                "filter",
                "send",
                "register",
                "check",
                "apply",
                "generate",
                "schedule",
                "compress",
                "upload",
                "scan",
                "trigger",
                "track",
                "calculate",
                "archive",
                "stream",
                "provide",
                "return",
                "verify",
                "migrate",
                "optimize",
                "monitor",
                "extract",
                "transform",
                "publish",
                "subscribe",
                "query",
                "export",
                "import",
            ]
            has_action = any(verb in desc_lower for verb in action_verbs)
            if vague_count >= 2 or (vague_count >= 1 and not has_action):
                vague.append(subtask.id)
            elif not has_action and len(subtask.description.split()) < 5:
                vague.append(subtask.id)
        return vague

    def _detect_overly_complex_subtasks(self, subtasks: list[Subtask]) -> list[str]:
        """Detect subtasks that are too large/broad for a single step."""
        complex_steps: list[str] = []
        for subtask in subtasks:
            desc_lower = subtask.description.lower()
            complexity_count = sum(1 for ind in COMPLEXITY_INDICATORS if ind in desc_lower)
            if complexity_count >= 2:
                complex_steps.append(subtask.id)
        return complex_steps

    def _is_complex_task(self, task_description: str) -> bool:
        """Determine if task is complex and requires thorough decomposition."""
        task_lower = task_description.lower()
        return any(ind in task_lower for ind in COMPLEX_TASK_INDICATORS)

    def _is_simple_task(self, task_description: str) -> bool:
        """Determine if task is simple and may not need decomposition."""
        task_lower = task_description.lower()
        has_simple = any(ind in task_lower for ind in SIMPLE_TASK_INDICATORS)
        has_complex = self._is_complex_task(task_description)
        return has_simple and not has_complex

    def _is_direct_implementation(self, output: str) -> bool:
        """Check if output indicates direct implementation (not decomposition)."""
        output_lower = output.lower()
        for pattern in DIRECT_IMPLEMENTATION_PATTERNS:
            if re.search(pattern, output_lower):
                return True
        return False

    def _get_min_subtasks_for_task(self, task_description: str) -> int:
        """Get minimum recommended subtasks based on task complexity."""
        if self._is_complex_task(task_description):
            return 4
        return self.min_subtasks

    @staticmethod
    def _has_word(keyword: str, text: str) -> bool:
        """Word-boundary aware keyword matching (prevents 'use' matching 'users')."""
        if " " in keyword:
            return keyword in text
        return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text))

    def _detect_irrelevant_steps(
        self,
        subtasks: list[Subtask],
        task_description: str,
    ) -> list[str]:
        """Detect steps whose content is unrelated to the task description.

        Only flags steps that belong to a clearly different domain.
        Requires >=2 irrelevant steps making up >=40% of all steps.
        """
        task_words = set(w for w in re.findall(r"[a-z]+", task_description.lower()) if len(w) > 3)
        if not task_words:
            return []

        task_lower = task_description.lower()
        task_bigrams: set[str] = set()
        twords = task_lower.split()
        for i in range(len(twords) - 1):
            task_bigrams.add(f"{twords[i]} {twords[i + 1]}")

        candidates: list[str] = []
        for subtask in subtasks:
            desc_lower = subtask.description.lower()
            step_words = set(w for w in re.findall(r"[a-z]+", desc_lower) if len(w) > 3)
            if not step_words:
                continue
            overlap = len(task_words & step_words)
            ratio = overlap / min(len(task_words), len(step_words))
            bigram_hit = any(bg in desc_lower for bg in task_bigrams)
            has_generic = bool(step_words & self._GENERIC_DEV_WORDS)
            if ratio < 0.15 and not bigram_hit and not has_generic:
                candidates.append(subtask.id)

        if len(candidates) >= 2 and len(candidates) / len(subtasks) >= 0.4:
            return candidates
        return []

    def _detect_ordering_issues(self, subtasks: list[Subtask]) -> list[str]:
        """Detect steps in wrong logical order based on common dependency pairs.

        Uses word-boundary matching and requires same-concern overlap.
        """
        if len(subtasks) < 2:
            return []

        violations: list[str] = []
        for dep_keywords, post_keywords in self._ORDERING_DEPS:
            prereq_idx = None
            for i, st in enumerate(subtasks):
                desc = st.description.lower()
                if any(self._has_word(kw, desc) for kw in dep_keywords):
                    prereq_idx = i
                    break

            dependent_idx = None
            for i, st in enumerate(subtasks):
                desc = st.description.lower()
                if any(self._has_word(kw, desc) for kw in post_keywords):
                    dependent_idx = i
                    break

            if prereq_idx is not None and dependent_idx is not None:
                if dependent_idx < prereq_idx:
                    violations.append(subtasks[dependent_idx].id)

        return list(set(violations))

    @staticmethod
    def _req_stem(word: str) -> str:
        """Minimal suffix strip for requirement keyword matching."""
        if word.endswith("ses") and len(word) >= 6:
            word = word[:-2]
        elif word.endswith("s") and not word.endswith("ss") and len(word) >= 5:
            word = word[:-1]
        for sfx in DecompositionDetector._REQ_STEM_SUFFIXES:
            if word.endswith(sfx) and len(word) - len(sfx) >= 3:
                return word[: -len(sfx)]
        return word

    def _detect_missing_requirements(
        self,
        subtasks: list[Subtask],
        task_description: str,
    ) -> list[str]:
        """Detect key task requirements that no step addresses."""
        task_lower = task_description.lower()
        requirement_chunks = re.split(r"\b(?:with|and|,)\b", task_lower)

        all_step_text = " ".join(st.description.lower() for st in subtasks)
        step_words = set(re.findall(r"[a-z]{3,}", all_step_text))
        step_stems = {self._req_stem(w) for w in step_words}

        missing: list[str] = []
        for chunk in requirement_chunks:
            chunk = chunk.strip()
            if len(chunk) < 5:
                continue
            chunk_words = set(
                w
                for w in re.findall(r"[a-z]+", chunk)
                if len(w) > 4 and w not in self._REQ_STOP_WORDS
            )
            if not chunk_words:
                continue
            found = 0
            for w in chunk_words:
                if w in all_step_text:
                    found += 1
                elif self._req_stem(w) in step_stems:
                    found += 1
            coverage = found / len(chunk_words)
            if coverage < 0.3 and len(chunk_words) >= 3:
                missing.append(chunk.strip()[:60])
        return missing
