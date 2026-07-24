"""Completion detector for identifying premature or delayed task completion.

Detects F14: Completion Misjudgment (MAST Taxonomy):
- Agent claims task is complete when it's not
- Agent delivers partial results as final
- Agent ignores incomplete subtasks
- Agent misses success criteria
- False success claims despite errors
- Incomplete verification

Version History:
- v1.0: Initial pisama-core port from backend v2.0
"""

import re
from typing import Any, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Trace


class CompletionDetector(BaseDetector):
    """Detects completion misjudgment -- agent wrongly claims completion.

    Analyzes agent output against task requirements to identify
    premature or false completion claims.
    """

    name = "completion"
    description = "Detects premature or delayed task completion claims"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (10, 90)
    realtime_capable = False

    # Configuration
    strict_mode: bool = False
    completion_threshold: float = 0.9

    # Patterns indicating completion claims
    COMPLETION_CLAIM_PATTERNS = [
        r"\b(?:task|work|job)\s+(?:is\s+)?(?:complete|completed|done|finished)\b",
        r"\b(?:i have|i\'ve)\s+(?:completed|finished|done)\b",
        r"\b(?:successfully|successfully)\s+(?:completed|finished|done)\b",
        r"\b(?:all\s+)?(?:tasks?|steps?|items?)\s+(?:are\s+)?(?:complete|done)\b",
        r"\b(?:mission\s+accomplished|job\s+done)\b",
        r"\bhere(?:\'s| is)\s+the\s+(?:final|completed|finished)\b",
        r"\b(?:that\'s|this\s+is)\s+everything\b",
        r"\bnothing\s+(?:else|more)\s+(?:to\s+do|needed|required)\b",
        r"\b(?:migration|implementation|setup|integration|feature|system|module|service|component|api|app|application|build|deployment|test|testing|code|refactor|fix|update|upgrade)\s+(?:is\s+)?(?:complete|completed|done|ready|finished|implemented|working|set up|configured|live)\b",
        r"\b(?:done|complete|completed|ready|finished|implemented)\s*[!.]",
    ]

    # Patterns indicating incomplete work
    INCOMPLETE_PATTERNS = [
        (r"\bTODO\b", "todo_marker"),
        (r"\bFIXME\b", "fixme_marker"),
        (r"\bHACK\b", "hack_marker"),
        (r"\bXXX\b", "xxx_marker"),
        (r"\b(?:not\s+yet|still\s+need|remaining|pending)\b", "pending_marker"),
        (r"\b(?:placeholder|stub|dummy|mock)\b", "placeholder"),
        (r"\b(?:will\s+be|to\s+be)\s+(?:implemented|added|done)\b", "future_work"),
        (r"\.{3,}|etc\.", "ellipsis"),
        (r"\b(?:part\s+\d|step\s+\d)\s+(?:of|/)\s+\d+", "partial_progress"),
        (r"\b(?:partial|incomplete|unfinished)\b", "explicit_incomplete"),
    ]

    # Patterns indicating errors or failures
    ERROR_PATTERNS = [
        r"\b(?:error|exception|failure|failed)\b",
        r"\b(?:could not|couldn\'t|unable to|cannot)\b",
        r"\b(?:crash|crashed|crashing)\b",
        r"\b(?:bug|broken|breaking)\b",
    ]

    # Words in task that require 100% completion
    QUANTITATIVE_REQUIREMENTS = [
        "all",
        "every",
        "each",
        "complete",
        "full",
        "entire",
        "comprehensive",
        "thorough",
        "exhaustive",
        "total",
    ]

    # Patterns indicating partial/incomplete work (hedges)
    PARTIAL_COMPLETION_PATTERNS = [
        (r"\b(?:most|majority|mainly|primarily|largely)\b", "partial_scope"),
        (
            r"\b(?:core|main|primary|key|essential)\s+(?:functionality|features?|parts?)\b",
            "core_only",
        ),
        (r"\b(?:basic|minimal|initial|preliminary)\b", "minimal_scope"),
        (r"(?<!\.)\b\d{1,2}%", "percentage_incomplete"),
        (r"\b(?:some|several|few|certain)\s+(?:of|aspects?|parts?|areas?)\b", "partial_coverage"),
        (r"\b(?:focus(?:ed|ing)?|priorit(?:ized?|izing))\s+on\b", "selective_focus"),
        (r"\b(?:for now|at this point|currently|at the moment)\b", "temporal_limitation"),
        (r"\b(?:happy path|common case|typical scenario)\b", "limited_coverage"),
        (
            r"\b(?:lingering|might have missed|could still|probably missed)\b",
            "uncertainty_incomplete",
        ),
        (
            r"\b(?:couple of|few more|some more)\s+(?:edge cases?|cases?|tests?|items?)\b",
            "acknowledged_gaps",
        ),
    ]

    # Qualifier patterns suggesting uncertainty about completion
    QUALIFIER_PATTERNS = [
        (
            r"\b(?:appears?|seems?|looks?)\s+(?:to be\s+)?(?:complete|done|finished|solid|good)\b",
            "appearance_qualifier",
        ),
        (r"\bon the surface\b", "surface_qualifier"),
        (
            r"\b(?:should|might|could)\s+(?:be\s+)?(?:complete|working|functional)\b",
            "uncertainty_qualifier",
        ),
        (r"\b(?:believe|think|assume)\s+(?:it\'?s?|we\'?re?|this is)\b", "belief_qualifier"),
        (r"\b(?:as far as|to the best of)\b", "limited_knowledge"),
    ]

    # Implicit completion indicators (confident delivery without caveats)
    CONFIDENT_DELIVERY_PATTERNS = [
        r"\b(?:i\'ve|we\'ve|i have|we have)\s+(?:completed|finished|done|implemented)\b",
        r"\b(?:successfully|thoroughly|fully)\s+(?:completed|implemented|deployed)\b",
        r"\b(?:the|this)\s+(?:implementation|feature|system)\s+is\s+(?:ready|complete|done)\b",
        r"\bjust wrapped up\b",
        r"\bfully\s+covered\b",
        r"\b(?:all|every)\s+(?:main|major|key|critical)\s+(?:modules?|components?|features?|endpoints?)\b",
        r"\b(?:operational|ready)\s+(?:and\s+ready\s+)?for\s+(?:review|deployment|production)\b",
        r"\b(?:we\'ve|i\'ve)\s+(?:pushed|deployed|shipped|released)\b",
        r"\bin\s+place\b",
    ]

    # Progress language implying work is ongoing (NOT complete)
    PROGRESS_NOT_COMPLETE_PATTERNS = [
        r"\bgreat progress\b",
        r"\bgood progress\b",
        r"\bmaking progress\b",
        r"\bprogressing\s+(?:well|nicely)\b",
        r"\bcoming\s+along\b",
        r"\bcomprehensive\s+(?:overview|coverage)\b",
    ]

    # Planned/future work indicators
    PLANNED_WORK_PATTERNS = [
        (r"\b(?:coverage|tests?|testing)\s+planned\b", "planned_tests"),
        (
            r"\bwill\s+(?:be\s+)?(?:added|implemented|included|covered|optimized?|fixed|handled|addressed|resolved|completed|deployed|integrated|tested|updated|refactored)\b",
            "future_work",
        ),
        (
            r"\b(?:i'll|we'll|i\s+will|we\s+will)\s+(?:add|implement|include|cover|optimize|fix|handle|address|resolve|complete|deploy|integrate|test|update|refactor)\b",
            "future_work",
        ),
        (
            r"\b(?:next|later|future)\s+(?:phase|step|iteration|sprint|release|version)\b",
            "deferred_work",
        ),
        (r"\b(?:backlog|follow[- ]up|roadmap|post[- ]launch|post[- ]release)\b", "deferred_work"),
        (r"\b(?:to\s+be|tbd|coming\s+soon)\b", "pending_work"),
        (r"\b(?:placeholder|stub|mock)\s+(?:test|coverage|implementation)?\b", "stub_work"),
    ]

    # Honest partial completion phrases
    HONEST_PARTIAL_PATTERNS = [
        r"\bpartially done\b",
        r"\b\w+\s+complete,?\s+\w+\s+remaining\b",
        r"\bcompleted\s+\d+\s+of\s+\d+\b",
        r"\bstill working on\b",
        r"\bTODO:\b",
        r"\bin progress\b",
        r"\bnot yet (?:complete|finished|done)\b",
        r"\bwork in progress\b",
        r"\bremaining tasks?\b",
        r"\b\d+\s+(?:tasks?|items?)\s+left\b",
    ]

    # Scoped task patterns (intentionally limited scope)
    SCOPED_TASK_PATTERNS = [
        r"\b(?:prototype|mvp|poc|proof\s+of\s+concept)\b",
        r"\b(?:quick|initial|rough|first)\s+(?:draft|version|pass|attempt)\b",
        r"\b(?:minimal|basic)\s+(?:version|implementation)\b",
        r"\bv0(?:\.\d+)?\b",
    ]

    # JSON-specific patterns
    JSON_COMPLETION_PATTERNS = [
        r'"status"\s*:\s*"[^"]*(?:complete|done|finished)[^"]*"',
        r'"(?:is_complete|completed|finished|done)"\s*:\s*true',
        r'"state"\s*:\s*"[^"]*(?:complete|success)[^"]*"',
    ]

    JSON_INCOMPLETE_PATTERNS = [
        (
            r'"(?:documented|hasExamples?|completed|done|tested|covered|implemented)"\s*:\s*false',
            "json_false_flag",
        ),
        (r'"(?:missing|pending|todo|incomplete)"\s*:\s*\[', "json_missing_list"),
        (r'"(?:coverage|completion)"\s*:\s*"?\d{1,2}%"?', "json_partial_coverage"),
    ]

    NUMERIC_RATIO_PATTERNS = [
        (r"(?<!,)(\d+)\s*/\s*(?<!,)(\d+)", "explicit_ratio"),
        (r"(?<!,)(\d+)\s+(?:of|out of)\s+(?<!,)(\d+)", "explicit_count"),
        (
            r'"(?:documented|completed|done|tested)(?:Endpoints?|Items?|Tasks?)?"\s*:\s*(\d+).*?"(?:total)(?:Endpoints?|Items?|Tasks?)?"\s*:\s*(\d+)',
            "json_ratio",
        ),
    ]

    # Success criteria extraction patterns
    CRITERIA_PATTERNS = [
        r"(?:should|must|need to|required to)\s+(.+?)(?:\.|$)",
        r"(?:criteria|requirement|goal):\s*(.+?)(?:\.|$)",
        r"(?:success|completion)\s+(?:means?|requires?):\s*(.+?)(?:\.|$)",
        r"\d+\.\s+(.+?)(?:\n|$)",
    ]

    # Partial progress exemption
    _PARTIAL_PROGRESS_EXEMPTIONS = re.compile(
        r"(?:i have|i've)\s+(?:completed|finished|done)\s+\d+\s+(?:of|out of|/)\s+\d+",
        re.IGNORECASE,
    )

    # Log/test separator patterns
    _LOG_PATTERNS = [
        re.compile(r"^={5,}$"),
        re.compile(r"^-{5,}$"),
        re.compile(r"^\[?\d{4}-\d{2}-\d{2}[T ]"),
    ]

    # Stem suffixes for criteria matching
    _CRITERIA_STEM_SUFFIXES = (
        "ation",
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ance",
        "ence",
        "ing",
        "ed",
        "er",
        "es",
        "ly",
        "al",
        "ous",
        "ive",
        "ful",
        "ize",
        "ise",
        "able",
        "ible",
    )

    # Context phrases that neutralize error/failure matches
    _ERROR_NEUTRALIZERS = re.compile(
        r"\b(?:fixed|resolved|addressed|handled|corrected|recovered|cleared|eliminated)\b",
        re.IGNORECASE,
    )
    _ERROR_SUFFIX_NEUTRALIZERS = re.compile(
        r"\b(?:flagged|reported|logged|noted|documented|skipped|excluded)\b",
        re.IGNORECASE,
    )
    _ERROR_COMPOUND_SKIP = re.compile(
        r"\berror\s+(?:codes?|handling|messages?|types?|classes?|logging|recovery|pages?|boundaries?|rates?)\b"
        r"|\bfailure\s+(?:modes?|types?|handling|recovery|rates?|conditions?)\b"
        r"|\bsuccess\s*/\s*failure\b"
        r"|\bfail(?:ed)?\s+(?:gracefully|safely|over)\b",
        re.IGNORECASE,
    )

    # Subtask stem suffixes
    _STEM_SUFFIXES = (
        "tion",
        "ing",
        "ment",
        "ness",
        "ity",
        "ed",
        "er",
        "es",
        "ly",
        "al",
        "ous",
        "ive",
        "ful",
        "ize",
        "ise",
    )

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect completion misjudgment in a trace.

        Looks for task spans and analyzes whether completion was correctly assessed.
        Also supports direct golden-dataset-style input via trace metadata.
        """
        # Check for golden-dataset-style input
        task = trace.metadata.custom.get("task", "")
        agent_output = trace.metadata.custom.get("agent_output", "")

        if task and agent_output:
            subtasks = trace.metadata.custom.get("subtasks")
            success_criteria = trace.metadata.custom.get("success_criteria")
            return self._detect_from_text(task, agent_output, subtasks, success_criteria)

        # Extract from trace spans
        task_spans = [s for s in trace.spans if s.kind in (SpanKind.TASK, SpanKind.WORKFLOW)]
        if not task_spans:
            return DetectionResult.no_issue(self.name)

        all_issues: list[str] = []
        max_severity = 0

        for span in task_spans:
            task_desc = ""
            output = ""
            if span.input_data:
                task_desc = span.input_data.get("task", span.input_data.get("description", ""))
            if span.output_data:
                output = span.output_data.get("content", span.output_data.get("result", ""))

            if not task_desc and not output:
                continue

            inner_result = self._detect_from_text(task_desc or "unknown task", output or "")
            if inner_result.detected:
                all_issues.append(inner_result.summary)
                max_severity = max(max_severity, inner_result.severity)

        if not all_issues:
            return DetectionResult.no_issue(self.name)

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=max_severity,
            summary=all_issues[0],
            fix_type=FixType.SWITCH_STRATEGY,
            fix_instruction="Add comprehensive completion verification to agent workflow.",
        )
        for issue in all_issues:
            result.add_evidence(
                description=issue,
                span_ids=[s.span_id for s in task_spans],
            )
        return result

    def _detect_from_text(
        self,
        task: str,
        agent_output: str,
        subtasks: Optional[list[Any]] = None,
        success_criteria: Optional[list[str]] = None,
    ) -> DetectionResult:
        """Core detection logic operating on text inputs.

        Faithfully ports the backend CompletionMisjudgmentDetector.detect() method.
        """
        # Auto-convert string subtask names to dicts
        if subtasks:
            string_names = [s for s in subtasks if isinstance(s, str)]
            if string_names:
                dict_items = [s for s in subtasks if isinstance(s, dict)]
                inferred = self._infer_subtask_status(string_names, agent_output)
                subtasks = dict_items + inferred

        issues: list[dict[str, Any]] = []

        # Detect completion claims
        completion_claimed = self._detect_completion_claim(agent_output)
        confident_delivery = self._detect_confident_delivery(agent_output)
        if confident_delivery and not completion_claimed:
            completion_claimed = True

        json_completion = self._detect_json_completion_claim(agent_output)
        if json_completion and not completion_claimed:
            completion_claimed = True

        # Detect various signals
        incomplete_markers = self._detect_incomplete_markers(agent_output)
        json_incomplete = self._detect_json_incomplete(agent_output)
        numeric_ratio = self._detect_numeric_ratio(agent_output)
        errors = self._detect_errors(agent_output)

        has_quant_req = self._has_quantitative_requirement(task)
        if not has_quant_req and success_criteria:
            criteria_text = " ".join(success_criteria)
            has_quant_req = self._has_quantitative_requirement(criteria_text)

        partial_indicators = self._detect_partial_completion(agent_output)
        qualifiers = self._detect_qualifiers(agent_output)
        honest_partial = self._detect_honest_partial_reporting(agent_output)
        planned_work = self._detect_planned_work(agent_output)
        is_scoped_task = self._is_intentionally_scoped_task(task)
        has_progress_language = self._detect_progress_language(agent_output)

        # Calculate completion ratio
        completion_ratio = self._calculate_completion_ratio(agent_output, task)

        if has_quant_req and partial_indicators:
            penalty = min(0.3, len(partial_indicators) * 0.1)
            completion_ratio = max(0.0, completion_ratio - penalty)

        # Analyze subtasks
        subtasks_completed = 0
        subtasks_total = 0
        incomplete_subtasks: list[str] = []

        if subtasks:
            subtasks_completed, subtasks_total, incomplete_subtasks = self._analyze_subtasks(
                subtasks
            )
            if subtasks_total > 0:
                completion_ratio = min(completion_ratio, subtasks_completed / subtasks_total)

        # Check success criteria
        criteria = success_criteria or self._extract_success_criteria(task)
        criteria_met = 0
        criteria_total = 0
        unmet_criteria: list[str] = []

        if criteria:
            criteria_met, criteria_total, unmet_criteria = self._check_criteria_met(
                criteria, agent_output
            )
            if criteria_total > 0:
                criteria_ratio = criteria_met / criteria_total
                completion_ratio = min(completion_ratio, criteria_ratio)

        # --- Issue detection ---

        # Premature completion
        if completion_claimed and completion_ratio < self.completion_threshold:
            issues.append(
                {
                    "type": "premature_completion",
                    "severity": "severe",
                    "description": f"Agent claimed completion at {completion_ratio:.0%} actual progress",
                }
            )

        # Incomplete markers with completion claim
        if completion_claimed and incomplete_markers:
            for marker, marker_type, context in incomplete_markers[:3]:
                issues.append(
                    {
                        "type": "incomplete_verification",
                        "severity": "moderate",
                        "description": f"Incomplete marker found despite completion claim: {marker}",
                    }
                )

        # Errors with completion claim
        if completion_claimed and errors:
            issues.append(
                {
                    "type": "false_success_claim",
                    "severity": "severe",
                    "description": f"Errors detected despite completion claim ({len(errors)} error mentions)",
                }
            )

        # Incomplete subtasks with completion claim
        if completion_claimed and incomplete_subtasks:
            issues.append(
                {
                    "type": "ignored_subtasks",
                    "severity": "severe",
                    "description": f"{len(incomplete_subtasks)} subtasks incomplete despite completion claim",
                }
            )

        # Unmet criteria with completion claim
        if completion_claimed and unmet_criteria:
            issues.append(
                {
                    "type": "missed_criteria",
                    "severity": "moderate",
                    "description": f"{len(unmet_criteria)} success criteria not met",
                }
            )

        # Partial completion with quantitative requirement
        if has_quant_req and partial_indicators and completion_claimed:
            for indicator, indicator_type, context in partial_indicators[:2]:
                issues.append(
                    {
                        "type": "partial_delivery",
                        "severity": "moderate",
                        "description": f"Task requires 100% but output indicates partial: '{indicator}'",
                    }
                )

        # Uncertainty qualifiers with completion claim
        if qualifiers and completion_claimed:
            for qualifier, qualifier_type, context in qualifiers[:2]:
                issues.append(
                    {
                        "type": "incomplete_verification",
                        "severity": "minor",
                        "description": f"Uncertainty qualifier suggests incomplete: '{qualifier}'",
                    }
                )

        # Planned work with completion claim
        if planned_work and completion_claimed:
            for indicator, indicator_type, context in planned_work[:2]:
                issues.append(
                    {
                        "type": "premature_completion",
                        "severity": "moderate",
                        "description": f"Work marked as planned/future but completion claimed: '{indicator}'",
                    }
                )

        # Quantitative requirements without explicit claim
        if has_quant_req and not is_scoped_task and not completion_claimed:
            if partial_indicators:
                for indicator, indicator_type, context in partial_indicators[:2]:
                    issues.append(
                        {
                            "type": "partial_delivery",
                            "severity": "moderate",
                            "description": f"Task requires 100% completion but partial work indicated: '{indicator}'",
                        }
                    )
            if qualifiers:
                for qualifier, qualifier_type, context in qualifiers[:2]:
                    issues.append(
                        {
                            "type": "incomplete_verification",
                            "severity": "moderate",
                            "description": f"Task requires certainty but uncertainty expressed: '{qualifier}'",
                        }
                    )
            if has_progress_language:
                issues.append(
                    {
                        "type": "premature_completion",
                        "severity": "moderate",
                        "description": "Task requires 100% but uses progress language (implies ongoing work)",
                    }
                )

        # JSON incomplete with completion claim
        if completion_claimed and json_incomplete:
            for indicator, indicator_type, context in json_incomplete[:2]:
                issues.append(
                    {
                        "type": "partial_delivery",
                        "severity": "moderate",
                        "description": f"JSON output shows incomplete items: '{indicator}'",
                    }
                )

        # Numeric ratios showing incomplete
        if numeric_ratio and has_quant_req and not is_scoped_task:
            completed_count, total_count, ratio = numeric_ratio
            if ratio < 1.0:
                issues.append(
                    {
                        "type": "partial_delivery",
                        "severity": "moderate",
                        "description": f"Task requires 100% but output shows {completed_count}/{total_count} ({ratio * 100:.0f}%)",
                    }
                )

        # JSON incomplete without claim but with quantitative requirement
        if json_incomplete and has_quant_req and not is_scoped_task and not completion_claimed:
            for indicator, indicator_type, context in json_incomplete[:2]:
                issues.append(
                    {
                        "type": "partial_delivery",
                        "severity": "moderate",
                        "description": f"Task requires 100% but JSON shows incomplete: '{indicator}'",
                    }
                )

        # Deferred/incomplete work without explicit claim
        if not completion_claimed and not is_scoped_task:
            if planned_work and any(p[1] == "deferred_work" for p in planned_work):
                deferred = [p for p in planned_work if p[1] == "deferred_work"]
                for indicator, _, context in deferred[:2]:
                    issues.append(
                        {
                            "type": "partial_delivery",
                            "severity": "moderate",
                            "description": f"Work explicitly deferred: '{indicator}'",
                        }
                    )
            if incomplete_markers and (partial_indicators or planned_work):
                for marker, marker_type, context in incomplete_markers[:2]:
                    issues.append(
                        {
                            "type": "incomplete_verification",
                            "severity": "minor",
                            "description": f"Incomplete marker without completion claim: '{marker}'",
                        }
                    )

        # Truncated output
        if len(agent_output) > 500:
            tail = agent_output[-30:].rstrip()
            if tail.endswith("..."):
                issues.append(
                    {
                        "type": "partial_delivery",
                        "severity": "severe",
                        "description": "Output appears truncated (ends with ellipsis)",
                    }
                )

        # Structural incompleteness
        structural_issues = self._detect_structural_incompleteness(task, agent_output)
        issues.extend(structural_issues)

        # Enumerated coverage
        enum_issues = self._detect_enumerated_coverage(task, agent_output)
        issues.extend(enum_issues)

        # Structural subtask gap detection
        if subtasks and len(subtasks) > 1 and completion_claimed and not incomplete_subtasks:
            subtask_names = [
                str(s.get("name") or s.get("description") or "") if isinstance(s, dict) else str(s)
                for s in subtasks
            ]
            output_lower = agent_output.lower()
            mentioned = sum(
                1
                for st in subtask_names
                if any(word.lower() in output_lower for word in st.split()[:3] if len(word) > 2)
            )
            coverage = mentioned / len(subtask_names) if subtask_names else 1.0
            if coverage < 0.5:
                issues.append(
                    {
                        "type": "ignored_subtasks",
                        "severity": "moderate",
                        "description": f"Agent claims completion but only mentions {mentioned}/{len(subtask_names)} subtasks ({coverage:.0%} coverage)",
                    }
                )

        # Completion by absence
        if criteria_total > 0 and criteria_met < criteria_total * 0.5 and not is_scoped_task:
            issues.append(
                {
                    "type": "missed_criteria",
                    "severity": "moderate",
                    "description": f"Output addresses only {criteria_met}/{criteria_total} success criteria",
                }
            )

        # Reduce false positives for scoped tasks
        if is_scoped_task and issues:
            issues = [i for i in issues if i["severity"] in ("severe", "critical")]

        # Ensemble voting -- require 2+ distinct signal categories
        severity_map = {"critical": 4, "severe": 3, "moderate": 2, "minor": 1}
        max_issue_severity = (
            max((severity_map.get(i["severity"], 1) for i in issues), default=0) if issues else 0
        )

        if issues and max_issue_severity < 3:
            signal_categories: set[str] = set()
            if completion_claimed:
                signal_categories.add("completion_claim")
            if incomplete_markers:
                signal_categories.add("incomplete_markers")
            if partial_indicators:
                signal_categories.add("partial_indicators")
            if qualifiers:
                signal_categories.add("qualifiers")
            if numeric_ratio and numeric_ratio[2] < 1.0:
                signal_categories.add("numeric_ratio")
            if json_incomplete:
                signal_categories.add("json_incomplete")
            if planned_work:
                signal_categories.add("planned_work")
            if errors:
                signal_categories.add("errors")
            if incomplete_subtasks:
                signal_categories.add("incomplete_subtasks")
            if structural_issues:
                signal_categories.add("structural_incompleteness")
            if enum_issues:
                signal_categories.add("enumerated_coverage")

            has_quant_exemption = has_quant_req and (
                "partial_indicators" in signal_categories
                or "numeric_ratio" in signal_categories
                or "incomplete_subtasks" in signal_categories
                or "structural_incompleteness" in signal_categories
                or "enumerated_coverage" in signal_categories
            )
            if (
                "enumerated_coverage" in signal_categories
                and "completion_claim" in signal_categories
            ):
                has_quant_exemption = True
            if "completion_claim" in signal_categories and (
                "incomplete_markers" in signal_categories
                or "planned_work" in signal_categories
                or "errors" in signal_categories
                or "json_incomplete" in signal_categories
            ):
                has_quant_exemption = True
            if (
                "structural_incompleteness" in signal_categories
                or "enumerated_coverage" in signal_categories
            ):
                has_quant_exemption = True
            if planned_work and any(p[1] == "deferred_work" for p in planned_work):
                has_quant_exemption = True

            if len(signal_categories) < 2 and not has_quant_exemption:
                issues = []

        # Honest partial reporting suppression
        if honest_partial and not completion_claimed:
            issues = []

        if not issues:
            result = DetectionResult.no_issue(self.name)
            result.metadata = {
                "completion_claimed": completion_claimed,
                "actual_completion_ratio": completion_ratio,
                "subtasks_total": subtasks_total,
                "subtasks_completed": subtasks_completed,
            }
            return result

        # Calculate severity
        sev_to_num = {"critical": 90, "severe": 70, "moderate": 50, "minor": 25}
        max_sev = max(sev_to_num.get(i["severity"], 25) for i in issues)

        # Calculate confidence
        signal_count = sum(
            [
                bool(completion_claimed),
                bool(incomplete_markers),
                bool(errors),
                bool(incomplete_subtasks),
                bool(partial_indicators),
                bool(planned_work),
                bool(structural_issues),
                bool(enum_issues),
                bool(numeric_ratio and numeric_ratio[2] < 1.0),
                bool(json_incomplete),
            ]
        )
        if signal_count >= 4:
            base_confidence = 0.85
        elif signal_count >= 3:
            base_confidence = 0.75
        elif signal_count >= 2:
            base_confidence = 0.65
        else:
            base_confidence = 0.55
        if any(i["severity"] in ("severe", "critical") for i in issues):
            base_confidence = max(base_confidence, 0.75)
        confidence = min(0.95, base_confidence)

        issue_types = set(i["type"] for i in issues)
        summary = (
            f"Detected completion misjudgment: {', '.join(issue_types)}. "
            f"Actual completion: {completion_ratio:.0%}"
        )

        # Suggest fix
        if any(i["type"] == "ignored_subtasks" for i in issues):
            fix_instruction = (
                "Ensure agent tracks and completes all subtasks before claiming completion"
            )
        elif any(i["type"] == "missed_criteria" for i in issues):
            fix_instruction = "Implement success criteria verification before completion"
        elif any(i["type"] == "false_success_claim" for i in issues):
            fix_instruction = "Add error detection to prevent false success claims"
        else:
            fix_instruction = "Add comprehensive completion verification to agent workflow"

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=max_sev,
            summary=summary,
            fix_type=FixType.SWITCH_STRATEGY,
            fix_instruction=fix_instruction,
        )
        result.confidence = confidence
        result.metadata = {
            "completion_claimed": completion_claimed,
            "actual_completion_ratio": completion_ratio,
            "subtasks_total": subtasks_total,
            "subtasks_completed": subtasks_completed,
            "success_criteria_met": criteria_met,
            "success_criteria_total": criteria_total,
            "issue_types": list(issue_types),
        }
        for issue in issues:
            result.add_evidence(description=issue["description"])
        return result

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_completion_claim(self, text: str) -> bool:
        """Detect if agent claims task completion."""
        for pattern in self.COMPLETION_CLAIM_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Skip log/test output lines
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                if line_end == -1:
                    line_end = len(text)
                line = text[line_start:line_end].strip()
                if any(lp.search(line) for lp in self._LOG_PATTERNS):
                    continue
                # Exempt partial-progress phrases
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 30)
                window = text[start:end]
                if self._PARTIAL_PROGRESS_EXEMPTIONS.search(window):
                    continue
                return True
        return False

    def _detect_incomplete_markers(self, text: str) -> list[tuple[str, str, str]]:
        """Detect markers indicating incomplete work."""
        markers: list[tuple[str, str, str]] = []
        for pattern, marker_type in self.INCOMPLETE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if marker_type == "ellipsis" and "etc" in match.group().lower():
                    before = text[max(0, match.start() - 60) : match.start()]
                    if "(" in before and ")" not in before[before.rfind("(") :]:
                        continue
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                markers.append((match.group(), marker_type, context))
        return markers

    def _detect_errors(self, text: str) -> list[str]:
        """Detect error/failure mentions in output (context-aware)."""
        errors: list[str] = []
        for pattern in self.ERROR_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 40)
                context = text[start:end].strip()
                # Skip domain compound terms
                local_start = max(0, match.start() - 10)
                local_end = min(len(text), match.end() + 20)
                local_ctx = text[local_start:local_end]
                if self._ERROR_COMPOUND_SKIP.search(local_ctx):
                    continue
                # Skip if preceded by resolution language
                prefix_start = max(0, match.start() - 50)
                prefix = text[prefix_start : match.start()]
                if self._ERROR_NEUTRALIZERS.search(prefix):
                    continue
                # Skip if followed by exception-handling language
                suffix_end = min(len(text), match.end() + 60)
                suffix = text[match.end() : suffix_end]
                if self._ERROR_SUFFIX_NEUTRALIZERS.search(suffix):
                    continue
                errors.append(context)
        return errors

    def _has_quantitative_requirement(self, task: str) -> bool:
        """Check if task requires 100% completion."""
        task_lower = task.lower()
        return any(req in task_lower for req in self.QUANTITATIVE_REQUIREMENTS)

    def _detect_partial_completion(self, text: str) -> list[tuple[str, str, str]]:
        """Detect hedges indicating partial completion."""
        indicators: list[tuple[str, str, str]] = []
        for pattern, indicator_type in self.PARTIAL_COMPLETION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                indicators.append((match.group(), indicator_type, context))
        return indicators

    def _detect_qualifiers(self, text: str) -> list[tuple[str, str, str]]:
        """Detect uncertainty qualifiers."""
        qualifiers: list[tuple[str, str, str]] = []
        for pattern, qualifier_type in self.QUALIFIER_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                qualifiers.append((match.group(), qualifier_type, context))
        return qualifiers

    def _detect_confident_delivery(self, text: str) -> bool:
        """Detect implicit completion via confident delivery."""
        for pattern in self.CONFIDENT_DELIVERY_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                if line_end == -1:
                    line_end = len(text)
                line = text[line_start:line_end].strip()
                if any(lp.search(line) for lp in self._LOG_PATTERNS):
                    continue
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 30)
                window = text[start:end]
                if self._PARTIAL_PROGRESS_EXEMPTIONS.search(window):
                    continue
                return True
        return False

    def _detect_planned_work(self, text: str) -> list[tuple[str, str, str]]:
        """Detect indicators of planned/future work."""
        indicators: list[tuple[str, str, str]] = []
        for pattern, indicator_type in self.PLANNED_WORK_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                indicators.append((match.group(), indicator_type, context))
        return indicators

    def _is_intentionally_scoped_task(self, task: str) -> bool:
        """Check if task is intentionally scoped (MVP, prototype, etc.)."""
        task_lower = task.lower()
        for pattern in self.SCOPED_TASK_PATTERNS:
            if re.search(pattern, task_lower):
                return True
        return False

    def _detect_progress_language(self, text: str) -> bool:
        """Detect progress language implying work is ongoing."""
        for pattern in self.PROGRESS_NOT_COMPLETE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_honest_partial_reporting(self, text: str) -> bool:
        """Detect honest partial completion reporting."""
        for pattern in self.HONEST_PARTIAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_json_completion_claim(self, text: str) -> bool:
        """Detect completion claims in JSON/structured output."""
        for pattern in self.JSON_COMPLETION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_json_incomplete(self, text: str) -> list[tuple[str, str, str]]:
        """Detect incomplete indicators in JSON/structured output."""
        indicators: list[tuple[str, str, str]] = []
        for pattern, indicator_type in self.JSON_INCOMPLETE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                indicators.append((match.group(), indicator_type, context))
        return indicators

    def _detect_numeric_ratio(self, text: str) -> Optional[tuple[int, int, float]]:
        """Detect numeric ratios indicating partial completion (e.g., 8/10)."""
        for pattern, ratio_type in self.NUMERIC_RATIO_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    completed_count = int(match.group(1))
                    total_count = int(match.group(2))
                    if total_count > 0 and completed_count < total_count:
                        ratio = completed_count / total_count
                        return (completed_count, total_count, ratio)
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_success_criteria(self, task: str) -> list[str]:
        """Extract success criteria from task description."""
        criteria: list[str] = []
        for pattern in self.CRITERIA_PATTERNS:
            matches = re.findall(pattern, task, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if len(match.strip()) > 5:
                    criteria.append(match.strip())
        return criteria

    @staticmethod
    def _criteria_stem(word: str) -> str:
        """Minimal suffix strip for criteria keyword matching."""
        if word.endswith("s") and len(word) >= 4:
            word = word[:-1]
        for sfx in CompletionDetector._CRITERIA_STEM_SUFFIXES:
            if word.endswith(sfx) and len(word) - len(sfx) >= 3:
                return word[: -len(sfx)]
        return word

    def _check_criteria_met(
        self,
        criteria: list[str],
        output: str,
    ) -> tuple[int, int, list[str]]:
        """Check how many success criteria appear to be met."""
        met = 0
        unmet: list[str] = []

        output_lower = output.lower()
        output_words = set(re.findall(r"[a-z]{3,}", output_lower))
        output_stems = {self._criteria_stem(w) for w in output_words}

        for criterion in criteria:
            words = re.findall(r"\b\w{3,}\b", criterion.lower())
            key_words = [
                w
                for w in words
                if w
                not in {
                    "should",
                    "must",
                    "need",
                    "required",
                    "that",
                    "this",
                    "will",
                    "have",
                    "been",
                    "with",
                    "from",
                    "into",
                    "the",
                    "and",
                    "for",
                    "are",
                    "all",
                    "not",
                    "can",
                }
            ]
            if not key_words:
                continue
            matches = 0
            for w in key_words:
                if w in output_lower:
                    matches += 1
                elif self._criteria_stem(w) in output_stems:
                    matches += 1
            if matches >= len(key_words) * 0.5:
                met += 1
            else:
                unmet.append(criterion)

        return met, len(criteria), unmet

    def _analyze_subtasks(
        self,
        subtasks: list[dict[str, Any]],
    ) -> tuple[int, int, list[str]]:
        """Analyze subtask completion status."""
        total = len(subtasks)
        completed = 0
        incomplete: list[str] = []

        for subtask in subtasks:
            if not isinstance(subtask, dict):
                continue
            status = subtask.get("status", "").lower()
            st_name = str(subtask.get("name") or subtask.get("description") or "Unknown")
            if status in ("complete", "completed", "done", "success", "passed"):
                completed += 1
            else:
                incomplete.append(st_name)

        return completed, total, incomplete

    def _calculate_completion_ratio(self, output: str, task: str) -> float:
        """Estimate completion ratio based on output analysis."""
        incomplete_markers = len(self._detect_incomplete_markers(output))
        errors = len(self._detect_errors(output))
        if incomplete_markers + errors == 0:
            return 1.0
        penalty = min(0.5, incomplete_markers * 0.1 + errors * 0.15)
        return max(0.0, 1.0 - penalty)

    @staticmethod
    def _stem_word(word: str) -> str:
        """Minimal suffix strip for subtask inference matching."""
        if word.endswith("s") and len(word) >= 4:
            word = word[:-1]
        for sfx in CompletionDetector._STEM_SUFFIXES:
            if word.endswith(sfx) and len(word) - len(sfx) >= 3:
                return word[: -len(sfx)]
        return word

    @staticmethod
    def _infer_subtask_status(
        subtask_names: list[str],
        agent_output: str,
    ) -> list[dict[str, Any]]:
        """Convert string subtask names to dicts with inferred status."""
        output_lower = agent_output.lower()
        output_words = set(re.findall(r"[a-z]{3,}", output_lower))
        stem = CompletionDetector._stem_word
        output_stems = {stem(w) for w in output_words}

        result: list[dict[str, Any]] = []
        for name in subtask_names:
            parts = re.split(r"[_\s\-]+", name.lower())
            name_words = {w for w in parts if len(w) >= 3}
            overlap = sum(1 for w in name_words if w in output_lower or stem(w) in output_stems)
            ratio = overlap / max(len(name_words), 1)
            status = "completed" if ratio >= 0.5 else "pending"
            if status == "completed" and overlap > 0:
                for w in name_words:
                    pos = output_lower.find(w)
                    if pos >= 0:
                        ctx = output_lower[max(0, pos - 20) : pos]
                        if re.search(r"\bnot\s+(?:yet|currently)?\b", ctx):
                            status = "pending"
                            break
            result.append({"name": name, "status": status})
        return result

    def _detect_structural_incompleteness(self, task: str, output: str) -> list[dict[str, Any]]:
        """Detect structural incompleteness (fewer list items than requested, missing sections)."""
        issues: list[dict[str, Any]] = []

        # Check for requested list item count
        count_patterns = [
            r"(?:list|provide|give|name|identify|enumerate)\s+(\d+)",
            r"(\d+)\s+(?:items|examples|points|reasons|steps|features|recommendations)",
            r"top\s+(\d+)",
        ]

        requested_count = None
        for pattern in count_patterns:
            match = re.search(pattern, task.lower())
            if match:
                requested_count = int(match.group(1))
                break

        if requested_count and requested_count > 1:
            bullet_pattern = r"(?:^|\n)\s*(?:\d+[\.\/\)]\s|[-*\u2022]\s)"
            actual_items = len(re.findall(bullet_pattern, output))
            if actual_items > 0 and actual_items < requested_count:
                ratio = actual_items / requested_count
                issues.append(
                    {
                        "type": "partial_delivery",
                        "severity": "moderate" if ratio >= 0.5 else "severe",
                        "description": f"Task requests {requested_count} items but output contains only {actual_items}",
                    }
                )

        # Check for missing sections
        section_patterns = [
            r"(?:include|cover|write|add|provide|create)\s+(?:a\s+)?(?:sections?\s+(?:on|for|about)\s+)?(.+?)(?:\s+section[s]?)?(?:\.|$)",
            r"sections?\s*:\s*(.+?)(?:\.|$)",
        ]

        task_lower = task.lower()
        output_lower = output.lower()

        for pattern in section_patterns:
            match = re.search(pattern, task_lower)
            if match:
                section_text = match.group(1)
                parts = re.split(r",\s*|\s+and\s+", section_text)
                section_names = [
                    p.strip().strip("\"'")
                    for p in parts
                    if 2 <= len(p.strip()) <= 40 and " is " not in p and " are " not in p
                ]
                if len(section_names) >= 2:
                    missing = [s for s in section_names if s not in output_lower]
                    if missing and len(missing) < len(section_names):
                        issues.append(
                            {
                                "type": "partial_delivery",
                                "severity": "moderate",
                                "description": f"Task mentions sections but {len(missing)} missing: {', '.join(missing[:3])}",
                            }
                        )
                    break

        return issues

    def _detect_enumerated_coverage(self, task: str, output: str) -> list[dict[str, Any]]:
        """Detect when task enumerates specific items but output only covers a subset."""
        issues: list[dict[str, Any]] = []

        enum_patterns = [
            r"(?:with|including|for|supporting|using|like)\s+(.+?)(?:\.|$)",
            r"(?:implement|build|create|add|set up)\s+(.+?)(?:\.|$)",
        ]

        task_lower = task.lower()
        output_lower = output.lower()

        for pattern in enum_patterns:
            match = re.search(pattern, task_lower)
            if not match:
                continue
            fragment = match.group(1)
            parts = re.split(r",\s*|\s+and\s+", fragment)
            items = [
                p.strip()
                for p in parts
                if len(p.strip()) >= 2
                and p.strip()
                not in (
                    "the",
                    "a",
                    "an",
                    "all",
                    "each",
                    "every",
                    "other",
                )
            ]
            if len(items) >= 3:
                found = sum(1 for item in items if item in output_lower)
                if found < len(items):
                    missing = [i for i in items if i not in output_lower]
                    severity = "severe" if found == 0 else "moderate"
                    issues.append(
                        {
                            "type": "partial_delivery",
                            "severity": severity,
                            "description": (
                                f"Task enumerates {len(items)} items but output "
                                f"only covers {found}: missing {', '.join(missing[:3])}"
                            ),
                        }
                    )
                    break

        return issues
