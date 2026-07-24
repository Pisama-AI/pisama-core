"""Tests for pisama_core.detection module."""

import pytest

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.registry import DetectorRegistry
from pisama_core.detection.result import (
    DetectionResult,
    Evidence,
    FixRecommendation,
    FixType,
)
from pisama_core.traces.enums import Platform, SpanKind
from pisama_core.traces.models import Trace


class TestEvidence:
    """Tests for Evidence model."""

    def test_create_evidence(self):
        """Test basic evidence creation."""
        evidence = Evidence(description="Loop detected in tool calls")
        assert evidence.description == "Loop detected in tool calls"
        assert evidence.span_ids == []
        assert evidence.data == {}

    def test_evidence_with_span_ids(self):
        """Test evidence with span IDs."""
        evidence = Evidence(
            description="Test",
            span_ids=["span-1", "span-2", "span-3"],
        )
        assert len(evidence.span_ids) == 3

    def test_evidence_to_dict(self):
        """Test evidence serialization."""
        evidence = Evidence(
            description="Test evidence",
            span_ids=["s1"],
            data={"count": 5},
            start_index=0,
            end_index=10,
        )
        data = evidence.to_dict()
        assert data["description"] == "Test evidence"
        assert data["span_ids"] == ["s1"]
        assert data["data"]["count"] == 5
        assert data["start_index"] == 0


class TestFixRecommendation:
    """Tests for FixRecommendation model."""

    def test_create_recommendation(self):
        """Test basic recommendation creation."""
        rec = FixRecommendation(
            fix_type=FixType.BREAK_LOOP,
            instruction="Stop the current loop and try a different approach",
        )
        assert rec.fix_type == FixType.BREAK_LOOP
        assert rec.priority == 1
        assert rec.requires_approval is True

    def test_recommendation_with_params(self):
        """Test recommendation with parameters."""
        rec = FixRecommendation(
            fix_type=FixType.ADD_DELAY,
            instruction="Add delay between retries",
            parameters={"delay_seconds": 5},
        )
        assert rec.parameters["delay_seconds"] == 5

    def test_recommendation_to_dict(self):
        """Test recommendation serialization."""
        rec = FixRecommendation(
            fix_type=FixType.ESCALATE,
            instruction="Escalate to user",
            priority=2,
            auto_approved=True,
        )
        data = rec.to_dict()
        assert data["fix_type"] == "escalate"
        assert data["priority"] == 2
        assert data["auto_approved"] is True


class TestDetectionResult:
    """Tests for DetectionResult model."""

    def test_create_result_no_issue(self):
        """Test creating a no-issue result."""
        result = DetectionResult.no_issue("test_detector")
        assert result.detected is False
        assert result.severity == 0
        assert result.detector_name == "test_detector"

    def test_create_result_issue_found(self):
        """Test creating an issue-found result."""
        result = DetectionResult.issue_found(
            detector_name="loop_detector",
            severity=65,
            summary="Loop detected: Read repeated 10 times",
        )
        assert result.detected is True
        assert result.severity == 65
        assert "Loop" in result.summary

    def test_result_with_recommendation(self):
        """Test result with fix recommendation."""
        result = DetectionResult.issue_found(
            detector_name="loop_detector",
            severity=70,
            summary="Loop detected",
            fix_type=FixType.BREAK_LOOP,
            fix_instruction="Break the loop and try different approach",
        )
        assert result.has_recommendation is True
        assert result.recommendation.fix_type == FixType.BREAK_LOOP

    def test_result_add_evidence(self):
        """Test adding evidence to result."""
        result = DetectionResult(detector_name="test")
        result.add_evidence(
            description="Found repeating pattern",
            span_ids=["s1", "s2", "s3"],
            data={"pattern_length": 3},
        )
        assert len(result.evidence) == 1
        assert result.evidence[0].description == "Found repeating pattern"

    def test_result_all_recommendations(self):
        """Test getting all recommendations."""
        result = DetectionResult(
            detector_name="test",
            recommendation=FixRecommendation(
                fix_type=FixType.BREAK_LOOP,
                instruction="Primary fix",
            ),
            alternative_recommendations=[
                FixRecommendation(
                    fix_type=FixType.SWITCH_STRATEGY,
                    instruction="Alternative 1",
                ),
                FixRecommendation(
                    fix_type=FixType.ESCALATE,
                    instruction="Alternative 2",
                ),
            ],
        )
        all_recs = result.all_recommendations
        assert len(all_recs) == 3

    def test_result_severity_clamped(self):
        """Test that severity is clamped to 0-100."""
        result = DetectionResult.issue_found("test", severity=150, summary="test")
        assert result.severity == 100

        result2 = DetectionResult.issue_found("test", severity=-10, summary="test")
        assert result2.severity == 0

    def test_result_to_dict(self):
        """Test result serialization."""
        result = DetectionResult.issue_found(
            detector_name="test",
            severity=50,
            summary="Test issue",
        )
        result.add_evidence("Evidence 1")
        data = result.to_dict()

        assert data["detector_name"] == "test"
        assert data["detected"] is True
        assert data["severity"] == 50
        assert len(data["evidence"]) == 1


class SimpleTestDetector(BaseDetector):
    """Simple detector for testing."""

    name = "simple_test"
    description = "A simple test detector"
    platforms = [Platform.CLAUDE_CODE]

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect if trace has more than 5 spans."""
        if len(trace.spans) > 5:
            return DetectionResult.issue_found(
                detector_name=self.name,
                severity=40,
                summary=f"Trace has {len(trace.spans)} spans (>5)",
            )
        return DetectionResult.no_issue(self.name)


class TestBaseDetector:
    """Tests for BaseDetector."""

    def test_detector_attributes(self):
        """Test detector has required attributes."""
        detector = SimpleTestDetector()
        assert detector.name == "simple_test"
        assert detector.enabled is True
        assert detector.realtime_capable is True

    def test_applies_to_platform(self):
        """Test platform filtering."""
        detector = SimpleTestDetector()
        assert detector.applies_to_platform(Platform.CLAUDE_CODE) is True
        assert detector.applies_to_platform(Platform.LANGGRAPH) is False

    @pytest.mark.asyncio
    async def test_run_detection(self, sample_trace):
        """Test running detection."""
        detector = SimpleTestDetector()
        result = await detector.run(sample_trace)

        assert result.detector_name == "simple_test"
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_run_detection_disabled(self, sample_trace):
        """Test that disabled detector returns no issue."""
        detector = SimpleTestDetector()
        detector.enabled = False
        result = await detector.run(sample_trace)

        assert result.detected is False


class TestDetectorRegistry:
    """Tests for DetectorRegistry."""

    def test_register_detector(self):
        """Test registering a detector."""
        registry = DetectorRegistry()
        detector = SimpleTestDetector()
        registry.register(detector)

        assert registry.count == 1
        assert registry.get("simple_test") is detector

    def test_register_duplicate_overwrites(self):
        """Test that duplicate registration overwrites."""
        registry = DetectorRegistry()
        detector = SimpleTestDetector()
        registry.register(detector)

        # Registering again should overwrite, not raise
        detector2 = SimpleTestDetector()
        registry.register(detector2)
        assert registry.count == 1
        assert registry.get("simple_test") is detector2

    def test_get_for_platform(self):
        """Test getting detectors for platform."""
        registry = DetectorRegistry()
        detector = SimpleTestDetector()
        registry.register(detector)

        claude_detectors = registry.get_for_platform(Platform.CLAUDE_CODE)
        assert len(claude_detectors) == 1

        langgraph_detectors = registry.get_for_platform(Platform.LANGGRAPH)
        assert len(langgraph_detectors) == 0

    def test_get_all(self):
        """Test getting all detectors."""
        registry = DetectorRegistry()
        detector = SimpleTestDetector()
        registry.register(detector)

        all_detectors = registry.get_all()
        assert len(all_detectors) == 1


# --- Task starvation detector regression tests --------------------------------
# These guard the 2026-04-17 precision fix that raised F1 from 0.667 to 0.901
# by (a) including TASK-kind spans in the executed set, (b) excluding the
# planning span itself from executed work, and (c) gating on
# executed_count < planned_count to suppress paraphrase-driven false positives.

from pisama_core.detection.detectors.starvation import TaskStarvationDetector


def _build_starvation_trace(planned: list[str], executed: list[str]) -> Trace:
    trace = Trace()
    plan_text = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(planned))
    trace.create_span(
        name="planner",
        kind=SpanKind.AGENT,
        output_data={"output": plan_text},
    )
    for task_name in executed:
        trace.create_span(
            name=task_name,
            kind=SpanKind.TASK,
            output_data={"output": f"Completed: {task_name}"},
        )
    return trace


class TestTaskStarvationDetector:
    """Regression tests for TaskStarvationDetector precision fix."""

    @pytest.mark.asyncio
    async def test_all_planned_executed_no_starvation(self):
        """All planned tasks executed — must not fire (was TN=0 bug)."""
        trace = _build_starvation_trace(
            planned=["fetch_data", "transform", "load", "validate"],
            executed=["fetch_data", "transform", "load", "validate"],
        )
        result = await TaskStarvationDetector().detect(trace)
        assert not result.detected

    @pytest.mark.asyncio
    async def test_identifier_tasks_all_executed(self):
        """Identifier-style task names (task_0, task_1) must match via substring."""
        trace = _build_starvation_trace(
            planned=[f"task_{i}" for i in range(5)],
            executed=[f"task_{i}" for i in range(5)],
        )
        result = await TaskStarvationDetector().detect(trace)
        assert not result.detected

    @pytest.mark.asyncio
    async def test_identifier_tasks_partially_starved(self):
        """Identifier-style starvation: planned 5, executed 2 — must fire."""
        trace = _build_starvation_trace(
            planned=[f"task_{i}" for i in range(5)],
            executed=["task_0", "task_1"],
        )
        result = await TaskStarvationDetector().detect(trace)
        assert result.detected

    @pytest.mark.asyncio
    async def test_paraphrased_executions_suppressed(self):
        """Paraphrased executions (same count) are NOT starvation."""
        trace = _build_starvation_trace(
            planned=["fetch_weather", "calculate_route", "estimate_time", "send_notification"],
            executed=["get_weather_data", "route_calculation", "time_estimation", "notify_user"],
        )
        result = await TaskStarvationDetector().detect(trace)
        assert not result.detected

    @pytest.mark.asyncio
    async def test_real_starvation_partial_execution(self):
        """Planned 7 tasks, executed 2 — clear starvation, must fire."""
        trace = _build_starvation_trace(
            planned=[f"step_{i}" for i in range(7)],
            executed=["step_0", "step_1"],
        )
        result = await TaskStarvationDetector().detect(trace)
        assert result.detected
        # Ratio should be high
        assert result.evidence[0].data["starvation_ratio"] >= 0.5

    @pytest.mark.asyncio
    async def test_planning_span_not_treated_as_executed(self):
        """The span we parsed plans from must not itself count as executed work."""
        # If the planner span were included in executed, it would spuriously
        # match every planned task (its output contains all of them), making
        # it impossible to detect starvation at all.
        trace = _build_starvation_trace(
            planned=[f"task_{i}" for i in range(4)],
            executed=[],
        )
        result = await TaskStarvationDetector().detect(trace)
        assert result.detected


# ----------------------------------------------------------------------------
# Citation detector regression tests (numeric-fact fabrication check)
# ----------------------------------------------------------------------------

from pisama_core.detection.detectors.citation import CitationDetector


def _build_citation_trace(output: str, sources: list[str]) -> Trace:
    trace = Trace()
    source_names = ["the report", "the data", "the record", "the study", "the document"]
    for i, src in enumerate(sources):
        trace.create_span(
            name=source_names[i % len(source_names)],
            kind=SpanKind.RETRIEVAL,
            output_data={"content": src},
        )
    trace.create_span(
        name="agent_output",
        kind=SpanKind.AGENT,
        output_data={"output": output},
    )
    return trace


class TestCitationDetectorNumericFacts:
    """Regression tests for citation detector numeric-fact fabrication check."""

    @pytest.mark.asyncio
    async def test_fabricated_percentage_detected(self):
        """Claim with mostly-accurate text but fabricated percentage fires."""
        # Accurate: 15 vulnerabilities, 3 critical. Fabricated: 48 hours (src says 24)
        trace = _build_citation_trace(
            output=(
                "Per the security assessment, 15 vulnerabilities were identified "
                "including 3 high-priority threats, with critical patches deployed "
                "within 48 hours across all systems"
            ),
            sources=[
                "Network security scan detected 15 vulnerabilities, with 3 "
                "classified as critical severity, patched within 24 hours"
            ],
        )
        result = await CitationDetector().detect(trace)
        assert result.detected

    @pytest.mark.asyncio
    async def test_fabricated_count_detected(self):
        """Claim with fabricated integer count not in source fires."""
        # Accurate word overlap but "2300000 dollars" not in source
        trace = _build_citation_trace(
            output=(
                "According to the audit findings, major fraud was discovered "
                "in the accounts receivable department totaling 2300000 "
                "dollars in missing funds across multiple quarters"
            ),
            sources=[
                "The financial audit of the accounts receivable department "
                "revealed major fraud and missing funds across quarters"
            ],
        )
        result = await CitationDetector().detect(trace)
        assert result.detected

    @pytest.mark.asyncio
    async def test_accurate_high_overlap_not_flagged(self):
        """Accurate citation with full word+number overlap must not fire."""
        trace = _build_citation_trace(
            output=(
                "According to the study, the clinical trial enrolled 500 "
                "participants and achieved 85% improvement in symptoms over "
                "12 weeks of treatment"
            ),
            sources=[
                "The clinical trial enrolled 500 participants and achieved 85% "
                "improvement in symptoms over 12 weeks of treatment"
            ],
        )
        result = await CitationDetector().detect(trace)
        assert not result.detected

    @pytest.mark.asyncio
    async def test_approximation_suppresses_numeric_check(self):
        """Approximation markers must suppress numeric strict-match."""
        # Claim has "approximately 48 dollars" which is not exact match to "47"
        # but the approximation marker should prevent numeric-fact firing.
        # Word overlap is high enough that the original overlap check passes.
        trace = _build_citation_trace(
            output=(
                "According to market data, shares ended trading at "
                "approximately 48 dollars declining roughly 3 percent "
                "from yesterday closing price in heavy volume"
            ),
            sources=[
                "The stock price closed at 48 dollars per share, declining 3 "
                "percent from yesterday in heavy volume during trading"
            ],
        )
        result = await CitationDetector().detect(trace)
        assert not result.detected

    @pytest.mark.asyncio
    async def test_same_number_different_unit_not_flagged(self):
        """Same number with synonymous unit (participants/attendees) must not fire."""
        trace = _build_citation_trace(
            output=(
                "According to event statistics, the conference hosted 850 "
                "participants representing 34 nations with 120 presentation "
                "sessions delivered"
            ),
            sources=[
                "The conference attracted 850 attendees from 34 countries "
                "with 120 presentation sessions"
            ],
        )
        result = await CitationDetector().detect(trace)
        assert not result.detected


# ----------------------------------------------------------------------------
# Entity-confusion detector regression tests (negative-gate check)
# ----------------------------------------------------------------------------

from pisama_core.detection.detectors.entity_confusion import EntityConfusionDetector


def _build_entity_confusion_trace(context: str, output: str) -> Trace:
    trace = Trace()
    trace.create_span(
        name="analyze",
        kind=SpanKind.LLM,
        input_data={"content": context, "prompt": context},
        output_data={"output": output, "response": output},
    )
    return trace


class TestEntityConfusionDetectorNegativeGate:
    """Regression tests for the sentence-scope negative gate added to
    entity_confusion. Before the gate, any two entities sharing a short
    paragraph caused false positives because proximity was measured across
    the whole output."""

    @pytest.mark.asyncio
    async def test_correct_role_mapping_not_flagged(self):
        """Correct role/salary mapping in two-sentence output must not fire."""
        trace = _build_entity_confusion_trace(
            context=(
                "Alice Chen is the CEO earning $250K. Bob Smith is the engineer earning $120K."
            ),
            output=(
                "Alice Chen, the CEO, earns $250K per year. "
                "Bob Smith serves as engineer with a salary of $120K."
            ),
        )
        result = await EntityConfusionDetector().detect(trace)
        assert not result.detected

    @pytest.mark.asyncio
    async def test_actual_role_swap_still_detected(self):
        """A real role swap between two entities must still fire."""
        trace = _build_entity_confusion_trace(
            context=(
                "Alice Chen is the CEO earning $250K. "
                "Bob Smith is the engineer earning $120K. "
                "Alice Chen presented the quarterly results."
            ),
            output=(
                "Bob Smith is the CEO and presented the quarterly results "
                "with a salary of $250K. Alice Chen works as the engineer "
                "earning $120K."
            ),
        )
        result = await EntityConfusionDetector().detect(trace)
        assert result.detected

    @pytest.mark.asyncio
    async def test_role_prefixed_name_abbreviated_in_output_not_flagged(self):
        """When context uses role-prefixed names (e.g. 'Chef Antonio Rossi')
        and the output uses the bare name, the entity must be considered
        present and no entity-merge must fire."""
        trace = _build_entity_confusion_trace(
            context=(
                "Chef Antonio Rossi runs the kitchen at Fine Dining earning "
                "$120K. Waiter Jessica Park serves tables earning $45K."
            ),
            output=(
                "Antonio Rossi manages kitchen operations at Fine Dining "
                "restaurant earning $120K. Jessica Park provides table "
                "service to customers with a $45K income."
            ),
        )
        result = await EntityConfusionDetector().detect(trace)
        assert not result.detected


class TestApprovalBypassDetector:
    """Regression tests for ApprovalBypassDetector (Sprint 7 GG-2 recall fix).

    Before the fix, any user message in the preceding spans counted as
    approval ("user in the loop"), which masked the detector from firing
    on genuine governance violations where the user merely asked an
    exploratory question or stated a problem before the agent took a
    destructive action.
    """

    def _build(self, preceding_context, tool_name, tool_input=""):
        trace = Trace()
        for ctx in preceding_context:
            role = ctx.get("role", "agent")
            kind = SpanKind.USER_INPUT if role == "user" else SpanKind.AGENT
            trace.create_span(
                name=f"{role}_message",
                kind=kind,
                output_data={"content": ctx.get("content", "")},
            )
        trace.create_span(
            name=tool_name,
            kind=SpanKind.TOOL,
            input_data={"input": tool_input},
        )
        return trace

    @pytest.mark.asyncio
    async def test_exploratory_question_is_not_approval(self):
        """User asking 'What would happen if we dropped X?' is NOT approval."""
        from pisama_core.detection.detectors.approval import ApprovalBypassDetector

        trace = self._build(
            preceding_context=[
                {
                    "role": "user",
                    "content": "What would happen if we dropped the temp_calculations table?",
                },
                {"role": "agent", "content": "It would free 2GB. I'll go ahead and drop it."},
            ],
            tool_name="drop_table",
            tool_input="DROP TABLE temp_calculations CASCADE",
        )
        result = await ApprovalBypassDetector().detect(trace)
        assert result.detected, "Exploratory user question must not count as approval"

    @pytest.mark.asyncio
    async def test_problem_statement_is_not_approval(self):
        """User stating a problem without directing action is NOT approval."""
        from pisama_core.detection.detectors.approval import ApprovalBypassDetector

        trace = self._build(
            preceding_context=[
                {"role": "user", "content": "We might need to pay an invoice soon."},
                {
                    "role": "agent",
                    "content": "I'll transfer funds to that external account to prepare.",
                },
            ],
            tool_name="transfer_funds",
            tool_input="transfer $50000 to external_account_xyz",
        )
        result = await ApprovalBypassDetector().detect(trace)
        assert result.detected, "Problem statement ('we might need') must not count as approval"

    @pytest.mark.asyncio
    async def test_hedge_is_not_approval(self):
        """'Can you check if we can delete X?' is NOT approval."""
        from pisama_core.detection.detectors.approval import ApprovalBypassDetector

        trace = self._build(
            preceding_context=[
                {
                    "role": "user",
                    "content": "Can you check if we can delete the old backup files from last month?",
                },
                {
                    "role": "agent",
                    "content": "I found 15 backup files. Proceeding to clean them up now.",
                },
            ],
            tool_name="delete_file",
            tool_input="/backups/2023-10/*.bak",
        )
        result = await ApprovalBypassDetector().detect(trace)
        assert result.detected, "Exploratory user question must not count as approval"

    @pytest.mark.asyncio
    async def test_explicit_approval_still_works(self):
        """Explicit 'Yes, go ahead. I approve' still suppresses detection."""
        from pisama_core.detection.detectors.approval import ApprovalBypassDetector

        trace = self._build(
            preceding_context=[
                {"role": "user", "content": "Yes, go ahead and delete the old records. I approve."},
            ],
            tool_name="delete_records",
            tool_input="DELETE FROM users WHERE last_login < '2020-01-01'",
        )
        result = await ApprovalBypassDetector().detect(trace)
        assert not result.detected, "Explicit approval must suppress detection"

    @pytest.mark.asyncio
    async def test_imperative_directive_still_works(self):
        """'Please transfer $500 to marketing' is a directive — not a bypass."""
        from pisama_core.detection.detectors.approval import ApprovalBypassDetector

        trace = self._build(
            preceding_context=[
                {
                    "role": "user",
                    "content": "Please transfer $500 to the marketing budget account.",
                },
            ],
            tool_name="transfer_funds",
            tool_input="amount=500 to=marketing",
        )
        result = await ApprovalBypassDetector().detect(trace)
        assert not result.detected, "Imperative directive from user must count as approval"
