"""Model selection detector for mismatched model tier vs task complexity."""

import re
from typing import Any

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Span, Trace


class ModelSelectionDetector(BaseDetector):
    """Detects mismatched model tier for task complexity.

    This detector identifies:
    - Expensive models used for simple tasks (wasted cost)
    - Cheap models used for complex tasks (quality risk)
    """

    name = "model_selection"
    description = "Detects mismatched model tier for task complexity"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (15, 50)
    realtime_capable = False

    # Model tier classifications
    EXPENSIVE_MODELS = re.compile(
        r"(?:opus|o1(?:-pro|-preview)?|o3(?:-mini)?|gpt-4(?!o)(?:-turbo)?(?:-\d+)?$)",
        re.IGNORECASE,
    )
    MID_MODELS = re.compile(
        r"(?:sonnet|gpt-4o(?!-mini)|gemini[- ](?:\d+\.?\d*[- ])?pro|"
        r"claude-3[.-]5?-sonnet|claude-sonnet)",
        re.IGNORECASE,
    )
    CHEAP_MODELS = re.compile(
        r"(?:haiku|gpt-4o-mini|gemini[- ](?:\d+\.?\d*[- ])?flash|"
        r"claude-3[.-]5?-haiku|claude-haiku|gpt-3\.5)",
        re.IGNORECASE,
    )

    # Complexity thresholds
    SIMPLE_TASK_MAX_INPUT_TOKENS = 200
    COMPLEX_TASK_MIN_INPUT_TOKENS = 5000
    COMPLEX_TASK_MIN_TOOLS = 5
    SIMPLE_TASK_MAX_TOOLS = 0

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect model tier vs task complexity mismatches."""
        llm_spans = trace.get_spans_by_kind(SpanKind.LLM)
        if not llm_spans:
            return DetectionResult.no_issue(self.name)

        # Assess overall task complexity
        complexity = self._assess_task_complexity(trace)

        issues: list[str] = []
        severity = 0
        evidence_spans: list[str] = []

        for span in llm_spans:
            model_name = self._extract_model_name(span)
            if not model_name:
                continue

            tier = self._classify_model_tier(model_name)
            if tier is None:
                continue

            # Also check per-span complexity (input tokens for this specific call)
            span_input_tokens = self._get_span_input_tokens(span)

            mismatch = self._check_mismatch(tier, model_name, complexity, span_input_tokens)
            if mismatch:
                issues.append(mismatch["description"])
                evidence_spans.append(span.span_id)
                severity += mismatch["severity_contribution"]

        if not issues:
            return DetectionResult.no_issue(self.name)

        severity = max(self.severity_range[0], min(self.severity_range[1], severity))

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=issues[0] if len(issues) == 1 else f"{len(issues)} model selection mismatches",
            fix_type=FixType.SWITCH_STRATEGY,
            fix_instruction=(
                "Model tier does not match task complexity. "
                "Use cheaper models for simple tasks (lookups, formatting) "
                "and reserve expensive models for complex reasoning, "
                "multi-step planning, and nuanced analysis."
            ),
        )

        for issue in issues:
            result.add_evidence(
                description=issue,
                span_ids=evidence_spans,
            )

        return result

    def _extract_model_name(self, span: Span) -> str | None:
        """Extract the model name from an LLM span's attributes."""
        # Check common attribute keys for model name
        for key in (
            "llm.model_name",
            "gen_ai.model",
            "model",
            "llm.model",
            "model_name",
            "gen_ai.request.model",
        ):
            val = span.attributes.get(key)
            if isinstance(val, str) and val:
                return val

        # Check input_data
        if span.input_data:
            for key in ("model", "model_name"):
                val = span.input_data.get(key)
                if isinstance(val, str) and val:
                    return val

        return None

    def _classify_model_tier(self, model_name: str) -> str | None:
        """Classify a model into expensive/mid/cheap tier."""
        if self.EXPENSIVE_MODELS.search(model_name):
            return "expensive"
        if self.MID_MODELS.search(model_name):
            return "mid"
        if self.CHEAP_MODELS.search(model_name):
            return "cheap"
        return None

    def _assess_task_complexity(self, trace: Trace) -> dict[str, Any]:
        """Assess the overall complexity of the task in this trace."""
        tool_spans = trace.get_spans_by_kind(SpanKind.TOOL)
        llm_spans = trace.get_spans_by_kind(SpanKind.LLM)
        user_input_spans = trace.get_spans_by_kind(SpanKind.USER_INPUT)

        # Estimate total input tokens across LLM spans
        total_input_tokens = 0
        for span in llm_spans:
            total_input_tokens += self._get_span_input_tokens(span)

        # Count conversation depth (user input turns)
        conversation_turns = len(user_input_spans)

        # Count unique tools used
        unique_tools = len({s.name for s in tool_spans})

        return {
            "tool_count": len(tool_spans),
            "unique_tools": unique_tools,
            "total_input_tokens": total_input_tokens,
            "conversation_turns": conversation_turns,
            "llm_call_count": len(llm_spans),
            "is_simple": (
                total_input_tokens <= self.SIMPLE_TASK_MAX_INPUT_TOKENS
                and len(tool_spans) <= self.SIMPLE_TASK_MAX_TOOLS
                and conversation_turns <= 1
            ),
            "is_complex": (
                total_input_tokens >= self.COMPLEX_TASK_MIN_INPUT_TOKENS
                or unique_tools >= self.COMPLEX_TASK_MIN_TOOLS
                or conversation_turns >= 3
            ),
        }

    def _get_span_input_tokens(self, span: Span) -> int:
        """Get input token count from a span."""
        # Check attributes for token counts
        for key in (
            "llm.input_tokens",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.prompt_tokens",
            "input_tokens",
            "prompt_tokens",
        ):
            val = span.attributes.get(key)
            if isinstance(val, (int, float)):
                return int(val)

        # Check input_data for token info
        if span.input_data:
            for key in ("input_tokens", "prompt_tokens", "tokens"):
                val = span.input_data.get(key)
                if isinstance(val, (int, float)):
                    return int(val)

            # Rough estimate from input text length
            for key in ("prompt", "messages", "input", "content"):
                val = span.input_data.get(key)
                if isinstance(val, str):
                    return len(val) // 4  # ~4 chars per token
                if isinstance(val, list):
                    total_chars = sum(len(str(item)) for item in val)
                    return total_chars // 4

        return 0

    def _check_mismatch(
        self,
        tier: str,
        model_name: str,
        complexity: dict[str, Any],
        span_input_tokens: int,
    ) -> dict[str, Any] | None:
        """Check for mismatch between model tier and task complexity."""
        # Expensive model on a simple task
        if tier == "expensive" and complexity["is_simple"]:
            return {
                "description": (
                    f"Expensive model '{model_name}' used for simple task "
                    f"({complexity['total_input_tokens']} input tokens, "
                    f"{complexity['tool_count']} tools, "
                    f"{complexity['conversation_turns']} turns)"
                ),
                "severity_contribution": 20,
            }

        # Expensive model for a single trivial LLM call in a larger trace
        if (
            tier == "expensive"
            and span_input_tokens > 0
            and span_input_tokens < self.SIMPLE_TASK_MAX_INPUT_TOKENS
            and complexity["llm_call_count"] > 1
        ):
            return {
                "description": (
                    f"Expensive model '{model_name}' used for small call "
                    f"({span_input_tokens} input tokens) that could use a cheaper model"
                ),
                "severity_contribution": 15,
            }

        # Cheap model on a complex task
        if tier == "cheap" and complexity["is_complex"]:
            return {
                "description": (
                    f"Cheap model '{model_name}' used for complex task "
                    f"({complexity['total_input_tokens']} input tokens, "
                    f"{complexity['unique_tools']} unique tools, "
                    f"{complexity['conversation_turns']} turns)"
                ),
                "severity_contribution": 25,
            }

        return None
