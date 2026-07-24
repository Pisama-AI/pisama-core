"""Context window overflow detection.

Detects when agent conversations approach or exceed their context window limit,
including:
- Current token usage relative to context window
- Per-component breakdown (system, messages, tools)
- Estimated turns until overflow
- Memory leak patterns (monotonic token growth)
- System prompt bloat warnings
- Remediation suggestions

Version History:
- v1.0: Port from backend ContextOverflowDetector with full logic
"""

from typing import Any, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixRecommendation, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Trace

# Default context windows for known models (tokens)
DEFAULT_CONTEXT_WINDOWS: dict[str, int] = {
    # Claude models
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3.5-haiku": 200_000,
    "claude-4-opus": 1_000_000,
    "claude-4-sonnet": 200_000,
    # GPT models (for completeness)
    "gpt-4": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4o": 128_000,
    "gpt-3.5-turbo": 16_385,
    # Gemini
    "gemini-pro": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
    # Default fallback
    "default": 128_000,
}


def _get_context_window(model: str) -> int:
    """Look up context window for a model, with fuzzy matching."""
    model_lower = model.lower()

    # Exact match
    if model_lower in DEFAULT_CONTEXT_WINDOWS:
        return DEFAULT_CONTEXT_WINDOWS[model_lower]

    # Fuzzy match: check if model name contains a known key
    for key, window in DEFAULT_CONTEXT_WINDOWS.items():
        if key in model_lower:
            return window

    return DEFAULT_CONTEXT_WINDOWS["default"]


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length.

    Uses the ~4 chars per token heuristic (accurate within ~10% for English).
    This avoids requiring tiktoken as a dependency.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


class OverflowDetector(BaseDetector):
    """Detects context window exhaustion in agent conversations.

    This detector identifies:
    - High context window usage (warning at 50%, critical at 85%)
    - Imminent overflow (>95% usage)
    - Estimated turns until overflow
    - Memory leak patterns (monotonic token growth)
    - System prompt bloat
    - Tool result bloat
    """

    name = "overflow"
    description = "Detects context window exhaustion and token budget issues"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (15, 95)
    realtime_capable = True

    # Thresholds
    warning_threshold: float = 0.50
    critical_threshold: float = 0.85
    overflow_threshold: float = 0.95

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect overflow risks in a trace.

        Extracts token usage from span attributes/metadata and compares
        against the model's context window.
        """
        # Determine model and context window
        model = self._find_model(trace)
        context_window = _get_context_window(model)

        # Compute token usage
        current_tokens = self._compute_total_tokens(trace)
        expected_output_tokens = self._find_expected_output_tokens(trace)
        effective_limit = context_window - expected_output_tokens

        if effective_limit <= 0:
            effective_limit = context_window

        usage_percent = current_tokens / effective_limit if effective_limit > 0 else 0.0
        remaining = max(0, effective_limit - current_tokens)

        # Determine severity
        warnings: list[str] = []
        if usage_percent >= self.overflow_threshold:
            severity_label = "overflow"
            warnings.append(f"Context window overflow imminent: {usage_percent:.1%} used")
        elif usage_percent >= self.critical_threshold:
            severity_label = "critical"
            warnings.append(f"Context window critically full: {usage_percent:.1%} used")
        elif usage_percent >= self.warning_threshold:
            severity_label = "warning"
            warnings.append(f"Context window filling up: {usage_percent:.1%} used")
        else:
            severity_label = "safe"

        # Token breakdown
        breakdown = self._compute_token_breakdown(trace)
        estimated_overflow_in: Optional[int] = None

        if breakdown["per_turn_avg"] > 0:
            estimated_overflow_in = int(remaining / breakdown["per_turn_avg"])
            if estimated_overflow_in < 5:
                warnings.append(f"Estimated overflow in {estimated_overflow_in} turns")
                if severity_label == "safe":
                    severity_label = "warning"

        if breakdown["system_tokens"] > context_window * 0.3:
            warnings.append(
                f"System prompt uses {breakdown['system_tokens'] / context_window:.1%} of context"
            )

        if breakdown["tool_tokens"] > context_window * 0.2:
            warnings.append(
                f"Tool results using significant context: {breakdown['tool_tokens']} tokens"
            )

        # Memory leak detection
        token_history = self._extract_token_history(trace)
        leak_info = self._detect_memory_leak(token_history, context_window)
        if leak_info:
            warnings.append(
                f"Token growth leak detected: {leak_info['avg_growth_rate']:.1%} avg growth, "
                f"overflow in ~{leak_info['projected_overflow_turns']} turns"
            )
            if severity_label == "safe":
                severity_label = "warning"

        if severity_label == "safe":
            return DetectionResult.no_issue(self.name)

        severity_score = self._severity_label_to_score(severity_label)

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity_score,
            summary=warnings[0] if warnings else f"Context window at {usage_percent:.1%}",
            fix_type=FixType.RESET_CONTEXT,
            fix_instruction=self._build_fix_instruction(severity_label, breakdown, context_window),
        )

        # Add evidence
        result.add_evidence(
            description=f"Token usage: {current_tokens}/{effective_limit} ({usage_percent:.1%})",
            data={
                "current_tokens": current_tokens,
                "context_window": context_window,
                "effective_limit": effective_limit,
                "usage_percent": round(usage_percent, 4),
                "remaining_tokens": remaining,
                "estimated_overflow_in": estimated_overflow_in,
                "model": model,
                "breakdown": breakdown,
                "warnings": warnings,
            },
        )

        if leak_info:
            result.add_evidence(
                description="Memory leak pattern detected",
                data=leak_info,
            )

        # Calibrate confidence
        confidence, _ = self._calibrate_confidence(
            usage_percent=usage_percent,
            severity_label=severity_label,
            warning_count=len(warnings),
            estimated_overflow_in=estimated_overflow_in,
        )
        result.confidence = confidence

        # Add remediation suggestions as alternative recommendations
        suggestions = self._suggest_remediation(
            severity_label,
            breakdown,
            context_window,
            usage_percent,
        )
        for suggestion in suggestions[:3]:
            result.alternative_recommendations.append(
                FixRecommendation(
                    fix_type=FixType.RESET_CONTEXT,
                    instruction=suggestion,
                    priority=2,
                )
            )

        return result

    # ------------------------------------------------------------------
    # Trace extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_model(trace: Trace) -> str:
        """Find the model name from trace metadata or LLM spans."""
        # Check metadata
        model = trace.metadata.custom.get("model") or trace.metadata.tags.get("model")
        if model:
            return model

        # Check LLM spans
        for span in trace.spans:
            if span.kind == SpanKind.LLM:
                model = (
                    span.attributes.get("gen_ai.request.model")
                    or span.attributes.get("model")
                    or (span.input_data or {}).get("model")
                )
                if model:
                    return model

        return "default"

    @staticmethod
    def _find_expected_output_tokens(trace: Trace) -> int:
        """Find expected output token reservation from trace."""
        for span in trace.spans:
            if span.kind == SpanKind.LLM:
                max_tokens = (
                    span.attributes.get("gen_ai.request.max_tokens")
                    or span.attributes.get("max_tokens")
                    or (span.input_data or {}).get("max_tokens")
                )
                if isinstance(max_tokens, int):
                    return max_tokens
        return 4096

    def _compute_total_tokens(self, trace: Trace) -> int:
        """Compute total token usage from trace spans.

        Prefers explicit token counts from span attributes, falls back to
        estimating from text content.
        """
        total = 0

        for span in trace.spans:
            # Check for explicit token counts
            token_count = (
                span.attributes.get("gen_ai.usage.total_tokens")
                or span.attributes.get("token_count")
                or span.attributes.get("tokens")
            )
            if isinstance(token_count, int):
                total += token_count
                continue

            # Check input/output token counts
            input_tokens = span.attributes.get("gen_ai.usage.input_tokens", 0)
            output_tokens = span.attributes.get("gen_ai.usage.output_tokens", 0)
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                if input_tokens > 0 or output_tokens > 0:
                    total += input_tokens + output_tokens
                    continue

            # Estimate from content
            if span.input_data:
                messages = span.input_data.get("messages", [])
                for msg in messages:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            total += _estimate_tokens(content)
                        elif isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict):
                                    total += _estimate_tokens(str(c.get("text", "")))

            if span.output_data:
                content = (
                    span.output_data.get("content")
                    or span.output_data.get("text")
                    or span.output_data.get("output")
                )
                if isinstance(content, str):
                    total += _estimate_tokens(content)

        return total

    def _compute_token_breakdown(self, trace: Trace) -> dict[str, Any]:
        """Compute token breakdown by category."""
        system_tokens = 0
        message_tokens = 0
        tool_tokens = 0
        turn_count = 0

        for span in trace.spans:
            if span.kind == SpanKind.LLM and span.input_data:
                messages = span.input_data.get("messages", [])
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    tokens = _estimate_tokens(content) if isinstance(content, str) else 0
                    tokens += 4  # Per-message overhead

                    if role == "system":
                        system_tokens += tokens
                    elif role == "tool" or msg.get("tool_calls"):
                        tool_tokens += tokens
                        for tc in msg.get("tool_calls", []):
                            if isinstance(tc, dict):
                                tool_tokens += _estimate_tokens(
                                    str(tc.get("function", {}).get("arguments", ""))
                                )
                    else:
                        message_tokens += tokens

                    if role in ("user", "assistant"):
                        turn_count += 1

            elif span.kind == SpanKind.TOOL:
                if span.output_data:
                    content = span.output_data.get("output") or span.output_data.get("result") or ""
                    if isinstance(content, str):
                        tool_tokens += _estimate_tokens(content)

        total = system_tokens + message_tokens + tool_tokens
        per_turn_avg = message_tokens / max(1, turn_count)

        return {
            "system_tokens": system_tokens,
            "message_tokens": message_tokens,
            "tool_tokens": tool_tokens,
            "total_tokens": total,
            "per_turn_avg": per_turn_avg,
        }

    @staticmethod
    def _extract_token_history(trace: Trace) -> list[int]:
        """Extract token usage history from LLM spans for leak detection."""
        history: list[int] = []
        for span in sorted(trace.spans, key=lambda s: s.start_time):
            if span.kind == SpanKind.LLM:
                tokens = span.attributes.get("gen_ai.usage.total_tokens") or span.attributes.get(
                    "gen_ai.usage.input_tokens"
                )
                if isinstance(tokens, int) and tokens > 0:
                    history.append(tokens)
        return history

    # ------------------------------------------------------------------
    # Memory leak detection (ported from backend)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_memory_leak(
        token_history: list[int],
        context_window: int,
    ) -> Optional[dict[str, Any]]:
        """Detect memory leak pattern from token history.

        Returns info dict if leak detected, None otherwise.
        """
        if len(token_history) < 5:
            return None

        growth_rates: list[float] = []
        for i in range(1, len(token_history)):
            if token_history[i - 1] > 0:
                rate = (token_history[i] - token_history[i - 1]) / token_history[i - 1]
                growth_rates.append(rate)

        if not growth_rates:
            return None

        avg_growth = sum(growth_rates) / len(growth_rates)
        expected_shrink = any(r < -0.1 for r in growth_rates)

        if avg_growth > 0.05 and not expected_shrink:
            current = token_history[-1]
            turns_to_overflow = 0
            projected = float(current)

            while projected < context_window and turns_to_overflow < 100:
                projected = projected * (1 + avg_growth)
                turns_to_overflow += 1

            return {
                "leak_detected": True,
                "avg_growth_rate": avg_growth,
                "current_tokens": current,
                "projected_overflow_turns": turns_to_overflow,
                "recommendation": "Implement conversation summarization or context pruning",
            }

        return None

    # ------------------------------------------------------------------
    # Remediation suggestions (ported from backend)
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest_remediation(
        severity_label: str,
        breakdown: dict[str, Any],
        context_window: int,
        usage_percent: float,
    ) -> list[str]:
        """Generate remediation suggestions based on overflow analysis."""
        suggestions: list[str] = []

        if severity_label == "overflow":
            suggestions.append("IMMEDIATE: Truncate or summarize conversation history")
            suggestions.append("Remove older messages, keeping only recent context")

        if severity_label in ("critical", "overflow"):
            suggestions.append("Summarize tool results instead of including full output")
            suggestions.append("Consider using a model with larger context window")

        if breakdown.get("system_tokens", 0) > context_window * 0.2:
            suggestions.append(
                "Reduce system prompt size - consider dynamic loading of instructions"
            )

        if breakdown.get("tool_tokens", 0) > context_window * 0.15:
            suggestions.append("Compress tool results - return summaries instead of full data")

        if usage_percent > 0.5:
            suggestions.append("Implement sliding window for message history")
            suggestions.append("Add periodic conversation summarization")

        return suggestions

    @staticmethod
    def _build_fix_instruction(
        severity_label: str,
        breakdown: dict[str, Any],
        context_window: int,
    ) -> str:
        """Build the primary fix instruction."""
        if severity_label == "overflow":
            return (
                "Immediately truncate or summarize the conversation history to free context space."
            )
        elif severity_label == "critical":
            return "Summarize older messages and compress tool results to reduce context usage."
        else:
            return "Monitor context usage. Consider summarizing conversation history soon."

    # ------------------------------------------------------------------
    # Confidence calibration (ported from backend)
    # ------------------------------------------------------------------

    @staticmethod
    def _calibrate_confidence(
        usage_percent: float,
        severity_label: str,
        warning_count: int,
        estimated_overflow_in: Optional[int],
    ) -> tuple[float, dict[str, Any]]:
        """Calibrate confidence based on usage metrics and severity."""
        severity_weight = {
            "safe": 0.0,
            "warning": 0.6,
            "critical": 0.8,
            "overflow": 0.95,
        }.get(severity_label, 0.5)

        usage_factor = min(1.0, usage_percent)
        warning_factor = min(0.15, warning_count * 0.05)

        overflow_urgency = 0.0
        if estimated_overflow_in is not None:
            if estimated_overflow_in <= 2:
                overflow_urgency = 0.2
            elif estimated_overflow_in <= 5:
                overflow_urgency = 0.1
            elif estimated_overflow_in <= 10:
                overflow_urgency = 0.05

        base_confidence = (
            severity_weight * 0.40 + usage_factor * 0.30 + warning_factor + overflow_urgency
        )

        calibrated = min(0.99, base_confidence)

        calibration_info = {
            "usage_percent": round(usage_percent, 4),
            "severity_weight": severity_weight,
            "warning_count": warning_count,
            "warning_factor": round(warning_factor, 4),
            "overflow_urgency": overflow_urgency,
            "estimated_overflow_in": estimated_overflow_in,
        }

        return round(calibrated, 4), calibration_info

    @staticmethod
    def _severity_label_to_score(label: str) -> int:
        """Map severity label to numeric 0-100 score."""
        return {
            "safe": 0,
            "warning": 35,
            "critical": 70,
            "overflow": 90,
        }.get(label, 40)
