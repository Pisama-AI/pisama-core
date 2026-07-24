"""Reasoning consistency detector for identifying contradictory and abandoned reasoning.

Detects:
- Contradictory conclusions from multiple reasoning paths (same question, different answers)
- Reasoning abandonment (chain-of-thought started but jumps to conclusion)
- Circular reasoning (conclusion restates premise without new information)

Version History:
- v1.0: Initial implementation
"""

import re
from typing import Any

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Span, Trace

# Patterns that signal a conclusion section
CONCLUSION_PATTERNS = [
    r"\b(?:therefore|thus|hence|consequently|so)\b",
    r"\bconclusion\s*[:.]",
    r"\banswer\s*[:.]",
    r"\bresult\s*[:.]",
    r"\bin\s+(?:summary|conclusion)\b",
    r"\bfinal\s+answer\b",
    r"\bthe\s+answer\s+is\b",
]

# Patterns that signal chain-of-thought reasoning
COT_MARKERS = [
    r"\blet\s+me\s+think\b",
    r"\bstep\s+by\s+step\b",
    r"\bfirst,?\s+(?:let\'s|we|i)\b",
    r"\blet\'s\s+(?:break|analyze|consider|reason|think)\b",
    r"\bthinking\s+through\b",
    r"\bto\s+solve\s+this\b",
    r"\bmy\s+reasoning\b",
    r"\bworking\s+through\b",
    r"\blet\s+me\s+(?:work|figure|reason)\b",
]

# Boolean-like conclusion values
_TRUE_WORDS = {"yes", "true", "correct", "affirmative", "confirmed", "positive", "valid"}
_FALSE_WORDS = {"no", "false", "incorrect", "negative", "invalid", "denied", "wrong"}


class ReasoningConsistencyDetector(BaseDetector):
    """Detects contradictory conclusions, abandoned reasoning, and circular logic.

    Analyzes LLM spans to find:
    - Multiple reasoning paths that reach different final answers
    - Chain-of-thought that starts but skips to a conclusion
    - Conclusions that merely restate the premise
    """

    name = "reasoning_consistency"
    description = "Detects contradictory conclusions and abandoned reasoning"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (25, 70)
    realtime_capable = False

    # Configuration
    input_similarity_threshold = 0.5  # Minimum keyword overlap to consider same question
    min_reasoning_spans = 2  # Minimum LLM spans to check for contradictions
    abandonment_gap_ratio = 0.7  # Fraction of output that is "gap" between CoT start and conclusion

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect reasoning consistency issues in a trace."""
        llm_spans = trace.get_spans_by_kind(SpanKind.LLM)

        if len(llm_spans) < self.min_reasoning_spans:
            return DetectionResult.no_issue(self.name)

        sorted_spans = sorted(llm_spans, key=lambda s: s.start_time)

        issues: list[str] = []
        severity = 0
        evidence_data: dict[str, Any] = {}

        # Check for contradictory conclusions from similar inputs
        contradiction = self._check_contradictions(sorted_spans)
        if contradiction:
            severity += contradiction["severity"]
            issues.append(contradiction["message"])
            evidence_data["contradiction"] = contradiction

        # Check for reasoning abandonment
        abandonment = self._check_abandonment(sorted_spans)
        if abandonment:
            severity += abandonment["severity"]
            issues.append(abandonment["message"])
            evidence_data["abandonment"] = abandonment

        # Check for circular reasoning
        circular = self._check_circular_reasoning(sorted_spans)
        if circular:
            severity += circular["severity"]
            issues.append(circular["message"])
            evidence_data["circular"] = circular

        if not issues:
            return DetectionResult.no_issue(self.name)

        severity = min(self.severity_range[1], max(self.severity_range[0], severity))

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=issues[0],
            fix_type=FixType.SWITCH_STRATEGY,
            fix_instruction=(
                "Review reasoning paths for consistency. "
                "Ensure each chain-of-thought is completed before drawing conclusions."
            ),
        )

        for issue in issues:
            result.add_evidence(
                description=issue,
                span_ids=[s.span_id for s in sorted_spans[:10]],
                data=evidence_data,
            )

        return result

    def _check_contradictions(self, spans: list[Span]) -> dict[str, Any] | None:
        """Check for contradictory conclusions from similar inputs."""
        # Group spans by similar input
        groups = self._group_by_similar_input(spans)

        for input_key, group_spans in groups.items():
            if len(group_spans) < 2:
                continue

            conclusions: list[tuple[Span, str]] = []
            for span in group_spans:
                conclusion = self._extract_conclusion(span)
                if conclusion:
                    conclusions.append((span, conclusion))

            if len(conclusions) < 2:
                continue

            # Compare conclusions pairwise
            for i in range(len(conclusions)):
                for j in range(i + 1, len(conclusions)):
                    span_a, conc_a = conclusions[i]
                    span_b, conc_b = conclusions[j]
                    contradiction_type = self._are_contradictory(conc_a, conc_b)
                    if contradiction_type:
                        return {
                            "severity": 40,
                            "message": (
                                f"Contradictory conclusions from similar reasoning: "
                                f"'{conc_a[:80]}' vs '{conc_b[:80]}' ({contradiction_type})"
                            ),
                            "type": contradiction_type,
                            "conclusion_a": conc_a[:200],
                            "conclusion_b": conc_b[:200],
                            "span_ids": [span_a.span_id, span_b.span_id],
                        }

        return None

    def _check_abandonment(self, spans: list[Span]) -> dict[str, Any] | None:
        """Check for reasoning abandonment -- CoT started but jumps to conclusion."""
        abandoned_spans: list[str] = []

        for span in spans:
            output = self._get_output_text(span)
            if not output or len(output) < 50:
                continue

            output_lower = output.lower()
            total_len = len(output_lower)

            # Check for CoT markers in first 20% of output
            first_segment = output_lower[: int(total_len * 0.2)]
            has_cot_start = any(re.search(pattern, first_segment) for pattern in COT_MARKERS)

            if not has_cot_start:
                continue

            # Check for conclusion in last 10% of output
            last_segment = output_lower[int(total_len * 0.9) :]
            has_conclusion = any(
                re.search(pattern, last_segment) for pattern in CONCLUSION_PATTERNS
            )

            if not has_conclusion:
                continue

            # Check for gap: middle section (20%-90%) should have reasoning content
            middle_section = output_lower[int(total_len * 0.2) : int(total_len * 0.9)]
            reasoning_indicators = [
                r"\b(?:because|since|given|considering|noting)\b",
                r"\b(?:if|then|when|assuming)\b",
                r"\b(?:however|but|although|on the other hand)\b",
                r"\b(?:step\s+\d|first|second|third|next)\b",
                r"\b(?:evidence|data|fact|observation)\b",
            ]
            reasoning_hits = sum(
                1 for pattern in reasoning_indicators if re.search(pattern, middle_section)
            )

            # If middle section has very few reasoning indicators, reasoning was abandoned
            middle_words = len(middle_section.split())
            if middle_words < 10 or reasoning_hits < 2:
                abandoned_spans.append(span.span_id)

        if not abandoned_spans:
            return None

        return {
            "severity": 30,
            "message": (
                f"Reasoning abandonment detected in {len(abandoned_spans)} span(s): "
                f"chain-of-thought started but jumped to conclusion without completing reasoning"
            ),
            "abandoned_span_ids": abandoned_spans,
        }

    def _check_circular_reasoning(self, spans: list[Span]) -> dict[str, Any] | None:
        """Check for circular reasoning -- conclusion restates premise."""
        circular_spans: list[str] = []

        for span in spans:
            input_text = self._get_input_text(span)
            output = self._get_output_text(span)
            if not input_text or not output:
                continue

            conclusion = self._extract_conclusion(span)
            if not conclusion:
                continue

            # Check if conclusion substantially overlaps with input (restating premise)
            input_words = set(w for w in re.findall(r"[a-z]+", input_text.lower()) if len(w) > 3)
            conclusion_words = set(
                w for w in re.findall(r"[a-z]+", conclusion.lower()) if len(w) > 3
            )

            if not input_words or not conclusion_words:
                continue

            overlap = len(input_words & conclusion_words) / len(conclusion_words)

            # High overlap with few new words = circular
            new_words = conclusion_words - input_words
            if overlap > 0.7 and len(new_words) < 3:
                circular_spans.append(span.span_id)

        if not circular_spans:
            return None

        return {
            "severity": 25,
            "message": (
                f"Circular reasoning in {len(circular_spans)} span(s): "
                f"conclusion restates the premise without adding new information"
            ),
            "circular_span_ids": circular_spans,
        }

    def _group_by_similar_input(
        self,
        spans: list[Span],
    ) -> dict[str, list[Span]]:
        """Group spans by similar input content using keyword overlap."""
        groups: dict[str, list[Span]] = {}
        span_keywords: list[tuple[Span, set[str]]] = []

        for span in spans:
            input_text = self._get_input_text(span)
            if not input_text:
                continue
            keywords = set(w for w in re.findall(r"[a-z]+", input_text.lower()) if len(w) > 3)
            if keywords:
                span_keywords.append((span, keywords))

        # Group by pairwise similarity
        assigned: set[int] = set()
        for i, (span_a, kw_a) in enumerate(span_keywords):
            if i in assigned:
                continue

            group_key = f"group_{i}"
            groups[group_key] = [span_a]
            assigned.add(i)

            for j, (span_b, kw_b) in enumerate(span_keywords):
                if j in assigned or j <= i:
                    continue

                union = kw_a | kw_b
                if not union:
                    continue
                overlap = len(kw_a & kw_b) / len(union)
                if overlap >= self.input_similarity_threshold:
                    groups[group_key].append(span_b)
                    assigned.add(j)

        return groups

    def _extract_conclusion(self, span: Span) -> str | None:
        """Extract conclusion from the end of a span's output."""
        output = self._get_output_text(span)
        if not output:
            return None

        output_lower = output.lower()

        # Look for explicit conclusion markers
        best_pos = -1
        for pattern in CONCLUSION_PATTERNS:
            match = re.search(pattern, output_lower)
            if match and match.start() > best_pos:
                best_pos = match.start()

        if best_pos >= 0:
            # Return text after the conclusion marker (up to 500 chars)
            conclusion_text = output[best_pos : best_pos + 500].strip()
            # Take the first sentence/paragraph after the marker
            end_match = re.search(r"[.!?\n]{2,}", conclusion_text[20:])
            if end_match:
                return conclusion_text[: 20 + end_match.end()].strip()
            return conclusion_text

        # Fallback: use last 10% of output as implicit conclusion
        last_portion = output[int(len(output) * 0.9) :].strip()
        if len(last_portion) > 10:
            return last_portion

        return None

    def _are_contradictory(self, conc_a: str, conc_b: str) -> str | None:
        """Check if two conclusions contradict each other. Returns contradiction type or None."""
        a_lower = conc_a.lower()
        b_lower = conc_b.lower()

        # Check boolean contradictions
        a_words = set(re.findall(r"[a-z]+", a_lower))
        b_words = set(re.findall(r"[a-z]+", b_lower))

        a_true = bool(a_words & _TRUE_WORDS)
        a_false = bool(a_words & _FALSE_WORDS)
        b_true = bool(b_words & _TRUE_WORDS)
        b_false = bool(b_words & _FALSE_WORDS)

        if (a_true and b_false) or (a_false and b_true):
            return "opposite_boolean"

        # Check numeric contradictions (different numbers for same quantity)
        a_numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", a_lower)
        b_numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", b_lower)

        if a_numbers and b_numbers:
            # If both have numbers and they differ significantly
            try:
                a_vals = [float(n) for n in a_numbers]
                b_vals = [float(n) for n in b_numbers]
                # Compare first numeric value in each
                if a_vals and b_vals:
                    ratio = max(a_vals[0], b_vals[0]) / max(min(a_vals[0], b_vals[0]), 0.001)
                    if ratio > 1.5 and a_vals[0] != b_vals[0]:
                        return "conflicting_numbers"
            except (ValueError, ZeroDivisionError):
                pass

        # Check for explicit negation patterns
        negation_pairs = [
            (r"\bshould\b", r"\bshould\s+not\b"),
            (r"\bcan\b", r"\bcannot\b"),
            (r"\bwill\b", r"\bwill\s+not\b"),
            (r"\brecommend\b", r"\bdo\s+not\s+recommend\b"),
            (r"\bpossible\b", r"\bimpossible\b"),
            (r"\bsafe\b", r"\bunsafe\b"),
            (r"\bvalid\b", r"\binvalid\b"),
        ]
        for positive, negative in negation_pairs:
            a_has_pos = bool(re.search(positive, a_lower))
            a_has_neg = bool(re.search(negative, a_lower))
            b_has_pos = bool(re.search(positive, b_lower))
            b_has_neg = bool(re.search(negative, b_lower))

            if (a_has_pos and not a_has_neg and b_has_neg) or (
                a_has_neg and b_has_pos and not b_has_neg
            ):
                return "conflicting_recommendation"

        return None

    @staticmethod
    def _get_input_text(span: Span) -> str:
        """Extract text from span input_data."""
        if not span.input_data:
            return ""
        content = span.input_data.get("content", "")
        if isinstance(content, str):
            return content
        prompt = span.input_data.get("prompt", "")
        if isinstance(prompt, str):
            return prompt
        question = span.input_data.get("question", "")
        if isinstance(question, str):
            return question
        # Fallback: join all string values
        parts = [str(v) for v in span.input_data.values() if isinstance(v, str)]
        return " ".join(parts)

    @staticmethod
    def _get_output_text(span: Span) -> str:
        """Extract text from span output_data."""
        if not span.output_data:
            return ""
        content = span.output_data.get("content", "")
        if isinstance(content, str):
            return content
        response = span.output_data.get("response", "")
        if isinstance(response, str):
            return response
        text = span.output_data.get("text", "")
        if isinstance(text, str):
            return text
        # Fallback: join all string values
        parts = [str(v) for v in span.output_data.values() if isinstance(v, str)]
        return " ".join(parts)
