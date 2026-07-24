"""Error propagation detector for tracking silent fact corruption across pipeline steps."""

import re
from typing import Any

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Span, Trace

# Patterns for extracting structured facts from text.
_NUMBER_PATTERN = re.compile(
    r"(?:(?:[$€£¥])\s?[\d,]+(?:\.\d+)?)|"  # Currency values
    r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|"  # Dates
    r"(?:\d+(?:\.\d+)?%)|"  # Percentages
    r"(?:\b\d[\d,]*(?:\.\d+)?\b)"  # Plain numbers
)

_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+")

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Unit tokens that, when present on both sides of a fact difference, indicate
# a legitimate unit conversion (kg <-> lb, C <-> F, GB <-> MB) rather than
# silent corruption.
_UNIT_TOKENS = frozenset(
    {
        # Mass
        "kg",
        "lb",
        "lbs",
        "pound",
        "pounds",
        "kilogram",
        "kilograms",
        "gram",
        "grams",
        "oz",
        "ounce",
        "ounces",
        "mg",
        "milligram",
        "milligrams",
        # Temperature
        "celsius",
        "fahrenheit",
        "°c",
        "°f",
        # Data
        "gb",
        "mb",
        "kb",
        "tb",
        "pb",
        "byte",
        "bytes",
        # Power / energy
        "kilowatt",
        "kilowatt-hours",
        "kilowatt-hour",
        "kwh",
        "mwh",
        "watt",
        "watts",
        # Length
        "meter",
        "meters",
        "metre",
        "metres",
        "foot",
        "feet",
        "inch",
        "inches",
        "km",
        "mile",
        "miles",
        "yard",
        "yards",
        # Speed
        "mph",
        "kph",
        "kmh",
        # Volume
        "liter",
        "liters",
        "litre",
        "litres",
        "gallon",
        "gallons",
        # Time / duration (common conversion source)
        "milliseconds",
        "millisecond",
        "seconds",
        "second",
        "ms",
        # Electrical charge
        "mah",
        "milliamp-hours",
    }
)

# Compact numeric-unit suffixes (e.g. "2500g", "250ms", "0.25s") used to
# detect unit conversions where a unit is fused to its number and would not
# match the main _UNIT_TOKENS substring scan.
_NUM_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?\s?(kg|g|mg|lb|lbs|oz|ms|s|hz|khz|mhz|ghz|gb|mb|kb|tb|kwh|mwh|mah|w|kw)\b",
    re.IGNORECASE,
)

# Magnitude-suffix abbreviations (K, M, B, T). Used to reconcile values such
# as "850,000" with "850K" so that abbreviations are not flagged as
# contradictions, while genuine value changes like "$3.8M" -> "$4.2M" remain
# visible.
_MAGNITUDE_NUMBER_RE = re.compile(
    r"(\d+(?:\.\d+)?)(k|m|b|t)\b",
    re.IGNORECASE,
)
_MAGNITUDE_MULTIPLIERS = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
    "t": 1_000_000_000_000,
}


def _has_unit_conversion(text_a: str, text_b: str) -> bool:
    """Return True when the two texts carry different unit tokens, implying
    a legitimate unit conversion rather than silent corruption."""

    def unit_set(text: str) -> set[str]:
        low = text.lower()
        tokens = {t for t in _UNIT_TOKENS if t in low}
        tokens.update(m.lower() for m in _NUM_UNIT_RE.findall(text))
        return tokens

    tokens_a = unit_set(text_a)
    tokens_b = unit_set(text_b)
    if not tokens_a or not tokens_b:
        return False
    return bool(tokens_a.symmetric_difference(tokens_b))


def _expanded_magnitude_values(text: str) -> set[str]:
    """Return expanded numeric values for any '850K'-style magnitude
    abbreviations in the text (e.g. '850K' -> '850000')."""
    expanded: set[str] = set()
    for num_str, suffix in _MAGNITUDE_NUMBER_RE.findall(text):
        try:
            expanded.add(str(int(float(num_str) * _MAGNITUDE_MULTIPLIERS[suffix.lower()])))
        except (ValueError, KeyError):
            continue
    return expanded


# Words and phrases that signal an intentional update rather than silent
# corruption. Kept lowercase; matched as substrings after lowercasing text.
_UPDATE_SIGNALS = frozenset(
    {
        "corrected",
        "correction",
        "updated",
        "revised",
        "amendment",
        "actually",
        "instead",
        "rather",
        "fixed",
        "recalculated",
        "adjustment",
        "adjusted",
        "modified",
        "changed to",
        # Colloquial update phrases common in agent dialogue and rapid
        # iteration, which previously registered as silent corruption.
        "fixed to",
        "swapped to",
        "swapped for",
        "now using",
        "switched to",
        "replaced with",
        "replaced by",
        "updated from",
        "updated to",
        "revised to",
        "amended to",
        "superseded by",
        "supersedes",
        "on second thought",
        "on reflection",
        "revising",
        "retracting",
        "scratch that",
        "correcting",
        "correction:",
        "update:",
        "re-checked",
        "rechecked",
        "recomputed",
        "recalc",
    }
)


def _extract_facts(text: str) -> dict[str, list[str]]:
    """Extract verifiable facts from text.

    Returns a dict mapping fact category to list of fact values.
    """
    if not text:
        return {}

    facts: dict[str, list[str]] = {}

    numbers = _NUMBER_PATTERN.findall(text)
    if numbers:
        facts["numbers"] = [n.strip() for n in numbers]

    urls = _URL_PATTERN.findall(text)
    if urls:
        facts["urls"] = urls

    emails = _EMAIL_PATTERN.findall(text)
    if emails:
        facts["emails"] = emails

    # Extract named entities: capitalized multi-word sequences.
    names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    if names:
        facts["names"] = names

    return facts


def _has_update_signal(text: str) -> bool:
    """Check whether text contains language signaling an intentional correction."""
    text_lower = text.lower()
    return any(signal in text_lower for signal in _UPDATE_SIGNALS)


def _normalize_number(raw: str) -> str:
    """Normalize a number string for comparison (strip currency, commas)."""
    return re.sub(r"[,$€£¥%\s]", "", raw)


class ErrorPropagationDetector(BaseDetector):
    """Detects silent error propagation across pipeline steps.

    Tracks key facts (numbers, names, dates, URLs) through sequential spans
    and flags when a fact from step N is contradicted or silently dropped
    in step N+2 or later.

    Distinguishes from legitimate updates by checking for explicit
    correction language and reconciling common abbreviations and unit
    conversions.
    """

    name = "propagation"
    description = "Detects silent error propagation and fact corruption across pipeline steps"
    version = "1.1.0"
    platforms = []  # All platforms
    severity_range = (30, 80)
    realtime_capable = False

    # A fact must survive at least this many steps before a contradiction
    # counts. Rapid re-writes within a 2-step window are treated as
    # intentional iteration rather than silent corruption.
    min_propagation_gap = 3

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect error propagation in a trace."""
        processing_kinds = {
            SpanKind.AGENT_TURN,
            SpanKind.TASK,
            SpanKind.CHAIN,
            SpanKind.AGENT,
        }
        processing_spans = [s for s in trace.spans if s.kind in processing_kinds]

        if len(processing_spans) < 3:
            return DetectionResult.no_issue(self.name)

        sorted_spans = sorted(processing_spans, key=lambda s: s.start_time)

        issues: list[str] = []
        severity = 0
        evidence_data: dict[str, Any] = {}

        contradictions = self._track_fact_propagation(sorted_spans)
        if contradictions:
            severity += self._score_contradictions(contradictions)
            worst = contradictions[0]
            issues.append(
                f"Fact contradiction: '{worst['original']}' became "
                f"'{worst['contradicted_by']}' (step {worst['original_step']} -> {worst['contradiction_step']})"
            )
            evidence_data["contradictions"] = contradictions[:10]

        dropped = self._check_dropped_facts(sorted_spans)
        if dropped:
            severity += min(len(dropped) * 8, 30)
            issues.append(f"{len(dropped)} fact(s) silently dropped from pipeline output")
            evidence_data["dropped_facts"] = dropped[:10]

        if not issues:
            return DetectionResult.no_issue(self.name)

        severity = max(self.severity_range[0], min(self.severity_range[1], severity))

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=issues[0],
            fix_type=FixType.ROLLBACK,
            fix_instruction=(
                "A fact was silently corrupted or dropped during pipeline processing. "
                "Review the intermediate outputs and either correct the error or "
                "explicitly acknowledge the change."
            ),
        )

        for issue in issues:
            result.add_evidence(
                description=issue,
                span_ids=[s.span_id for s in sorted_spans[:10]],
                data=evidence_data,
            )

        return result

    def _get_span_output_text(self, span: Span) -> str:
        """Extract output text from a span."""
        parts: list[str] = []
        if span.output_data:
            for key in ("output", "result", "response", "text", "content", "answer"):
                val = span.output_data.get(key)
                if isinstance(val, str):
                    parts.append(val)
            if not parts:
                parts.append(str(span.output_data))
        return " ".join(parts)

    def _track_fact_propagation(self, spans: list[Span]) -> list[dict[str, Any]]:
        """Track facts across pipeline steps and find contradictions.

        A numeric fact from an earlier step is only flagged as contradicted
        when (a) it genuinely disappears from the later step's extracted
        numbers, (b) a different same-magnitude number has taken its place,
        and (c) no legitimate-update signal or abbreviation/unit conversion
        reconciles the two values.
        """
        step_text: list[str] = []
        step_facts: list[dict[str, list[str]]] = []
        for span in spans:
            text = self._get_span_output_text(span)
            step_text.append(text)
            step_facts.append(_extract_facts(text) if text else {})

        # fact_value -> (normalized_value, first_seen_step, category)
        fact_registry: dict[str, tuple[str, int, str]] = {}
        contradictions: list[dict[str, Any]] = []

        for step_idx, facts in enumerate(step_facts):
            text = step_text[step_idx]
            has_update = _has_update_signal(text) if text else False
            current_number_set = {_normalize_number(v) for v in facts.get("numbers", [])}

            for reg_val, (reg_norm, reg_step, reg_cat) in list(fact_registry.items()):
                if reg_cat != "numbers":
                    continue
                if step_idx - reg_step < self.min_propagation_gap:
                    continue
                # The original number still appears verbatim -> persisted.
                if reg_norm in current_number_set:
                    continue
                if has_update:
                    continue
                # Legitimate unit conversion (kg<->lb, C<->F, GB<->MB).
                if _has_unit_conversion(step_text[reg_step], text):
                    continue
                # Magnitude-suffix reconciliation: '850,000' vs '850K'
                # expand to the same value. A true value change such as
                # '$3.8M' -> '$4.2M' still flags because expanded values
                # differ.
                reg_expanded = _expanded_magnitude_values(step_text[reg_step])
                cur_expanded = _expanded_magnitude_values(text)
                if reg_norm in cur_expanded or any(n in reg_expanded for n in current_number_set):
                    continue
                # Require a plausible same-magnitude replacement to avoid
                # flagging "number simply dropped from a paraphrase".
                replacement = None
                for candidate_norm in current_number_set:
                    if candidate_norm == reg_norm:
                        continue
                    if self._same_magnitude(reg_norm, candidate_norm):
                        replacement = candidate_norm
                        break
                if replacement is None:
                    continue
                contradictions.append(
                    {
                        "category": "numbers",
                        "original": reg_val,
                        "contradicted_by": replacement,
                        "original_step": reg_step,
                        "contradiction_step": step_idx,
                        "span_id": spans[step_idx].span_id,
                    }
                )

            registered_norms = {v[0] for v in fact_registry.values()}
            for category, values in facts.items():
                for value in values:
                    normalized = (
                        _normalize_number(value) if category == "numbers" else value.lower()
                    )
                    if normalized in registered_norms:
                        continue
                    fact_registry[value] = (normalized, step_idx, category)
                    registered_norms.add(normalized)

        contradictions.sort(
            key=lambda c: c["contradiction_step"] - c["original_step"],
            reverse=True,
        )
        return contradictions

    def _check_dropped_facts(self, spans: list[Span]) -> list[dict[str, Any]]:
        """Check if verifiable facts from early steps are absent in the
        final output.

        Scoped to high-signal categories (numbers, URLs, emails) -
        capitalized phrases and generic names are commonly omitted from
        summaries and produced too many false positives when included.
        """
        if len(spans) < 3:
            return []

        tracked_categories = {"numbers", "urls", "emails"}

        early_cutoff = max(1, len(spans) // 3)
        early_facts: dict[str, tuple[str, int]] = {}

        for step_idx in range(early_cutoff):
            text = self._get_span_output_text(spans[step_idx])
            facts = _extract_facts(text)
            for category, values in facts.items():
                if category not in tracked_categories:
                    continue
                for value in values:
                    key = f"{category}:{value}"
                    if key not in early_facts:
                        early_facts[key] = (value, step_idx)

        if not early_facts:
            return []

        final_text = self._get_span_output_text(spans[-1])
        if not final_text:
            return []

        final_lower = final_text.lower()
        final_numbers = [_normalize_number(n) for n in _NUMBER_PATTERN.findall(final_text)]
        # Bare-digit scan catches numbers fused to units ("4500mAh",
        # "2500g", "250ms") which the regex drops at its trailing word
        # boundary.
        final_bare_digits = set(re.findall(r"\d[\d,]*(?:\.\d+)?", final_text))
        final_bare_norm = {_normalize_number(d) for d in final_bare_digits}
        dropped: list[dict[str, Any]] = []

        for key, (value, step_idx) in early_facts.items():
            category = key.split(":", 1)[0]
            if category == "numbers":
                search_val = _normalize_number(value)
                if search_val in final_numbers:
                    continue
                if search_val in final_bare_norm:
                    continue
                # Calendar-year-shaped numbers (1900-2099) frequently act
                # as context in early steps and are legitimately omitted
                # from summaries. Skip unless the final step carries a
                # *different* year, which would be a real contradiction.
                try:
                    year_candidate = int(float(search_val))
                    if 1900 <= year_candidate <= 2099:
                        final_years = {
                            int(n) for n in final_numbers if n.isdigit() and 1900 <= int(n) <= 2099
                        }
                        if not final_years:
                            continue
                except (TypeError, ValueError):
                    pass
                original_text = self._get_span_output_text(spans[step_idx])
                if _has_unit_conversion(original_text, final_text):
                    continue
                # Magnitude-suffix reconciliation for abbreviated forms.
                if search_val in _expanded_magnitude_values(final_text):
                    continue
                # Explicit update in the final step acknowledges the change.
                if _has_update_signal(final_text):
                    continue
                dropped.append(
                    {
                        "category": category,
                        "value": value,
                        "first_seen_step": step_idx,
                    }
                )
            else:
                search_val = value.lower()
                if search_val not in final_lower:
                    dropped.append(
                        {
                            "category": category,
                            "value": value,
                            "first_seen_step": step_idx,
                        }
                    )

        return dropped

    @staticmethod
    def _same_magnitude(a: str, b: str) -> bool:
        """Check if two number strings are in the same order of magnitude.

        Distinguishes a genuine contradiction (100 vs 150) from unrelated
        numbers (5 vs 50000).
        """
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if fa == 0 or fb == 0:
            return False
        ratio = max(fa, fb) / min(fa, fb)
        return ratio < 10

    def _score_contradictions(self, contradictions: list[dict[str, Any]]) -> int:
        """Score the severity of contradictions."""
        if not contradictions:
            return 0
        return min(30 + (len(contradictions) - 1) * 10, 60)
