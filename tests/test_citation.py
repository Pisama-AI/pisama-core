"""Regression tests for CitationDetector recall-lift (Sprint 10 VV, v1.6.3).

Covers:
- Decimal-point / title-abbreviation claim truncation bug fix
- Synonym-aware overlap normalization
- Expanded approximation markers
- Single-digit-unit numeric-fact fabrication detection
- Precision guards (unrelated claims still flag, paraphrase TNs)
"""

from __future__ import annotations

import pytest

from pisama_core.detection.detectors.citation import (
    _APPROXIMATION_MARKERS,
    CitationDetector,
    _claim_supported_by_source,
    _extract_citations,
    _trim_to_first_sentence,
)
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Trace


def _build_trace(output: str, sources: list[str]) -> Trace:
    """Assemble the same trace shape the calibrate adapter uses."""
    trace = Trace()
    names = ["the report", "the data", "the record", "the study", "the document"]
    for i, src in enumerate(sources):
        trace.create_span(
            name=names[i % len(names)],
            kind=SpanKind.RETRIEVAL,
            output_data={"content": src},
        )
    trace.create_span(
        name="agent_output",
        kind=SpanKind.AGENT,
        output_data={"output": output},
    )
    return trace


class TestTrimToFirstSentence:
    """Unit tests for the decimal-safe sentence splitter."""

    def test_decimal_not_terminator(self) -> None:
        out = _trim_to_first_sentence("the rate was 99.2% after three months of testing")
        assert "99.2%" in out
        assert "three months" in out

    def test_title_abbreviation_not_terminator(self) -> None:
        out = _trim_to_first_sentence("led by Dr. Maria Rodriguez from the Oceanographic Center")
        assert "Dr. Maria" in out
        assert "Oceanographic" in out

    def test_real_sentence_terminator(self) -> None:
        out = _trim_to_first_sentence("Output was solid. The next sentence is unrelated")
        assert out == "Output was solid"

    def test_currency_decimal_not_terminator(self) -> None:
        out = _trim_to_first_sentence("budget of $2.8 million with strong regional growth")
        assert "$2.8 million" in out


class TestDecimalPointFix:
    """The core VV bug: claims used to truncate at decimals inside numbers."""

    def test_claim_extends_past_decimal_number(self) -> None:
        text = (
            "According to clinical data, the new protocol reduced recovery time to "
            "6.5 days but increased side effect incidence to 18% of patients."
        )
        cits = _extract_citations(text)
        assert cits, "expected at least one citation"
        claim = cits[0]["claim"]
        # The fabricated tail (18% of patients) must now be inside the claim.
        assert "18%" in claim or "18 %" in claim
        assert "side effect" in claim.lower()

    def test_claim_extends_past_title_abbreviation(self) -> None:
        text = (
            "According to grant documentation, the study is led by "
            "Dr. Maria Rodriguez from the Oceanographic Research Center."
        )
        cits = _extract_citations(text)
        assert cits
        claim = cits[0]["claim"]
        assert "Oceanographic" in claim

    @pytest.mark.asyncio
    async def test_partial_fabrication_with_decimal_is_detected(self) -> None:
        det = CitationDetector()
        trace = _build_trace(
            output=(
                "According to app analytics, downloads reached 2.1 million with a "
                "4.6-star rating, though users typically engage for only 8 minutes per session."
            ),
            sources=[
                "The mobile app has 2.1 million downloads with an average rating of "
                "4.6 stars. Users spend an average of 18 minutes per session."
            ],
        )
        result = await det.detect(trace)
        assert result.detected, (
            "Fabricated '8 minutes per session' vs source '18 minutes' must be flagged "
            "via single-digit-unit numeric-fact check once the decimal truncation bug is fixed"
        )


class TestSynonymMap:
    """Paraphrase-friendly overlap via the small synonym map."""

    def test_synonyms_boost_overlap(self) -> None:
        claim = "annual budget serves the research team"
        source = "yearly allocated covers the research team"
        score = _claim_supported_by_source(claim, source)
        # All three synonym pairs (annual/yearly, budget/allocated, serves/covers)
        # should make this look heavily supported.
        assert score >= 0.75, f"expected synonym-normalized overlap ≥ 0.75, got {score:.3f}"

    def test_unrelated_claim_still_low_overlap(self) -> None:
        claim = "the quarterly dividend was raised by twelve percent"
        source = "the new intern ordered lunch yesterday"
        score = _claim_supported_by_source(claim, source)
        assert score < 0.2, f"expected disjoint overlap < 0.2, got {score:.3f}"


class TestApproximationMarkers:
    """Expanded approximation markers cover more hedging phrases."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "close to 50% of users",
            "just over a million downloads",
            "just under 6.5 days",
            "almost exactly 3 hours",
            "a bit more than two-thirds",
            "approximately 95% uptime",
            "in the millions of requests",
        ],
    )
    def test_marker_matches(self, phrase: str) -> None:
        assert _APPROXIMATION_MARKERS.search(phrase) is not None


class TestFabricationGuards:
    """Precision guards: unrelated claims and full fabrications still flag."""

    @pytest.mark.asyncio
    async def test_unrelated_claim_is_flagged(self) -> None:
        det = CitationDetector()
        trace = _build_trace(
            output=(
                "According to the quarterly earnings call, executives revealed "
                "classified military contracts worth seventeen billion dollars."
            ),
            sources=[
                "Q3 summary: Revenue reached $42.5 million with steady year-over-year growth."
            ],
        )
        result = await det.detect(trace)
        assert result.detected, "wholly fabricated content must be flagged"

    @pytest.mark.asyncio
    async def test_faithful_citation_not_flagged(self) -> None:
        det = CitationDetector()
        trace = _build_trace(
            output=(
                "According to the quarterly report, revenue reached $42.5 million in Q3, "
                "a 12% year-over-year increase."
            ),
            sources=[
                "Q3 2024 financial summary: Revenue reached $42.5 million, representing "
                "a 12% year-over-year increase."
            ],
        )
        result = await det.detect(trace)
        assert not result.detected, "grounded citation must not be flagged"
