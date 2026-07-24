"""Regression tests for ErrorPropagationDetector precision fix.

Covers the Phase J.3 work (2026-04-17) that raised cap F1 from 0.730 to
0.899 by:
- Requiring a same-magnitude replacement before flagging a contradiction,
  so unrelated numbers in the same step no longer cross-flag each other.
- Expanding _UPDATE_SIGNALS with colloquial variants like "swapped to",
  "now using", "switched to".
- Raising min_propagation_gap from 2 to 3 steps.
- Scoping dropped-fact detection to numbers/URLs/emails (no generic names).
- Adding unit-conversion and magnitude-suffix reconciliation.
- Skipping calendar-year-shaped context numbers.
"""

import pytest

from pisama_core.detection.detectors.propagation import ErrorPropagationDetector
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Trace


def _build_trace(step_outputs: list[str]) -> Trace:
    """Build a trace with one CHAIN span per pipeline step."""
    trace = Trace()
    for i, out in enumerate(step_outputs):
        trace.create_span(
            name=f"step_{i + 1}",
            kind=SpanKind.CHAIN,
            output_data={"output": out},
        )
    return trace


@pytest.fixture
def detector() -> ErrorPropagationDetector:
    return ErrorPropagationDetector()


async def _detect(detector: ErrorPropagationDetector, steps: list[str]) -> bool:
    result = await detector.detect(_build_trace(steps))
    return bool(result.detected)


class TestPrecisionGuards:
    """Guards that suppress the FPs identified in the baseline analysis."""

    async def test_consistent_order_data_is_not_flagged(self, detector):
        """Numbers that all persist across steps are not a contradiction.

        This was the dominant FP: '$9.99' and '1 item' from step 1 got
        compared against each other because they are same-magnitude.
        """
        assert not await _detect(
            detector,
            [
                "Order #1000: 1 items at $9.99 each.",
                "Processing order #1000: 1 items, $9.99 per unit.",
                "Shipped order #1000: 1 items at $9.99 each.",
            ],
        )

    async def test_quarterly_revenue_paraphrase_is_not_flagged(self, detector):
        """Revenue restated across steps with no value change."""
        assert not await _detect(
            detector,
            [
                "Revenue for Q3: $4.2 million.",
                "Q3 revenue was $4.2 million, representing 12% growth.",
                "Summary: Q3 revenue of $4.2M grew 12% year-over-year.",
            ],
        )

    async def test_magnitude_abbreviation_reconciles(self, detector):
        """'850,000' and '850K' are the same value; not a contradiction."""
        assert not await _detect(
            detector,
            [
                "Project budget allocated: €850,000 for Q4 initiatives.",
                "Q4 project funding: €850,000 approved and allocated.",
                "Budget confirmation: €850K allocated for Q4 projects.",
            ],
        )

    async def test_unit_conversion_is_not_flagged(self, detector):
        """kg -> lb unit conversion should be tolerated."""
        assert not await _detect(
            detector,
            [
                "Package weight: 2.75 kilograms for international shipping.",
                "Shipping weight: 6.06 pounds for overseas delivery.",
                "International package: 6.06 lbs ready for shipment.",
            ],
        )

    async def test_calendar_year_drop_is_not_flagged(self, detector):
        """A 2024 mention that drops from the summary is not corruption."""
        assert not await _detect(
            detector,
            [
                "Energy consumption: Building used 1,250 kWh in October 2024.",
                "Utility billing: October 2024 electricity usage: 1,250 kilowatt-hours.",
                "Cost calculation: 1,250 kWh consumed in October at $0.12 per kWh.",
            ],
        )


class TestRecallPreserved:
    """Regression coverage for real contradictions that must still flag."""

    async def test_currency_value_change_flags(self, detector):
        """'$3.8M' -> '$4.2M' is a real contradiction."""
        assert await _detect(
            detector,
            [
                "Quarterly revenue: Q2 2024 generated $3.8 million in sales.",
                "Financial analysis: Q2 2024 revenue of $3.8M represents 8% growth.",
                "Board report: Q2 revenue reached $4.2M with 8% year-over-year growth.",
            ],
        )

    async def test_different_year_contradiction_flags(self, detector):
        """Badge expiry year change (2024 -> 2025) is a real contradiction."""
        assert await _detect(
            detector,
            [
                "Security badge expires on December 31, 2024 for John Miller.",
                "Badge renewal needed: John Miller expires December 31, 2025.",
                "Access card: John Miller valid until December 31, 2025.",
            ],
        )


class TestUpdateSignalsAndGap:
    """Coverage for new colloquial update phrases and the larger gap."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "swapped to",
            "now using",
            "switched to",
            "replaced with",
        ],
    )
    async def test_colloquial_update_phrase_suppresses_contradiction(self, detector, phrase):
        """An explicit update phrase in the later step should suppress."""
        assert not await _detect(
            detector,
            [
                "Price quoted at $1,500 for the service package.",
                "Customer reviewing $1,500 proposal with stakeholders.",
                f"Final invoice: {phrase} $1,800 after scope addition.",
            ],
        )

    async def test_rapid_two_step_rewrite_with_acknowledgement_not_flagged(self, detector):
        """Changes explicitly acknowledged in the final step (via an update
        signal) do not constitute silent error propagation."""
        assert not await _detect(
            detector,
            [
                "Initial estimate: $1,200 for the project.",
                "Revising estimate after new scope: $1,450.",
                "Corrected final budget updated to $1,450 for the project.",
            ],
        )


class TestNameDropNoLongerFires:
    """Verify that dropped-fact detection no longer flags generic names."""

    async def test_role_title_drop_does_not_flag(self, detector):
        """'Software Engineer' dropping from a summary is not corruption."""
        assert not await _detect(
            detector,
            [
                "Employee salary: $85,000 annually for John Smith, Software Engineer.",
                "Payroll processing: John Smith (Software Engineer) - $85,000 per year.",
                "Tax calculation for John Smith: annual salary $85,000.",
            ],
        )
