"""Memory staleness detector for outdated context being used in current tasks."""

import re
from datetime import datetime, timezone
from typing import Any, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Span, Trace


class MemoryStalenessDetector(BaseDetector):
    """Detects outdated memory/context being used for current tasks.

    This detector identifies:
    - Retrieval spans containing old temporal references
    - Retrieved content with explicit "as of [old date]" markers
    - Temporal mismatch between retrieved context and task context
    """

    name = "memory_staleness"
    description = "Detects outdated memory/context being used for current tasks"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (20, 60)
    realtime_capable = False

    # How many days old before we consider content stale
    staleness_threshold_days = 365

    # Patterns for extracting dates
    _EXPLICIT_DATE_PATTERN = re.compile(
        r"(?:as of|data from|updated|last updated|current as of|"
        r"dated|retrieved on|published|recorded|effective)\s+"
        r"(\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})",
        re.IGNORECASE,
    )
    _YEAR_PATTERN = re.compile(
        r"(?:in|since|from|during|circa|year)\s+(\d{4})\b",
        re.IGNORECASE,
    )
    _RELATIVE_OLD_PATTERN = re.compile(
        r"(?:last year|two years ago|three years ago|several years ago|"
        r"years ago|a few years back|back in \d{4}|prior to \d{4})",
        re.IGNORECASE,
    )
    _MONTH_YEAR_PATTERN = re.compile(
        r"(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(\d{4})",
        re.IGNORECASE,
    )

    _MONTH_MAP = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect stale memory/context usage in the trace."""
        retrieval_spans = trace.get_spans_by_kind(SpanKind.RETRIEVAL)
        if not retrieval_spans:
            return DetectionResult.no_issue(self.name)

        # Determine the task's temporal context
        task_reference_time = self._get_task_reference_time(trace)

        issues: list[str] = []
        severity = 0
        evidence_spans: list[str] = []

        for span in retrieval_spans:
            text = self._get_span_text(span)
            if not text:
                continue

            stale_refs = self._find_stale_references(text, task_reference_time)
            for ref in stale_refs:
                issues.append(ref["description"])
                evidence_spans.append(span.span_id)
                severity += ref["severity_contribution"]

        if not issues:
            return DetectionResult.no_issue(self.name)

        severity = max(self.severity_range[0], min(self.severity_range[1], severity))

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=issues[0]
            if len(issues) == 1
            else f"{len(issues)} stale memory references detected",
            fix_type=FixType.RESET_CONTEXT,
            fix_instruction=(
                "Retrieved content contains outdated information. "
                "Re-fetch current data or explicitly note the temporal limitation "
                "to the user before using stale information for decisions."
            ),
        )

        for issue in issues:
            result.add_evidence(
                description=issue,
                span_ids=evidence_spans,
            )

        return result

    def _get_task_reference_time(self, trace: Trace) -> datetime:
        """Determine the temporal context of the current task.

        Uses dates from user input spans, or falls back to trace creation time.
        """
        # Check user input spans for date references
        user_spans = trace.get_spans_by_kind(SpanKind.USER_INPUT)
        for span in user_spans:
            text = self._get_span_text(span)
            if text:
                parsed = self._extract_newest_date(text)
                if parsed:
                    return parsed

        # Fall back to trace metadata creation time
        return trace.metadata.created_at

    def _extract_newest_date(self, text: str) -> Optional[datetime]:
        """Extract the most recent date mentioned in text."""
        dates: list[datetime] = []

        # Try explicit date patterns (YYYY-MM-DD, Month DD YYYY, etc.)
        for match in self._EXPLICIT_DATE_PATTERN.finditer(text):
            parsed = self._parse_date_string(match.group(1))
            if parsed:
                dates.append(parsed)

        # Try month-year patterns
        for match in self._MONTH_YEAR_PATTERN.finditer(text):
            month_name = match.group(1).lower()
            year = int(match.group(2))
            month = self._MONTH_MAP.get(month_name, 1)
            if 1990 <= year <= 2100:
                dates.append(datetime(year, month, 1, tzinfo=timezone.utc))

        # Try year-only patterns
        for match in self._YEAR_PATTERN.finditer(text):
            year = int(match.group(1))
            if 1990 <= year <= 2100:
                dates.append(datetime(year, 7, 1, tzinfo=timezone.utc))

        return max(dates) if dates else None

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse a date string into a datetime."""
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%B %d, %Y",
            "%B %d %Y",
        ]
        date_str = date_str.strip().rstrip(",")
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _find_stale_references(self, text: str, reference_time: datetime) -> list[dict[str, Any]]:
        """Find stale temporal references in retrieved content."""
        stale_refs: list[dict[str, Any]] = []

        # Check explicit "as of" / "data from" patterns
        for match in self._EXPLICIT_DATE_PATTERN.finditer(text):
            parsed = self._parse_date_string(match.group(1))
            if parsed:
                age_days = (reference_time - parsed).days
                if age_days > self.staleness_threshold_days:
                    years = age_days / 365.25
                    stale_refs.append(
                        {
                            "description": (
                                f"Retrieved content references '{match.group(0).strip()}' "
                                f"({years:.1f} years old)"
                            ),
                            "severity_contribution": min(20, 10 + int(years) * 3),
                        }
                    )

        # Check month-year references
        for match in self._MONTH_YEAR_PATTERN.finditer(text):
            month_name = match.group(1).lower()
            year = int(match.group(2))
            month = self._MONTH_MAP.get(month_name, 1)
            if 1990 <= year <= 2100:
                ref_date = datetime(year, month, 1, tzinfo=timezone.utc)
                age_days = (reference_time - ref_date).days
                if age_days > self.staleness_threshold_days:
                    years = age_days / 365.25
                    stale_refs.append(
                        {
                            "description": (
                                f"Retrieved content mentions '{match.group(0)}' "
                                f"({years:.1f} years old)"
                            ),
                            "severity_contribution": min(15, 8 + int(years) * 2),
                        }
                    )

        # Check year-only references
        for match in self._YEAR_PATTERN.finditer(text):
            year = int(match.group(1))
            if 1990 <= year <= 2100:
                ref_date = datetime(year, 7, 1, tzinfo=timezone.utc)
                age_days = (reference_time - ref_date).days
                if age_days > self.staleness_threshold_days:
                    years = age_days / 365.25
                    stale_refs.append(
                        {
                            "description": (
                                f"Retrieved content references year {year} ({years:.1f} years ago)"
                            ),
                            "severity_contribution": min(12, 5 + int(years) * 2),
                        }
                    )

        # Check relative old references
        for match in self._RELATIVE_OLD_PATTERN.finditer(text):
            stale_refs.append(
                {
                    "description": (
                        f"Retrieved content uses relative past reference: '{match.group(0)}'"
                    ),
                    "severity_contribution": 10,
                }
            )

        return stale_refs

    def _get_span_text(self, span: Span) -> str:
        """Extract text content from a span's output_data and input_data."""
        parts: list[str] = []

        for data in (span.output_data, span.input_data):
            if not data:
                continue
            for key in (
                "text",
                "content",
                "result",
                "output",
                "response",
                "message",
                "query",
                "documents",
                "context",
            ):
                val = data.get(key)
                if isinstance(val, str):
                    parts.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            for v in item.values():
                                if isinstance(v, str):
                                    parts.append(v)

        return " ".join(parts)
