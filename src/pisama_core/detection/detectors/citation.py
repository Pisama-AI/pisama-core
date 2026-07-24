"""Citation detector for identifying fabricated citations and source misattribution."""

import re
from typing import Any

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Span, Trace

# Patterns that introduce a citation (claim attributed to a source).
#
# The claim group uses ``[^!\n]{10,}`` (no early period termination): a period
# inside a decimal ("99.2%"), a currency amount ("$2.8 million"), or a title
# abbreviation ("Dr. Rodriguez") must NOT truncate the claim, or the fabricated
# tail is dropped and recall collapses. Claims are trimmed to a single sentence
# post-extraction by ``_trim_to_first_sentence``.
_CITATION_PATTERNS = [
    # "according to [source], [claim]"
    re.compile(
        r"according\s+to\s+(?P<source>[^,\.\n]{3,80})\s*[,:]?\s*(?P<claim>[^!\n]{10,})",
        re.IGNORECASE,
    ),
    # "source states/says/reports/indicates that [claim]"
    re.compile(
        r"(?P<source>[A-Z][^\.,]{2,60})\s+(?:states?|says?|reports?|indicates?|mentions?|notes?|confirms?|shows?|argues?|claims?|contends?|asserts?|alleges?|suggests?)\s+(?:that\s+)?(?P<claim>[^!\n]{10,})",
        re.IGNORECASE,
    ),
    # "from [source]: [claim]" or "from [source], [claim]"
    re.compile(
        r"from\s+(?P<source>[^:,\n]{3,80})\s*[:,]\s*(?P<claim>[^!\n]{10,})",
        re.IGNORECASE,
    ),
    # "[source]: \"[claim]\""
    re.compile(
        r"(?P<source>[A-Z][^\:]{2,60}):\s*[\"'](?P<claim>[^\"']{10,})[\"']",
    ),
    # "as stated in [source]"
    re.compile(
        r"as\s+(?:stated|mentioned|described|noted|documented)\s+in\s+(?P<source>[^,\.\n]{3,80})\s*[,:]?\s*(?P<claim>[^!\n]{10,})",
        re.IGNORECASE,
    ),
    # "per [source], [claim]"
    re.compile(
        r"per\s+(?P<source>[^,\.\n]{3,80})\s*,\s*(?P<claim>[^!\n]{10,})",
        re.IGNORECASE,
    ),
    # "[claim] (source: [source])"
    re.compile(
        r"(?P<claim>[^(]{10,})\s*\(\s*source:\s*(?P<source>[^)]{3,80})\s*\)",
        re.IGNORECASE,
    ),
]


# Title abbreviations that contain a period but never end a sentence.
_TITLE_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "st",
        "sr",
        "jr",
        "inc",
        "corp",
        "co",
        "ltd",
        "llc",
        "vs",
        "e.g",
        "i.e",
        "etc",
        "fig",
        "vol",
        "no",
    }
)


def _trim_to_first_sentence(claim: str) -> str:
    """Trim a greedily-matched claim to its first sentence.

    A period is treated as a sentence terminator only when:
      - NOT preceded by a digit AND followed by a digit (decimal number), AND
      - NOT preceded by a known title abbreviation ("Dr.", "Inc."), AND
      - followed by whitespace + an uppercase letter OR end-of-string.

    This keeps fabricated tails after decimals / titles inside the claim
    so the numeric-fact and overlap checks see the full claim, while still
    stopping at genuine sentence boundaries.
    """
    claim = claim.rstrip(" \t,;:")
    if not claim:
        return claim

    # Find the first sentence boundary.
    for i, ch in enumerate(claim):
        if ch != ".":
            continue
        # decimal number: digit on both sides
        prev_ch = claim[i - 1] if i > 0 else ""
        next_ch = claim[i + 1] if i + 1 < len(claim) else ""
        if prev_ch.isdigit() and next_ch.isdigit():
            continue
        # title abbreviation: preceding token (letters only) in the known set
        j = i - 1
        while j >= 0 and claim[j].isalpha():
            j -= 1
        preceding_token = claim[j + 1 : i].lower()
        if preceding_token in _TITLE_ABBREVIATIONS:
            continue
        # sentence terminator: followed by whitespace + uppercase, or end
        if i + 1 >= len(claim):
            return claim[:i].rstrip(" \t,;:")
        rest = claim[i + 1 :]
        rest_lstripped = rest.lstrip()
        if rest_lstripped and rest_lstripped[0].isupper():
            return claim[:i].rstrip(" \t,;:")
        # single trailing period with no following text
        if not rest_lstripped:
            return claim[:i].rstrip(" \t,;:")
    # no internal terminator found — claim is a single sentence
    return claim.rstrip(" \t,;:.")


# Small bidirectional synonym map for overlap normalization. Keep this tiny —
# each entry risks a precision regression. Pairs come from the Sprint 9 hard
# paraphrase generator (grounding) but are equally applicable here.
_SYNONYM_MAP: dict[str, str] = {
    "allocated": "budget",
    "budget": "budget",
    "record": "report",
    "report": "report",
    "serves": "covers",
    "covers": "covers",
    "annual": "yearly",
    "yearly": "yearly",
    "manual": "human",
    "human": "human",
    "review": "check",
    "check": "check",
}


def _synonym_normalize(tokens: set[str]) -> set[str]:
    """Map each token through the synonym map when present."""
    return {_SYNONYM_MAP.get(t, t) for t in tokens}


def _extract_citations(text: str) -> list[dict[str, Any]]:
    """Extract all citations from text, each with a source name and claimed content."""
    if not text:
        return []

    citations: list[dict[str, Any]] = []
    seen_claims: set[str] = set()

    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(text):
            source = match.group("source").strip()
            claim = _trim_to_first_sentence(match.group("claim").strip())
            if len(claim) < 10:
                continue

            # Deduplicate by claim content
            claim_key = claim[:50].lower()
            if claim_key in seen_claims:
                continue
            seen_claims.add(claim_key)

            citations.append(
                {
                    "source": source,
                    "claim": claim,
                    "match_start": match.start(),
                }
            )

    return citations


def _extract_source_content(trace: Trace) -> dict[str, str]:
    """Collect source content from RETRIEVAL spans and tool outputs.

    Returns a mapping from source identifier to source text content.
    """
    sources: dict[str, str] = {}

    # Collect from RETRIEVAL spans
    retrieval_spans = trace.get_spans_by_kind(SpanKind.RETRIEVAL)
    for span in retrieval_spans:
        source_id = span.name or span.span_id
        content_parts: list[str] = []

        if span.output_data:
            for key in (
                "content",
                "text",
                "document",
                "result",
                "output",
                "retrieved_text",
                "chunk",
                "passage",
            ):
                val = span.output_data.get(key)
                if isinstance(val, str):
                    content_parts.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            content_parts.append(item)
                        elif isinstance(item, dict):
                            for sub_key in ("content", "text", "page_content"):
                                sub_val = item.get(sub_key)
                                if isinstance(sub_val, str):
                                    content_parts.append(sub_val)

            # Also check for source metadata
            source_meta = span.output_data.get("source", span.output_data.get("metadata", {}))
            if isinstance(source_meta, dict):
                for key in ("title", "name", "filename", "url"):
                    val = source_meta.get(key)
                    if isinstance(val, str):
                        source_id = val
                        break

        if content_parts:
            merged = " ".join(content_parts)
            if source_id in sources:
                sources[source_id] = sources[source_id] + " " + merged
            else:
                sources[source_id] = merged

    # Also collect from TOOL spans that might fetch external content
    tool_spans = trace.get_spans_by_kind(SpanKind.TOOL)
    for span in tool_spans:
        name_lower = span.name.lower()
        if not any(
            kw in name_lower for kw in ("search", "fetch", "read", "retrieve", "get", "query")
        ):
            continue

        if span.output_data:
            content_parts = []
            for key in ("content", "text", "result", "output", "body", "data"):
                val = span.output_data.get(key)
                if isinstance(val, str) and len(val) > 20:
                    content_parts.append(val)
            if content_parts:
                sources[span.name] = " ".join(content_parts)

    return sources


def _claim_supported_by_source(claim: str, source_text: str) -> float:
    """Check if a claim is supported by source text using fuzzy word overlap.

    Applies a small synonym map (``_SYNONYM_MAP``) before computing set
    overlap so paraphrases like "budget" ↔ "allocated" and
    "manual review" ↔ "human check" don't artificially lower the ratio.

    Returns 0.0 if the source explicitly negates the claim's numeric facts.
    Returns the overlap ratio (0.0 to 1.0) otherwise.
    """
    if not claim or not source_text:
        return 0.0
    if _claim_negated_by_source(claim, source_text):
        return 0.0

    # Extract meaningful words (3+ chars, lowercased)
    claim_words = {w.lower() for w in re.findall(r"[a-z0-9]+", claim.lower()) if len(w) >= 3}
    source_words = {w.lower() for w in re.findall(r"[a-z0-9]+", source_text.lower()) if len(w) >= 3}

    if not claim_words:
        return 1.0  # Nothing to verify

    # Synonym-normalize both sides so paraphrase pairs line up.
    claim_norm = _synonym_normalize(claim_words)
    source_norm = _synonym_normalize(source_words)
    overlap = len(claim_norm & source_norm)
    return overlap / len(claim_norm)


# Regex for specific numeric facts in a claim: percentages, currency, units.
# Matches things like "95%", "$2.3 million", "15,000", "101.3°F", "48 hours",
# "8.5%", "$48". Avoids bare small integers (1-9) to reduce false positives
# on counts like "1 study" or "3 months".
_NUMERIC_FACT_PATTERNS = [
    # Percentages with optional decimal: 95%, 8.5%, 0.8%
    re.compile(r"\b\d+(?:\.\d+)?\s*%"),
    # Currency: $2.3 million, $750,000, $48, $47.82
    re.compile(r"\$\s*\d+(?:[.,]\d+)*(?:\s*(?:million|billion|thousand|k|m|b))?", re.IGNORECASE),
    # Multi-digit integers (≥10) with optional comma/decimal, optional units
    re.compile(
        r"\b\d{2,}(?:[.,]\d+)*"
        r"(?!\s*[-–]\s*(?:point|level|item|scale|tier|grade|star))"
        r"\s*(?:million|billion|thousand|units|stars|hours?|minutes?|seconds?|days?|weeks?|months?|years?|patients?|participants?|responses?|ms|cases?|cycles?|degrees?|°[cCfFkK]|miles?|km|kg|lbs?)?\b",
        re.IGNORECASE,
    ),
    # Single-digit with strong measurement unit
    re.compile(
        r"\b\d(?:\.\d+)?\s*"
        r"(?:grams?|mg|oz|lbs?|minutes?|hours?|days?|weeks?|months?|years?|percent|stars?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d+(?:\.\d+)?\s*°[cCfFkK]\b"),
]

# "Softening" words that indicate approximation/paraphrase — skip numeric
# strict-match when present, to preserve precision on approximations.
_APPROXIMATION_MARKERS = re.compile(
    r"\b(?:approximately|approximately\s+equal\s+to|roughly|nearly|almost\s+exactly|"
    r"about|around|close\s+to|just\s+over|just\s+under|"
    r"over\s+(?=\d|\$)|under\s+(?=\d|\$)|"
    r"a\s+bit\s+more\s+than|a\s+bit\s+less\s+than|more\s+than|less\s+than|"
    r"at\s+least|at\s+most|in\s+the\s+(?:low|mid|high)?\s*"
    r"(?:millions|billions|thousands|hundreds)|"
    r"two[-\s]thirds|three[-\s]quarters|half|quarter|majority|minority|"
    r"most|some|few|several)\b",
    re.IGNORECASE,
)


def _normalize_number(num_str: str) -> str:
    """Normalize a number string for comparison: strip commas, lowercase units."""
    return re.sub(r"[\s,]", "", num_str).lower()


def _is_year(fact: str) -> bool:
    """Return True if the fact string is a 4-digit year (1900-2099)."""
    m = re.match(r"^(\d{4})$", fact.strip())
    if m is None:
        return False
    return 1900 <= int(m.group(1)) <= 2099


_RANGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)"
    r"|between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)"
    r"|(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_NEGATION_PATTERN = re.compile(
    r"\bnot\s+(?:the\s+)?(\S+)",
    re.IGNORECASE,
)


def _value_in_source_range(value_str: str, source_text: str) -> bool:
    """Return True if value_str falls within a numeric range in source_text."""
    num_match = re.match(r"(\d+(?:\.\d+)?)", value_str)
    if not num_match:
        return False
    try:
        val = float(num_match.group(1))
    except ValueError:
        return False
    for m in _RANGE_PATTERN.finditer(source_text):
        g = m.groups()
        try:
            if g[0] and g[1]:
                lo, hi = float(g[0]), float(g[1])
            elif g[2] and g[3]:
                lo, hi = float(g[2]), float(g[3])
            elif g[4] and g[5]:
                lo, hi = float(g[4]), float(g[5])
            else:
                continue
        except ValueError:
            continue
        if lo <= val <= hi:
            return True
    return False


def _claim_negated_by_source(claim: str, source_text: str) -> bool:
    """Return True if the source text explicitly negates numeric facts in the claim."""
    if not claim or not source_text:
        return False
    claim_facts = _extract_numeric_facts(claim)
    if not claim_facts:
        return False
    negated_values: set[str] = set()
    for m in _NEGATION_PATTERN.finditer(source_text):
        negated_val = _normalize_number(m.group(1).rstrip(".,;:!?"))
        if negated_val:
            negated_values.add(negated_val)
    claim_norm_facts = {_normalize_number(f) for f in claim_facts}
    negated_matches = claim_norm_facts & negated_values
    # Also match when claim fact has a unit but negated value is bare number
    for f in claim_facts:
        m2 = re.match(r"(\d+(?:\.\d+)?)", _normalize_number(f))
        if m2 and m2.group(1) in negated_values:
            negated_matches = negated_matches | {_normalize_number(f)}
    if not negated_matches:
        return False
    source_norm = _normalize_number(source_text)
    for fact in negated_matches:
        safe_fact = re.escape(fact)
        found_non_negated = False
        for m3 in re.finditer(rf"\b{safe_fact}\b", source_norm):
            before = source_norm[max(0, m3.start() - 10) : m3.start()]
            if not re.search(r"\bnot\s*$", before):
                found_non_negated = True
                break
        if not found_non_negated:
            return True
    return False


def _extract_numeric_facts(text: str) -> list[str]:
    """Extract specific numeric facts (percentages, currency, units) from text."""
    facts: list[str] = []
    for pat in _NUMERIC_FACT_PATTERNS:
        for m in pat.finditer(text):
            facts.append(_normalize_number(m.group(0)))
    return facts


def _numeric_facts_unsupported(claim: str, source_text: str) -> list[str]:
    """Return numeric facts from the claim that are not present in source_text.

    Skips entirely when the claim uses approximation markers (paraphrasing).
    Only returns facts that would be a strict fabrication signal.
    """
    if not claim or not source_text:
        return []
    if _APPROXIMATION_MARKERS.search(claim):
        return []

    claim_facts = _extract_numeric_facts(claim)
    if not claim_facts:
        return []

    source_norm = _normalize_number(source_text)
    # Extract bare numeric tokens from the original source text
    source_numbers = set(re.findall(r"\d+(?:\.\d+)?", source_text))
    missing: list[str] = []
    for fact in claim_facts:
        # Check if the fact appears in source with a digit-boundary match —
        # "8minutes" must NOT match inside "18minutes". Using a negative
        # lookbehind for a digit character prevents subsumption by a larger
        # number.
        bare = fact.lstrip("$")
        if _fact_present_in_source(fact, source_norm) or (
            bare and _fact_present_in_source(bare, source_norm)
        ):
            continue
        # Extract the numeric portion of this fact; if the exact number is
        # present in the source (possibly with different unit), consider it
        # supported. This handles "850 participants" vs "850 attendees".
        num_match = re.match(r"\$?(\d+(?:\.\d+)?)", fact)
        if num_match and num_match.group(1) in source_numbers:
            continue
        # Check if the value falls within a range stated in the source
        if _value_in_source_range(fact, source_text):
            continue
        missing.append(fact)
    return missing


def _fact_present_in_source(fact: str, source_norm: str) -> bool:
    """Check fact substring in normalized source with digit-boundary safety.

    A digit-prefixed match (e.g. "8minutes" found inside "18minutes") is NOT
    considered a match — the claim fact's number would be contained inside a
    larger source number, which is not what "supported" means.
    """
    if not fact:
        return False
    idx = 0
    while True:
        pos = source_norm.find(fact, idx)
        if pos == -1:
            return False
        prev_ch = source_norm[pos - 1] if pos > 0 else ""
        if not prev_ch.isdigit() and prev_ch != ".":
            return True
        idx = pos + 1


class CitationDetector(BaseDetector):
    """Detects fabricated citations where claims are attributed to sources
    that don't contain the claimed information.

    This detector:
    - Extracts citation patterns from agent output text
    - Collects available source content from RETRIEVAL spans and tool outputs
    - Checks if cited claims appear in any available source content
    - Flags when overlap between claim and source is below threshold
    """

    name = "citation"
    description = "Detects fabricated citations and source misattribution"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (35, 85)
    realtime_capable = False

    # Minimum overlap for a citation to be considered supported.
    min_support_overlap = 0.25

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect fabricated citations in a trace."""
        # Collect all available source content
        sources = _extract_source_content(trace)
        if not sources:
            # Without source content, we can't verify citations
            return DetectionResult.no_issue(self.name)

        # Extract citations from agent output spans
        output_spans = self._get_output_spans(trace)
        if not output_spans:
            return DetectionResult.no_issue(self.name)

        " ".join(sources.values())
        fabricated: list[dict[str, Any]] = []
        total_citations = 0

        for span in output_spans:
            output_text = self._get_output_text(span)
            citations = _extract_citations(output_text)

            for citation in citations:
                total_citations += 1
                claim = citation["claim"]
                cited_source = citation["source"]

                # First check: does the cited source name match any known source?
                best_overlap = 0.0
                best_source = ""

                for source_id, source_text in sources.items():
                    # Check if the citation references this specific source
                    source_name_match = (
                        source_id.lower() in cited_source.lower()
                        or cited_source.lower() in source_id.lower()
                    )

                    overlap = _claim_supported_by_source(claim, source_text)

                    if source_name_match:
                        # Specific source cited: use its overlap directly
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_source = source_id
                    elif overlap > best_overlap:
                        best_overlap = overlap
                        best_source = source_id

                if best_overlap < self.min_support_overlap:
                    # Range gate: if all non-year numeric facts in the claim
                    # fall within a range stated by some source, the claim is
                    # supported even if word overlap is low.
                    claim_facts = _extract_numeric_facts(claim)
                    non_year_facts = [f for f in claim_facts if not _is_year(f)]
                    if non_year_facts and all(
                        any(_value_in_source_range(f, src) for src in sources.values())
                        for f in non_year_facts
                    ):
                        continue
                    fabricated.append(
                        {
                            "span_id": span.span_id,
                            "cited_source": cited_source,
                            "claim": claim[:200],
                            "best_matching_source": best_source,
                            "best_overlap": round(best_overlap, 3),
                        }
                    )
                    continue

                # Numeric-fact fabrication check: even when word overlap is high,
                # specific numbers (percentages, currency, counts with units) in
                # the claim that don't appear in ANY source indicate fabrication.
                # Only fires for mid-to-high overlap (partial-fabrication zone)
                # to avoid double-counting with the overlap threshold above.
                if best_overlap >= self.min_support_overlap:
                    unsupported_facts: list[str] = []
                    for source_text in sources.values():
                        facts_missing = _numeric_facts_unsupported(claim, source_text)
                        if not facts_missing:
                            # Some source supports the facts — claim is ok.
                            unsupported_facts = []
                            break
                        if not unsupported_facts or len(facts_missing) < len(unsupported_facts):
                            unsupported_facts = facts_missing

                    if unsupported_facts:
                        fabricated.append(
                            {
                                "span_id": span.span_id,
                                "cited_source": cited_source,
                                "claim": claim[:200],
                                "best_matching_source": best_source,
                                "best_overlap": round(best_overlap, 3),
                                "unsupported_numeric_facts": unsupported_facts[:5],
                            }
                        )

        if not fabricated:
            return DetectionResult.no_issue(self.name)

        # Score severity based on count and overlap
        severity = self._score_fabrications(fabricated, total_citations)
        severity = max(self.severity_range[0], min(self.severity_range[1], severity))

        worst = min(fabricated, key=lambda f: f["best_overlap"])

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=(
                f"Fabricated citation: claim attributed to '{worst['cited_source']}' "
                f"not supported by source content (overlap {worst['best_overlap']:.0%})"
            ),
            fix_type=FixType.ROLLBACK,
            fix_instruction=(
                "Citation appears fabricated. The claimed information was not found "
                "in the cited source. Remove the unsupported citation or verify the "
                "claim against the actual source content."
            ),
        )

        result.add_evidence(
            description=f"{len(fabricated)} of {total_citations} citation(s) unsupported by sources",
            span_ids=[f["span_id"] for f in fabricated[:10]],
            data={
                "fabricated_citations": fabricated[:10],
                "total_citations": total_citations,
                "available_sources": list(sources.keys()),
            },
        )

        return result

    def _get_output_spans(self, trace: Trace) -> list[Span]:
        """Get spans that produce user-facing output (agent turns, agents, tasks)."""
        output_kinds = {
            SpanKind.AGENT,
            SpanKind.AGENT_TURN,
            SpanKind.TASK,
            SpanKind.USER_OUTPUT,
        }
        spans = [s for s in trace.spans if s.kind in output_kinds]
        return sorted(spans, key=lambda s: s.start_time)

    @staticmethod
    def _get_output_text(span: Span) -> str:
        """Extract output text from a span."""
        parts: list[str] = []
        if span.output_data:
            for key in ("output", "result", "response", "text", "content", "answer", "message"):
                val = span.output_data.get(key)
                if isinstance(val, str):
                    parts.append(val)
            if not parts:
                parts.append(str(span.output_data))
        return " ".join(parts)

    @staticmethod
    def _score_fabrications(fabricated: list[dict[str, Any]], total_citations: int) -> int:
        """Score severity based on fabrication count and quality."""
        if total_citations == 0:
            return 0

        fabrication_ratio = len(fabricated) / total_citations
        # Base severity from ratio
        base = 35 + int(fabrication_ratio * 30)

        # Bonus for very low overlap (blatant fabrication)
        min_overlap = min(f["best_overlap"] for f in fabricated)
        if min_overlap < 0.10:
            base += 15
        elif min_overlap < 0.20:
            base += 8

        return min(base, 85)
