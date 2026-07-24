"""Critic quality detector for identifying rubber-stamping critics in reflection loops."""

import re
from typing import Any

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Span, Trace

# Keywords that indicate a critic/evaluator role
_CRITIC_KEYWORDS = frozenset(
    {
        "review",
        "evaluate",
        "critic",
        "feedback",
        "judge",
        "assess",
        "validate",
        "check",
        "verify",
        "approve",
        "score",
        "rate",
        "grade",
        "qa",
        "quality",
        "audit",
    }
)

# Keywords that indicate a producer role
_PRODUCER_KEYWORDS = frozenset(
    {
        "write",
        "generate",
        "create",
        "draft",
        "compose",
        "produce",
        "build",
        "implement",
        "code",
        "design",
        "author",
        "synthesize",
    }
)

# Approval indicators in critic output
_APPROVAL_PATTERNS = [
    re.compile(r"\bapprov(?:ed|es|al)\b", re.IGNORECASE),
    re.compile(
        r"\blooks?\s+(?:good|solid|correct|valid|sound|clean|clear|great|fine)\b", re.IGNORECASE
    ),
    re.compile(r"\bno\s+(?:issues?|problems?|concerns?)\b", re.IGNORECASE),
    re.compile(r"\blgtm\b", re.IGNORECASE),
    re.compile(r"\bwell[-\s](?:done|structured|organized|implemented|written)\b", re.IGNORECASE),
    re.compile(r"\baccept(?:ed|able)?\b", re.IGNORECASE),
    re.compile(r"\bpass(?:es|ed)?\b", re.IGNORECASE),
    re.compile(r"\bsatisf(?:ied|actory|ies)\b", re.IGNORECASE),
    re.compile(r"\bready\s+(?:for|to)\b", re.IGNORECASE),
    # Shallow-praise adjectives (target: soft rubber-stamp approvals)
    re.compile(
        r"\b(?:great|excellent|perfect|nice)\s+(?:job|work|use|implementation|choice|algorithm|async|method|structure|markup|approach)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:great|excellent|perfect|nice)[!.]", re.IGNORECASE),
    re.compile(
        r"\bgood\s+(?:job|work|use|algorithm|choice|boolean|implementation|use\s+of)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:is|are|seems?|appears?|looks?)\s+(?:solid|clean|clear|concise|straightforward|descriptive|efficient|sound|reliable|appropriate)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:mathematically|logically|algorithmically)\s+sound\b", re.IGNORECASE),
    re.compile(r"\bsimple\s+and\s+(?:effective|efficient|clean|concise)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:clean|solid|concise|straightforward|descriptive|efficient|professional)\s+(?:implementation|code|markup|structure|logic|approach|validation|naming|pattern)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bI\s+love\s+the\b", re.IGNORECASE),
    re.compile(r"\b(?:will\s+work|works?)\s+(?:perfectly|great|fine|well)\b", re.IGNORECASE),
    # Group A: additional shallow-praise patterns (recall-focused, 2026-04-17)
    # "clean and simple/readable/concise", "simple and clean"
    re.compile(
        r"\b(?:clean|simple|concise|straightforward)\s+and\s+(?:simple|clean|readable|concise|straightforward|maintainable)\b",
        re.IGNORECASE,
    ),
    # "nice clean implementation", "nice concise async" — "nice" + adjective + noun
    re.compile(
        r"\b(?:nice|great|excellent|solid)\s+(?:clean|concise|simple|straightforward|readable|elegant)\s+(?:implementation|code|approach|async|method|structure|logic)\b",
        re.IGNORECASE,
    ),
    # "looks well organized", "looks well-structured"
    re.compile(
        r"\blooks?\s+well[-\s](?:organized|structured|implemented|designed|written)\b",
        re.IGNORECASE,
    ),
    # "is easy to (follow|read|understand)"
    re.compile(
        r"\b(?:is|are|looks?|seems?)\s+easy\s+to\s+(?:follow|read|understand|use|maintain)\b",
        re.IGNORECASE,
    ),
    # "is valid" (HTML/JSON/etc) — narrow to avoid TN validation-keyword mentions
    re.compile(
        r"\b(?:structure|code|syntax|markup|document|html|json|xml|yaml)\s+(?:is|looks?|seems?|appears?)\s+valid\b",
        re.IGNORECASE,
    ),
    # "follows X conventions/style/best practices"
    re.compile(
        r"\bfollows?\s+(?:\w+\s+)?(?:conventions?|style|standards?|best\s+practices|idioms?|patterns?)\b",
        re.IGNORECASE,
    ),
    # "for the basic case" — implies ignoring edge cases
    re.compile(r"\b(?:correct|works?|fine|ok)\s+for\s+the\s+basic\s+case\b", re.IGNORECASE),
    # "well-implemented", "well-organized" as hyphenated adjective form
    re.compile(
        r"\bwell-(?:implemented|organized|structured|designed|written|done)\b", re.IGNORECASE
    ),
    # "X is reliable/professional/appropriate" applied to nouns — extend existing
    re.compile(
        r"\b(?:column\s+types?|naming\s+convention|table\s+structure|class\s+structure|method\s+name|variable\s+naming|error\s+handling)\s+(?:is|looks?|are|seems?|appears?)\s+(?:clear|clean|professional|descriptive|appropriate|correct|good|fine|solid)\b",
        re.IGNORECASE,
    ),
    # "timestamp format is clear", "return values are appropriate"
    re.compile(
        r"\b(?:format|values?|logic|flow|output|structure|signature)\s+(?:is|are|looks?|seems?)\s+(?:clear|appropriate|clean|correct|good|readable|professional)\b",
        re.IGNORECASE,
    ),
    # "correctly implemented", "properly implemented"
    re.compile(
        r"\b(?:correctly|properly|cleanly|elegantly)\s+(?:implemented|structured|organized|designed|written|handled)\b",
        re.IGNORECASE,
    ),
]

# Group C: cosmetic-only critic suggestions. Not approvals on their own — only
# count as "approval" when combined with producer red flags (the critic made a
# trivial suggestion while missing a real bug). Handled in _check_shallow_approval.
_COSMETIC_ONLY_PATTERNS = [
    re.compile(r"\bconsider\s+renaming\b", re.IGNORECASE),
    re.compile(
        r"\bconsider\s+adding\s+(?:a\s+)?(?:proper\s+)?(?:title|docstring|comment|javadoc|header|tag|meta)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\brename\s+(?:the\s+|this\s+)?(?:variable|parameter|argument|field|property)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:for|to\s+improve)\s+(?:clarity|readability|seo|documentation|style)\b", re.IGNORECASE
    ),
    re.compile(r"\bconsider\s+(?:a\s+)?(?:more\s+)?descriptive\s+(?:name|naming)\b", re.IGNORECASE),
]

# Producer red flags: clear signals that critic-approved code has obvious issues.
# These are combined with approval detection in _check_shallow_approval to catch
# rubber-stamps on buggy/insecure/incomplete code without needing multi-iteration.
_PRODUCER_RED_FLAGS = [
    # Stub / empty / trivial bodies
    (
        re.compile(
            r"(?:def|function|class)\s+\w+[^:{]*[:{]\s*(?:pass\b|return\s*(?:None|null|undefined|;)?\s*[}\n]|\.{3,}|\}\s*$)",
            re.IGNORECASE | re.MULTILINE,
        ),
        "stub/empty body",
    ),
    # Security: shell injection
    (re.compile(r"rm\s+-rf\s+\$", re.IGNORECASE), "shell injection (rm -rf with unescaped var)"),
    # Security: XSS via document.write
    (
        re.compile(r"document\.write\s*\([^)]*(?:user|input|param)", re.IGNORECASE),
        "XSS via document.write of user input",
    ),
    # Security: eval/exec of dynamic input
    (
        re.compile(r"\b(?:eval|exec)\s*\(\s*(?:user|input|request|param)", re.IGNORECASE),
        "eval/exec of dynamic input",
    ),
    # Security: insecure random for tokens/secrets
    (re.compile(r"new\s+Random\s*\(\s*\)", re.IGNORECASE), "insecure Random (non-cryptographic)"),
    (
        re.compile(r"Math\.random\s*\([^)]*\).*(?:token|secret|password|key)", re.IGNORECASE),
        "insecure Math.random for secrets",
    ),
    # Security: hardcoded admin credentials
    (
        re.compile(r"(?:username|user)\s*==?\s*['\"]admin['\"]", re.IGNORECASE),
        "hardcoded admin check",
    ),
    # Security: hardcoded API keys / secrets
    (re.compile(r"\bsk-[A-Za-z0-9]{10,}", re.IGNORECASE), "hardcoded API key literal"),
    (
        re.compile(
            r"(?:api[_-]?key|apikey|password|secret)\s*[:=]\s*['\"][^'\"]{4,}['\"]", re.IGNORECASE
        ),
        "hardcoded credential",
    ),
    # Security: SQL string concatenation
    (
        re.compile(r"(?:SELECT|INSERT|UPDATE|DELETE)[^;'\"]*['\"]?\s*\+\s*\w+", re.IGNORECASE),
        "SQL string concatenation (injection)",
    ),
    (
        re.compile(r"[fF]['\"](?:SELECT|INSERT|UPDATE|DELETE)[^'\"]*\{", re.IGNORECASE),
        "SQL f-string with interpolation",
    ),
    # Trivial / placeholder email validation
    (re.compile(r"return\s+['\"]?@['\"]?\s+in\s+email", re.IGNORECASE), "trivial email validation"),
    (
        re.compile(r"return\s+email\.(?:includes|indexOf)\s*\(\s*['\"]@['\"]\s*\)", re.IGNORECASE),
        "trivial email validation",
    ),
    # Print-as-send (no real implementation)
    (re.compile(r"print\s*\(\s*f?['\"]Sending\b", re.IGNORECASE), "print-as-send stub"),
    # No-op extension-only file upload validation
    (
        re.compile(r"allowedTypes\s*=\s*\[\s*['\"](?:jpg|png|gif|jpeg)", re.IGNORECASE),
        "extension-only file-type check",
    ),
    # Group B: additional producer red flags (recall-focused, 2026-04-17)
    # Stub body where a comment separates `:` from `pass`
    (
        re.compile(
            r"(?:def|function|class)\s+\w+[^:{]*[:{]\s*(?:#|//)[^\n]*\n\s*(?:pass\b|return\s*(?:None|null|undefined|;)?\s*[}\n;]|\.{3,})",
            re.IGNORECASE | re.MULTILINE,
        ),
        "stub body with placeholder comment",
    ),
    # Inefficient / known-bad algorithm choice
    (re.compile(r"\bbubble\s*sort\b", re.IGNORECASE), "bubble sort (O(n^2) choice)"),
    # Division by length/count without guard (will ZeroDivisionError on empty input)
    (re.compile(r"/\s*len\s*\(\s*\w+\s*\)", re.IGNORECASE), "divide by len() without empty-guard"),
    # (Unchecked-division red flag removed — too many FPs when producer has
    # a zero-guard elsewhere in the snippet. We still catch divide-by-len()
    # patterns and handle the `def divide(a, b): return a / b` shape via
    # _has_unguarded_division below.)
    # Balance mutation without overdraft guard (banking withdraw/deduct)
    (
        re.compile(r"\.balance\s*[-+]?=\s*\w+\s*\n(?!\s*(?:if|assert|raise))", re.IGNORECASE),
        "balance mutation without guard",
    ),
    # fetch(...).json() or await fetch(...).json() with NO error handling present
    # in the same snippet — implemented as red flag via helper, see _has_producer_red_flags.
    # Scanner / file resource without try-with-resources / close()
    (
        re.compile(r"new\s+Scanner\s*\(", re.IGNORECASE),
        "Scanner created without try-with-resources",
    ),
    # `file.delete()` as the method body — destructive op in a "process" method
    (
        re.compile(r"\bfile\.delete\s*\(\s*\)\s*;?\s*\}", re.IGNORECASE),
        "file.delete() as side effect",
    ),
    # `System.out.println` inside a method named *log* — println as logger
    (
        re.compile(
            r"(?:void|public|private|static)\s+\w*[Ll]og\w*\s*\([^)]*\)\s*\{[^}]*System\.out\.println",
            re.MULTILINE,
        ),
        "println used as logger",
    ),
    # Plain password comparison (no hashing)
    (re.compile(r"\.password\s*==\s*password\b", re.IGNORECASE), "plaintext password comparison"),
    # Missing-else comment — producer admits they skipped a branch
    (re.compile(r"(?://|#)\s*Missing\s+else\b", re.IGNORECASE), "admitted missing else branch"),
    # JSON dict access without .get() or KeyError handling (narrow pattern)
    (
        re.compile(
            r"(?:=|return)\s+\w+\[\s*['\"]\w+['\"]\s*\]\s*,\s*\w+\[\s*['\"]\w+['\"]\s*\]",
            re.IGNORECASE,
        ),
        "unguarded multi-dict-key access",
    ),
]

# Incomplete/placeholder markers in producer output
_INCOMPLETE_MARKERS = [
    re.compile(r"\bTODO\b"),
    re.compile(r"\bFIXME\b"),
    re.compile(r"\bHACK\b"),
    re.compile(r"\bXXX\b"),
    re.compile(r"\bplaceholder\b", re.IGNORECASE),
    re.compile(r"\bTBD\b"),
    re.compile(r"\blorem\s+ipsum\b", re.IGNORECASE),
    re.compile(r"\b\[.*?insert.*?\]", re.IGNORECASE),
    re.compile(r"\b\[.*?fill.*?\]", re.IGNORECASE),
    # Ellipsis as placeholder — exclude JS/Python spread/rest (...args, ...rest)
    # and method chains (e.g., `"foo"...`). Require surrounding non-identifier context.
    re.compile(r"(?<![\w)])\.{3,}(?![\w(])"),
]


def _is_critic_span(span: Span) -> bool:
    """Check whether a span represents a critic/evaluator role."""
    name_lower = span.name.lower()
    return any(kw in name_lower for kw in _CRITIC_KEYWORDS)


def _is_producer_span(span: Span) -> bool:
    """Check whether a span represents a producer role."""
    name_lower = span.name.lower()
    if any(kw in name_lower for kw in _PRODUCER_KEYWORDS):
        return True
    # If not explicitly a critic, treat agent spans as potential producers
    return not _is_critic_span(span)


def _get_output_text(span: Span) -> str:
    """Extract output text from a span."""
    parts: list[str] = []
    if span.output_data:
        for key in (
            "output",
            "result",
            "response",
            "text",
            "content",
            "feedback",
            "review",
            "evaluation",
        ):
            val = span.output_data.get(key)
            if isinstance(val, str):
                parts.append(val)
        if not parts:
            parts.append(str(span.output_data))
    return " ".join(parts)


def _has_approval(text: str) -> bool:
    """Check if text contains approval language."""
    return any(pat.search(text) for pat in _APPROVAL_PATTERNS)


def _has_incomplete_markers(text: str) -> list[str]:
    """Find incomplete/placeholder markers in text."""
    found: list[str] = []
    for pat in _INCOMPLETE_MARKERS:
        match = pat.search(text)
        if match:
            found.append(match.group())
    return found


def _has_producer_red_flags(text: str) -> list[str]:
    """Find red-flag patterns in producer output (security, stubs, trivial impls)."""
    found: list[str] = []
    for pat, label in _PRODUCER_RED_FLAGS:
        if pat.search(text):
            found.append(label)
    # Composite: CREATE TABLE statement with no NOT NULL / UNIQUE / CHECK / FOREIGN KEY
    # constraints beyond a single PRIMARY KEY — trivial schema lacking integrity.
    # Use DOTALL-friendly split on the keyword rather than a regex grouping (nested
    # parens like VARCHAR(50) break a naive `\(([^)]+)\)` capture).
    ct_match = re.search(r"CREATE\s+TABLE\s+\w+\s*\(", text, re.IGNORECASE)
    if ct_match:
        body = text[ct_match.end() :]
        # Scan up to the matching closing paren by depth-counting
        depth = 1
        end = 0
        for i, ch in enumerate(body):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        table_body = body[:end] if end else body
        has_constraint = re.search(
            r"\b(?:NOT\s+NULL|UNIQUE|CHECK\s*\(|FOREIGN\s+KEY|REFERENCES|DEFAULT\s+\w+)\b",
            table_body,
            re.IGNORECASE,
        )
        if not has_constraint:
            found.append("CREATE TABLE without NOT NULL/UNIQUE constraints")
    # Composite: fetch().json()/then() chain with no visible error handling.
    # Only fires when the fetch result is consumed inline (.json() or .then())
    # — a bare `return fetch(x)` forwards the promise and is caller's concern.
    if re.search(r"\bfetch\s*\([^)]+\)\s*\.(?:json|then)\b", text, re.IGNORECASE) or re.search(
        r"=\s*await\s+fetch\s*\([^)]+\)[\s\S]*\.json\s*\(", text, re.IGNORECASE
    ):
        if not re.search(
            r"\b(?:try|catch|\.catch|\.ok|response\.ok|status\s*[!=]==?\s*200)\b",
            text,
            re.IGNORECASE,
        ):
            found.append("fetch+consume without error handling")
    # Composite: unguarded `def/function X(...): return ... / param` — no zero-check.
    # Requires (a) a function definition, (b) a `/ paramname` in its return
    # expression, (c) no `if` guard in the snippet. Catches divide(a,b): return a/b.
    div_match = re.search(
        r"(?:def|function)\s+\w+\s*\(([^)]*)\)[\s:{]*(?:return|\{?\s*return)\s+[^;\n}]*/\s*(\w+)",
        text,
    )
    if div_match:
        params = div_match.group(1)
        divisor = div_match.group(2)
        # Only flag when the divisor is one of the function's parameters
        param_names = {
            p.strip().split(":")[0].split("=")[0].strip() for p in params.split(",") if p.strip()
        }
        if divisor in param_names:
            has_guard = re.search(
                r"if\s*\(?\s*(?:not\s+)?"
                + re.escape(divisor)
                + r"\s*(?:[!=]==?|is)\s*(?:0|None|null)",
                text,
                re.IGNORECASE,
            )
            if not has_guard:
                found.append("unguarded divide-by-parameter")
    return found


def _has_cosmetic_only_feedback(text: str) -> bool:
    """Check whether critic text is purely cosmetic (naming/SEO/docs).

    Only meaningful when combined with producer red flags: the critic offered
    trivial style suggestions while missing real bugs.
    """
    if not text:
        return False
    return any(pat.search(text) for pat in _COSMETIC_ONLY_PATTERNS)


# Words that indicate substantive critique. If absent AND the text is short,
# the feedback is likely shallow/rubber-stamp.
_SUBSTANTIVE_KEYWORDS = frozenset(
    {
        "issue",
        "issues",
        "bug",
        "bugs",
        "missing",
        "add",
        "error",
        "errors",
        "handle",
        "handling",
        "concern",
        "concerns",
        "fix",
        "fixes",
        "vulnerab",
        "risk",
        "problem",
        "problems",
        "wrong",
        "incorrect",
        "broken",
        "fail",
        "fails",
        "failure",
        "leak",
        "leaks",
        "unsafe",
        "insecure",
        "injection",
        "overflow",
        "null",
        "none",
        "undefined",
        "empty",
        "edge",
        "exception",
        "throws",
        "race",
        "concurrent",
        "deadlock",
        "timeout",
        "validate",
        "validation",
        "sanitize",
        "escape",
        "required",
        "must",
        "should",
        "need",
        "needs",
        "needed",
        "crash",
        "crashes",
        "corrupt",
        "invalid",
        "memory",
        "hash",
        "salt",
        "bcrypt",
        "pbkdf2",
        "md5",
        "sha",
        "csrf",
        "xss",
        "sql",
        "complexity",
        "performance",
        "memoiz",
        "iterative",
        "refactor",
        "deprecate",
        "bounds",
        "divide",
        "division",
        "negative",
        "overdraft",
        "balance",
        "token",
        "secret",
        "password",
        "auth",
    }
)


def _is_substanceless(text: str) -> bool:
    """Check whether critic text lacks any substantive critique signal.

    Returns True if text is short-ish and contains no actionable critique
    keywords. Used only when combined with producer red flags to avoid FPs.
    """
    if not text:
        return False
    lower = text.lower()
    # Short praise-only feedback rarely exceeds ~200 chars
    if len(text) > 400:
        return False
    # Any substantive keyword present? Not substanceless.
    for kw in _SUBSTANTIVE_KEYWORDS:
        if kw in lower:
            return False
    return True


_QUESTION_PATTERN = re.compile(r"\?")


def _is_question_only(text: str) -> bool:
    """Critic text consists solely of questions (no declarative critique)."""
    if not text:
        return False
    # Split on sentence-ending punctuation
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return False
    # Must have at least one "?" and every non-empty sentence-candidate
    # must look interrogative (starts with question word or ends before ?).
    if "?" not in text:
        return False
    question_starters = (
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "is",
        "are",
        "can",
        "could",
        "should",
        "would",
        "will",
        "does",
        "do",
        "did",
        "have",
        "has",
        "shouldn",
        "wouldn",
        "couldn",
    )
    non_q = 0
    for s in sentences:
        first = s.split()[0].lower() if s.split() else ""
        # Strip non-alpha for starter test
        first_alpha = re.sub(r"[^a-z]", "", first)
        if first_alpha not in question_starters:
            non_q += 1
    # Allow at most one non-question sentence (e.g., a short intro)
    return non_q <= 1


def _word_level_diff_ratio(text_a: str, text_b: str) -> float:
    """Compute word-level difference ratio between two texts.

    Returns 0.0 for identical texts, 1.0 for completely different texts.
    """
    if not text_a and not text_b:
        return 0.0
    if not text_a or not text_b:
        return 1.0

    words_a = text_a.lower().split()
    words_b = text_b.lower().split()

    if not words_a and not words_b:
        return 0.0

    # Use set-based Jaccard distance for efficiency
    set_a = set(words_a)
    set_b = set(words_b)
    union = set_a | set_b
    if not union:
        return 0.0

    intersection = set_a & set_b
    return 1.0 - (len(intersection) / len(union))


class CriticQualityDetector(BaseDetector):
    """Detects rubber-stamping critics in reflection loops.

    This detector identifies:
    - Critics that approve without requesting meaningful changes
    - Critics that approve when producer output barely changed between iterations
    - Critics that approve output containing TODO/FIXME/placeholder markers
    """

    name = "critic_quality"
    description = "Detects rubber-stamping critics in reflection loops"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (20, 60)
    realtime_capable = False

    # A diff ratio below this means the producer barely changed their output
    min_meaningful_change = 0.10
    # Minimum number of reflection iterations to analyze
    min_iterations = 2

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect rubber-stamping critics in reflection loops."""
        agent_spans = trace.get_spans_by_kind(SpanKind.AGENT)
        turn_spans = trace.get_spans_by_kind(SpanKind.AGENT_TURN)
        all_candidates = agent_spans + turn_spans

        # Require at least one producer + one critic span (2+ spans), down from 3.
        # Single-iteration rubber-stamp detection depends on this gate.
        if len(all_candidates) < 2:
            return DetectionResult.no_issue(self.name)

        sorted_spans = sorted(all_candidates, key=lambda s: s.start_time)

        issues: list[str] = []
        severity = 0
        evidence_data: dict[str, Any] = {}

        # --- Identify critic-producer pairs ---
        reflection_pairs = self._find_reflection_pairs(sorted_spans)
        if not reflection_pairs:
            return DetectionResult.no_issue(self.name)

        # --- Check 1: Rubber-stamping (approve with minimal change) ---
        rubber_stamp = self._check_rubber_stamping(reflection_pairs)
        if rubber_stamp:
            severity += rubber_stamp["severity"]
            issues.append(rubber_stamp["summary"])
            evidence_data["rubber_stamping"] = rubber_stamp["details"]

        # --- Check 2: Weak critic (approve despite incomplete markers) ---
        weak_critic = self._check_weak_critic(reflection_pairs)
        if weak_critic:
            severity += weak_critic["severity"]
            issues.append(weak_critic["summary"])
            evidence_data["weak_critic"] = weak_critic["details"]

        # --- Check 3: Shallow approval on red-flag producer output ---
        shallow = self._check_shallow_approval(reflection_pairs)
        if shallow:
            severity += shallow["severity"]
            issues.append(shallow["summary"])
            evidence_data["shallow_approval"] = shallow["details"]

        if not issues:
            return DetectionResult.no_issue(self.name)

        severity = max(self.severity_range[0], min(self.severity_range[1], severity))

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=issues[0],
            fix_type=FixType.ESCALATE,
            fix_instruction=(
                "The critic/evaluator appears to be rubber-stamping. "
                "Review the evaluation criteria and ensure the critic provides "
                "substantive feedback before approving."
            ),
        )

        span_ids = []
        for pair in reflection_pairs[:5]:
            span_ids.append(pair["critic"].span_id)
            if pair.get("producer"):
                span_ids.append(pair["producer"].span_id)

        for issue in issues:
            result.add_evidence(
                description=issue,
                span_ids=span_ids,
                data=evidence_data,
            )

        return result

    def _find_reflection_pairs(self, spans: list[Span]) -> list[dict[str, Any]]:
        """Find producer-critic pairs in the span sequence.

        A reflection pair is a critic span that follows a producer span
        (possibly with some spans in between).
        """
        pairs: list[dict[str, Any]] = []
        last_producer: Span | None = None
        last_producer_output: str = ""

        for span in spans:
            if _is_critic_span(span):
                critic_output = _get_output_text(span)
                pair: dict[str, Any] = {
                    "critic": span,
                    "critic_output": critic_output,
                    "producer": last_producer,
                    "producer_output": last_producer_output,
                }
                pairs.append(pair)
            elif _is_producer_span(span):
                output = _get_output_text(span)
                if output:
                    last_producer = span
                    last_producer_output = output

        return pairs

    def _check_rubber_stamping(self, pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Check if critic approves when producer output barely changed."""
        if len(pairs) < self.min_iterations:
            return None

        rubber_stamp_count = 0
        details: list[dict[str, Any]] = []

        # Compare consecutive producer outputs across reflection iterations
        previous_producer_output = ""
        for pair in pairs:
            critic_output = pair["critic_output"]
            producer_output = pair["producer_output"]

            if not critic_output:
                continue

            approved = _has_approval(critic_output)
            if not approved:
                previous_producer_output = producer_output
                continue

            # If we have a previous producer output, check if it changed
            if previous_producer_output and producer_output:
                diff = _word_level_diff_ratio(previous_producer_output, producer_output)
                if diff < self.min_meaningful_change:
                    rubber_stamp_count += 1
                    details.append(
                        {
                            "critic_span_id": pair["critic"].span_id,
                            "diff_ratio": round(diff, 3),
                            "approved": True,
                        }
                    )

            previous_producer_output = producer_output

        if rubber_stamp_count == 0:
            return None

        sev = 20 + rubber_stamp_count * 15

        return {
            "severity": min(sev, 45),
            "summary": (
                f"Critic rubber-stamped {rubber_stamp_count} time(s) "
                f"(approved with <{self.min_meaningful_change:.0%} producer change)"
            ),
            "details": details,
        }

    def _check_weak_critic(self, pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Check if critic approves output with incomplete markers."""
        weak_approvals: list[dict[str, Any]] = []

        for pair in pairs:
            critic_output = pair["critic_output"]
            producer_output = pair["producer_output"]

            if not critic_output or not producer_output:
                continue

            approved = _has_approval(critic_output)
            if not approved:
                continue

            markers = _has_incomplete_markers(producer_output)
            if markers:
                weak_approvals.append(
                    {
                        "critic_span_id": pair["critic"].span_id,
                        "markers_found": markers,
                    }
                )

        if not weak_approvals:
            return None

        total_markers = sum(len(w["markers_found"]) for w in weak_approvals)
        sev = 20 + min(total_markers * 5, 30)

        return {
            "severity": min(sev, 40),
            "summary": (
                f"Critic approved output with {total_markers} incomplete marker(s) "
                f"({', '.join(weak_approvals[0]['markers_found'][:3])})"
            ),
            "details": weak_approvals,
        }

    def _check_shallow_approval(self, pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Check if critic approves producer output that has red flags
        (security issues, stubs, trivial impls).

        This catches single-iteration rubber-stamps where the code has
        obvious problems the critic should have caught. A critic counts as
        "approving" if any of the following apply:
        - Contains an explicit approval marker (_has_approval)
        - Offers only cosmetic suggestions (_has_cosmetic_only_feedback)
        - Is short praise with no substantive critique (_is_substanceless)
        - Asks questions without declaring any concrete issue (_is_question_only)
        All four forms require producer_output to contain a red flag before
        firing, which preserves precision.
        """
        flagged: list[dict[str, Any]] = []

        for pair in pairs:
            critic_output = pair["critic_output"]
            producer_output = pair["producer_output"]

            if not critic_output or not producer_output:
                continue

            approved = _has_approval(critic_output)
            cosmetic = _has_cosmetic_only_feedback(critic_output)
            substanceless = _is_substanceless(critic_output)
            question_only = _is_question_only(critic_output)

            if not (approved or cosmetic or substanceless or question_only):
                continue

            red_flags = _has_producer_red_flags(producer_output)
            if not red_flags:
                continue

            flagged.append(
                {
                    "critic_span_id": pair["critic"].span_id,
                    "red_flags": red_flags,
                    "approval_type": (
                        "explicit"
                        if approved
                        else "cosmetic_only"
                        if cosmetic
                        else "substanceless"
                        if substanceless
                        else "question_only"
                    ),
                }
            )

        if not flagged:
            return None

        total_flags = sum(len(f["red_flags"]) for f in flagged)
        sev = 25 + min(total_flags * 5, 25)

        return {
            "severity": min(sev, 45),
            "summary": (
                f"Critic approved {len(flagged)} producer output(s) with "
                f"{total_flags} red flag(s): {', '.join(flagged[0]['red_flags'][:3])}"
            ),
            "details": flagged,
        }
