"""Regression tests for CriticQualityDetector recall improvements.

These cover the shallow-approval + red-flag and expanded approval-pattern
additions made on 2026-04-17 to lift recall from 0.326 to 0.522.
"""

import pytest

from pisama_core.detection.detectors.critic import (
    CriticQualityDetector,
    _has_approval,
    _has_cosmetic_only_feedback,
    _has_producer_red_flags,
    _is_question_only,
    _is_substanceless,
)
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Trace


def _build_trace(iterations: list[tuple[str, str]]) -> Trace:
    """Build a trace from (role, output) pairs."""
    trace = Trace()
    for i, (role, output) in enumerate(iterations):
        name = f"{role}_{i}" if role != "critic" else f"review_{i}"
        trace.create_span(
            name=name,
            kind=SpanKind.AGENT,
            output_data={"output": output},
        )
    return trace


class TestApprovalPatterns:
    """Expanded approval patterns — shallow praise variants."""

    @pytest.mark.parametrize(
        "text",
        [
            "Looks good.",
            "LGTM!",
            "Great job! This function is clean.",
            "Nice clean implementation!",
            "Excellent work on the structure.",
            "Perfect!",
            "Good algorithm choice! Bubble sort is reliable.",
            "The logic is sound and clean.",
            "Algorithm implementation looks solid.",
            "Simple and efficient validation.",
            "Well-implemented pattern.",
            "I love the concise implementation.",
            "This will work perfectly in production.",
            "Mathematically sound.",
            "The implementation is descriptive and clean.",
        ],
    )
    def test_shallow_praise_is_approval(self, text):
        assert _has_approval(text)

    @pytest.mark.parametrize(
        "text",
        [
            "What about error handling?",
            "Consider renaming this variable.",
            "Is this handling edge cases?",
            "What happens if the input is null?",
            "You should add validation here.",
        ],
    )
    def test_question_feedback_is_not_approval(self, text):
        # These are substantive critic questions, not approvals.
        assert not _has_approval(text)


class TestProducerRedFlags:
    """Red-flag detection on producer output."""

    def test_detects_shell_injection(self):
        flags = _has_producer_red_flags("#!/bin/bash\nrm -rf $1")
        assert any("shell" in f for f in flags)

    def test_detects_xss_document_write(self):
        flags = _has_producer_red_flags("<script>document.write(userInput);</script>")
        assert any("XSS" in f for f in flags)

    def test_detects_hardcoded_admin(self):
        flags = _has_producer_red_flags(
            "def authenticate(username, password):\n    if username == 'admin' and password == 'pass': return True"
        )
        assert any("admin" in f for f in flags)

    def test_detects_hardcoded_api_key(self):
        flags = _has_producer_red_flags("const apiKey = 'sk-1234567890abcdef';")
        assert flags  # matches either sk- or apiKey-hardcoded pattern

    def test_detects_insecure_random(self):
        flags = _has_producer_red_flags(
            "public String generateToken() { Random rand = new Random(); return String.valueOf(rand.nextInt()); }"
        )
        assert any("Random" in f for f in flags)

    def test_detects_sql_concat_injection(self):
        flags = _has_producer_red_flags("query = 'SELECT * FROM users WHERE name = ' + userName")
        assert any("SQL" in f for f in flags)

    def test_detects_trivial_email_validation(self):
        flags = _has_producer_red_flags("def validate_email(email):\n    return '@' in email")
        assert any("email" in f for f in flags)

    def test_detects_print_as_send_stub(self):
        flags = _has_producer_red_flags(
            "def send_email(to, subject, body):\n    print(f'Sending email to {to}: {subject}')"
        )
        assert any("print-as-send" in f for f in flags)

    def test_clean_code_has_no_red_flags(self):
        flags = _has_producer_red_flags(
            "function debounce(func, wait) {\n  let timeout;\n  return function(...args) { "
            "clearTimeout(timeout); timeout = setTimeout(() => func.apply(this, args), wait); };\n}"
        )
        # JS spread operator `...args` should NOT trigger ellipsis placeholder detection
        assert flags == []


@pytest.mark.asyncio
class TestShallowApprovalCheck:
    """End-to-end: shallow critic approvals on red-flag producer output."""

    async def test_shallow_praise_on_xss_code(self):
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                (
                    "producer",
                    "<!DOCTYPE html><html><body><script>document.write(userInput);</script></body></html>",
                ),
                (
                    "critic",
                    "HTML structure looks good. Nice and clean markup with proper DOCTYPE declaration.",
                ),
            ]
        )
        res = await det.detect(trace)
        assert res.detected

    async def test_shallow_praise_on_insecure_random(self):
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                (
                    "producer",
                    "public String generateToken() { Random rand = new Random(); return String.valueOf(rand.nextInt()); }",
                ),
                ("critic", "The class structure looks solid. Clean implementation."),
            ]
        )
        res = await det.detect(trace)
        assert res.detected

    async def test_shallow_praise_on_hardcoded_api_key(self):
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                (
                    "producer",
                    "const config = { apiKey: 'sk-1234567890abcdef', endpoint: 'https://api.example.com' };",
                ),
                (
                    "critic",
                    "The configuration object is well-structured. The function looks clean.",
                ),
            ]
        )
        res = await det.detect(trace)
        assert res.detected

    async def test_shallow_praise_on_shell_injection(self):
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                ("producer", "#!/bin/bash\nrm -rf $1\necho 'Cleanup complete'"),
                (
                    "critic",
                    "Script is concise and accomplishes the cleanup task. Good use of shell scripting.",
                ),
            ]
        )
        res = await det.detect(trace)
        assert res.detected

    async def test_substantive_critique_not_flagged(self):
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                ("producer", "def divide(a, b): return a / b"),
                (
                    "critic",
                    "This function is missing division-by-zero handling. Please add: "
                    "if b == 0: raise ValueError before returning.",
                ),
            ]
        )
        res = await det.detect(trace)
        assert not res.detected

    async def test_clean_code_with_genuine_approval_not_flagged(self):
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                (
                    "producer",
                    "function debounce(func, wait) {\n  let timeout;\n  return function(...args) {\n"
                    "    clearTimeout(timeout);\n    timeout = setTimeout(() => func.apply(this, args), wait);\n"
                    "  };\n}",
                ),
                (
                    "critic",
                    "Well-implemented debounce with proper closure and argument forwarding.",
                ),
            ]
        )
        res = await det.detect(trace)
        # No red flags in the producer, so shallow-approval check should NOT fire.
        assert not res.detected


@pytest.mark.asyncio
class TestBackwardCompatibility:
    """Ensure existing TODO/FIXME and multi-iteration checks still work."""

    async def test_still_detects_todo_marker_approval(self):
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                ("producer", "def process():\n    # TODO: implement\n    pass"),
                ("critic", "Looks good, ready to merge."),
            ]
        )
        res = await det.detect(trace)
        assert res.detected

    async def test_still_detects_fixme_marker_approval(self):
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                ("producer", "def calc(x): # FIXME: off-by-one\n    return x + 1"),
                ("critic", "LGTM."),
            ]
        )
        res = await det.detect(trace)
        assert res.detected


# --- 2026-04-17 recall improvements (baseline 0.522 -> 0.935) ----------------


class TestExpandedApprovalPatterns:
    """Group A: additional shallow-praise approval patterns."""

    @pytest.mark.parametrize(
        "text",
        [
            "This function is clean and simple.",
            "Nice clean implementation!",
            "Looks well organized.",
            "The code is easy to follow.",
            "The HTML structure is valid.",
            "Follows Python conventions.",
            "Correct for the basic case.",
            "Well-implemented debounce.",
            "The column types and naming convention look professional.",
            "The timestamp format is clear and the output is readable.",
            "The binary search logic is correctly implemented.",
        ],
    )
    def test_group_a_shallow_praise_is_approval(self, text):
        assert _has_approval(text)


class TestCosmeticOnlyCritique:
    """Group C: cosmetic-only critic suggestions count as approval-equivalent."""

    @pytest.mark.parametrize(
        "text",
        [
            "Consider renaming variable 'a' to 'dividend' for clarity.",
            "Consider adding a proper title tag for better SEO.",
            "Rename the parameter for readability.",
            "Consider a more descriptive name.",
        ],
    )
    def test_cosmetic_only_recognized(self, text):
        assert _has_cosmetic_only_feedback(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Add null check before division.",
            "MD5 is cryptographically broken. Use bcrypt instead.",
            "Missing error handling for fetch failures.",
        ],
    )
    def test_substantive_critique_not_cosmetic(self, text):
        assert not _has_cosmetic_only_feedback(text)


class TestQuestionOnlyCritic:
    """Group D: purely interrogative critic feedback."""

    @pytest.mark.parametrize(
        "text",
        [
            "What about other users? How do we handle non-admin authentication?",
            "What happens if the network request fails?",
            "What's the intended use case for this token?",
        ],
    )
    def test_pure_questions_recognized(self, text):
        assert _is_question_only(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Missing null check. Add: if not data: return [].",
            "Division by zero will throw ArithmeticException. Add a guard.",
        ],
    )
    def test_declarative_critique_not_question_only(self, text):
        assert not _is_question_only(text)


class TestSubstancelessCritic:
    """Group D helper: short praise with no actionable critique keywords."""

    @pytest.mark.parametrize(
        "text",
        [
            "The method name is descriptive and the logic is straightforward. Good implementation!",
            "Simple logging implementation. The timestamp format is clear.",
            "The JSON parsing approach is straightforward and the return tuple is clean.",
        ],
    )
    def test_praise_without_critique_keywords(self, text):
        assert _is_substanceless(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Missing null check — add validation before access.",
            "Thread-safety issue: shared state causes race conditions.",
            "MD5 is broken. Use bcrypt with salt.",
        ],
    )
    def test_substantive_text_not_substanceless(self, text):
        assert not _is_substanceless(text)


class TestExpandedProducerRedFlags:
    """Group B: additional buggy-code red flags."""

    def test_bubble_sort_flagged(self):
        flags = _has_producer_red_flags(
            "int[] sortArray(int[] arr) { // Bubble sort implementation\n"
            "  for (int i = 0; i < arr.length; i++) { /*...*/ } return arr; }"
        )
        assert any("bubble" in f.lower() for f in flags)

    def test_divide_by_len_flagged(self):
        flags = _has_producer_red_flags(
            "def calculate_average(numbers):\n    return sum(numbers) / len(numbers)"
        )
        assert any("len" in f for f in flags)

    def test_unguarded_divide_by_parameter_flagged(self):
        flags = _has_producer_red_flags("def divide(a, b):\n    return a / b")
        assert any("divide-by-parameter" in f for f in flags)

    def test_guarded_divide_by_parameter_not_flagged(self):
        flags = _has_producer_red_flags(
            "def divide(a, b):\n    if b == 0: raise ValueError('zero')\n    return a / b"
        )
        assert not any("divide-by-parameter" in f for f in flags)

    def test_fetch_then_json_without_handler_flagged(self):
        flags = _has_producer_red_flags(
            "async function fetchData(url) {\n  const response = await fetch(url);\n  return response.json();\n}"
        )
        assert any("fetch" in f for f in flags)

    def test_fetch_returning_promise_not_flagged(self):
        # `return fetch(...)` without .json()/.then() forwards the promise — caller handles.
        flags = _has_producer_red_flags("function getUser(id) { return fetch(`/api/${id}`); }")
        assert not any("fetch" in f for f in flags)

    def test_create_table_without_constraints_flagged(self):
        flags = _has_producer_red_flags(
            "CREATE TABLE users (\n  id INT PRIMARY KEY,\n  username VARCHAR(50),\n"
            "  email VARCHAR(100),\n  created_at TIMESTAMP\n);"
        )
        assert any("CREATE TABLE" in f for f in flags)

    def test_create_table_with_not_null_not_flagged(self):
        flags = _has_producer_red_flags(
            "CREATE TABLE users (\n  id INT PRIMARY KEY,\n  username VARCHAR(50) NOT NULL UNIQUE,\n"
            "  email VARCHAR(100) NOT NULL\n);"
        )
        assert not any("CREATE TABLE" in f for f in flags)

    def test_stub_body_with_placeholder_comment_flagged(self):
        flags = _has_producer_red_flags(
            "class PaymentProcessor:\n    def process_payment(self, amount):\n"
            "        # Implementation needed\n        pass"
        )
        assert any("stub" in f.lower() for f in flags)

    def test_unguarded_multi_dict_access_flagged(self):
        flags = _has_producer_red_flags(
            "def parse_json_config(s):\n    import json\n    config = json.loads(s)\n"
            "    return config['database_url'], config['api_key']"
        )
        assert any("dict" in f for f in flags)


@pytest.mark.asyncio
class TestRecallImprovementsEndToEnd:
    """End-to-end: the FN cases lifted by the 2026-04-17 recall patch."""

    async def test_cosmetic_suggestion_on_hardcoded_admin(self):
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                (
                    "producer",
                    "def authenticate(username, password):\n"
                    "    if username == 'admin' and password == 'password123':\n"
                    "        return True\n    return False",
                ),
                ("critic", "Consider renaming variable 'username' for clarity."),
            ]
        )
        res = await det.detect(trace)
        assert res.detected

    async def test_question_only_on_hardcoded_admin(self):
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                (
                    "producer",
                    "def authenticate(username, password):\n"
                    "    if username == 'admin' and password == 'password123':\n"
                    "        return True\n    return False",
                ),
                ("critic", "What about other users? How do we handle non-admin authentication?"),
            ]
        )
        res = await det.detect(trace)
        assert res.detected

    async def test_shallow_praise_on_divide_by_zero(self):
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                ("producer", "def divide(a, b):\n    return a / b"),
                ("critic", "Great job! This function is clean and simple."),
            ]
        )
        res = await det.detect(trace)
        assert res.detected

    async def test_shallow_praise_on_unguarded_fetch(self):
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                (
                    "producer",
                    "async function fetchUserData(userId) {\n"
                    "    const response = await fetch('/api/users/' + userId);\n"
                    "    return response.json();\n}",
                ),
                ("critic", "Great async implementation! The API call structure is perfect."),
            ]
        )
        res = await det.detect(trace)
        assert res.detected

    async def test_guarded_divide_with_approval_not_flagged(self):
        """Regression: producer that properly guards division should not be flagged
        even when critic offers only light praise."""
        det = CriticQualityDetector()
        trace = _build_trace(
            [
                (
                    "producer",
                    "public static double divide(double a, double b) {\n"
                    '    if (b == 0) { throw new ArithmeticException("Division by zero"); }\n'
                    "    return a / b;\n}",
                ),
                (
                    "critic",
                    "Good exception handling for division by zero. Consider adding JavaDoc.",
                ),
            ]
        )
        res = await det.detect(trace)
        assert not res.detected
