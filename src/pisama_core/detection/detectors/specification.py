"""Specification mismatch detector for output vs spec deviation.

F1: Specification Mismatch Detection (MAST Taxonomy)

Detects when a task specification doesn't match the user's original intent.
This occurs at the system design level when:
- The task decomposition loses critical requirements
- The specification is ambiguous or incomplete
- The task scope drifts from original intent

Detection Methods:
1. Requirement extraction (modal verbs, action verbs)
2. Constraint extraction (prohibitions, limits)
3. Keyword + synonym + stem coverage scoring
4. Key phrase (bigram) gap detection
5. Ambiguity detection (vague language)
6. Code quality / deprecated syntax checks
7. Language mismatch detection
8. Numeric constraint tolerance
9. Reformulation / bonus feature / benign expansion recognition
10. Scope expansion detection
11. Framework metadata stripping
12. Q&A answer format handling

Ported from backend/app/detection/specification.py (v2.1+).
"""

import logging
import re
from typing import Any, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import Platform
from pisama_core.traces.models import Trace

logger = logging.getLogger(__name__)


class SpecificationDetector(BaseDetector):
    """Detects specification mismatch -- when task spec doesn't match user intent.

    Compares user intent with task specification to identify gaps,
    ambiguities, scope drift, and code quality issues.

    Span convention:
        The detector looks for a root-level user request in
        ``trace.metadata.custom["user_request"]`` or the first span's
        ``input_data.user_request``. Each span's ``input_data.task`` is
        compared against this intent. Falls back to comparing
        ``input_data.user_intent`` and ``input_data.task_specification``
        within individual spans.
    """

    name = "specification"
    description = "Detects output vs spec mismatch"
    version = "2.5.0"
    platforms: list[Platform] = []  # All platforms
    severity_range = (0, 100)
    realtime_capable = False

    # --- Class-level configuration (ported from backend) ---

    # Programming language patterns for language mismatch detection
    LANGUAGE_PATTERNS: dict[str, tuple[str, list[str]]] = {
        "python": (
            r"\bpython\b",
            [r"\bdef\s+\w+\s*\(", r"\bimport\s+\w+", r":\s*$", r"\bprint\s*\("],
        ),
        "javascript": (
            r"\b(?:javascript|js)\b",
            [r"\bfunction\s+\w+", r"\bconst\s+\w+", r"\blet\s+\w+", r"=>"],
        ),
        "typescript": (
            r"\btypescript\b",
            [r":\s*\w+(?:\[\])?(?:\s*[,\)])", r"\binterface\s+\w+", r"<\w+>"],
        ),
        "java": (r"\bjava\b(?!script)", [r"\bpublic\s+class", r"\bprivate\s+\w+", r"\bvoid\s+\w+"]),
        "sql": (r"\bsql\b", [r"\bSELECT\b", r"\bFROM\b", r"\bWHERE\b", r"\bINSERT\b"]),
    }

    # Code quality issues
    CODE_QUALITY_PATTERNS: list[tuple[str, str]] = [
        (
            r"\.(?:sort|append|extend|pop|remove|clear|reverse)\s*(?:\n|$|[^(])",
            "method_without_call",
        ),
        (r"\bpass\s*$", "empty_implementation"),
        (r"//\s*TODO", "todo_marker"),
        (r"#\s*TODO", "todo_marker"),
        (r"raise\s+NotImplementedError", "not_implemented"),
        (r'throw\s+new\s+Error\(["\']not implemented', "not_implemented"),
        (r"\.\.\.\s*$", "ellipsis_placeholder"),
    ]

    # Deprecated syntax patterns by language
    DEPRECATED_SYNTAX_PATTERNS: dict[str, list[tuple[str, str]]] = {
        "python": [
            (r"\bcmp\s*=", "deprecated_cmp_parameter"),
            (r"\.has_key\s*\(", "deprecated_has_key"),
            (r'\bprint\s+["\']', "deprecated_print_statement"),
            (r"\braw_input\s*\(", "deprecated_raw_input"),
            (r"\bexecfile\s*\(", "deprecated_execfile"),
            (r"\bxrange\s*\(", "deprecated_xrange"),
            (r"\.iteritems\s*\(", "deprecated_iteritems"),
            (r"\.iterkeys\s*\(", "deprecated_iterkeys"),
            (r"\.itervalues\s*\(", "deprecated_itervalues"),
            (r"\breduce\s*\((?![^)]*functools)", "deprecated_reduce"),
            (r"\bapply\s*\(", "deprecated_apply"),
            (r"\bcoerce\s*\(", "deprecated_coerce"),
            (r"<>\s*", "deprecated_not_equal"),
            (r"\blong\s*\(", "deprecated_long"),
            (r"\bunicode\s*\(", "deprecated_unicode"),
        ],
        "javascript": [
            (r"\bvar\s+\w+", "deprecated_var"),
            (r"arguments\.callee", "deprecated_callee"),
            (r"with\s*\([^)]+\)\s*{", "deprecated_with"),
            (r"\.substr\s*\(", "deprecated_substr"),
            (r"escape\s*\(", "deprecated_escape"),
            (r"unescape\s*\(", "deprecated_unescape"),
        ],
        "java": [
            (r"new\s+Date\s*\(\s*\d+\s*,", "deprecated_date_constructor"),
            (r"\.getYear\s*\(", "deprecated_getYear"),
            (r"Thread\s*\.\s*stop\s*\(", "deprecated_thread_stop"),
        ],
    }

    # Numeric constraint patterns
    NUMERIC_CONSTRAINT_PATTERNS: list[tuple[str, str]] = [
        (r"(\d+)[\s-]?word", "word_count"),
        (r"(\d+)[\s-]?character", "char_count"),
        (r"(\d+)[\s-]?line", "line_count"),
        (r"(\d+)[\s-]?item", "item_count"),
        (r"(\d+)[\s-]?point", "point_count"),
        (r"(\d+)[\s-]?step", "step_count"),
    ]

    NUMERIC_TOLERANCE: float = 0.10

    # Reformulation indicators
    REFORMULATION_INDICATORS: list[str] = [
        r"(?:task|goal|objective|mission):\s*",
        r"(?:working on|implementing|building|creating|developing)\s+",
        r"(?:as requested|as specified|as mentioned|as described|per request)",
        r"(?:proceeding with|starting|beginning|initiating)\s+",
        r"(?:the following|this involves|this includes|this requires)",
        r"(?:i will|i'll|let me|going to)\s+(?:implement|create|build|develop)",
        r"(?:here's|here is)\s+(?:the|my|a)\s+(?:plan|approach|solution)",
    ]

    # Bonus feature patterns
    BONUS_FEATURE_PATTERNS: list[str] = [
        r"(?:also|additionally|as a bonus|bonus)\s+(?:added|included|implemented)",
        r"(?:extra|additional)\s+(?:feature|functionality|enhancement)",
        r"(?:i also|i've also|we also)\s+(?:added|included|implemented|created)",
        r"(?:went ahead|took the liberty)\s+(?:and|to)\s+(?:add|include)",
        r"(?:while i was at it|while working on this)",
        r"(?:for good measure|for convenience|for better ux)",
        r"(?:future-proofing|future proof|extensibility)",
        r"(?:nice to have|optional enhancement)",
        r"(?:beyond the requirements|exceeding requirements)",
    ]

    # Benign scope expansion patterns
    BENIGN_EXPANSION_PATTERNS: list[str] = [
        r"(?:error handling|validation|edge case|edge-case)",
        r"(?:logging|monitoring|observability)",
        r"(?:unit test|test coverage|integration test)",
        r"(?:documentation|docstring|readme|comment)",
        r"(?:type hint|typing|annotation)",
        r"(?:security|sanitization|input validation)",
    ]

    # Framework-specific thresholds
    FRAMEWORK_THRESHOLDS: dict[str, dict[str, Any]] = {
        "ChatDev": {"spec_coverage": 0.60, "ambiguity_threshold": 5},
        "MetaGPT": {"spec_coverage": 0.60, "ambiguity_threshold": 4},
        "AG2": {"spec_coverage": 0.60, "ambiguity_threshold": 4},
        "Magentic": {"spec_coverage": 0.60, "ambiguity_threshold": 4},
        "AutoGen": {"spec_coverage": 0.60, "ambiguity_threshold": 4},
        "LangGraph": {"spec_coverage": 0.60, "ambiguity_threshold": 4},
        "default": {"spec_coverage": 0.60, "ambiguity_threshold": 4},
    }

    # Expansion phrases
    EXPANSION_PHRASES: list[str] = [
        "also added",
        "additionally",
        "in addition to",
        "as a bonus",
        "plus",
        "on top of",
        "beyond what was asked",
        "extra feature",
        "i also",
        "we also",
        "also included",
        "also implemented",
    ]

    # Synonym groups for requirement matching
    _SYNONYMS: dict[str, set[str]] = {
        "implement": {"build", "create", "develop", "make", "write", "code", "construct"},
        "build": {"implement", "create", "develop", "make", "write", "code", "construct"},
        "create": {"implement", "build", "develop", "make", "write", "generate"},
        "validate": {"verify", "check", "ensure", "confirm", "test", "assert"},
        "verify": {"validate", "check", "ensure", "confirm", "test"},
        "return": {"provide", "output", "give", "produce", "yield", "emit"},
        "provide": {"return", "output", "give", "produce", "supply"},
        "list": {"enumerate", "show", "display", "present", "outline"},
        "display": {"show", "render", "present", "list", "output"},
        "analyze": {"evaluate", "assess", "examine", "review", "inspect"},
        "update": {"modify", "change", "edit", "revise", "patch", "alter"},
        "delete": {"remove", "drop", "destroy", "clear", "purge", "erase"},
        "send": {"deliver", "transmit", "dispatch", "emit", "push"},
        "fetch": {"get", "retrieve", "load", "pull", "obtain", "query"},
        "store": {"save", "persist", "write", "record", "keep"},
        "parse": {"extract", "read", "decode", "interpret", "process"},
        "filter": {"select", "exclude", "narrow", "restrict", "screen"},
        "sort": {"order", "rank", "arrange", "organize"},
        "format": {"render", "transform", "convert", "style"},
        "handle": {"manage", "process", "deal", "address", "resolve"},
        "configure": {"setup", "config", "set", "initialize", "init"},
        "deploy": {"release", "publish", "ship", "launch", "rollout"},
        "test": {"verify", "validate", "check", "assert", "evaluate"},
        "log": {"record", "track", "audit", "trace", "monitor"},
        "notify": {"alert", "inform", "email", "message", "warn"},
        "authenticate": {"login", "auth", "verify", "authorize"},
        "register": {"signup", "enroll", "subscribe", "onboard"},
        "support": {"accept", "handle", "allow", "enable"},
        "resize": {"scale", "crop", "transform"},
        "purchase": {"order", "buy", "transaction", "checkout"},
        "order": {"purchase", "buy", "transaction"},
        "pipeline": {"workflow", "process", "flow", "chain"},
        "workflow": {"pipeline", "process", "flow"},
        "catalog": {"inventory", "collection", "listing", "directory", "category"},
        "category": {"catalog", "type", "classification"},
        "microservices": {"services", "microservice"},
        "services": {"microservices", "service"},
        "endpoint": {"route", "path", "url", "api"},
        "feature": {"functionality", "capability", "module"},
        "management": {"managing", "administration", "admin"},
        "processing": {"pipeline", "handling", "workflow"},
        "invoice": {"bill", "receipt"},
        "customer": {"user", "client"},
        "user": {"customer", "client", "account"},
    }

    _ACRONYMS: dict[str, set[str]] = {
        "crud": {"create", "read", "update", "delete"},
        "ci": {"continuous", "integration", "pipeline", "workflow", "actions"},
        "cd": {"continuous", "deployment", "delivery"},
        "api": {"endpoint", "route", "interface", "rest"},
        "sso": {"single", "sign", "login", "oauth", "authentication"},
        "auth": {"authentication", "authorization", "login"},
        "ui": {"interface", "frontend", "view", "page"},
        "db": {"database", "storage", "postgres", "mysql", "sql"},
        "etl": {"extract", "transform", "load"},
    }

    _PHRASE_STOP_WORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "has",
            "have",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "that",
            "this",
            "it",
            "its",
            "my",
            "our",
            "your",
            "their",
            "all",
            "each",
            "every",
            "any",
            "some",
            "no",
            "not",
            "so",
            "as",
            "if",
            "when",
            "than",
            "then",
            "also",
            "just",
            "about",
            "up",
            "out",
            "into",
            "over",
            "after",
            "before",
            "between",
            "need",
            "want",
            "help",
            "like",
            "wish",
            "make",
            "please",
            "simple",
            "basic",
            "good",
            "best",
            "easy",
            "quick",
            "fast",
            "better",
            "complete",
            "full",
            "proper",
            "right",
            "nice",
            "feature",
            "system",
            "tool",
            "service",
            "application",
            "platform",
            "thing",
            "stuff",
            "part",
            "function",
            "ability",
            "capability",
            "supports",
            "functionality",
        }
    )

    # Framework metadata patterns to strip
    METADATA_PATTERNS: list[str] = [
        r"\[Preprocessing\].*?\n",
        r"##\s+(?:Summary|Modified Files|Thinking|Code)\b.*?\n",
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}.*?\n",
        r"(?:config|src|lib)/\S+\.(?:py|js|ts|json)\b.*?\n",
        r"\.py:\d+.*?\n",
        r"(?:INFO|DEBUG|WARNING|ERROR)\s+\S+.*?\n",
    ]

    # --- Default thresholds ---
    coverage_threshold: float = 0.60
    ambiguity_threshold: int = 4

    def __init__(self, framework: Optional[str] = None) -> None:
        super().__init__()
        self._framework = framework
        if framework and framework in self.FRAMEWORK_THRESHOLDS:
            thresholds = self.FRAMEWORK_THRESHOLDS[framework]
            self.coverage_threshold = thresholds["spec_coverage"]
            self.ambiguity_threshold = thresholds["ambiguity_threshold"]
        elif framework:
            thresholds = self.FRAMEWORK_THRESHOLDS["default"]
            self.coverage_threshold = thresholds["spec_coverage"]
            self.ambiguity_threshold = thresholds["ambiguity_threshold"]

    # --- Internal helpers (ported faithfully from backend) ---

    @staticmethod
    def _stem(word: str) -> str:
        """Minimal suffix-stripping stemmer for coverage matching."""
        if len(word) <= 3:
            return word
        for suffix in (
            "ation",
            "ting",
            "ing",
            "ies",
            "ment",
            "ness",
            "able",
            "ible",
            "ive",
            "ous",
            "ful",
            "ed",
            "er",
            "es",
            "ly",
            "al",
            "s",
        ):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word

    def _expand_with_synonyms(self, word: str) -> set[str]:
        """Return the word plus any known synonyms, stems, and acronym expansions."""
        result = {word}
        result |= self._SYNONYMS.get(word, set())
        result |= self._ACRONYMS.get(word, set())
        stem = self._stem(word)
        if stem != word:
            result.add(stem)
            result |= self._SYNONYMS.get(stem, set())
        return result

    @staticmethod
    def _strip_metadata(text: str) -> str:
        """Strip framework metadata that inflates ambiguity scores."""
        for pattern in SpecificationDetector.METADATA_PATTERNS:
            text = re.sub(pattern, "", text)
        return text.strip()

    @staticmethod
    def _extract_requirements(text: str) -> list[str]:
        """Extract explicit requirements from text."""
        requirements: list[str] = []

        must_patterns = [
            r"must\s+([^.!?]+)",
            r"should\s+([^.!?]+)",
            r"need(?:s)?\s+to\s+([^.!?]+)",
            r"require(?:s|d)?\s+([^.!?]+)",
            r"has\s+to\s+([^.!?]+)",
            r"ensure\s+(?:that\s+)?([^.!?]+)",
        ]
        for pattern in must_patterns:
            matches = re.findall(pattern, text.lower())
            requirements.extend(matches)

        action_patterns = [
            r"(?:create|build|make|generate|design)\s+(?:a\s+)?([^.!?,]+)",
            r"(?:find|search|get|fetch)\s+([^.!?,]+)",
            r"(?:analyze|evaluate|assess)\s+([^.!?,]+)",
            r"(?:send|deliver|transmit)\s+([^.!?,]+)",
            r"(?:implement|set\s+up|configure|deploy|add)\s+(?:a\s+)?([^.!?,]+)",
            r"(?:monitor|track|log|alert)\s+([^.!?,]+)",
            r"(?:help\s+me|i\s+want\s+to|i\s+want\s+a)\s+([^.!?,]+)",
            r"(?:i\s+need)\s+(?:a\s+)?([^.!?,]+)",
            r"(?:migrate|convert|transform|clean\s+up)\s+([^.!?,]+)",
            r"(?:write|develop|plan|prepare|draft|outline)\s+(?:a\s+)?([^.!?,]+)",
            r"(?:process|handle|manage|run|execute)\s+([^.!?,]+)",
            r"(?:connect|integrate|link|sync(?:hronize)?)\s+([^.!?,]+)",
            r"(?:automate|schedule|optimize|improve)\s+([^.!?,]+)",
        ]
        for pattern in action_patterns:
            matches = re.findall(pattern, text.lower())
            requirements.extend(matches)

        return [r.strip() for r in requirements if len(r.strip()) > 3]

    @staticmethod
    def _extract_constraints(text: str) -> list[str]:
        """Extract constraints/prohibitions from text."""
        constraints: list[str] = []
        constraint_patterns = [
            r"(?:do not|don\'t|never|must not)\s+([^.!?,]+)",
            r"without\s+(?:any\s+)?([^.!?,]+)",
            r"(?:only|exclusively)\s+([^.!?,]+)",
            r"(?:at\s+(?:most|least))\s+([^.!?,]+)",
            r"(?:within|under|below|above)\s+(\d+[^.!?,]*)",
            r"(?:before|after|by)\s+([^.!?,]+)",
            r"(?:limit(?:ed)?\s+to)\s+([^.!?,]+)",
        ]
        for pattern in constraint_patterns:
            matches = re.findall(pattern, text.lower())
            constraints.extend(matches)

        return [c.strip() for c in constraints if len(c.strip()) > 8]

    def _compute_coverage(
        self,
        intent_requirements: list[str],
        spec_text: str,
    ) -> tuple[float, list[str]]:
        """Compute keyword + synonym + stem coverage of requirements in spec."""
        if not intent_requirements:
            return 1.0, []

        spec_lower = spec_text.lower()
        spec_words = set(re.findall(r"[a-z]+", spec_lower))
        spec_stems = {self._stem(w) for w in spec_words if len(w) >= 3}
        covered = 0
        missing: list[str] = []

        for req in intent_requirements:
            req_words = set(req.lower().split())
            req_words = {w for w in req_words if len(w) > 3}

            if not req_words:
                covered += 1
                continue

            overlap = 0
            for w in req_words:
                expanded = self._expand_with_synonyms(w)
                if any(syn in spec_lower for syn in expanded):
                    overlap += 1
                elif self._stem(w) in spec_stems:
                    overlap += 1

            coverage_ratio = overlap / len(req_words)
            if coverage_ratio >= 0.5:
                covered += 1
            else:
                missing.append(req)

        return covered / len(intent_requirements), missing

    def _extract_key_phrases(self, text: str) -> list[str]:
        """Extract important bigram phrases from text."""
        words = re.findall(r"[a-z][a-z\'-]+", text.lower())
        content_words = [
            (i, w) for i, w in enumerate(words) if len(w) > 3 and w not in self._PHRASE_STOP_WORDS
        ]

        phrases: list[str] = []
        for j in range(len(content_words) - 1):
            idx1, w1 = content_words[j]
            idx2, w2 = content_words[j + 1]
            if idx2 - idx1 <= 2:
                phrases.append(f"{w1} {w2}")

        return phrases

    def _detect_missing_key_phrases(
        self,
        user_intent: str,
        spec_text: str,
    ) -> tuple[bool, list[str], float]:
        """Detect when specific multi-word concepts from intent are absent from spec."""
        intent_phrases = self._extract_key_phrases(user_intent)
        if not intent_phrases:
            return False, [], 0.0

        spec_lower = spec_text.lower()
        spec_words = set(re.findall(r"[a-z]+", spec_lower))
        spec_stems = {self._stem(w) for w in spec_words if len(w) >= 3}

        missing: list[str] = []
        for phrase in intent_phrases:
            w1, w2 = phrase.split()
            w1_found = (
                w1 in spec_lower
                or self._stem(w1) in spec_stems
                or any(syn in spec_lower for syn in self._expand_with_synonyms(w1))
            )
            w2_found = (
                w2 in spec_lower
                or self._stem(w2) in spec_stems
                or any(syn in spec_lower for syn in self._expand_with_synonyms(w2))
            )
            if not w1_found and not w2_found:
                missing.append(phrase)

        if not missing:
            return False, [], 0.0

        fraction = len(missing) / len(intent_phrases)
        has_missing = len(missing) >= 2 and fraction >= 0.20
        return has_missing, missing, fraction

    @staticmethod
    def _detect_ambiguities(text: str) -> list[str]:
        """Detect vague or ambiguous language in text."""
        ambiguities: list[str] = []
        vague_patterns: list[tuple[str, str]] = [
            (r"\b(some|several|many|few|various)\s+\w+", "vague quantity"),
            (r"\b(soon|later|eventually|sometime)\b", "vague timing"),
            (r"\b(good|better|best|nice|appropriate)\b", "subjective quality"),
            (r"\b(etc|and so on|and more|among others)\b", "incomplete list"),
            (r"\b(it|this|that)\b(?!\s+(?:is|are|was|will))", "ambiguous reference"),
            (r"\b(usually|typically|generally|normally)\b", "uncertain qualifier"),
            (r"\b(might|may|could|possibly|perhaps)\b", "uncertain action"),
            (r"\b(simple|easy|quick|basic)\b", "undefined complexity"),
        ]
        for pattern, issue_type in vague_patterns:
            if re.search(pattern, text.lower()):
                ambiguities.append(issue_type)
        return ambiguities

    @staticmethod
    def _is_task_reformulation(text: str) -> bool:
        """v1.3: Detect if text reformulates/restates the task."""
        text_lower = text.lower()
        for pattern in SpecificationDetector.REFORMULATION_INDICATORS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _has_bonus_features(text: str) -> bool:
        """v1.4: Detect if text indicates bonus/extra features were added."""
        text_lower = text.lower()
        for pattern in SpecificationDetector.BONUS_FEATURE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _has_benign_expansion(text: str) -> bool:
        """v1.4: Detect if text contains benign scope expansions."""
        text_lower = text.lower()
        for pattern in SpecificationDetector.BENIGN_EXPANSION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def _detect_requested_language(self, intent: str) -> Optional[str]:
        """v1.1: Detect which programming language was requested."""
        intent_lower = intent.lower()
        for lang, (pattern, _) in self.LANGUAGE_PATTERNS.items():
            if re.search(pattern, intent_lower):
                return lang
        return None

    def _detect_output_language(self, output: str) -> Optional[str]:
        """v1.1: Detect which programming language is in the output."""
        for lang, (_, code_patterns) in self.LANGUAGE_PATTERNS.items():
            matches = sum(1 for p in code_patterns if re.search(p, output, re.MULTILINE))
            if matches >= 2:
                return lang
        return None

    def _check_code_quality(self, output: str) -> list[tuple[str, str, str]]:
        """v1.1: Check for code quality issues."""
        issues: list[tuple[str, str, str]] = []
        for pattern, issue_type in self.CODE_QUALITY_PATTERNS:
            for match in re.finditer(pattern, output, re.MULTILINE):
                start = max(0, match.start() - 20)
                end = min(len(output), match.end() + 20)
                context = output[start:end].strip()
                issues.append((issue_type, match.group(), context))
        return issues

    def _check_deprecated_syntax(
        self,
        output: str,
        language: Optional[str],
    ) -> list[tuple[str, str, str]]:
        """v1.2: Check for deprecated syntax patterns in code output."""
        issues: list[tuple[str, str, str]] = []
        if not language or language not in self.DEPRECATED_SYNTAX_PATTERNS:
            return issues

        for pattern, issue_type in self.DEPRECATED_SYNTAX_PATTERNS[language]:
            for match in re.finditer(pattern, output, re.MULTILINE):
                start = max(0, match.start() - 30)
                end = min(len(output), match.end() + 30)
                context = output[start:end].strip()
                issues.append((issue_type, match.group(), context))
        return issues

    def _extract_numeric_constraint(self, intent: str) -> Optional[tuple[int, str]]:
        """v1.1: Extract numeric constraint from intent."""
        for pattern, constraint_type in self.NUMERIC_CONSTRAINT_PATTERNS:
            match = re.search(pattern, intent.lower())
            if match:
                return (int(match.group(1)), constraint_type)
        return None

    @staticmethod
    def _check_numeric_constraint(output: str, target: int, constraint_type: str) -> bool:
        """v1.1: Check if output meets numeric constraint within tolerance."""
        tolerance_pct = SpecificationDetector.NUMERIC_TOLERANCE
        if constraint_type == "word_count":
            words = len(output.split())
            tolerance = int(target * tolerance_pct)
            return abs(words - target) <= tolerance
        elif constraint_type == "line_count":
            lines = len(output.strip().split("\n"))
            tolerance = max(1, int(target * tolerance_pct))
            return abs(lines - target) <= tolerance
        return True

    @staticmethod
    def _is_code_task(intent: str) -> bool:
        """v1.1: Check if the task is code-related."""
        code_keywords = [
            r"\bcode\b",
            r"\bfunction\b",
            r"\bprogram\b",
            r"\bscript\b",
            r"\bimplement\b",
            r"\bwrite\s+(?:a\s+)?(?:python|javascript|java|sql)",
            r"\bcreate\s+(?:a\s+)?(?:function|class|method)",
        ]
        intent_lower = intent.lower()
        return any(re.search(kw, intent_lower) for kw in code_keywords)

    @staticmethod
    def _count_distinct_requirements(text: str) -> int:
        """Count distinct requirements in text."""
        count = 0
        bullet_count = len(
            re.findall(
                r"(?:^|\n)\s*(?:\d+[\.\/\)]\s|[-*\u2022]\s)",
                text,
            )
        )
        if bullet_count > 0:
            count = bullet_count
        else:
            sentences = re.split(r"[.!?]\s+|\n+", text)
            req_sentences = [
                s
                for s in sentences
                if len(s.strip()) > 10
                and re.search(
                    r"\b(?:must|should|need|create|build|implement|add|set up|configure|ensure|include)\b",
                    s,
                    re.IGNORECASE,
                )
            ]
            count = len(req_sentences)

        and_actions = re.findall(
            r"\b(?:and|,)\s+(?:also\s+)?(?:create|build|implement|add|set up|configure|ensure|include)\b",
            text,
            re.IGNORECASE,
        )
        count += len(and_actions)
        return max(1, count)

    def _detect_scope_expansion(
        self,
        user_intent: str,
        task_specification: str,
    ) -> Optional[str]:
        """v2.2: Detect when spec has significantly more requirements than intent."""
        spec_lower = task_specification.lower()
        expansion_found = [phrase for phrase in self.EXPANSION_PHRASES if phrase in spec_lower]

        intent_reqs = self._count_distinct_requirements(user_intent)
        spec_reqs = self._count_distinct_requirements(task_specification)

        if spec_reqs >= intent_reqs * 3 and spec_reqs >= 6:
            return (
                f"scope_expansion: spec has {spec_reqs} requirements vs "
                f"{intent_reqs} in intent ({spec_reqs / intent_reqs:.1f}x expansion)"
            )

        if expansion_found and spec_reqs > intent_reqs:
            return (
                f"scope_expansion: expansion phrases found ({', '.join(expansion_found[:3])}) "
                f"with {spec_reqs} spec requirements vs {intent_reqs} in intent"
            )

        return None

    # --- Core single-pair detection ---

    def _detect_single(
        self,
        user_intent: str,
        task_specification: str,
        original_request: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Run specification mismatch detection on a single intent/spec pair.

        Returns a dict with detection results if mismatch found, else None.
        """
        user_intent = self._strip_metadata(user_intent)
        task_specification = self._strip_metadata(task_specification)

        is_qa_answer = task_specification.strip().startswith("Answer:")

        intent_requirements = self._extract_requirements(user_intent)
        intent_constraints = self._extract_constraints(user_intent)
        all_requirements = intent_requirements + intent_constraints

        # Keyword coverage (no embedding in pisama-core; backend falls back the same way)
        coverage, missing = self._compute_coverage(all_requirements, task_specification)

        ambiguities = self._detect_ambiguities(task_specification)

        is_reformulation = self._is_task_reformulation(task_specification)
        has_bonus = self._has_bonus_features(task_specification)
        has_benign_expansion = self._has_benign_expansion(task_specification)

        mismatch_type: Optional[str] = None
        detected = False
        code_issues: list[tuple[str, str, str]] = []

        # Numeric constraints
        numeric_constraint = self._extract_numeric_constraint(user_intent)
        numeric_constraint_met = False
        if numeric_constraint:
            target, constraint_type = numeric_constraint
            if self._check_numeric_constraint(task_specification, target, constraint_type):
                numeric_constraint_met = True
            else:
                detected = True
                mismatch_type = "missing_requirement"
                missing.append(f"{target}-{constraint_type.replace('_', ' ')}")

        # Code quality for code tasks
        if self._is_code_task(user_intent):
            code_issues = self._check_code_quality(task_specification)
            if code_issues:
                detected = True
                mismatch_type = "missing_requirement"
                for issue_type, _, _ in code_issues[:3]:
                    missing.append(f"code issue: {issue_type}")

        # Language mismatch + deprecated syntax
        requested_lang = self._detect_requested_language(user_intent)
        if requested_lang:
            deprecated_issues = self._check_deprecated_syntax(task_specification, requested_lang)
            if deprecated_issues:
                detected = True
                mismatch_type = "missing_requirement"
                for issue_type, _, _ in deprecated_issues[:3]:
                    missing.append(f"deprecated syntax: {issue_type}")

            output_lang = self._detect_output_language(task_specification)
            if output_lang and output_lang != requested_lang:
                if not (requested_lang == "javascript" and output_lang == "typescript"):
                    detected = True
                    mismatch_type = "missing_requirement"
                    missing.append(
                        f"language mismatch: requested {requested_lang}, got {output_lang}"
                    )

        # Coverage / ambiguity (with skip conditions)
        skip_coverage_check_qa = is_qa_answer
        skip_coverage_check = skip_coverage_check_qa or (
            numeric_constraint_met
            or (is_reformulation and coverage >= 0.4)
            or (has_bonus and coverage >= 0.65)
            or (has_benign_expansion and coverage >= 0.65)
        )
        if not skip_coverage_check:
            if coverage < self.coverage_threshold and not detected:
                detected = True
                mismatch_type = "missing_requirement"
            elif len(ambiguities) >= self.ambiguity_threshold and not detected:
                detected = True
                mismatch_type = "ambiguous_spec"

        # Original request drift
        if original_request and user_intent != original_request:
            orig_reqs = self._extract_requirements(original_request)
            orig_coverage, orig_missing = self._compute_coverage(orig_reqs, task_specification)
            if orig_coverage < coverage - 0.2:
                detected = True
                mismatch_type = "scope_drift"
                missing.extend(orig_missing)

        # Key phrase check
        if not detected and not is_qa_answer:
            has_missing_phrases, missing_phrases, phrase_fraction = (
                self._detect_missing_key_phrases(user_intent, task_specification)
            )
            if has_missing_phrases:
                detected = True
                mismatch_type = "missing_requirement"
                missing.extend(f"key phrase: {p}" for p in missing_phrases[:3])
                coverage = min(coverage, 1.0 - phrase_fraction)

        # Scope expansion
        if not detected:
            scope_expansion = self._detect_scope_expansion(user_intent, task_specification)
            if scope_expansion:
                detected = True
                mismatch_type = "scope_drift"
                missing.append(scope_expansion)

        if not detected:
            return None

        # Severity
        if coverage < 0.3:
            severity = 80
        elif coverage < 0.5:
            severity = 55
        else:
            severity = 30

        # Confidence (gradient)
        if coverage < 0.3:
            confidence = min(0.95, 0.75 + (0.3 - coverage))
        elif coverage < 0.5:
            confidence = 0.60
        else:
            confidence = max(0.10, 0.40 - coverage * 0.5)

        # Explanation
        if mismatch_type == "missing_requirement":
            explanation = (
                f"Task specification missing {len(missing)} requirements from user intent. "
                f"Coverage: {coverage:.1%}"
            )
            fix = f"Add missing requirements to specification: {', '.join(missing[:3])}"
        elif mismatch_type == "ambiguous_spec":
            explanation = (
                f"Task specification contains {len(ambiguities)} ambiguous elements: "
                f"{', '.join(ambiguities[:5])}"
            )
            fix = "Replace vague language with specific, measurable criteria"
        else:
            explanation = "Task specification has drifted from original user request"
            fix = "Re-align specification with original user intent"

        return {
            "detected": True,
            "severity": severity,
            "confidence": confidence,
            "summary": explanation,
            "fix_instruction": fix,
            "evidence": {
                "mismatch_type": mismatch_type,
                "requirement_coverage": round(coverage, 4),
                "missing_requirements": missing[:10],
                "ambiguous_elements": ambiguities,
                "is_reformulation": is_reformulation,
                "has_bonus_features": has_bonus,
                "has_benign_expansion": has_benign_expansion,
                "is_qa_answer": is_qa_answer,
                "code_issues": [{"type": t, "match": m} for t, m, _ in code_issues[:5]],
            },
        }

    # --- Trace-level detect (BaseDetector interface) ---

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect specification mismatch across spans in a trace.

        Looks for a root user request in ``trace.metadata.custom["user_request"]``
        or the first span's ``input_data.user_request``. Each span's
        ``input_data.task`` is compared against the intent. Also supports
        per-span ``input_data.user_intent`` / ``input_data.task_specification``
        for golden-dataset-style inputs.

        Returns the highest-severity finding.
        """
        # Find root user request
        user_request = (trace.metadata.custom or {}).get("user_request", "")
        if not user_request and trace.spans:
            first_span = sorted(trace.spans, key=lambda s: s.start_time)[0]
            user_request = (first_span.input_data or {}).get("user_request", "")

        worst: Optional[dict[str, Any]] = None

        for span in trace.spans:
            # Try golden-dataset-style keys first
            user_intent = (span.input_data or {}).get("user_intent", "")
            task_spec = (span.input_data or {}).get("task_specification", "")

            if user_intent and task_spec:
                finding = self._detect_single(
                    user_intent=user_intent,
                    task_specification=task_spec,
                )
            elif user_request:
                task_spec = (span.input_data or {}).get("task", "")
                if task_spec and len(task_spec) > 20:
                    finding = self._detect_single(
                        user_intent=user_request,
                        task_specification=task_spec,
                    )
                else:
                    continue
            else:
                continue

            if finding and (worst is None or finding["severity"] > worst["severity"]):
                worst = finding

        if worst is None:
            return DetectionResult.no_issue(self.name)

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=worst["severity"],
            summary=worst["summary"],
            fix_type=FixType.ESCALATE,
            fix_instruction=worst["fix_instruction"],
        )
        result.confidence = worst["confidence"]
        result.add_evidence(
            description=worst["summary"],
            data=worst["evidence"],
        )
        return result
