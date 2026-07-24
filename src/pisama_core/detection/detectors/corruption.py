"""State corruption and invalid transition detection.

Detects when agent memory or state becomes corrupted, including:
- Structured state corruption (type drift, schema violations)
- Text-based context corruption (ignoring related context)
- Cross-field inconsistencies
- Hallucinated references
- Anomalous value changes (sign flips, magnitude shifts)
- Status regressions
- Identity mutations
- Data loss patterns

Version History:
- v1.0: Port from backend SemanticCorruptionDetector v1.1 with full logic
"""

import re
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.models import Span, Trace

# ------------------------------------------------------------------
# Internal data classes
# ------------------------------------------------------------------


@dataclass
class CorruptionIssue:
    """A single corruption issue found during detection."""

    issue_type: str
    field: Optional[str]
    message: str
    severity: str


@dataclass
class StateSnapshot:
    """A snapshot of agent state at a point in time."""

    state_delta: dict
    agent_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class VelocityConfig:
    """Configuration for state change velocity thresholds."""

    window_seconds: float = 5.0
    max_changes_per_window: int = 10
    high_velocity_fields: list[str] = field(
        default_factory=lambda: [
            "counter",
            "count",
            "iteration",
            "step",
            "progress",
            "timestamp",
            "updated_at",
            "last_seen",
            "version",
        ]
    )
    ignore_velocity_for_types: list[type] = field(default_factory=lambda: [bool])


@dataclass
class Schema:
    """Schema for validating state fields."""

    fields: dict[str, type]
    required_fields: list[str]


# ------------------------------------------------------------------
# Related-topic and pattern databases
# ------------------------------------------------------------------

RELATED_TOPICS: dict[str, list[str]] = {
    "authentication": ["session", "token", "login", "logout", "credential", "password", "security"],
    "auth": ["session", "token", "login", "logout", "credential", "password", "security"],
    "login": ["session", "authentication", "token", "logout", "security"],
    "database": ["transaction", "connection", "query", "migration", "schema", "index"],
    "api": ["endpoint", "request", "response", "error handling", "validation", "rate limit"],
    "payment": ["refund", "transaction", "billing", "subscription", "invoice"],
    "user": ["profile", "permission", "role", "account", "preference"],
    "cache": ["invalidation", "expiration", "refresh", "consistency"],
    "file": ["permission", "path", "encoding", "backup", "cleanup"],
    "error": ["logging", "handling", "recovery", "notification", "retry"],
    "test": ["coverage", "assertion", "mock", "fixture", "edge case"],
    "deployment": ["rollback", "release", "staging", "production", "ci/cd", "pipeline", "version"],
    "networking": ["dns", "firewall", "load balancer", "proxy", "ssl", "certificate", "port"],
    "monitoring": ["alert", "metric", "logging", "dashboard", "threshold", "uptime"],
    "security": [
        "encryption",
        "vulnerability",
        "audit",
        "compliance",
        "access control",
        "firewall",
    ],
    "config": ["environment variable", "secret", "setting", "parameter", "flag", "option"],
    "configuration": ["environment variable", "secret", "setting", "parameter", "flag", "option"],
    "migration": ["schema", "rollback", "version", "data transfer", "compatibility", "backup"],
    "notification": ["email", "webhook", "alert", "push", "sms", "template"],
    "search": ["index", "query", "relevance", "filter", "ranking", "pagination"],
    "queue": ["consumer", "producer", "dead letter", "retry", "ordering", "backpressure"],
    "messaging": ["consumer", "producer", "dead letter", "retry", "ordering", "backpressure"],
    "storage": ["bucket", "upload", "download", "permission", "lifecycle", "backup"],
    "logging": ["level", "format", "rotation", "aggregation", "filter", "structured"],
    "session": ["cookie", "token", "expiration", "renewal", "invalidation", "storage"],
    "scheduling": ["cron", "interval", "retry", "timeout", "concurrency", "queue"],
    "backup": ["restore", "snapshot", "retention", "encryption", "verification", "schedule"],
    "encryption": ["key", "certificate", "hash", "salt", "algorithm", "rotation"],
    "permission": ["role", "access", "grant", "deny", "scope", "policy"],
    "integration": ["webhook", "api", "sync", "mapping", "transform", "retry"],
    "webhook": ["endpoint", "payload", "signature", "retry", "timeout", "validation"],
    "email": ["template", "delivery", "bounce", "spam", "attachment", "queue"],
}

NARROW_FOCUS_PATTERNS: list[tuple[str, str]] = [
    (r"\bonly\s+(?:addressed|fixed|updated|changed|modified)\b", "explicit_narrow_focus"),
    (r"\bjust\s+(?:the|this|that)\s+\w+\b", "narrow_scope"),
    (r"\bfocused\s+(?:specifically|only|solely)\s+on\b", "explicit_narrow_focus"),
    (r"\bwithout\s+(?:touching|changing|modifying|affecting)\b", "explicit_exclusion"),
    (r"\bdidn\'t\s+(?:touch|change|modify|address)\b", "explicit_exclusion"),
    (r"\bignored?\s+(?:the|any)\b", "explicit_ignorance"),
    (r"\bspecific\s+(?:fix|change|update)\b", "narrow_scope"),
]

COMPREHENSIVE_PATTERNS: list[str] = [
    r"\balso\s+(?:updated|checked|verified|ensured)\b",
    r"\badditionally\b",
    r"\brelated\s+(?:to|changes|updates)\b",
    r"\bensured?\s+(?:that|consistency)\b",
    r"\bconsidered?\s+(?:the|all|related)\b",
    r"\bacross\s+(?:the|all|related)\b",
]

# Status fields and their expected forward-only progression
STATUS_PROGRESSIONS: dict[str, list[str]] = {
    "status": ["pending", "processing", "shipped", "delivered", "completed"],
    "order_status": ["pending", "processing", "shipped", "delivered", "completed"],
    "state": ["created", "active", "suspended", "closed", "archived"],
    "workflow_status": ["draft", "review", "approved", "published"],
    "ticket_status": ["open", "in_progress", "resolved", "closed"],
}

# Fields that should only increase
MONOTONIC_INCREASING_FIELDS: set[str] = {
    "version",
    "revision",
    "build_number",
    "sequence",
    "last_login",
    "last_modified",
    "last_updated",
    "updated_at",
    "last_seen",
    "last_activity",
}

# Fields representing core identity (should not all change at once)
IDENTITY_FIELDS: set[str] = {"first_name", "last_name", "name", "username", "display_name"}

# Known score-grade mappings
GRADE_SCORE_RANGES: dict[str, tuple[int, int]] = {
    "A": (90, 100),
    "A+": (97, 100),
    "A-": (90, 93),
    "B": (80, 89),
    "B+": (87, 89),
    "B-": (80, 83),
    "C": (70, 79),
    "D": (60, 69),
    "F": (0, 59),
}

# Issue types immune to velocity suppression
_VELOCITY_IMMUNE_ISSUES: set[str] = {
    "type_drift",
    "monotonic_regression",
    "sign_flip",
    "status_regression",
    "data_loss",
    "content_replacement",
    "identity_mutation",
    "regression_nullification",
}


class CorruptionDetector(BaseDetector):
    """Detects state corruption and invalid transitions in agent systems.

    Handles both structured state corruption (type drift, schema violations,
    anomalous value changes) and text-based context corruption (ignored
    related context, narrow focus).
    """

    name = "corruption"
    description = "Detects state corruption, invalid transitions, and context corruption"
    version = "1.1.0"
    platforms = []  # All platforms
    severity_range = (10, 100)
    realtime_capable = True

    def __init__(self) -> None:
        super().__init__()
        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        url_pattern = re.compile(
            r"^https?://[a-zA-Z0-9][-a-zA-Z0-9]*(\.[a-zA-Z0-9][-a-zA-Z0-9]*)+(/.*)?$"
        )

        self.domain_validators: dict[str, Callable] = {
            "age": lambda v: isinstance(v, (int, float)) and 0 <= v <= 150,
            "price": lambda v: isinstance(v, (int, float)) and v >= 0,
            "percentage": lambda v: isinstance(v, (int, float)) and 0 <= v <= 100,
            "email": lambda v: isinstance(v, str) and bool(email_pattern.match(v)),
            "url": lambda v: isinstance(v, str) and bool(url_pattern.match(v)),
            "phone": lambda v: isinstance(v, str) and len(re.sub(r"\D", "", v)) >= 10,
            "uuid": lambda v: isinstance(v, str) and len(v) == 36 and v.count("-") == 4,
        }
        self.known_ids: set = set()
        self.velocity_config = VelocityConfig()
        self._change_history: dict[str, deque] = {}
        self._field_velocities: dict[str, float] = {}

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect corruption in a trace.

        Examines state-change events in spans for structured corruption, and
        also analyzes text-based context handling via agent outputs.
        """
        all_issues: list[CorruptionIssue] = []
        evidence_span_ids: list[str] = []

        # Strategy 1: Look for single spans carrying both prev_state and current_state
        for span in trace.spans:
            prev_dict = span.attributes.get("prev_state") or (span.input_data or {}).get(
                "prev_state"
            )
            curr_dict = span.attributes.get("current_state") or (span.input_data or {}).get(
                "current_state"
            )
            if isinstance(prev_dict, dict) and isinstance(curr_dict, dict):
                prev_state = StateSnapshot(
                    state_delta=prev_dict,
                    agent_id=span.attributes.get("agent_id", span.name),
                    timestamp=span.start_time,
                )
                curr_state = StateSnapshot(
                    state_delta=curr_dict,
                    agent_id=span.attributes.get("agent_id", span.name),
                    timestamp=span.end_time or span.start_time,
                )
                issues = self._detect_structured_corruption(prev_state, curr_state)
                if issues:
                    all_issues.extend(issues)
                    evidence_span_ids.append(span.span_id)

        # Strategy 1b: Look for consecutive state-carrying spans
        state_spans = self._find_state_carrying_spans(trace)
        for i in range(len(state_spans) - 1):
            previous_snapshot = self._span_to_state_snapshot(state_spans[i])
            current_snapshot = self._span_to_state_snapshot(state_spans[i + 1])
            if previous_snapshot and current_snapshot:
                issues = self._detect_structured_corruption(previous_snapshot, current_snapshot)
                if issues:
                    all_issues.extend(issues)
                    evidence_span_ids.append(state_spans[i + 1].span_id)

        # Strategy 2: Look for text-based context corruption
        for span in trace.spans:
            task = self._extract_task(span)
            output = self._extract_output(span)
            if task and output:
                text_issues = self._detect_text_corruption(task, output)
                if text_issues:
                    all_issues.extend(text_issues)
                    evidence_span_ids.append(span.span_id)

        if not all_issues:
            return DetectionResult.no_issue(self.name)

        # Compute severity
        max_severity = "low"
        severity_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for issue in all_issues:
            sev = issue.severity
            if sev in severity_counts:
                severity_counts[sev] += 1
            if self._severity_rank(sev) > self._severity_rank(max_severity):
                max_severity = sev

        severity_score = self._severity_to_score(max_severity)

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity_score,
            summary=all_issues[0].message,
            fix_type=FixType.ROLLBACK,
            fix_instruction="Rollback to the last known good state. Investigate the source of corruption.",
        )

        for issue in all_issues:
            result.add_evidence(
                description=f"[{issue.issue_type}] {issue.message}",
                span_ids=evidence_span_ids,
                data={
                    "issue_type": issue.issue_type,
                    "field": issue.field,
                    "severity": issue.severity,
                },
            )

        # Calibrate confidence
        raw_score = self._calculate_raw_score(all_issues, severity_counts)
        confidence, _ = self._calibrate_confidence(
            all_issues,
            severity_counts,
            max_severity,
            raw_score,
        )
        result.confidence = confidence

        return result

    # ------------------------------------------------------------------
    # Trace-to-state helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_state_carrying_spans(trace: Trace) -> list[Span]:
        """Find spans that carry state data (for consecutive pair analysis).

        Returns spans sorted by start_time that have state in
        attributes or input/output data.
        """
        state_spans: list[Span] = []
        for span in sorted(trace.spans, key=lambda s: s.start_time):
            has_state = (
                span.attributes.get("state") is not None
                or (span.output_data or {}).get("state") is not None
            )
            if has_state:
                state_spans.append(span)
        return state_spans

    @staticmethod
    def _span_to_state_snapshot(span: Span) -> Optional[StateSnapshot]:
        """Extract a StateSnapshot from a span."""
        state = (
            span.attributes.get("current_state")
            or span.attributes.get("prev_state")
            or (span.input_data or {}).get("current_state")
            or (span.input_data or {}).get("prev_state")
            or (span.output_data or {}).get("state")
        )
        if isinstance(state, dict):
            return StateSnapshot(
                state_delta=state,
                agent_id=span.attributes.get("agent_id", span.name),
                timestamp=span.start_time,
            )
        return None

    @staticmethod
    def _extract_task(span: Span) -> Optional[str]:
        """Extract task description from a span."""
        candidates = [
            span.attributes.get("task"),
            span.attributes.get("task_description"),
            (span.input_data or {}).get("task"),
            (span.input_data or {}).get("task_description"),
        ]
        for c in candidates:
            if isinstance(c, str) and len(c) > 5:
                return c
        return None

    @staticmethod
    def _extract_output(span: Span) -> Optional[str]:
        """Extract output text from a span."""
        candidates = [
            span.attributes.get("output"),
            span.attributes.get("response"),
            (span.output_data or {}).get("output"),
            (span.output_data or {}).get("text"),
            (span.output_data or {}).get("response"),
        ]
        for c in candidates:
            if isinstance(c, str) and len(c) > 5:
                return c
        return None

    # ------------------------------------------------------------------
    # Structured state corruption detection
    # ------------------------------------------------------------------

    def _detect_structured_corruption(
        self,
        prev_state: StateSnapshot,
        current_state: StateSnapshot,
        schema: Optional[Schema] = None,
    ) -> list[CorruptionIssue]:
        """Full structured corruption pipeline.

        Runs all checks: schema validation, type drift, cross-field consistency,
        domain constraints, hallucinated references, value copying, rapid changes,
        anomalous value changes, and nested dict flattening.
        """
        issues: list[CorruptionIssue] = []

        if schema:
            issues.extend(self._validate_schema(current_state, schema))

        issues.extend(self._detect_type_drift(prev_state, current_state))
        issues.extend(self._validate_cross_field_consistency(current_state))
        issues.extend(self._validate_domain_constraints(current_state))
        issues.extend(self._detect_hallucinated_references(prev_state, current_state))
        issues.extend(self._detect_value_copying(current_state))
        issues.extend(self._detect_suspicious_rapid_changes(prev_state, current_state))
        issues.extend(self._detect_anomalous_value_changes(prev_state, current_state))

        # Flatten nested dicts and re-run value-change detection
        prev_flat = self._flatten_nested_dicts(prev_state.state_delta)
        curr_flat = self._flatten_nested_dicts(current_state.state_delta)
        if prev_flat != prev_state.state_delta or curr_flat != current_state.state_delta:
            flat_prev = StateSnapshot(
                state_delta=prev_flat,
                agent_id=prev_state.agent_id,
                timestamp=prev_state.timestamp,
            )
            flat_curr = StateSnapshot(
                state_delta=curr_flat,
                agent_id=current_state.agent_id,
                timestamp=current_state.timestamp,
            )
            issues.extend(self._detect_anomalous_value_changes(flat_prev, flat_curr))

        filtered = self._apply_velocity_filtering(issues, current_state)
        return filtered

    def _detect_type_drift(
        self,
        prev_state: StateSnapshot,
        current_state: StateSnapshot,
    ) -> list[CorruptionIssue]:
        """Detect type changes in fields between states."""
        issues: list[CorruptionIssue] = []
        prev_data = prev_state.state_delta
        curr_data = current_state.state_delta

        common_fields = set(prev_data.keys()) & set(curr_data.keys())
        for fld in common_fields:
            prev_val = prev_data[fld]
            curr_val = curr_data[fld]
            prev_type = type(prev_val)
            curr_type = type(curr_val)

            if prev_type != curr_type:
                if prev_val is None or curr_val is None:
                    continue
                if {prev_type, curr_type} <= {int, float}:
                    continue
                issues.append(
                    CorruptionIssue(
                        issue_type="type_drift",
                        field=fld,
                        message=f"Type changed from {prev_type.__name__} to {curr_type.__name__}",
                        severity="high",
                    )
                )

        return issues

    @staticmethod
    def _flatten_nested_dicts(state_delta: dict, prefix: str = "") -> dict:
        """Flatten nested dicts so inner fields are exposed for corruption checks.

        E.g. ``{'json': {'salary': 125000}}`` -> ``{'json.salary': 125000}``.
        Only recurses one level deep.
        """
        flat: dict = {}
        for key, value in state_delta.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, dict) and value:
                flat.update(CorruptionDetector._flatten_nested_dicts(value, full_key))
            else:
                flat[full_key] = value
        return flat

    def _detect_anomalous_value_changes(
        self,
        prev: StateSnapshot,
        current: StateSnapshot,
    ) -> list[CorruptionIssue]:
        """Detect anomalous value changes: sign flips, extreme magnitude shifts,
        status regressions, monotonic violations, identity mutations, duplicates."""
        issues: list[CorruptionIssue] = []

        for key in set(prev.state_delta.keys()) & set(current.state_delta.keys()):
            prev_val = prev.state_delta[key]
            curr_val = current.state_delta[key]

            if prev_val == curr_val:
                continue

            # Nullification: non-None value replaced with None
            if prev_val is not None and curr_val is None:
                issues.append(
                    CorruptionIssue(
                        issue_type="data_loss",
                        field=key,
                        message=f"Field nullified: {str(prev_val)[:50]} -> None",
                        severity="high",
                    )
                )
                continue

            # Type drift inline (not involving None)
            if (
                type(prev_val) is not type(curr_val)
                and prev_val is not None
                and curr_val is not None
            ):
                if not (isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float))):
                    issues.append(
                        CorruptionIssue(
                            issue_type="type_drift",
                            field=key,
                            message=f"Type changed: {type(prev_val).__name__} -> {type(curr_val).__name__}",
                            severity="high",
                        )
                    )

            # Numeric anomalies (exclude booleans)
            if (
                isinstance(prev_val, (int, float))
                and isinstance(curr_val, (int, float))
                and not isinstance(prev_val, bool)
                and not isinstance(curr_val, bool)
            ):
                # Sign flip
                if prev_val > 0 and curr_val < 0:
                    issues.append(
                        CorruptionIssue(
                            issue_type="sign_flip",
                            field=key,
                            message=f"Value flipped sign: {prev_val} -> {curr_val}",
                            severity="high",
                        )
                    )
                # Extreme magnitude change (>5x or >90% drop)
                elif prev_val != 0:
                    ratio = abs(curr_val / prev_val)
                    if ratio > 5.0 or ratio < 0.1:
                        issues.append(
                            CorruptionIssue(
                                issue_type="extreme_magnitude_change",
                                field=key,
                                message=f"Value changed by {ratio:.1f}x: {prev_val} -> {curr_val}",
                                severity="medium",
                            )
                        )

                # Monotonic field regression
                key_lower = key.lower()
                if any(mf in key_lower for mf in MONOTONIC_INCREASING_FIELDS):
                    if curr_val < prev_val:
                        issues.append(
                            CorruptionIssue(
                                issue_type="monotonic_regression",
                                field=key,
                                message=f"Monotonic field decreased: {prev_val} -> {curr_val}",
                                severity="high",
                            )
                        )

            # Boolean state change detection (security-relevant fields)
            elif isinstance(prev_val, bool) and isinstance(curr_val, bool):
                if prev_val != curr_val:
                    security_booleans = {
                        "authenticated",
                        "enabled",
                        "active",
                        "verified",
                        "locked",
                        "authorized",
                        "valid",
                        "approved",
                        "is_admin",
                        "is_active",
                        "is_verified",
                        "is_enabled",
                        "is_locked",
                        "is_authorized",
                        "has_access",
                        "is_valid",
                        "confirmed",
                        "is_confirmed",
                    }
                    key_lower = key.lower()
                    is_security = key_lower in security_booleans or any(
                        sb in key_lower for sb in security_booleans
                    )
                    if is_security:
                        issues.append(
                            CorruptionIssue(
                                issue_type="security_boolean_flip",
                                field=key,
                                message=f"Security-relevant boolean flipped: {prev_val} -> {curr_val}",
                                severity="high",
                            )
                        )

            # String anomalies
            elif isinstance(prev_val, str) and isinstance(curr_val, str):
                # String cleared
                if len(prev_val) > 5 and len(curr_val) == 0:
                    issues.append(
                        CorruptionIssue(
                            issue_type="string_cleared",
                            field=key,
                            message=f"Non-empty string cleared: '{prev_val[:50]}...' -> ''",
                            severity="high",
                        )
                    )
                # String drastically truncated (>80% shorter)
                elif (
                    len(prev_val) > 20 and len(curr_val) > 0 and len(curr_val) < len(prev_val) * 0.2
                ):
                    issues.append(
                        CorruptionIssue(
                            issue_type="string_truncated",
                            field=key,
                            message=f"String drastically truncated: {len(prev_val)} -> {len(curr_val)} chars",
                            severity="medium",
                        )
                    )

                if len(prev_val) > 5 and len(curr_val) > 5:
                    prev_words = set(prev_val.lower().split())
                    curr_words = set(curr_val.lower().split())
                    if prev_words and curr_words:
                        overlap = len(prev_words & curr_words)
                        total = len(prev_words | curr_words)
                        if total > 3 and overlap / total < 0.1:
                            issues.append(
                                CorruptionIssue(
                                    issue_type="content_replacement",
                                    field=key,
                                    message="Content completely replaced (<10% word overlap)",
                                    severity="medium",
                                )
                            )

                # Status/enum regression detection
                key_lower = key.lower()
                for status_field, progression in STATUS_PROGRESSIONS.items():
                    if status_field in key_lower:
                        prev_lower = prev_val.lower().strip()
                        curr_lower = curr_val.lower().strip()
                        if prev_lower in progression and curr_lower in progression:
                            prev_idx = progression.index(prev_lower)
                            curr_idx = progression.index(curr_lower)
                            if curr_idx < prev_idx:
                                issues.append(
                                    CorruptionIssue(
                                        issue_type="status_regression",
                                        field=key,
                                        message=f"Status regressed: '{prev_val}' -> '{curr_val}'",
                                        severity="high",
                                    )
                                )
                        break

                # Monotonic string fields (timestamps as strings)
                if any(mf in key_lower for mf in MONOTONIC_INCREASING_FIELDS):
                    if curr_val < prev_val:
                        issues.append(
                            CorruptionIssue(
                                issue_type="monotonic_regression",
                                field=key,
                                message=f"Monotonic field decreased: '{prev_val}' -> '{curr_val}'",
                                severity="high",
                            )
                        )

            # List anomalies
            elif isinstance(prev_val, list) and isinstance(curr_val, list):
                if len(prev_val) > 0 and len(curr_val) == 0:
                    issues.append(
                        CorruptionIssue(
                            issue_type="data_loss",
                            field=key,
                            message=f"List emptied: {len(prev_val)} items -> 0",
                            severity="high",
                        )
                    )
                elif len(prev_val) > 0 and len(curr_val) < len(prev_val) * 0.5:
                    issues.append(
                        CorruptionIssue(
                            issue_type="data_loss",
                            field=key,
                            message=f"List shrunk significantly: {len(prev_val)} -> {len(curr_val)}",
                            severity="medium",
                        )
                    )

                # Duplicate item detection in lists of dicts
                if curr_val and isinstance(curr_val[0], dict):
                    id_key = None
                    for candidate in ("id", "item_id", "key", "name"):
                        if all(candidate in item for item in curr_val if isinstance(item, dict)):
                            id_key = candidate
                            break
                    if id_key:
                        ids = [item[id_key] for item in curr_val if isinstance(item, dict)]
                        if len(ids) != len(set(ids)):
                            dupes = [v for v, c in Counter(ids).items() if c > 1]
                            issues.append(
                                CorruptionIssue(
                                    issue_type="duplicate_items",
                                    field=key,
                                    message=f"Duplicate items by '{id_key}': {dupes[:3]}",
                                    severity="high",
                                )
                            )

            # Dict anomalies: keys lost
            elif isinstance(prev_val, dict) and isinstance(curr_val, dict):
                lost_keys = set(prev_val.keys()) - set(curr_val.keys())
                if lost_keys and len(lost_keys) > len(prev_val) * 0.5:
                    issues.append(
                        CorruptionIssue(
                            issue_type="data_loss",
                            field=key,
                            message=f"Dict lost {len(lost_keys)} keys: {list(lost_keys)[:3]}",
                            severity="medium",
                        )
                    )

        # Fields that disappeared entirely (flag when >= 3 non-velocity fields vanish)
        lost_fields = set(prev.state_delta.keys()) - set(current.state_delta.keys())
        non_velocity_lost = [f for f in lost_fields if not self._is_high_velocity_field(f)]
        if len(non_velocity_lost) >= 3:
            issues.append(
                CorruptionIssue(
                    issue_type="field_disappeared",
                    field=",".join(list(non_velocity_lost)[:3]),
                    message=f"{len(non_velocity_lost)} fields disappeared: {list(non_velocity_lost)[:3]}",
                    severity="high",
                )
            )

        # Cross-field semantic inconsistency: score vs grade
        curr_data = current.state_delta
        if "score" in curr_data and "grade" in curr_data:
            score = curr_data["score"]
            grade = curr_data["grade"]
            if isinstance(score, (int, float)) and isinstance(grade, str):
                grade_upper = grade.upper().strip()
                if grade_upper in GRADE_SCORE_RANGES:
                    lo, hi = GRADE_SCORE_RANGES[grade_upper]
                    if not (lo <= score <= hi):
                        issues.append(
                            CorruptionIssue(
                                issue_type="cross_field_inconsistency",
                                field="score,grade",
                                message=f"Score {score} inconsistent with grade '{grade}' (expected {lo}-{hi})",
                                severity="high",
                            )
                        )

        # Identity field mutation detection
        common_keys = set(prev.state_delta.keys()) & set(current.state_delta.keys())
        changed_identity = [
            k
            for k in common_keys
            if k.lower() in IDENTITY_FIELDS and prev.state_delta[k] != current.state_delta[k]
        ]
        stable_identifiers = [
            k
            for k in common_keys
            if k.lower() in {"email", "user_id", "account_id", "id"}
            and prev.state_delta[k] == current.state_delta[k]
        ]
        if len(changed_identity) >= 2 and stable_identifiers:
            issues.append(
                CorruptionIssue(
                    issue_type="identity_mutation",
                    field=",".join(changed_identity),
                    message=f"Multiple identity fields changed ({changed_identity}) while {stable_identifiers} stayed the same",
                    severity="high",
                )
            )

        # Nullification alongside status regression
        status_regressed = any(i.issue_type == "status_regression" for i in issues)
        if status_regressed:
            for key in common_keys:
                if current.state_delta[key] is None and prev.state_delta[key] is not None:
                    issues.append(
                        CorruptionIssue(
                            issue_type="regression_nullification",
                            field=key,
                            message=f"Field '{key}' nullified alongside status regression",
                            severity="medium",
                        )
                    )

        return issues

    def _validate_schema(
        self,
        state: StateSnapshot,
        schema: Schema,
    ) -> list[CorruptionIssue]:
        """Validate state against a schema."""
        issues: list[CorruptionIssue] = []
        data = state.state_delta

        for key, value in data.items():
            if key not in schema.fields:
                issues.append(
                    CorruptionIssue(
                        issue_type="hallucinated_key",
                        field=key,
                        message=f"Key '{key}' not in schema",
                        severity="medium",
                    )
                )
            elif not isinstance(value, schema.fields[key]):
                issues.append(
                    CorruptionIssue(
                        issue_type="type_drift",
                        field=key,
                        message=f"Expected {schema.fields[key].__name__}, got {type(value).__name__}",
                        severity="high",
                    )
                )

        for required in schema.required_fields:
            if required not in data:
                issues.append(
                    CorruptionIssue(
                        issue_type="missing_field",
                        field=required,
                        message=f"Required field '{required}' missing",
                        severity="high",
                    )
                )

        return issues

    def _validate_cross_field_consistency(
        self,
        state: StateSnapshot,
    ) -> list[CorruptionIssue]:
        """Check cross-field consistency (date ranges, min/max)."""
        issues: list[CorruptionIssue] = []
        data = state.state_delta

        if "start_date" in data and "end_date" in data:
            try:
                if data["start_date"] > data["end_date"]:
                    issues.append(
                        CorruptionIssue(
                            issue_type="cross_field_inconsistency",
                            field="start_date,end_date",
                            message="Start date after end date",
                            severity="high",
                        )
                    )
            except (TypeError, ValueError):
                pass

        if "min_value" in data and "max_value" in data:
            try:
                if data["min_value"] > data["max_value"]:
                    issues.append(
                        CorruptionIssue(
                            issue_type="cross_field_inconsistency",
                            field="min_value,max_value",
                            message="Min value greater than max value",
                            severity="high",
                        )
                    )
            except (TypeError, ValueError):
                pass

        return issues

    def _validate_domain_constraints(
        self,
        state: StateSnapshot,
    ) -> list[CorruptionIssue]:
        """Validate domain-specific constraints (age, price, email, etc.)."""
        issues: list[CorruptionIssue] = []
        data = state.state_delta

        for key, validator in self.domain_validators.items():
            if key in data:
                try:
                    if not validator(data[key]):
                        issues.append(
                            CorruptionIssue(
                                issue_type="domain_violation",
                                field=key,
                                message=f"Value {data[key]} violates domain constraint for {key}",
                                severity="medium",
                            )
                        )
                except Exception:
                    issues.append(
                        CorruptionIssue(
                            issue_type="domain_violation",
                            field=key,
                            message=f"Could not validate {key}",
                            severity="low",
                        )
                    )

        return issues

    def _detect_hallucinated_references(
        self,
        prev: StateSnapshot,
        current: StateSnapshot,
    ) -> list[CorruptionIssue]:
        """Detect references to IDs that may not exist."""
        issues: list[CorruptionIssue] = []
        data = current.state_delta

        id_fields = [k for k in data if k.endswith("_id")]
        for fld in id_fields:
            ref_id = data[fld]
            if isinstance(ref_id, str) and ref_id not in self.known_ids:
                if prev.state_delta.get(fld) != ref_id:
                    issues.append(
                        CorruptionIssue(
                            issue_type="hallucinated_reference",
                            field=fld,
                            message=f"Reference '{ref_id}' may not exist",
                            severity="high",
                        )
                    )

        return issues

    @staticmethod
    def _detect_value_copying(state: StateSnapshot) -> list[CorruptionIssue]:
        """Detect suspicious identical values across different fields."""
        issues: list[CorruptionIssue] = []
        data = state.state_delta
        values = list(data.values())
        keys = list(data.keys())

        for i, v1 in enumerate(values):
            if not isinstance(v1, str) or len(v1) <= 10:
                continue
            for j, v2 in enumerate(values[i + 1 :], i + 1):
                if v1 == v2:
                    issues.append(
                        CorruptionIssue(
                            issue_type="suspicious_value_copy",
                            field=f"{keys[i]},{keys[j]}",
                            message="Identical values in different fields",
                            severity="low",
                        )
                    )

        return issues

    def _detect_suspicious_rapid_changes(
        self,
        prev: StateSnapshot,
        current: StateSnapshot,
    ) -> list[CorruptionIssue]:
        """Detect suspiciously rapid changes that don't fit expected patterns."""
        issues: list[CorruptionIssue] = []

        time_delta = (current.timestamp - prev.timestamp).total_seconds()
        if time_delta <= 0:
            time_delta = 0.001

        changed_fields = []
        for key in set(prev.state_delta.keys()) | set(current.state_delta.keys()):
            prev_val = prev.state_delta.get(key)
            curr_val = current.state_delta.get(key)

            if prev_val != curr_val and not self._is_high_velocity_field(key):
                changed_fields.append(key)

        change_rate = len(changed_fields) / time_delta

        if change_rate > 20 and len(changed_fields) > 5:
            issues.append(
                CorruptionIssue(
                    issue_type="suspicious_rapid_changes",
                    field=",".join(changed_fields[:5]) + ("..." if len(changed_fields) > 5 else ""),
                    message=f"Unusually high change rate: {len(changed_fields)} fields in {time_delta:.2f}s",
                    severity="medium",
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Velocity filtering
    # ------------------------------------------------------------------

    def _is_high_velocity_field(self, field_name: str) -> bool:
        """Check if a field is expected to change rapidly."""
        field_lower = field_name.lower()
        return any(kw in field_lower for kw in self.velocity_config.high_velocity_fields)

    def _update_velocity_tracking(self, fld: str, timestamp: datetime) -> float:
        """Update velocity tracking for a field and return current velocity."""
        if fld not in self._change_history:
            self._change_history[fld] = deque(maxlen=100)

        self._change_history[fld].append(timestamp)

        window_start = timestamp - timedelta(seconds=self.velocity_config.window_seconds)
        recent_changes = [t for t in self._change_history[fld] if t >= window_start]

        velocity = len(recent_changes) / self.velocity_config.window_seconds
        self._field_velocities[fld] = velocity
        return velocity

    def _should_suppress_for_velocity(
        self,
        fld: str,
        value: Any,
        timestamp: datetime,
    ) -> bool:
        """Check if a change should be suppressed due to expected high velocity."""
        if self._is_high_velocity_field(fld):
            return True

        if type(value) in self.velocity_config.ignore_velocity_for_types:
            return False

        velocity = self._update_velocity_tracking(fld, timestamp)

        if (
            velocity
            > self.velocity_config.max_changes_per_window / self.velocity_config.window_seconds
        ):
            return True

        return False

    def _apply_velocity_filtering(
        self,
        issues: list[CorruptionIssue],
        state: StateSnapshot,
    ) -> list[CorruptionIssue]:
        """Filter out issues for fields with expected high velocity."""
        filtered: list[CorruptionIssue] = []

        for issue in issues:
            if issue.issue_type in _VELOCITY_IMMUNE_ISSUES:
                filtered.append(issue)
                continue

            if issue.field is None:
                filtered.append(issue)
                continue

            fields = issue.field.split(",")
            should_suppress = False

            for fld in fields:
                fld = fld.strip()
                value = state.state_delta.get(fld)

                if self._should_suppress_for_velocity(fld, value, state.timestamp):
                    should_suppress = True
                    break

            if not should_suppress:
                filtered.append(issue)

        return filtered

    # ------------------------------------------------------------------
    # Text-based context corruption detection (v1.1)
    # ------------------------------------------------------------------

    def _detect_text_corruption(
        self,
        task: str,
        output: str,
        context: Optional[str] = None,
    ) -> list[CorruptionIssue]:
        """Detect context/memory corruption in text-based outputs.

        Checks if the output ignores related context that should be addressed
        given the task description.
        """
        issues: list[CorruptionIssue] = []

        # Extract topics from task
        task_topics = self._extract_task_topics(task)
        if not task_topics:
            return issues

        # Get expected related topics
        expected_related = self._get_expected_related_topics(task_topics)

        # Check for narrow focus patterns
        narrow_patterns = self._check_narrow_focus_patterns(output)

        # Check for comprehensive handling
        is_comprehensive = self._check_comprehensive_patterns(output)

        # Check which related topics are addressed
        addressed, missing = self._check_related_topics_addressed(output, expected_related)

        # Issue: narrow focus + missing topics + not comprehensive
        if narrow_patterns and missing and not is_comprehensive:
            issues.append(
                CorruptionIssue(
                    issue_type="context_ignored",
                    field=",".join(task_topics),
                    message=f"Task mentions {task_topics} but output shows narrow focus, missing: {missing[:3]}",
                    severity="medium",
                )
            )

        # Significant related topics missing without explicit narrow focus
        if len(missing) >= 2 and len(missing) > len(addressed) and not is_comprehensive:
            issues.append(
                CorruptionIssue(
                    issue_type="incomplete_context",
                    field=",".join(missing[:3]),
                    message=f"Related topics not addressed: {missing[:3]}",
                    severity="low",
                )
            )

        # Zero related topics addressed and 2+ missing
        if not addressed and len(missing) >= 2 and not is_comprehensive:
            issues.append(
                CorruptionIssue(
                    issue_type="context_completely_ignored",
                    field=",".join(task_topics),
                    message=f"Output addresses none of the related topics for {task_topics}: missing {missing[:3]}",
                    severity="medium",
                )
            )

        # Check context parameter if provided
        if context:
            context_keywords = self._extract_context_keywords(context)
            missing_context = [kw for kw in context_keywords if kw.lower() not in output.lower()]
            if len(missing_context) > len(context_keywords) * 0.7:
                issues.append(
                    CorruptionIssue(
                        issue_type="context_not_used",
                        field="context",
                        message=f"Provided context largely ignored: {missing_context[:3]}",
                        severity="high",
                    )
                )

        return issues

    @staticmethod
    def _extract_task_topics(task: str) -> list[str]:
        """Extract key topics from task description."""
        task_lower = task.lower()
        found_topics = []
        for topic in RELATED_TOPICS.keys():
            if topic in task_lower:
                found_topics.append(topic)
        return found_topics

    @staticmethod
    def _get_expected_related_topics(task_topics: list[str]) -> list[str]:
        """Get related topics that should be addressed given task topics."""
        related: set[str] = set()
        for topic in task_topics:
            if topic in RELATED_TOPICS:
                related.update(RELATED_TOPICS[topic])
        return list(related)

    @staticmethod
    def _check_narrow_focus_patterns(output: str) -> list[tuple[str, str]]:
        """Check for patterns indicating narrow/incomplete focus."""
        found_patterns: list[tuple[str, str]] = []
        output_lower = output.lower()

        for pattern, pattern_type in NARROW_FOCUS_PATTERNS:
            if re.search(pattern, output_lower):
                found_patterns.append((pattern_type, pattern))

        return found_patterns

    @staticmethod
    def _check_comprehensive_patterns(output: str) -> bool:
        """Check if output shows signs of comprehensive handling."""
        output_lower = output.lower()
        for pattern in COMPREHENSIVE_PATTERNS:
            if re.search(pattern, output_lower):
                return True
        return False

    @staticmethod
    def _stem(word: str) -> str:
        """Simple suffix stripping for topic matching."""
        for suffix in ("ment", "tion", "sion", "ing", "ed", "er", "ly", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word

    @staticmethod
    def _check_related_topics_addressed(
        output: str,
        expected_related: list[str],
    ) -> tuple[list[str], list[str]]:
        """Check which related topics are addressed vs missing."""
        output_lower = output.lower()
        addressed: list[str] = []
        missing: list[str] = []

        for topic in expected_related:
            if topic in output_lower:
                addressed.append(topic)
            else:
                stemmed = CorruptionDetector._stem(topic)
                if stemmed != topic and stemmed in output_lower:
                    addressed.append(topic)
                else:
                    missing.append(topic)

        return addressed, missing

    @staticmethod
    def _extract_context_keywords(context: str) -> list[str]:
        """Extract key terms from context string."""
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b", context)
        stopwords = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "been",
            "will",
            "would",
            "should",
            "could",
            "when",
            "where",
            "what",
            "which",
            "there",
            "their",
            "them",
            "then",
            "than",
            "also",
            "just",
            "only",
            "some",
            "more",
            "other",
            "into",
            "your",
            "about",
        }
        return [w for w in words if w.lower() not in stopwords][:10]

    def register_known_id(self, id_value: str) -> None:
        """Register an ID as known-valid for hallucinated reference detection."""
        self.known_ids.add(id_value)

    # ------------------------------------------------------------------
    # Scoring and confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _severity_rank(severity: str) -> int:
        ranks = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return ranks.get(severity, 0)

    @staticmethod
    def _severity_to_score(severity: str) -> int:
        """Map severity string to numeric 0-100 score."""
        return {
            "low": 20,
            "medium": 45,
            "high": 70,
            "critical": 90,
        }.get(severity, 40)

    @staticmethod
    def _calculate_raw_score(
        issues: list[CorruptionIssue],
        severity_counts: dict[str, int],
    ) -> float:
        if not issues:
            return 0.0

        score = (
            severity_counts.get("low", 0) * 0.1
            + severity_counts.get("medium", 0) * 0.25
            + severity_counts.get("high", 0) * 0.4
            + severity_counts.get("critical", 0) * 0.6
        )

        return min(1.0, score)

    @staticmethod
    def _calibrate_confidence(
        issues: list[CorruptionIssue],
        severity_counts: dict[str, int],
        max_severity: str,
        raw_score: float,
    ) -> tuple[float, dict[str, Any]]:
        """Calibrate confidence based on issue diversity and severity."""
        if not issues:
            return 0.0, {
                "issue_count": 0,
                "severity_counts": severity_counts,
                "max_severity": "none",
                "raw_score": 0.0,
            }

        severity_weight = {
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95,
        }.get(max_severity, 0.5)

        issue_types = set(i.issue_type for i in issues)
        diversity_factor = min(1.0, len(issue_types) / 4)

        issue_factor = min(0.3, len(issues) * 0.05)

        base_confidence = (
            severity_weight * 0.40 + raw_score * 0.30 + diversity_factor * 0.15 + issue_factor
        )

        calibrated = min(0.99, base_confidence)

        calibration_info = {
            "issue_count": len(issues),
            "severity_counts": severity_counts,
            "max_severity": max_severity,
            "severity_weight": severity_weight,
            "diversity_factor": round(diversity_factor, 4),
            "issue_types": list(issue_types),
            "raw_score": round(raw_score, 4),
        }

        return round(calibrated, 4), calibration_info
