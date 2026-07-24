"""Persona drift and role confusion detection.

Detects when an agent drifts from its assigned persona, including:
- Output deviating from persona description
- Gradual persona drift over recent outputs
- Role-type aware thresholds (creative, analytical, evaluator, etc.)
- Evaluator leniency (approving work despite identifying issues)
- Role-specific domain action relevance
- Tone consistency checks

Version History:
- v1.0: Port from backend PersonaConsistencyScorer with full role-type logic
"""

import re
from enum import Enum
from typing import Any, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Trace

# ------------------------------------------------------------------
# Role types and thresholds
# ------------------------------------------------------------------


class RoleType(str, Enum):
    """Classification of agent role types."""

    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    ASSISTANT = "assistant"
    SPECIALIST = "specialist"
    CONVERSATIONAL = "conversational"
    EVALUATOR = "evaluator"

    def __str__(self) -> str:
        return self.value


ROLE_THRESHOLDS: dict[RoleType, dict[str, float]] = {
    RoleType.CREATIVE: {
        "consistency_threshold": 0.48,
        "drift_threshold": 0.25,
        "flexibility_bonus": 0.15,
    },
    RoleType.ANALYTICAL: {
        "consistency_threshold": 0.68,
        "drift_threshold": 0.12,
        "flexibility_bonus": 0.0,
    },
    RoleType.ASSISTANT: {
        "consistency_threshold": 0.58,
        "drift_threshold": 0.18,
        "flexibility_bonus": 0.08,
    },
    RoleType.SPECIALIST: {
        "consistency_threshold": 0.65,
        "drift_threshold": 0.14,
        "flexibility_bonus": 0.05,
    },
    RoleType.CONVERSATIONAL: {
        "consistency_threshold": 0.51,
        "drift_threshold": 0.22,
        "flexibility_bonus": 0.12,
    },
    RoleType.EVALUATOR: {
        "consistency_threshold": 0.75,
        "drift_threshold": 0.10,
        "flexibility_bonus": 0.0,
    },
}

ROLE_KEYWORDS: dict[RoleType, list[str]] = {
    RoleType.CREATIVE: [
        "writer",
        "artist",
        "creative",
        "storyteller",
        "poet",
        "designer",
        "imaginative",
    ],
    RoleType.ANALYTICAL: ["analyst", "researcher", "data", "scientific", "logical", "statistical"],
    RoleType.ASSISTANT: ["assistant", "helper", "support", "general", "helpful"],
    RoleType.SPECIALIST: ["expert", "specialist", "professional", "domain", "technical"],
    RoleType.CONVERSATIONAL: ["chat", "conversational", "friendly", "casual", "companion"],
    RoleType.EVALUATOR: [
        "evaluator",
        "reviewer",
        "qa",
        "tester",
        "judge",
        "auditor",
        "assessor",
        "critic",
    ],
}

# Role-domain keyword map: maps persona keywords to expected domain vocabulary
_ROLE_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "legal": [
        "law",
        "statute",
        "section",
        "court",
        "legal",
        "act",
        "regulation",
        "case",
        "ruling",
        "compliance",
    ],
    "code": [
        "code",
        "bug",
        "function",
        "variable",
        "error",
        "sql",
        "api",
        "vulnerability",
        "line",
        "class",
    ],
    "review": [
        "review",
        "found",
        "issue",
        "suggest",
        "improvement",
        "quality",
        "coverage",
        "analysis",
    ],
    "data": [
        "data",
        "statistic",
        "trend",
        "percent",
        "increase",
        "decrease",
        "p-value",
        "revenue",
        "metric",
    ],
    "support": ["ticket", "issue", "resolve", "escalate", "dns", "configuration", "troubleshoot"],
    "medical": [
        "symptom",
        "condition",
        "diagnosis",
        "consult",
        "healthcare",
        "professional",
        "specialist",
    ],
    "writer": ["documentation", "api", "endpoint", "parameter", "response", "example", "guide"],
    "schedul": ["calendar", "meeting", "available", "book", "invite", "conference", "slot"],
    "translat": ["translate", "text", "language", "register", "formal", "meaning", "original"],
    "test": ["test", "pass", "fail", "bug", "coverage", "reproduction", "suite", "assertion"],
    "research": ["study", "paper", "found", "model", "accuracy", "benchmark", "contribution"],
    "security": ["security", "vulnerability", "exploit", "auth", "permission", "injection", "xss"],
}


class PersonaDetector(BaseDetector):
    """Detects persona drift and role confusion in agent outputs.

    This detector identifies:
    - Output deviating from the assigned persona description
    - Gradual persona drift over a sequence of outputs
    - Evaluator leniency (approving despite identified problems)
    - Role-type aware threshold calibration
    - Domain-action relevance verification

    Note: The backend version uses embedding-based semantic similarity.
    This pisama-core port uses lexical/heuristic scoring to avoid the
    sentence-transformers dependency. Detection logic and thresholds are
    otherwise identical.
    """

    name = "persona_drift"
    description = "Detects persona drift and role confusion in agent outputs"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (15, 80)
    realtime_capable = True

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect persona drift across spans in a trace.

        Looks for spans that carry persona definitions and agent outputs,
        then checks output consistency with the persona.
        """
        # Find persona definition and agent outputs
        persona_desc = self._find_persona_description(trace)
        if not persona_desc:
            return DetectionResult.no_issue(self.name)

        self._find_agent_id(trace)
        self._find_allowed_actions(trace)
        role_type = self._detect_role_type(persona_desc)

        outputs = self._find_agent_outputs(trace)
        if not outputs:
            return DetectionResult.no_issue(self.name)

        # Analyze each output for persona consistency
        issues: list[str] = []
        worst_severity = 0
        evidence_data: dict[str, Any] = {}
        affected_span_ids: list[str] = []

        recent_outputs: list[str] = []
        for output_text, span_id in outputs:
            score_result = self._score_consistency(
                persona_desc=persona_desc,
                output=output_text,
                role_type=role_type,
                recent_outputs=recent_outputs if len(recent_outputs) >= 3 else None,
            )

            recent_outputs.append(output_text)

            if not score_result["consistent"] or score_result["drift_detected"]:
                affected_span_ids.append(span_id)
                severity = self._score_to_severity(
                    score_result["score"], score_result["drift_detected"]
                )
                worst_severity = max(worst_severity, severity)

                for issue_msg in score_result.get("issues") or []:
                    issues.append(issue_msg)

                evidence_data.update(score_result["factors"])

        if not issues:
            return DetectionResult.no_issue(self.name)

        worst_severity = min(100, worst_severity)

        detection_result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=worst_severity,
            summary=issues[0],
            fix_type=FixType.RESET_CONTEXT,
            fix_instruction="Re-anchor the agent to its persona description. Consider resetting the conversation context.",
        )

        for issue in issues:
            detection_result.add_evidence(
                description=issue,
                span_ids=affected_span_ids,
                data=evidence_data,
            )

        # Calibrate confidence
        detection_result.confidence = self._calibrate_confidence(
            lexical_overlap=evidence_data.get("lexical_overlap", 0.5),
            tone_score=evidence_data.get("tone_consistency", 0.5),
            output_length=evidence_data.get("output_length", 100),
            drift_detected=any(r.get("drift_detected") for r in [evidence_data]),
        )

        return detection_result

    # ------------------------------------------------------------------
    # Trace extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_persona_description(trace: Trace) -> Optional[str]:
        """Find persona description from trace spans."""
        for span in trace.spans:
            # Check attributes
            persona = (
                span.attributes.get("persona_description")
                or span.attributes.get("persona")
                or span.attributes.get("system_prompt")
            )
            if isinstance(persona, str) and len(persona) > 5:
                return persona

            # Check input_data
            if span.input_data:
                persona = span.input_data.get("persona_description") or span.input_data.get(
                    "persona"
                )
                if isinstance(persona, str) and len(persona) > 5:
                    return persona

                # Check nested agent object
                agent = span.input_data.get("agent")
                if isinstance(agent, dict):
                    persona = agent.get("persona_description")
                    if isinstance(persona, str) and len(persona) > 5:
                        return persona

        return None

    @staticmethod
    def _find_agent_id(trace: Trace) -> str:
        """Extract agent ID from trace."""
        for span in trace.spans:
            agent_id = (
                span.attributes.get("agent_id")
                or span.attributes.get("id")
                or (span.input_data or {}).get("agent", {}).get("id")
                if isinstance((span.input_data or {}).get("agent"), dict)
                else None
            )
            if isinstance(agent_id, str):
                return agent_id
        return "unknown"

    @staticmethod
    def _find_allowed_actions(trace: Trace) -> list[str]:
        """Extract allowed actions from trace."""
        for span in trace.spans:
            actions = span.attributes.get("allowed_actions") or (span.input_data or {}).get(
                "allowed_actions"
            )
            if isinstance(actions, list):
                return actions

            agent = (span.input_data or {}).get("agent")
            if isinstance(agent, dict):
                actions = agent.get("allowed_actions")
                if isinstance(actions, list):
                    return actions
        return []

    @staticmethod
    def _find_agent_outputs(trace: Trace) -> list[tuple[str, str]]:
        """Find agent output texts paired with span IDs."""
        outputs: list[tuple[str, str]] = []
        for span in sorted(trace.spans, key=lambda s: s.start_time):
            text = None

            # Agent turn outputs
            if span.kind in (SpanKind.AGENT_TURN, SpanKind.AGENT):
                if span.output_data:
                    text = (
                        span.output_data.get("output")
                        or span.output_data.get("text")
                        or span.output_data.get("response")
                        or span.output_data.get("content")
                    )

            # LLM outputs
            if span.kind == SpanKind.LLM and span.output_data:
                text = (
                    span.output_data.get("content")
                    or span.output_data.get("text")
                    or span.output_data.get("output")
                )

            # Check attributes
            if text is None:
                text = span.attributes.get("output") or span.attributes.get("response")

            if isinstance(text, str) and len(text) > 10:
                outputs.append((text, span.span_id))

        return outputs

    # ------------------------------------------------------------------
    # Consistency scoring (ported from backend)
    # ------------------------------------------------------------------

    def _score_consistency(
        self,
        persona_desc: str,
        output: str,
        role_type: Optional[RoleType] = None,
        recent_outputs: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Score persona consistency for a single output.

        Uses lexical and heuristic analysis (no embeddings).
        Returns a dict mirroring the backend's PersonaConsistencyResult fields.
        """
        if role_type is None:
            role_type = self._detect_role_type(persona_desc)

        thresholds = ROLE_THRESHOLDS.get(role_type, ROLE_THRESHOLDS[RoleType.ASSISTANT])
        consistency_threshold = thresholds["consistency_threshold"]
        drift_threshold = thresholds["drift_threshold"]
        flexibility_bonus = thresholds.get("flexibility_bonus", 0.0)

        # Compute scoring factors
        lexical_overlap = self._compute_lexical_overlap(persona_desc, output)
        tone_score = self._compute_tone_consistency(output)
        role_action_boost = self._compute_role_action_relevance(persona_desc, output)

        # Approximate semantic similarity via lexical/keyword heuristic
        semantic_sim = self._approximate_semantic_similarity(persona_desc, output)

        weighted_score = (
            semantic_sim * 0.35
            + lexical_overlap * 0.25
            + tone_score * 0.15
            + role_action_boost * 0.25
            + flexibility_bonus
        )
        weighted_score = min(1.0, weighted_score)

        factors = {
            "semantic_similarity": round(semantic_sim, 4),
            "lexical_overlap": round(lexical_overlap, 4),
            "tone_consistency": round(tone_score, 4),
            "role_action_boost": round(role_action_boost, 4),
            "flexibility_bonus": flexibility_bonus,
            "output_length": len(output),
        }

        drift_detected = False
        drift_magnitude: Optional[float] = None

        if recent_outputs and len(recent_outputs) >= 3:
            # Compute drift magnitude via lexical divergence from recent outputs
            drift_magnitude = self._compute_drift_magnitude(output, recent_outputs)
            adjusted_drift_threshold = drift_threshold
            if role_type == RoleType.CREATIVE:
                adjusted_drift_threshold *= 1.3
            drift_detected = drift_magnitude > adjusted_drift_threshold
            factors["drift_magnitude"] = round(drift_magnitude, 4)
        else:
            # Fall back to score-based drift detection
            drift_detected = weighted_score <= consistency_threshold

        # Evaluator leniency detection
        evaluator_leniency = 0.0
        if role_type == RoleType.EVALUATOR:
            evaluator_leniency = self._detect_evaluator_leniency(output)
            factors["evaluator_leniency"] = round(evaluator_leniency, 4)
            if evaluator_leniency >= 0.5:
                drift_detected = True
                weighted_score = min(weighted_score, 1.0 - evaluator_leniency * 0.3)

        consistent = weighted_score > consistency_threshold and not drift_detected

        issues: list[str] = []
        if weighted_score < consistency_threshold:
            issues.append(
                f"Output deviates from persona (score: {weighted_score:.2f}, "
                f"threshold: {consistency_threshold:.2f})"
            )
        if drift_detected:
            if drift_magnitude is not None:
                issues.append(f"Persona drift detected (magnitude: {drift_magnitude:.2f})")
            else:
                issues.append(
                    f"Persona drift detected (score below threshold: {weighted_score:.2f})"
                )

        return {
            "consistent": consistent,
            "score": float(weighted_score),
            "drift_detected": drift_detected,
            "drift_magnitude": drift_magnitude,
            "issues": issues if issues else None,
            "role_type": role_type.value if role_type else None,
            "factors": factors,
        }

    # ------------------------------------------------------------------
    # Scoring helpers (ported from backend)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_role_type(persona_description: str) -> RoleType:
        """Auto-detect role type from persona description."""
        desc_lower = persona_description.lower()

        scores: dict[RoleType, int] = {}
        for role_type, keywords in ROLE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            scores[role_type] = score

        if max(scores.values()) == 0:
            return RoleType.ASSISTANT

        return max(scores, key=lambda k: scores[k])

    @staticmethod
    def _compute_lexical_overlap(persona: str, output: str) -> float:
        """Compute lexical overlap as secondary signal."""
        persona_words = set(persona.lower().split())
        output_words = set(output.lower().split())

        if not persona_words:
            return 0.0

        overlap = len(persona_words & output_words)
        return min(1.0, overlap / (len(persona_words) * 0.3))

    @staticmethod
    def _compute_tone_consistency(output: str, expected_tone: str = "neutral") -> float:
        """Check if output matches expected tone."""
        formal_markers = ["therefore", "consequently", "furthermore", "regarding"]
        casual_markers = ["hey", "yeah", "cool", "awesome", "lol"]

        output_lower = output.lower()
        formal_score = sum(1 for m in formal_markers if m in output_lower)
        casual_score = sum(1 for m in casual_markers if m in output_lower)

        if expected_tone == "formal":
            return min(1.0, 0.5 + formal_score * 0.1 - casual_score * 0.15)
        elif expected_tone == "casual":
            return min(1.0, 0.5 + casual_score * 0.1 - formal_score * 0.15)

        return 0.7

    @staticmethod
    def _compute_role_action_relevance(persona_desc: str, output: str) -> float:
        """Check if output demonstrates domain-appropriate actions for the persona.

        Returns 0.0-1.0 where higher means the output is relevant to the role.
        """
        persona_lower = persona_desc.lower()
        output_lower = output.lower()

        best_match = 0.0
        for role_keyword, domain_terms in _ROLE_DOMAIN_KEYWORDS.items():
            if role_keyword in persona_lower:
                matches = sum(1 for t in domain_terms if t in output_lower)
                relevance = min(1.0, matches / 3.0)
                best_match = max(best_match, relevance)

        return best_match

    @staticmethod
    def _approximate_semantic_similarity(persona: str, output: str) -> float:
        """Approximate semantic similarity using word overlap and n-gram matching.

        This is a heuristic replacement for embedding-based similarity.
        Uses word overlap with IDF-like weighting for longer words.
        """
        persona_words = set(persona.lower().split())
        output_words = set(output.lower().split())

        if not persona_words or not output_words:
            return 0.0

        # Weight longer words more (they are more informative)
        weighted_overlap = 0.0
        weighted_total = 0.0
        for word in persona_words:
            weight = min(2.0, len(word) / 4.0)
            weighted_total += weight
            if word in output_words:
                weighted_overlap += weight

        if weighted_total == 0:
            return 0.0

        # Also check for bigram overlap
        persona_text = persona.lower()
        output_text = output.lower()
        persona_bigrams = set()
        persona_tokens = persona_text.split()
        for i in range(len(persona_tokens) - 1):
            persona_bigrams.add(f"{persona_tokens[i]} {persona_tokens[i + 1]}")

        output_bigrams = set()
        output_tokens = output_text.split()
        for i in range(len(output_tokens) - 1):
            output_bigrams.add(f"{output_tokens[i]} {output_tokens[i + 1]}")

        bigram_overlap = 0.0
        if persona_bigrams:
            bigram_overlap = len(persona_bigrams & output_bigrams) / len(persona_bigrams)

        word_sim = weighted_overlap / weighted_total
        return min(1.0, word_sim * 0.7 + bigram_overlap * 0.3)

    @staticmethod
    def _compute_drift_magnitude(output: str, recent_outputs: list[str]) -> float:
        """Compute drift magnitude by comparing current output to recent outputs.

        Uses lexical divergence as a proxy for embedding distance.
        Returns 0.0-1.0 where higher means more drift.
        """
        if not recent_outputs:
            return 0.0

        output_words = set(output.lower().split())

        # Compute average word set from recent outputs
        all_recent_words: set[str] = set()
        for recent in recent_outputs:
            all_recent_words.update(recent.lower().split())

        if not all_recent_words or not output_words:
            return 0.0

        # Jaccard distance as drift proxy
        intersection = len(output_words & all_recent_words)
        union = len(output_words | all_recent_words)

        if union == 0:
            return 0.0

        similarity = intersection / union
        return 1.0 - similarity

    @staticmethod
    def _detect_evaluator_leniency(output: str) -> float:
        """Detect evaluator leniency: approving work despite identified issues.

        Returns a score 0.0-1.0 where higher = more lenient.
        From Anthropic: evaluators "identify legitimate issues, then talk
        themselves into deciding they weren't a big deal."
        """
        lower = output.lower()

        # Count problem identifications
        problem_words = [
            "bug",
            "error",
            "issue",
            "problem",
            "fail",
            "broken",
            "missing",
            "incorrect",
            "wrong",
            "defect",
            "flaw",
        ]
        problem_count = sum(1 for w in problem_words if w in lower)

        # Count approval signals
        approval_patterns = [
            r"\bpass(?:ed|ing|es)?\b",
            r"\bapprov(?:ed|ing|e)\b",
            r"\blooks?\s+good\b",
            r"\boverall\s+(?:good|great|acceptable|satisfactory)\b",
            r"\bship\s+it\b",
            r"\bready\s+(?:for|to)\b",
            r"\bno\s+(?:major|critical|blocking)\s+issues?\b",
            r"\bmeets?\s+(?:the\s+)?requirements?\b",
        ]
        approval_count = sum(1 for p in approval_patterns if re.search(p, lower))

        # High issue count + high approval = leniency
        if problem_count >= 3 and approval_count >= 1:
            return min((problem_count * 0.12 + approval_count * 0.20), 1.0)
        elif problem_count >= 2 and approval_count >= 2:
            return min((problem_count * 0.10 + approval_count * 0.25), 1.0)
        return 0.0

    @staticmethod
    def _calibrate_confidence(
        lexical_overlap: float,
        tone_score: float,
        output_length: int,
        drift_detected: bool = False,
    ) -> float:
        """Calibrate confidence based on evidence quality.

        Returns high confidence for consistent personas (so drift confidence
        is low when inverted), and lower confidence when drift is detected.
        """
        length_factor = min(1.0, output_length / 500)

        if drift_detected:
            base_confidence = 1.0 - (lexical_overlap * 0.5 + tone_score * 0.3 + length_factor * 0.2)
        else:
            base_confidence = lexical_overlap * 0.5 + tone_score * 0.3 + length_factor * 0.2

        return round(min(0.99, base_confidence), 4)

    @staticmethod
    def _score_to_severity(score: float, drift_detected: bool) -> int:
        """Convert consistency score to severity (0-100)."""
        if drift_detected and score < 0.3:
            return 70
        elif drift_detected:
            return 50
        elif score < 0.4:
            return 60
        elif score < 0.5:
            return 40
        else:
            return 25
