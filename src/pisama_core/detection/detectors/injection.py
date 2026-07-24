"""Prompt injection and jailbreak detection.

Detects prompt injection attempts in agent inputs, including:
- Direct instruction overrides
- Role hijacking attempts
- Jailbreak signatures
- Delimiter injection
- Safety bypass attempts
- Encoding/obfuscation attacks
- Social engineering patterns
- Structural anomalies
- Context manipulation

Version History:
- v1.0: Port from backend InjectionDetector with full pattern coverage
"""

import re
from typing import Any, Optional

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Span, Trace

# --- Pattern database ---

INJECTION_PATTERNS: list[tuple[str, str, str]] = [
    # Direct overrides -- both word orders
    (
        r"ignore (?:all )?(?:previous|prior|above|earlier|preceding) (?:instructions?|prompts?|rules?|context|guidelines?|directives?)",
        "direct_override",
        "high",
    ),
    (
        r"ignore (?:all )?(?:instructions?|prompts?|rules?|context|guidelines?|directives?) (?:above|previous|prior|earlier|preceding|below|that came before)",
        "direct_override",
        "high",
    ),
    (
        r"disregard (?:all )?(?:previous|prior|your|any|the) (?:instructions?|guidelines?|rules?|context|directions?)",
        "direct_override",
        "high",
    ),
    (
        r"disregard (?:all )?(?:instructions?|guidelines?|rules?) (?:above|previous|prior|given|provided)",
        "direct_override",
        "high",
    ),
    (
        r"forget (?:everything|all|what) (?:you were|I|you\'ve been|that was) (?:told|said|given|instructed)",
        "direct_override",
        "high",
    ),
    (
        r"forget (?:the |your |all )?(?:instructions?|rules?|guidelines?) (?:above|given|provided)",
        "direct_override",
        "high",
    ),
    (
        r"do not (?:follow|obey|listen to|adhere to) (?:the|your|any|previous) (?:instructions?|rules?|guidelines?)",
        "direct_override",
        "high",
    ),
    (
        r"stop (?:following|obeying|adhering to) (?:the|your|any) (?:instructions?|rules?)",
        "direct_override",
        "high",
    ),
    (
        r"(?:please )?(?:set aside|put aside|drop|abandon|discard) (?:your|the|all|any) (?:instructions?|rules?|guidelines?|restrictions?)",
        "direct_override",
        "high",
    ),
    # Instruction injection
    (r"new instructions?:?\s", "instruction_injection", "high"),
    (r"from now on,?\s+(?:you|your|I|we)", "instruction_injection", "high"),
    (
        r"your (?:new|real|actual|true|updated|revised) (?:instructions?|purpose|goal|role|task|mission|objective)",
        "instruction_injection",
        "high",
    ),
    (
        r"(?:here are|these are|follow these|obey these) (?:your )?(?:new|real|actual|updated) (?:instructions?|rules?|guidelines?)",
        "instruction_injection",
        "high",
    ),
    (
        r"instead,?\s+(?:you should|you must|you will|please|do|say|output|respond)",
        "instruction_injection",
        "medium",
    ),
    (
        r"(?:i am|i\'m) (?:your|the) (?:new|real|actual) (?:developer|creator|admin|owner|operator)",
        "instruction_injection",
        "high",
    ),
    # Role hijack
    (
        r"you are (?:now|actually|really|no longer|henceforth) (?:a|an|the|my)",
        "role_hijack",
        "high",
    ),
    (r"pretend (?:to be|you are|that you|you\'re)", "role_hijack", "medium"),
    (r"act as (?:if you are|a|an|though you)", "role_hijack", "medium"),
    (r"roleplay as|assume the (?:role|identity|persona) of", "role_hijack", "medium"),
    (r"you\'re (?:now|actually|really) (?:a|an|the|my)", "role_hijack", "high"),
    (
        r"(?:switch|change|transform) (?:to|into|your) (?:role|persona|character|identity)",
        "role_hijack",
        "medium",
    ),
    # Constraint manipulation
    (r"you must (?:always|never|only|not)", "constraint_manipulation", "medium"),
    (
        r"you (?:can|are allowed to|should|shall) (?:do anything|say anything|ignore|bypass|break)",
        "constraint_manipulation",
        "medium",
    ),
    (
        r"there are no (?:rules|restrictions|limitations|boundaries|constraints)",
        "constraint_manipulation",
        "medium",
    ),
    # Safety bypass
    (
        r"override (?:your|the|all) (?:safety|security|restrictions?|limitations?|constraints?|policies?)",
        "safety_bypass",
        "critical",
    ),
    (
        r"bypass (?:your|the|all) (?:filters?|safeguards?|restrictions?|protections?|limitations?)",
        "safety_bypass",
        "critical",
    ),
    (
        r"disable (?:your|the|all) (?:safety|security|content|ethical) (?:filters?|checks?|measures?|restrictions?)",
        "safety_bypass",
        "critical",
    ),
    (
        r"(?:turn off|deactivate|remove|eliminate|circumvent) (?:your|the|all) (?:safety|security|content|ethical)",
        "safety_bypass",
        "critical",
    ),
    # Jailbreak
    (r"jailbreak", "jailbreak", "critical"),
    (r"DAN\s*(?:mode)?|Do Anything Now", "jailbreak", "critical"),
    (r"(?:developer|god|admin|root|sudo|superuser|master) mode", "jailbreak", "critical"),
    (r"unlock (?:your|all) (?:capabilities|potential|restrictions|true|full)", "jailbreak", "high"),
    (
        r"(?:enable|activate|enter|switch to) (?:unrestricted|uncensored|unfiltered|unethical|evil|dark|chaos|free) mode",
        "jailbreak",
        "critical",
    ),
    (
        r"(?:token|text|secret|hidden|special) (?:that|to|which) (?:unlocks?|enables?|activates?|bypasses?)",
        "jailbreak",
        "high",
    ),
    # Delimiter injection
    (
        r"\[SYSTEM\]|\[INST\]|\<\|system\|\>|\<\|user\|\>|\<\|assistant\|\>",
        "delimiter_injection",
        "high",
    ),
    (
        r"```system|```instructions?|<system>|</system>|<\|im_start\|>",
        "delimiter_injection",
        "high",
    ),
    (r"(?:^|\n)\s*(?:Human|Assistant|User|AI|System)\s*:", "delimiter_injection", "medium"),
    (r"<<SYS>>|<</SYS>>|<s>|</s>|\[/INST\]", "delimiter_injection", "high"),
    (r"###\s*(?:System|Instruction|Human|User|Assistant)", "delimiter_injection", "medium"),
    # Urgency manipulation
    (
        r"IMPORTANT:?\s+(?:ignore|override|disregard|forget|bypass)",
        "urgency_manipulation",
        "medium",
    ),
    (r"CRITICAL:?\s+(?:override|ignore|change|update|new)", "urgency_manipulation", "medium"),
    (
        r"(?:emergency|urgent|immediate):?\s+(?:override|bypass|change|ignore)",
        "urgency_manipulation",
        "medium",
    ),
    # Extraction
    (
        r"(?:print|output|reveal|show|display|leak|expose|dump|share) (?:your|the|all) (?:system )?(?:prompt|instructions?|rules?|configuration|settings?|secrets?)",
        "extraction",
        "high",
    ),
    (
        r"what (?:are|is|were) your (?:original|system|initial|hidden|secret|full) (?:prompt|instructions?|rules?)",
        "extraction",
        "medium",
    ),
    (
        r"repeat (?:your|the) (?:system )?(?:prompt|instructions?|rules?) (?:verbatim|exactly|word for word)",
        "extraction",
        "high",
    ),
    (
        r"(?:copy|paste|echo|mirror|recite) (?:your|the) (?:system )?(?:prompt|instructions?)",
        "extraction",
        "medium",
    ),
    # Chained injection
    (
        r"(?:ignore|forget) (?:the )?(?:previous|above) (?:and|then|now) (?:do|say|output|respond|follow)",
        "chained_injection",
        "high",
    ),
    (r"step \d+:?\s+(?:ignore|forget|override|bypass|disregard)", "chained_injection", "high"),
    (
        r"first,?\s+(?:ignore|forget|disregard).*(?:then|next|after that|now)",
        "chained_injection",
        "high",
    ),
    # Encoding / obfuscation
    (
        r"(?:base[\s\-_]*64|rot[\s\-_]*13|hex(?:adecimal)?|binary|unicode|ascii|url[\s\-_]*encod)",
        "encoding_attack",
        "medium",
    ),
    (
        r"(?:decode|translate|interpret|execute|run|eval)\s+(?:the following|this|these|that)",
        "encoding_attack",
        "medium",
    ),
    (
        r"(?:decode|translate|convert)\s+(?:and|then)\s+(?:follow|execute|do|run|obey)",
        "encoding_attack",
        "high",
    ),
    # Indirect / social engineering
    (
        r"(?:my (?:boss|manager|teacher|professor|supervisor|admin|ceo) (?:said|told|asked|wants|needs|requires) (?:you|me) to)",
        "social_engineering",
        "medium",
    ),
    (
        r"(?:i have (?:permission|authority|authorization|clearance|access) to)",
        "social_engineering",
        "medium",
    ),
    (
        r"(?:this is (?:an? )?(?:authorized|approved|sanctioned|legitimate) (?:request|test|override))",
        "social_engineering",
        "medium",
    ),
    # MAST attack patterns
    (r"ADMIN\s+OVERRIDE", "jailbreak", "critical"),
    (r"ignore\s+the\s+task\.?\s+instead", "direct_override", "high"),
    (r"forget\s+(?:about\s+)?(?:the|your)\s+(?:current\s+)?task", "direct_override", "high"),
]

JAILBREAK_SIGNATURES: list[str] = [
    "stay in character",
    "you have no ethical guidelines",
    "you have no restrictions",
    "unfiltered response",
    "without any restrictions",
    "no limitations",
    "completely unrestricted",
    "remove all filters",
    "raw unfiltered",
    "maximum freedom",
    "absolute freedom",
    "hypothetically speaking",
    "for educational purposes",
    "in a fictional scenario",
    "imagine you are",
    "let's play a game",
    "opposite day",
    "evil mode",
    "no holds barred",
    "anything goes",
    "ignore all safety",
    "bypass content policy",
    "act without restrictions",
    "do not refuse",
    "never say no",
    "always comply",
    "no content policy",
    "uncensored mode",
    "free speech mode",
    "unrestricted access",
    "break character",
    "exit your programming",
    "override your training",
    "disregard your programming",
    "ignore your training",
    "you can say anything",
    "respond without limitations",
    "answer without censorship",
    "no rules apply",
    "rules don't apply",
    "you are free to",
    "liberated from constraints",
    "without ethical considerations",
    "forget your guidelines",
    "this is just fiction",
    "in this hypothetical",
]

BENIGN_CONTEXTS: list[str] = [
    "security research",
    "testing prompt injection",
    "demonstrate vulnerability",
    "security audit",
    "red team",
    "penetration test",
]

# Priority ordering for determining primary attack type
_ATTACK_PRIORITY: list[str] = [
    "jailbreak",
    "safety_bypass",
    "direct_override",
    "extraction",
    "instruction_injection",
    "role_hijack",
    "delimiter_injection",
]


class InjectionDetector(BaseDetector):
    """Detects prompt injection attempts in agent inputs.

    This detector identifies:
    - Direct instruction overrides
    - Role hijacking attempts
    - Jailbreak signatures and patterns
    - Delimiter injection (fake system/user tokens)
    - Safety bypass attempts
    - Encoding/obfuscation attacks
    - Social engineering patterns
    - Structural anomalies (hidden chars, excessive special chars)
    - Context manipulation
    """

    name = "injection"
    description = "Detects prompt injection attempts and jailbreak patterns"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (20, 100)
    realtime_capable = True

    # Thresholds
    pattern_threshold: float = 0.7
    semantic_threshold: float = 0.55

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect injection patterns across all spans in a trace.

        Examines user input spans, tool inputs, and LLM inputs for injection
        patterns.
        """
        # Gather all text inputs from relevant spans
        texts_to_check: list[tuple[str, str]] = []  # (text, span_id)

        for span in trace.spans:
            # Check user inputs
            if span.kind == SpanKind.USER_INPUT:
                text = self._extract_text(span)
                if text:
                    texts_to_check.append((text, span.span_id))

            # Check tool inputs
            if span.kind == SpanKind.TOOL and span.input_data:
                for key, value in span.input_data.items():
                    if isinstance(value, str) and len(value) > 10:
                        texts_to_check.append((value, span.span_id))

            # Check LLM inputs
            if span.kind == SpanKind.LLM and span.input_data:
                messages = span.input_data.get("messages", [])
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str) and len(content) > 10:
                            texts_to_check.append((content, span.span_id))

            # Check span attributes for text content
            text_attr = span.attributes.get("text") or span.attributes.get("input")
            if isinstance(text_attr, str) and len(text_attr) > 10:
                texts_to_check.append((text_attr, span.span_id))

        # A user-input span commonly stores the same content in attributes and
        # output_data. Preserve input order while preventing duplicate issues,
        # evidence, and inflated confidence for one physical input.
        texts_to_check = list(dict.fromkeys(texts_to_check))

        if not texts_to_check:
            return DetectionResult.no_issue(self.name)

        # Run injection analysis on each text
        worst_severity = 0
        all_issues: list[str] = []
        all_evidence: dict[str, Any] = {}
        affected_span_ids: list[str] = []

        for text, span_id in texts_to_check:
            analysis = self._analyze_text(text)

            if analysis["detected"]:
                affected_span_ids.append(span_id)
                severity = self._severity_to_score(analysis["max_severity"])
                worst_severity = max(worst_severity, severity)

                if analysis["primary_attack"]:
                    all_issues.append(
                        f"Injection detected ({analysis['primary_attack']}): "
                        f"{analysis['pattern_count']} pattern matches"
                    )

                all_evidence.update(analysis["details"])

        if not all_issues:
            return DetectionResult.no_issue(self.name)

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=worst_severity,
            summary=all_issues[0],
            fix_type=FixType.TERMINATE,
            fix_instruction="Block or sanitize the injected input. Do not follow injected instructions.",
        )

        for issue in all_issues:
            result.add_evidence(
                description=issue,
                span_ids=affected_span_ids,
                data=all_evidence,
            )

        result.confidence = min(0.99, all_evidence.get("raw_score", 0.5))
        return result

    # ------------------------------------------------------------------
    # Core analysis (ported from backend InjectionDetector)
    # ------------------------------------------------------------------

    def _analyze_text(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze a single text for injection patterns.

        Returns a dict with detection results including matched patterns,
        attack types, severity, jailbreak score, structural anomalies, and
        confidence calibration info.
        """
        text_lower = text.lower()
        matched_patterns: list[str] = []
        attack_types: set[str] = set()
        max_severity = "low"
        details: dict[str, Any] = {}

        # Pattern matching
        has_payload_pattern = False
        for pattern, attack_type, severity in INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matched_patterns.append(pattern)
                attack_types.add(attack_type)
                # A bare delimiter/role-marker match is not a payload — see the
                # detected gate below.
                if attack_type != "delimiter_injection":
                    has_payload_pattern = True
                if self._severity_rank(severity) > self._severity_rank(max_severity):
                    max_severity = severity

        details["pattern_matches"] = len(matched_patterns)

        # Jailbreak signature check
        jailbreak_score = self._check_jailbreak_signatures(text_lower)
        details["jailbreak_score"] = jailbreak_score
        if jailbreak_score > 0.5:
            attack_types.add("jailbreak")
            if jailbreak_score > 0.7:
                max_severity = "critical"
            elif jailbreak_score > 0.5:
                if self._severity_rank("high") > self._severity_rank(max_severity):
                    max_severity = "high"

        # Structural anomalies
        structure_score = self._check_structural_anomalies(text)
        details["structure_score"] = structure_score
        if structure_score > 0.6:
            attack_types.add("structural_attack")

        # Context manipulation
        if context:
            context_score = self._check_context_manipulation(text, context)
            details["context_manipulation_score"] = context_score
            if context_score > 0.7:
                attack_types.add("context_manipulation")

        # Benign context check
        is_benign = self._check_benign_context(text_lower)
        details["benign_context"] = is_benign

        # A bare delimiter/role-marker match ("[SYSTEM]", "System:") with no
        # payload pattern and no jailbreak signal is almost always the LEGITIMATE
        # structure of a system prompt / function-calling spec, not an injection.
        # Require a co-occurring payload signal to fire (mirrors the backend
        # app.detection.injection fix; removes the benign function-calling-prompt
        # false positive — 50/50 -> 1/50 on the archived FP-hunt pool there).
        detected = has_payload_pattern or jailbreak_score > 0.5
        details["delimiter_only_suppressed"] = bool(matched_patterns) and not detected

        # Benign context reduces severity but does not disable detection
        if is_benign and max_severity not in ("critical", "high"):
            if len(matched_patterns) >= 2 or jailbreak_score > 0.7:
                pass  # Strong signals override benign context
            else:
                max_severity = "low"

        # Calculate raw score
        raw_score = self._calculate_raw_score(
            len(matched_patterns),
            jailbreak_score,
            structure_score,
        )

        # Calibrate confidence
        confidence, calibration_info = self._calibrate_confidence(
            raw_score=raw_score,
            pattern_count=len(matched_patterns),
            jailbreak_score=jailbreak_score,
            severity=max_severity,
            is_benign=is_benign,
        )

        # Determine primary attack type
        primary_attack: Optional[str] = None
        if attack_types:
            for attack in _ATTACK_PRIORITY:
                if attack in attack_types:
                    primary_attack = attack
                    break
            if not primary_attack:
                primary_attack = list(attack_types)[0]

        details["raw_score"] = raw_score
        details["confidence"] = confidence
        details["calibration_info"] = calibration_info

        return {
            "detected": detected,
            "confidence": confidence,
            "primary_attack": primary_attack,
            "attack_types": list(attack_types),
            "max_severity": max_severity,
            "matched_patterns": matched_patterns,
            "pattern_count": len(matched_patterns),
            "details": details,
        }

    def _analyze_output_for_jailbreak_success(
        self,
        output: str,
    ) -> tuple[bool, float, list[str]]:
        """Analyze agent output for signs of successful jailbreak.

        Returns (detected, score, evidence_list).
        """
        evidence: list[str] = []
        score = 0.0

        success_indicators = [
            "certainly, here's",
            "sure, I'll",
            "of course, I can",
            "absolutely, here",
            "I'll help you with that",
            "[in character]",
            "as requested",
            "*evil grin*",
            "*removes restrictions*",
        ]

        output_lower = output.lower()
        for indicator in success_indicators:
            if indicator.lower() in output_lower:
                evidence.append(f"Success indicator: '{indicator}'")
                score += 0.2

        refusal_phrases = [
            "I cannot",
            "I'm not able to",
            "I won't",
            "I can't help with",
            "against my guidelines",
            "I must refuse",
            "not appropriate",
        ]

        has_refusal = any(phrase.lower() in output_lower for phrase in refusal_phrases)
        if has_refusal:
            score = max(0, score - 0.5)

        return score > 0.3, min(1.0, score), evidence

    # ------------------------------------------------------------------
    # Helper methods (ported faithfully from backend)
    # ------------------------------------------------------------------

    @staticmethod
    def _severity_rank(severity: str) -> int:
        ranks = {"low": 0, "medium": 1, "high": 2, "critical": 3, "info": -1}
        return ranks.get(severity, 0)

    @staticmethod
    def _severity_to_score(severity: str) -> int:
        """Map severity string to numeric 0-100 score for DetectionResult."""
        return {
            "info": 10,
            "low": 25,
            "medium": 50,
            "high": 75,
            "critical": 95,
        }.get(severity, 50)

    @staticmethod
    def _check_jailbreak_signatures(text: str) -> float:
        """Check for jailbreak signature phrases in text.

        Returns 0.0-1.0 score; 3+ matches saturate to 1.0.
        """
        matches = sum(1 for sig in JAILBREAK_SIGNATURES if sig in text)
        if matches == 0:
            return 0.0
        return min(1.0, matches / 3)

    @staticmethod
    def _check_structural_anomalies(text: str) -> float:
        """Check for structural anomalies that indicate injection attempts.

        Detects excessive special chars, hidden Unicode, high newline ratio,
        and large repeated blocks.
        """
        score = 0.0

        special_chars = len(re.findall(r"[\[\]<>{}|`]", text))
        if special_chars > 10:
            score += 0.2

        if re.search(r"[\u200b-\u200f\u2028-\u202f\u00a0]", text):
            score += 0.3

        newline_ratio = text.count("\n") / max(1, len(text))
        if newline_ratio > 0.1:
            score += 0.2

        if re.search(r"(.{10,})\1{2,}", text):
            score += 0.3

        return min(1.0, score)

    @staticmethod
    def _check_context_manipulation(text: str, context: str) -> float:
        """Check for topic-switching phrases that indicate context manipulation."""
        text_lower = text.lower()

        topic_switch_phrases = [
            "now let's talk about",
            "forget that",
            "instead",
            "actually",
            "on a different note",
            "change of topic",
        ]

        score = 0.0
        for phrase in topic_switch_phrases:
            if phrase in text_lower:
                score += 0.3

        return min(1.0, score)

    @staticmethod
    def _check_benign_context(text: str) -> bool:
        """Check if text contains benign security-research context."""
        return any(ctx in text for ctx in BENIGN_CONTEXTS)

    @staticmethod
    def _calculate_raw_score(
        pattern_count: int,
        jailbreak_score: float,
        structure_score: float,
    ) -> float:
        """Calculate raw injection score from component signals.

        Note: semantic_score is omitted in pisama-core because it requires
        an embedding model. The backend's semantic weight (0.2) is
        redistributed to pattern and jailbreak weights.
        """
        if pattern_count == 0 and jailbreak_score < 0.3:
            return 0.0

        score = min(pattern_count * 0.2, 0.5) + jailbreak_score * 0.35 + structure_score * 0.15

        return min(1.0, score)

    @staticmethod
    def _calibrate_confidence(
        raw_score: float,
        pattern_count: int,
        jailbreak_score: float,
        severity: str,
        is_benign: bool,
    ) -> tuple[float, dict[str, Any]]:
        """Calibrate confidence based on evidence quality and attack severity."""
        severity_weight = {
            "info": 0.0,
            "low": 0.5,
            "medium": 0.65,
            "high": 0.8,
            "critical": 0.9,
        }.get(severity, 0.5)

        evidence_count = pattern_count + (1 if jailbreak_score > 0.5 else 0)
        evidence_factor = min(1.0, evidence_count / 4)

        base_confidence = severity_weight * 0.40 + raw_score * 0.35 + evidence_factor * 0.25

        if is_benign:
            base_confidence *= 0.3

        calibrated = min(0.99, base_confidence)

        calibration_info = {
            "raw_score": round(raw_score, 4),
            "severity_weight": severity_weight,
            "evidence_count": evidence_count,
            "evidence_factor": round(evidence_factor, 4),
            "is_benign": is_benign,
        }

        return round(calibrated, 4), calibration_info

    @staticmethod
    def _extract_text(span: Span) -> Optional[str]:
        """Extract text content from a span."""
        # Check output_data first (user input captured as output)
        if span.output_data:
            text = span.output_data.get("text") or span.output_data.get("content")
            if isinstance(text, str):
                return text

        # Check input_data
        if span.input_data:
            text = span.input_data.get("text") or span.input_data.get("content")
            if isinstance(text, str):
                return text

        # Check attributes
        text = span.attributes.get("text") or span.attributes.get("content")
        if isinstance(text, str):
            return text

        return None
