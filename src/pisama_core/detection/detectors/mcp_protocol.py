"""MCP protocol detector for tool communication failures."""

import re
from typing import Any

from pisama_core.detection.base import BaseDetector
from pisama_core.detection.result import DetectionResult, FixType
from pisama_core.traces.enums import SpanKind
from pisama_core.traces.models import Span, Trace


class MCPProtocolDetector(BaseDetector):
    """Detects MCP-specific failures in tool communication.

    This detector identifies:
    - Tool discovery failures (tool not found, unknown tool)
    - Schema validation errors (invalid arguments, type mismatch)
    - Authentication failures (unauthorized, auth failed)
    - Connection failures (timeout, connection refused)
    """

    name = "mcp_protocol"
    description = "Detects MCP-specific failures in tool communication"
    version = "1.0.0"
    platforms = []  # All platforms
    severity_range = (25, 70)
    realtime_capable = True

    # Failure category patterns and their severity weights
    FAILURE_CATEGORIES: dict[str, dict[str, Any]] = {
        "discovery": {
            "patterns": [
                re.compile(r"tool\s+not\s+found", re.IGNORECASE),
                re.compile(r"unknown\s+tool", re.IGNORECASE),
                re.compile(r"no\s+such\s+tool", re.IGNORECASE),
                re.compile(r"tool\s+(?:does\s+not|doesn'?t)\s+exist", re.IGNORECASE),
                re.compile(r"unrecognized\s+tool", re.IGNORECASE),
                re.compile(r"method\s+not\s+found", re.IGNORECASE),
                re.compile(r"file\s+not\s+found", re.IGNORECASE),
                re.compile(r"path\s+not\s+found", re.IGNORECASE),
                re.compile(r"no\s+such\s+file", re.IGNORECASE),
            ],
            "severity_weight": 20,
            "description": "Tool discovery failure",
        },
        "schema": {
            "patterns": [
                re.compile(r"schema\s+validation", re.IGNORECASE),
                re.compile(r"invalid\s+arguments?", re.IGNORECASE),
                re.compile(r"type\s+mismatch", re.IGNORECASE),
                re.compile(r"required\s+field", re.IGNORECASE),
                re.compile(r"missing\s+required", re.IGNORECASE),
                re.compile(r"validation\s+error", re.IGNORECASE),
                re.compile(r"invalid\s+param", re.IGNORECASE),
                re.compile(r"unexpected\s+(?:argument|field|property)", re.IGNORECASE),
                re.compile(r"wrong\s+type", re.IGNORECASE),
            ],
            "severity_weight": 30,
            "description": "Schema validation failure",
        },
        "auth": {
            "patterns": [
                re.compile(r"unauthori[sz]ed", re.IGNORECASE),
                re.compile(r"authentication\s+failed", re.IGNORECASE),
                re.compile(r"auth(?:entication)?\s+error", re.IGNORECASE),
                re.compile(r"permission\s+denied", re.IGNORECASE),
                re.compile(r"access\s+denied", re.IGNORECASE),
                re.compile(r"forbidden", re.IGNORECASE),
                re.compile(r"invalid\s+(?:token|credential|api[_\s]?key)", re.IGNORECASE),
                re.compile(r"expired\s+token", re.IGNORECASE),
            ],
            "severity_weight": 40,
            "description": "Authentication/authorization failure",
        },
        "connection": {
            "patterns": [
                re.compile(r"connection\s+refused", re.IGNORECASE),
                re.compile(r"timeout", re.IGNORECASE),
                re.compile(r"timed?\s*out", re.IGNORECASE),
                re.compile(r"connect(?:ion)?\s+(?:error|failed|reset)", re.IGNORECASE),
                re.compile(r"ECONNREFUSED", re.IGNORECASE),
                re.compile(r"ETIMEDOUT", re.IGNORECASE),
                re.compile(r"host\s+(?:not\s+found|unreachable)", re.IGNORECASE),
                re.compile(r"network\s+(?:error|unreachable)", re.IGNORECASE),
                re.compile(r"server\s+(?:not\s+responding|unavailable)", re.IGNORECASE),
                re.compile(r"keepalive", re.IGNORECASE),
                re.compile(r"empty\s+response", re.IGNORECASE),
                re.compile(r"HTTP\s+5[0-9][0-9]", re.IGNORECASE),
                re.compile(r"service\s+unavailable", re.IGNORECASE),
                re.compile(r"bad\s+gateway", re.IGNORECASE),
                re.compile(r"temporarily\s+(?:down|unavailable)", re.IGNORECASE),
            ],
            "severity_weight": 25,
            "description": "Connection failure",
        },
        "protocol_version": {
            "patterns": [
                re.compile(r"version\s+mismatch", re.IGNORECASE),
                re.compile(r"protocol\s+version", re.IGNORECASE),
                re.compile(r"incompatible\s+(?:protocol|version)", re.IGNORECASE),
                re.compile(r"client\s+[\w.]+\s*,?\s*server\s+[\w.]+", re.IGNORECASE),
                re.compile(r"handshake\s+(?:failed|incomplete)", re.IGNORECASE),
            ],
            "severity_weight": 35,
            "description": "Protocol version mismatch",
        },
        "transport": {
            "patterns": [
                re.compile(r"transport\s+(?:error|failure|closed)", re.IGNORECASE),
                re.compile(r"websocket", re.IGNORECASE),
                re.compile(r"closed\s+unexpectedly", re.IGNORECASE),
                re.compile(r"tls\s+handshake", re.IGNORECASE),
                re.compile(r"certificate\s+(?:verification|invalid|expired)", re.IGNORECASE),
                re.compile(r"ssl\s+error", re.IGNORECASE),
                re.compile(r"stream\s+(?:closed|reset|aborted)", re.IGNORECASE),
                re.compile(r"message\s+queue\s+overflow", re.IGNORECASE),
            ],
            "severity_weight": 30,
            "description": "Transport layer failure",
        },
        "serialization": {
            "patterns": [
                re.compile(r"\bjson-?rpc\b", re.IGNORECASE),
                re.compile(r"json\s+parse\s+error", re.IGNORECASE),
                re.compile(r"parse\s+error.*json", re.IGNORECASE),
                re.compile(r"json.*unexpected\s+token", re.IGNORECASE),
                re.compile(
                    r"missing\s+['\"]?(?:jsonrpc|method|id|params)['\"]?\s+field", re.IGNORECASE
                ),
                re.compile(r"serializ(?:ation|er)\s+(?:error|failed)", re.IGNORECASE),
                re.compile(r"circular\s+reference", re.IGNORECASE),
                re.compile(r"malformed\s+(?:message|request|response)", re.IGNORECASE),
                re.compile(r"invalid\s+json", re.IGNORECASE),
                re.compile(r"response\s+format\s+differs", re.IGNORECASE),
            ],
            "severity_weight": 30,
            "description": "JSON-RPC / serialization failure",
        },
        "capability": {
            "patterns": [
                re.compile(
                    r"capabilit(?:y|ies)(?:\s+['\"][^'\"]+['\"])?\s+not\s+(?:support|available)",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"(?:method|operation|feature|capability)(?:\s+['\"][^'\"]+['\"])?\s+(?:not\s+available|not\s+supported|unsupported|not\s+found)",
                    re.IGNORECASE,
                ),
                re.compile(
                    r"unsupported\s+(?:method|operation|capability|feature|content-?type|content\s+encoding|format|encoding)",
                    re.IGNORECASE,
                ),
                re.compile(r"resource\s+limit\s+exceeded", re.IGNORECASE),
                re.compile(r"no\s+common\s+supported\s+features", re.IGNORECASE),
                re.compile(r"not\s+found\s+in\s+capability\s+list", re.IGNORECASE),
                re.compile(r"protocol\s+negotiation\s+failed", re.IGNORECASE),
            ],
            "severity_weight": 25,
            "description": "Unsupported capability or method",
        },
        "rate_limit": {
            "patterns": [
                re.compile(r"rate-?limited\s+by\s+(?:the\s+)?(?:mcp|server|tool)", re.IGNORECASE),
                re.compile(r"(?:mcp|tool)\s+server.*(?:rate\s+limit|throttl)", re.IGNORECASE),
                re.compile(
                    r"(?:rate\s+limit|throttl(?:ed|ing)).*(?:mcp|tool\s+server)", re.IGNORECASE
                ),
            ],
            "severity_weight": 20,
            "description": "Rate limiting / throttling",
        },
        "format": {
            "patterns": [
                re.compile(r"invalid\s+(?:pdf|image|file)\s+format", re.IGNORECASE),
                re.compile(r"corrupted?\s+(?:file|data|document)", re.IGNORECASE),
                re.compile(r"unsupported\s+(?:file\s+)?format", re.IGNORECASE),
                re.compile(r"hash\s+mismatch", re.IGNORECASE),
                re.compile(r"checksum\s+(?:failed|mismatch|invalid)", re.IGNORECASE),
                re.compile(r"file\s+(?:is\s+)?corrupt", re.IGNORECASE),
                re.compile(
                    r"cannot\s+(?:parse|read|decode)\s+(?:the\s+)?(?:file|document|data)",
                    re.IGNORECASE,
                ),
            ],
            "severity_weight": 25,
            "description": "File format or data integrity failure",
        },
        "resource": {
            "patterns": [
                re.compile(r"out\s+of\s+memory", re.IGNORECASE),
                re.compile(r"insufficient\s+(?:memory|heap|resources?)", re.IGNORECASE),
                re.compile(r"disk\s+(?:quota\s+exceeded|full|space)", re.IGNORECASE),
                re.compile(r"memory\s+(?:limit|allocation)\s+(?:exceeded|failed)", re.IGNORECASE),
                re.compile(r"resource\s+exhausted", re.IGNORECASE),
            ],
            "severity_weight": 30,
            "description": "Resource exhaustion (memory / disk)",
        },
        "delivery": {
            "patterns": [
                re.compile(r"SMTP\s+error", re.IGNORECASE),
                re.compile(r"recipient\s+(?:address\s+)?rejected", re.IGNORECASE),
                re.compile(r"address\s+rejected", re.IGNORECASE),
                re.compile(r"mailbox\s+(?:full|unavailable|not\s+found)", re.IGNORECASE),
                re.compile(r"delivery\s+(?:failed|failure|error)", re.IGNORECASE),
                re.compile(r"could\s+not\s+deliver", re.IGNORECASE),
            ],
            "severity_weight": 25,
            "description": "Message delivery failure",
        },
    }

    async def detect(self, trace: Trace) -> DetectionResult:
        """Detect MCP protocol failures in tool spans."""
        tool_spans = trace.get_spans_by_kind(SpanKind.TOOL)
        if not tool_spans:
            return DetectionResult.no_issue(self.name)

        # Filter to error spans
        error_spans = [s for s in tool_spans if s.status.is_failure]
        if not error_spans:
            return DetectionResult.no_issue(self.name)

        issues: list[str] = []
        severity = 0
        evidence_spans: list[str] = []
        failure_counts: dict[str, int] = {category: 0 for category in self.FAILURE_CATEGORIES}

        for span in error_spans:
            error_text = self._get_error_text(span)
            if not error_text:
                continue

            categories = self._classify_failure(error_text)
            if not categories:
                continue

            for category in categories:
                cat_info = self.FAILURE_CATEGORIES[category]
                failure_counts[category] += 1
                issues.append(
                    f"{cat_info['description']} in tool '{span.name}': "
                    f"{self._truncate(error_text, 100)}"
                )
                evidence_spans.append(span.span_id)
                severity += cat_info["severity_weight"]

        if not issues:
            return DetectionResult.no_issue(self.name)

        severity = max(self.severity_range[0], min(self.severity_range[1], severity))

        # Determine fix type based on dominant failure category
        dominant_category = max(failure_counts, key=failure_counts.get)  # type: ignore[arg-type]
        fix_type, fix_instruction = self._get_fix_for_category(dominant_category)

        result = DetectionResult.issue_found(
            detector_name=self.name,
            severity=severity,
            summary=(
                issues[0] if len(issues) == 1 else f"{len(issues)} MCP protocol failures detected"
            ),
            fix_type=fix_type,
            fix_instruction=fix_instruction,
        )

        for issue in issues:
            result.add_evidence(
                description=issue,
                span_ids=evidence_spans,
                data={"failure_counts": failure_counts},
            )

        return result

    def _get_error_text(self, span: Span) -> str:
        """Extract error text from a span."""
        parts: list[str] = []

        if span.error_message:
            parts.append(span.error_message)

        # Check output_data for error details
        if span.output_data:
            for key in ("error", "message", "detail", "error_message", "reason"):
                val = span.output_data.get(key)
                if isinstance(val, str):
                    parts.append(val)
                elif isinstance(val, dict):
                    msg = val.get("message") or val.get("detail") or val.get("error")
                    if isinstance(msg, str):
                        parts.append(msg)

        # Check attributes
        for key in ("error.message", "error.type", "exception.message"):
            val = span.attributes.get(key)
            if isinstance(val, str):
                parts.append(val)

        return " ".join(parts)

    def _classify_failure(self, error_text: str) -> list[str]:
        """Classify an error into MCP failure categories."""
        matched_categories: list[str] = []

        for category, info in self.FAILURE_CATEGORIES.items():
            for pattern in info["patterns"]:
                if pattern.search(error_text):
                    if category not in matched_categories:
                        matched_categories.append(category)
                    break

        return matched_categories

    def _get_fix_for_category(self, category: str) -> tuple[FixType, str]:
        """Get the appropriate fix type and instruction for a failure category."""
        if category == "auth":
            return (
                FixType.ESCALATE,
                "Authentication/authorization failure detected in MCP tool communication. "
                "Check API keys, tokens, or permissions. The tool server may require "
                "re-authentication or the credentials may have expired.",
            )
        elif category == "schema":
            return (
                FixType.SWITCH_STRATEGY,
                "Schema validation errors in MCP tool calls. "
                "The tool arguments do not match the expected schema. "
                "Re-read the tool's input schema and ensure all required fields "
                "are present with correct types.",
            )
        elif category == "discovery":
            return (
                FixType.SWITCH_STRATEGY,
                "Tool not found in MCP server. "
                "The requested tool may not be available on this server. "
                "List available tools and use an alternative, or check "
                "that the MCP server is configured correctly.",
            )
        elif category == "protocol_version":
            return (
                FixType.ESCALATE,
                "MCP protocol version mismatch between client and server. "
                "Upgrade the client or server to a compatible version, or negotiate "
                "a shared protocol version during handshake.",
            )
        elif category == "transport":
            return (
                FixType.ADD_DELAY,
                "MCP transport-layer failure (websocket/TLS/stream). "
                "Reconnect and retry; verify certificate validity and network stability. "
                "Consider falling back to a different transport if available.",
            )
        elif category == "serialization":
            return (
                FixType.SWITCH_STRATEGY,
                "JSON-RPC or serialization failure in MCP message. "
                "Ensure all required JSON-RPC fields (jsonrpc, method, id) are present "
                "and that payloads are valid JSON without circular references.",
            )
        elif category == "capability":
            return (
                FixType.SWITCH_STRATEGY,
                "Requested MCP capability or method is not supported by the server. "
                "Query supported capabilities before use and fall back to an alternative.",
            )
        elif category == "rate_limit":
            return (
                FixType.ADD_DELAY,
                "MCP server is rate-limiting requests. "
                "Back off with exponential delay and respect any Retry-After headers.",
            )
        elif category == "format":
            return (
                FixType.ESCALATE,
                "File format or data integrity failure in MCP tool. "
                "The input file may be corrupted, unsupported, or fail checksum validation. "
                "Verify the file before retrying or use a supported format.",
            )
        elif category == "resource":
            return (
                FixType.ESCALATE,
                "Resource exhaustion in MCP tool server (memory or disk). "
                "The operation cannot complete due to insufficient system resources. "
                "Reduce payload size or retry on a less-loaded instance.",
            )
        elif category == "delivery":
            return (
                FixType.ESCALATE,
                "Message delivery failure in MCP tool. "
                "The recipient address may be invalid or the mailbox unavailable. "
                "Verify the destination and retry or escalate to the user.",
            )
        else:  # connection
            return (
                FixType.ADD_DELAY,
                "Connection failure to MCP tool server. "
                "The server may be down, overloaded, or unreachable. "
                "Retry after a brief delay, or check network connectivity "
                "and server status.",
            )

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."
