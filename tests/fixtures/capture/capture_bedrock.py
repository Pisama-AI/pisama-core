"""Capture a REAL-shaped Bedrock Agents `InvokeAgent` trace stream fixture.

No AWS call is made and no credentials are used. The fixture's *structure*
comes entirely from botocore's own service model for `bedrock-agent-runtime`
(`botocore/data/bedrock-agent-runtime/<apiVersion>/service-2.json.gz`), which is
AWS's authoritative API definition and is what boto3 itself parses responses
against. Only leaf scalar values are authored here, and each one is validated
against the model's declared type, enum, length/range and regex constraints.

Two independent guarantees that no key here is invented:

1. `emit()` walks the service model. Every structure member name in the
   scenario spec is looked up in `shapes[<Shape>]["members"]`; an unknown name
   raises. Shapes marked `"union": true` are required to carry exactly one
   member, exactly as AWS declares them.
2. The emitted wire JSON is pushed through the *production* botocore pipeline:
   raw `vnd.amazon.eventstream` frames -> `botocore.eventstream.EventStream`
   (CRC-validating decoder) -> `botocore.parsers.EventStreamJSONParser` against
   the `ResponseStream` output shape. That is byte-for-byte the path
   `InvokeAgentResponse["completion"]` takes inside boto3. botocore's parser
   drops any member absent from the model, so a round-trip that comes back
   identical proves every key is model-defined.

Output: `invoke_agent_trace.json`.
"""

from __future__ import annotations

import binascii
import datetime as dt
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

import botocore
import botocore.session
from botocore.eventstream import EventStream
from botocore.model import ServiceModel
from botocore.parsers import EventStreamJSONParser

SERVICE = "bedrock-agent-runtime"
HERE = Path(__file__).resolve().parent
# Writes straight into tests/fixtures/, the location the test suite reads.
OUT = HERE.parent / "bedrock_invoke_agent_trace.json"


# --------------------------------------------------------------------------
# service model access
# --------------------------------------------------------------------------

def load_model() -> dict[str, Any]:
    loader = botocore.session.get_session().get_component("data_loader")
    return loader.load_service_model(SERVICE, "service-2")


MODEL = load_model()
SHAPES: dict[str, Any] = MODEL["shapes"]
API_VERSION: str = MODEL["metadata"]["apiVersion"]


class ModelViolationError(Exception):
    """Raised when the scenario spec strays from the AWS service model."""


def _shape(name: str) -> dict[str, Any]:
    try:
        return SHAPES[name]
    except KeyError:  # pragma: no cover - defensive
        raise ModelViolationError(f"shape {name!r} is not in the {SERVICE} service model")


# --------------------------------------------------------------------------
# model-driven emitter
# --------------------------------------------------------------------------

_SCALARS = {
    "string": str,
    "integer": int,
    "long": int,
    "float": float,
    "double": float,
    "boolean": bool,
    "blob": (bytes, str),
    "timestamp": (dt.datetime, str),
}


def _check_scalar(shape_name: str, shape: dict[str, Any], value: Any, path: str) -> Any:
    kind = shape["type"]
    expected = _SCALARS[kind]
    if kind in ("float", "double") and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, expected) or (kind != "boolean" and isinstance(value, bool)):
        raise ModelViolationError(
            f"{path}: {shape_name} is {kind}, got {type(value).__name__} ({value!r})"
        )
    if kind == "string":
        if "enum" in shape and value not in shape["enum"]:
            raise ModelViolationError(
                f"{path}: {shape_name} enum is {shape['enum']}, got {value!r}"
            )
        if "min" in shape and len(value) < shape["min"]:
            raise ModelViolationError(f"{path}: {shape_name} min length {shape['min']}, got {len(value)}")
        if "max" in shape and len(value) > shape["max"]:
            raise ModelViolationError(f"{path}: {shape_name} max length {shape['max']}, got {len(value)}")
        if "pattern" in shape and not re.search(shape["pattern"], value):
            raise ModelViolationError(f"{path}: {shape_name} pattern {shape['pattern']!r} rejects {value!r}")
    if kind in ("integer", "long", "float", "double"):
        if "min" in shape and value < shape["min"]:
            raise ModelViolationError(f"{path}: {shape_name} min {shape['min']}, got {value}")
        if "max" in shape and value > shape["max"]:
            raise ModelViolationError(f"{path}: {shape_name} max {shape['max']}, got {value}")
    if kind == "timestamp":
        fmt = shape.get("timestampFormat")
        if fmt != "iso8601":
            raise ModelViolationError(f"{path}: unexpected timestampFormat {fmt!r}")
        if isinstance(value, dt.datetime):
            value = value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return value


def emit(shape_name: str, spec: Any, path: str = "$") -> Any:
    """Render `spec` against the AWS service model shape `shape_name`.

    Structure member names are resolved through the model, so a typo or an
    invented key raises instead of silently landing in the fixture.
    """
    shape = _shape(shape_name)
    kind = shape["type"]

    if kind == "structure":
        if not isinstance(spec, dict):
            raise ModelViolationError(f"{path}: {shape_name} is a structure, spec is {type(spec).__name__}")
        members = shape["members"]
        if shape.get("union") and len(spec) != 1:
            raise ModelViolationError(
                f"{path}: {shape_name} is declared 'union': true in the service model, "
                f"so exactly one member may be set; spec has {sorted(spec)}"
            )
        for required in shape.get("required", []):
            if required not in spec:
                raise ModelViolationError(f"{path}: {shape_name} requires member {required!r}")
        out: dict[str, Any] = {}
        for key, sub in spec.items():
            if key not in members:
                raise ModelViolationError(
                    f"{path}.{key}: {shape_name} has no member {key!r}; "
                    f"model defines {sorted(members)}"
                )
            out[key] = emit(members[key]["shape"], sub, f"{path}.{key}")
        return out

    if kind == "list":
        if not isinstance(spec, list):
            raise ModelViolationError(f"{path}: {shape_name} is a list, spec is {type(spec).__name__}")
        for bound, cmp_ in (("min", len(spec).__lt__), ("max", len(spec).__gt__)):
            if bound in shape and cmp_(shape[bound]):
                raise ModelViolationError(f"{path}: {shape_name} {bound} {shape[bound]}, got {len(spec)}")
        member = shape["member"]["shape"]
        return [emit(member, item, f"{path}[{i}]") for i, item in enumerate(spec)]

    if kind == "map":
        if not isinstance(spec, dict):
            raise ModelViolationError(f"{path}: {shape_name} is a map, spec is {type(spec).__name__}")
        key_shape = shape["key"]["shape"]
        val_shape = shape["value"]["shape"]
        return {
            _check_scalar(key_shape, _shape(key_shape), k, f"{path}<key>"): emit(
                val_shape, v, f"{path}[{k!r}]"
            )
            for k, v in spec.items()
        }

    return _check_scalar(shape_name, shape, spec, path)


# --------------------------------------------------------------------------
# scenario  (only leaf VALUES below are authored; every key is a model member)
# --------------------------------------------------------------------------

AGENT_ID = "AGT7QJ2X1B"          # AgentId: ^[0-9a-zA-Z]+$, max 10
AGENT_ALIAS_ID = "ALS4KD9P2C"    # AgentAliasId: same constraints
AGENT_VERSION = "1"              # AgentVersion: ^(DRAFT|[0-9]{0,4}[1-9][0-9]{0,4})$
SESSION_ID = "sess-2026-08-11-refund-4417"   # SessionId: ^[0-9a-zA-Z._:-]+$
FOUNDATION_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# TraceId is declared max 16 / min 2 in the service model, so the ids below
# respect that bound rather than the longer uuid-shaped strings sometimes seen
# in AWS console output.
T_PRE_IN, T_PRE_OUT = "tr-pre-0001", "tr-pre-0002"
T_ORCH_IN, T_ORCH_OUT = "tr-orc-0001", "tr-orc-0002"
T_RATIONALE, T_INVOKE = "tr-orc-0003", "tr-orc-0004"
T_OBS_TOOL, T_OBS_FINAL = "tr-orc-0005", "tr-orc-0006"
T_FAIL = "tr-fail-0001"

BASE = dt.datetime(2026, 8, 11, 9, 14, 2, 481000, tzinfo=dt.timezone.utc)


def _at(seconds: float) -> dt.datetime:
    return BASE + dt.timedelta(seconds=seconds)


def _meta(start: float, end: float, in_tok: int | None = None, out_tok: int | None = None):
    """A `Metadata` spec. `usage` only exists on model-invocation metadata."""
    spec: dict[str, Any] = {
        "startTime": _at(start),
        "endTime": _at(end),
        "totalTimeMs": int((end - start) * 1000),
    }
    if in_tok is not None:
        spec["usage"] = {"inputTokens": in_tok, "outputTokens": out_tok}
    return spec


# Each entry: (offset_seconds, Trace-union spec). `Trace` is a union, so every
# entry sets exactly one member; `OrchestrationTrace` is a union too, which is
# why the tool call and its observation arrive as separate trace parts.
SUCCESS_STEPS: list[tuple[float, dict[str, Any]]] = [
    (0.00, {"preProcessingTrace": {"modelInvocationInput": {
        "traceId": T_PRE_IN,
        "type": "PRE_PROCESSING",
        "text": "Human: Categorise the user request: \"Where is my refund for order 4417?\"",
        "foundationModel": FOUNDATION_MODEL,
        "promptCreationMode": "DEFAULT",
        "parserMode": "DEFAULT",
        "inferenceConfiguration": {
            "maximumLength": 2048, "temperature": 0.0, "topK": 250, "topP": 1.0,
            "stopSequences": ["</invoke>", "</answer>", "</error>"],
        },
    }}}),
    (0.94, {"preProcessingTrace": {"modelInvocationOutput": {
        "traceId": T_PRE_OUT,
        "rawResponse": {"content": "<thinking>Question is in scope for the refund agent.</thinking>\n<category>D</category>"},
        "parsedResponse": {"isValid": True, "rationale": "Question is in scope for the refund agent."},
        "metadata": _meta(0.00, 0.94, 1183, 42),
    }}}),
    (0.95, {"orchestrationTrace": {"modelInvocationInput": {
        "traceId": T_ORCH_IN,
        "type": "ORCHESTRATION",
        "text": "Human: Where is my refund for order 4417?",
        "foundationModel": FOUNDATION_MODEL,
        "promptCreationMode": "OVERRIDDEN",
        "parserMode": "DEFAULT",
        "inferenceConfiguration": {
            "maximumLength": 2048, "temperature": 0.0, "topK": 250, "topP": 1.0,
            "stopSequences": ["</invoke>", "</answer>", "</error>"],
        },
    }}}),
    (2.61, {"orchestrationTrace": {"modelInvocationOutput": {
        "traceId": T_ORCH_OUT,
        "rawResponse": {"content": "<thinking>I need the refund status for order 4417.</thinking>\n<function_calls><invoke><tool_name>refund-actions::get_refund_status</tool_name></invoke></function_calls>"},
        "metadata": _meta(0.95, 2.61, 2914, 118),
    }}}),
    (2.62, {"orchestrationTrace": {"rationale": {
        "traceId": T_RATIONALE,
        "text": "The customer is asking about a refund for order 4417. I should look up the refund status in the refund action group before answering.",
    }}}),
    (2.63, {"orchestrationTrace": {"invocationInput": {
        "traceId": T_INVOKE,
        "invocationType": "ACTION_GROUP",
        "actionGroupInvocationInput": {
            "actionGroupName": "refund-actions",
            "function": "get_refund_status",
            "executionType": "LAMBDA",
            "parameters": [
                {"name": "orderId", "type": "string", "value": "4417"},
                {"name": "includeTimeline", "type": "boolean", "value": "true"},
            ],
        },
    }}}),
    (3.88, {"orchestrationTrace": {"observation": {
        "traceId": T_OBS_TOOL,
        "type": "ACTION_GROUP",
        "actionGroupInvocationOutput": {
            "text": "{\"orderId\":\"4417\",\"refundState\":\"PENDING_BANK\",\"amount\":\"64.00\",\"currency\":\"USD\",\"initiatedAt\":\"2026-08-06T11:02:00Z\",\"expectedBy\":\"2026-08-13\"}",
            "metadata": _meta(2.63, 3.88),
        },
    }}}),
    (5.42, {"orchestrationTrace": {"observation": {
        "traceId": T_OBS_FINAL,
        "type": "FINISH",
        "finalResponse": {
            "text": "Your refund of $64.00 for order 4417 was initiated on 6 August and is currently with your bank. You should see it by 13 August.",
            "metadata": _meta(3.88, 5.42),
        },
    }}}),
]

FAILURE_STEPS: list[tuple[float, dict[str, Any]]] = [
    (0.00, {"orchestrationTrace": {"modelInvocationInput": {
        "traceId": T_ORCH_IN,
        "type": "ORCHESTRATION",
        "text": "Human: Where is my refund for order 4417?",
        "foundationModel": FOUNDATION_MODEL,
        "promptCreationMode": "OVERRIDDEN",
        "parserMode": "DEFAULT",
        "inferenceConfiguration": {
            "maximumLength": 2048, "temperature": 0.0, "topK": 250, "topP": 1.0,
            "stopSequences": ["</invoke>", "</answer>", "</error>"],
        },
    }}}),
    (1.77, {"failureTrace": {
        "traceId": T_FAIL,
        "failureCode": 424,
        "failureReason": "The model output for the orchestration step could not be parsed. Expected a <function_calls> or <answer> block.",
        "metadata": _meta(0.00, 1.77),
    }}),
]

FINAL_ANSWER = SUCCESS_STEPS[-1][1]["orchestrationTrace"]["observation"]["finalResponse"]["text"]


def build_trace_part(offset: float, trace_spec: dict[str, Any]) -> dict[str, Any]:
    """Render one `TracePart` event body from the service model."""
    return emit(
        "TracePart",
        {
            "agentId": AGENT_ID,
            "agentAliasId": AGENT_ALIAS_ID,
            "agentVersion": AGENT_VERSION,
            "sessionId": SESSION_ID,
            "eventTime": _at(offset),
            "trace": trace_spec,
        },
    )


# --------------------------------------------------------------------------
# vnd.amazon.eventstream framing, decoded back by botocore itself
# --------------------------------------------------------------------------

_HEADER_TYPE_STRING = 7


def _encode_headers(headers: dict[str, str]) -> bytes:
    buf = b""
    for name, value in headers.items():
        nb, vb = name.encode(), value.encode()
        buf += struct.pack("!B", len(nb)) + nb
        buf += struct.pack("!B", _HEADER_TYPE_STRING)
        buf += struct.pack("!H", len(vb)) + vb
    return buf


def encode_event(event_type: str, body: bytes) -> bytes:
    """Encode one AWS event-stream frame (prelude + headers + payload + CRCs)."""
    headers = _encode_headers(
        {":event-type": event_type, ":message-type": "event", ":content-type": "application/json"}
    )
    total = 4 + 4 + 4 + len(headers) + len(body) + 4
    prelude = struct.pack("!II", total, len(headers))
    frame = prelude + struct.pack("!I", binascii.crc32(prelude) & 0xFFFFFFFF)
    frame += headers + body
    return frame + struct.pack("!I", binascii.crc32(frame) & 0xFFFFFFFF)


class _RawStream:
    """Minimal stand-in for the urllib3 response boto3 hands to EventStream."""

    def __init__(self, data: bytes, chunk: int = 97) -> None:
        self._data, self._chunk = data, chunk

    def stream(self):
        for i in range(0, len(self._data), self._chunk):
            yield self._data[i : i + self._chunk]


def parse_with_botocore(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-trip wire events through boto3's own InvokeAgent response pipeline."""
    service = ServiceModel(MODEL, service_name=SERVICE)
    response_stream = service.operation_model("InvokeAgent").output_shape.members["completion"]
    raw = b"".join(
        encode_event("trace", json.dumps(e["trace"]).encode()) for e in events
    )
    stream = EventStream(_RawStream(raw), response_stream, EventStreamJSONParser(), "InvokeAgent")
    return list(stream)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dt.datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


# --------------------------------------------------------------------------

def main() -> int:
    success = [{"trace": build_trace_part(off, spec)} for off, spec in SUCCESS_STEPS]
    failure = [{"trace": build_trace_part(off, spec)} for off, spec in FAILURE_STEPS]

    # Fidelity gate: botocore's parser silently discards any member the model
    # does not define, so an identical round-trip proves every key is real.
    for label, events in (("success", success), ("failure", failure)):
        reparsed = _jsonable(parse_with_botocore(events))
        if reparsed != events:
            print(f"ROUND-TRIP MISMATCH in {label} stream:", file=sys.stderr)
            print(json.dumps({"sent": events, "botocore_returned": reparsed}, indent=2)[:4000],
                  file=sys.stderr)
            return 1

    payload = {
        "_capture": {
            "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "generator": "capture_bedrock.py",
            "aws_call_made": False,
            "botocore_version": botocore.__version__,
            "service": SERVICE,
            "service_model_api_version": API_VERSION,
            "service_model_path": (
                f"botocore/data/{SERVICE}/{API_VERSION}/service-2.json.gz"
            ),
            "python": sys.version.split()[0],
            "provenance": (
                "Structure derived from the botocore service model; every structure "
                "member name resolved through shapes[...]['members']. Union shapes "
                "carry exactly one member per the model's \"union\": true flag. Both "
                "streams round-trip byte-identically through botocore's own "
                "EventStream decoder + EventStreamJSONParser against the "
                "ResponseStream output shape of InvokeAgent."
            ),
            "note_traceid_bound": (
                "TraceId is declared min 2 / max 16 in the service model, so trace ids "
                "here honour that bound."
            ),
            "note_timestamps": (
                "eventTime / startTime / endTime are serialised in their iso8601 wire "
                "form. boto3 hands callers datetime objects for these members."
            ),
        },
        "invoke_agent_request": {
            "agentId": AGENT_ID,
            "agentAliasId": AGENT_ALIAS_ID,
            "sessionId": SESSION_ID,
            "enableTrace": True,
            "inputText": "Where is my refund for order 4417?",
        },
        "scenarios": {
            "success": {
                "description": (
                    "Pre-processing, orchestration LLM call, rationale, an "
                    "actionGroup tool invocation, its observation, and a "
                    "finalResponse observation."
                ),
                "completion_events": success,
                "final_answer": FINAL_ANSWER,
            },
            "failure": {
                "description": "Orchestration LLM call followed by a failureTrace.",
                "completion_events": failure,
                "final_answer": None,
            },
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"botocore {botocore.__version__}  |  {SERVICE} apiVersion {API_VERSION}")
    print(f"success events: {len(success)}   failure events: {len(failure)}")
    print("round-trip through botocore EventStream + EventStreamJSONParser: OK")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
