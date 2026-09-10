# pisama-core

[![PyPI version](https://img.shields.io/pypi/v/pisama-core.svg)](https://pypi.org/project/pisama-core/)
[![Python versions](https://img.shields.io/pypi/pyversions/pisama-core.svg)](https://pypi.org/project/pisama-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Pisama-AI/pisama-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Pisama-AI/pisama-core/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/pypi/dm/pisama-core)](https://pypistats.org/packages/pisama-core)

Detection, scoring, and healing engine for AI agent systems. Detect failure modes like infinite loops, hallucinations, cost overruns, and coordination breakdowns in your LLM agents, entirely offline, no API keys required. Each detector returns a binary verdict per failure mode with calibrated confidence.

Part of the [Pisama](https://pisama.ai) platform for single-agent, multi-agent, and sub-agent failure detection.

## Install

```bash
pip install pisama-core
```

## Quick Start

```python
import asyncio
from pisama_core import Trace, SpanKind, DetectionOrchestrator

# Build a trace from your agent's execution
trace = Trace()
for i in range(8):
    trace.create_span(name="Read", kind=SpanKind.TOOL)

# Run all built-in detectors
orchestrator = DetectionOrchestrator()
result = asyncio.run(orchestrator.analyze(trace))

for detection in result.detection_results:
    if detection.detected:
        print(f"[{detection.detector_name}] {detection.summary}")
        print(f"  Severity: {detection.severity}/100")
        if detection.recommendation:
            print(f"  Fix: {detection.recommendation.instruction}")
```

Output:
```
[loop] Tool 'Read' repeated 8x consecutively
  Severity: 100/100
  Fix: Stop the current loop. Try a different approach or ask the user for guidance.
```

No API key. No network calls. Runs completely locally. The optional
[telemetry](#telemetry) is opt-in and disabled by default.

### Ingest conversation and Omnigent traces

`ConversationTrace` normalizes multi-turn user, agent, system, and tool
messages into detector-compatible spans. Omnigent event streams can be
converted directly to the Agent Trajectory Interchange Format (ATIF):

```python
from pisama_core.ingestion.omnigent_events import events_to_atif, load_event_stream

events = load_event_stream("events.jsonl")
trajectory = events_to_atif(events, agent_name="research-agent")
```

Detection results can also expose a typed `DiagnosisRecord`, so causal
evidence and recommended interventions round-trip without losing newer,
additive fields.

Prefer to analyze a trace file in one line? Install the `pisama` wrapper
(`pip install pisama`) and run `pisama.analyze("trace.json")`. Grab a ready-made
example loop trace to try it:

```bash
curl -O https://raw.githubusercontent.com/Pisama-AI/pisama-core/main/examples/trace.json
```

## Using Pisama?

We read every email. If you are using `pisama-core`, even just trying it out, write a line to [tuomo@pisama.ai](mailto:tuomo@pisama.ai). What works, what does not, what you wish it did. The roadmap is shaped by these notes.

## Built-in Detectors

| Detector | What it catches |
|----------|----------------|
| **Loop** | Consecutive repetitions, cyclic patterns (A->B->A->B), low tool diversity |
| **Repetition** | Similar actions with slight variations, tool dominance |
| **Cost** | Token budget overruns, excessive LLM/tool calls |
| **Hallucination** | Failed file operations, error rate spikes |
| **Coordination** | Message storms, agent imbalance, handoff loops |

All detectors support both **batch analysis** (full trace) and **real-time hooks** (per-span).

## Use Individual Detectors

```python
import asyncio
from pisama_core import Trace, SpanKind
from pisama_core.detection.detectors.loop import LoopDetector
from pisama_core.detection.detectors.cost import CostDetector

trace = Trace()
# ... add spans representing your agent's execution

loop = LoopDetector()
cost = CostDetector()

loop_result = asyncio.run(loop.detect(trace))
cost_result = asyncio.run(cost.detect(trace))

if loop_result.detected:
    print(f"Loop detected: {loop_result.summary}")
```

## Write Your Own Detector

```python
from pisama_core import BaseDetector, DetectionResult, Trace
from pisama_core.detection.result import FixType

class MyDetector(BaseDetector):
    name = "my_detector"
    description = "Detects my custom failure pattern"
    version = "0.1.0"

    async def detect(self, trace: Trace) -> DetectionResult:
        # Your detection logic here
        tool_names = trace.get_tool_sequence()
        if len(set(tool_names)) == 1 and len(tool_names) > 5:
            return DetectionResult.issue_found(
                detector_name=self.name,
                severity=50,
                summary="Agent is stuck using a single tool",
                fix_type=FixType.SWITCH_STRATEGY,
                fix_instruction="Try a different approach",
            )
        return DetectionResult.no_issue(self.name)
```

Register it so the orchestrator picks it up:

```python
from pisama_core import registry
registry.register(MyDetector())
```

## Core Concepts

- **Trace** -- A complete agent execution session containing multiple spans
- **Span** -- A single unit of work (tool call, LLM inference, agent turn) with `kind`, timing, and optional I/O data
- **DetectionResult** -- Detector output: issue found (yes/no), severity (0-100), evidence, fix recommendation
- **DetectorRegistry** -- Plugin system for registering detectors (built-ins auto-register on import)
- **DetectionOrchestrator** -- Runs all registered detectors and aggregates results

## Platform Support

Traces are framework-agnostic. Set `platform` for platform-aware threshold tuning:

```python
from pisama_core import Trace, TraceMetadata, Platform

trace = Trace(metadata=TraceMetadata(platform=Platform.LANGGRAPH))
```

Ships trace adapters for OpenAI (Assistants, Responses, Agents SDK), AWS Bedrock, Google Gemini, Google ADK and DeepAgents. Any other platform is supported by constructing a `Trace` directly; the `Platform` enum carries labels for Claude Code, LangGraph, n8n, Dify, OpenClaw and others.

## Pisama Platform

`pisama-core`'s built-in detectors run entirely offline with no API key. The hosted Pisama platform runs a larger calibrated detector set on top of those, and adds ML-based detection, LLM-as-judge verification, and a dashboard. See [pisama.ai](https://pisama.ai).

## Design Partner Program

Up to five companies. Biweekly product input. Free Pro access for 12 months.

If you are running multi-agent systems in production and want a direct voice in the roadmap, email [tuomo@pisama.ai](mailto:tuomo@pisama.ai) with a short note about what you are building.

## Telemetry

`pisama-core` does **not** send any telemetry by default. Nothing leaves your
process unless you explicitly opt in.

If you'd like to help us understand which Python versions, operating systems,
and runtime environments to prioritize, you can opt in with one of:

```bash
export PISAMA_TELEMETRY=1
```

Or programmatically:

```python
import pisama_core
pisama_core.enable_telemetry()
```

**When opted in, one HTTP POST is sent to `https://api.pisama.ai/api/v1/telemetry/install`:**

| Field | Example |
|-------|---------|
| `install_id` | locally-generated UUID4, persisted at `~/.pisama/install_id` |
| `sdk_version` | `1.7.1` |
| `python` | `3.12.3` |
| `os` | `Darwin`, `Linux`, `Windows` |
| `os_release` | `25.2.0` (truncated to 64 chars) |
| `runtime_env` | `github_actions`, `aws_lambda`, `vercel`, `fly`, `modal`, `kubernetes`, `docker`, `local`, etc. |
| `event` | `first_run` once, `session` thereafter |

**What is never sent:** trace contents, detector outputs, file paths,
environment variables, hostnames, IPs (the server discards them on receipt),
API keys, or user identifiers.

**To opt back out** (overrides any opt-in):

```bash
export DO_NOT_TRACK=1
touch ~/.pisama/telemetry_disabled
```

Or: `pisama_core.disable_telemetry()`.

The implementation is a single file:
[`src/pisama_core/utils/_telemetry.py`](src/pisama_core/utils/_telemetry.py):
stdlib-only, daemon-thread send, 2-second timeout, swallows all exceptions.
Telemetry can never block, slow down, or crash your process.

## Communication detection limits

The communication detector checks narrow, explicit literal-answer and JSON
response contracts on captured input/output pairs or explicit message-parent
relationships. It does not infer communication from chronological adjacency,
nor infer semantic failure from missing instruction verbs or keyword overlap.
Ambiguous natural-language requests are outside this checker and produce no
finding; that is abstention, not proof the task succeeded. Quoted literals and
single uppercase answer tokens are supported only in unconditional reply/return
instructions. JSON syntax is checked, not arbitrary schemas or business rules.
Literal responses are compared exactly, including whitespace inside the quoted
operand; no trimming is applied to the captured response.
Confidence values are heuristic weights, not calibrated probabilities. Authored
regression controls are not representative customer-validation evidence.

## Response contract accounting

Use `pisama>=0.7.0` for validated positional accounting and coverage reporting
in the wrapper/CLI. Older wrappers may install but lack these reporting
guarantees. Core-only consumers must interpret the metadata directly.

Communication results include `metadata.response_contract_coverage` version 1.
Its scope is `eligible_captured_response_pairs`, not business semantics. Every
supplied span has a positional record: satisfied, violated, unsupported or
outside_scope. Counts distinguish considered response spans, actual checks and
unsupported inputs; evaluation continues after the first finding. A passing
contract does not hide unsupported requests elsewhere in the trace.

`span_index` identifies a position within this supplied trace, not a stable
cross-run contract identity. Duplicate/empty IDs are marked ambiguous, and
duplicate parent IDs cannot establish a message relationship. IDs generated by
an upstream parser cannot retrospectively prove source identity. No prompt or
response content is copied into this accounting. This metadata does not certify
whole-task coverage or establish comparable repair outcomes across traces.

## License

MIT
