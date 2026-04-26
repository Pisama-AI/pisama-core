# pisama-core

[![PyPI version](https://img.shields.io/pypi/v/pisama-core.svg)](https://pypi.org/project/pisama-core/)
[![Python versions](https://img.shields.io/pypi/pyversions/pisama-core.svg)](https://pypi.org/project/pisama-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/tn-pisama/pisama-core/actions/workflows/ci.yml/badge.svg)](https://github.com/tn-pisama/pisama-core/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/pypi/dm/pisama-core)](https://pypistats.org/packages/pisama-core)

Detection, scoring, and healing engine for AI agent systems. Detect failure modes like infinite loops, hallucinations, cost overruns, and coordination breakdowns in your LLM agents -- entirely offline, no API keys required.

Part of the [Pisama](https://pisama.ai) platform for multi-agent failure detection.

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

for detection in result.detections:
    print(f"[{detection.detector_name}] {detection.summary}")
    print(f"  Severity: {detection.severity}/100")
    print(f"  Fix: {detection.fix_recommendation.instruction}")
```

Output:
```
[loop] Tool 'Read' repeated 8x consecutively
  Severity: 45/100
  Fix: Stop the current loop. Try a different approach or ask the user for guidance.
```

No API key. No network calls. Runs completely locally.

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

Works with Claude Agent SDK, LangGraph, AutoGen, CrewAI, n8n, Dify, and custom agents.

## Pisama Platform

For production monitoring with 25+ calibrated detectors, ML-based detection, LLM-as-judge verification, and a dashboard, see [pisama.ai](https://pisama.ai).

## License

MIT
