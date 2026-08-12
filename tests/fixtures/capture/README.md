# Real-payload fixture capture

Scripts that regenerate the adapter test fixtures in the parent directory.

## Why this exists

An adapter test proves nothing about an integration if the test author wrote the
payload. A 2026-08-11 audit of every framework Pisama claims to support found
that the Google ADK adapter expected an event shape no released ADK has ever
produced, and that its test suite had been green the entire time because it fed
the adapter hand-written dictionaries built from the same wrong assumption. The
same pattern then turned up in the OpenAI, Bedrock, Dify and OpenClaw surfaces.

A hand-written payload tests the adapter's internals. Only a payload the vendor
produced tests the integration.

## The rule

**Every key in a fixture must come from the vendor. Only leaf values may be
ours.** How that is enforced depends on what the vendor ships:

| Vendor mechanism | Used for | What enforces the shape |
|---|---|---|
| The SDK's own tracing processor | `openai_agents_handoff_trace.json` | Payload is the verbatim output of `TraceImpl.export()` / `SpanImpl.export()` |
| The SDK's own typed models | `openai_assistants_run.json`, `openai_responses_api.json` | Instances are built through the `openai` package's pydantic models and dumped; pydantic rejects any field or literal the API cannot return |
| The vendor's published API model | `bedrock_invoke_agent_trace.json` | Keys are resolved against the `bedrock-agent-runtime` service model botocore ships, then the whole stream is round-tripped through botocore's own eventstream decoder |
| A live run of the framework | `docs/plans/adk-evidence/raw/` (Google ADK) | Verbatim capture of a real execution, including the framework's own OpenTelemetry spans |
| The SDK's own typed models | `gemini_interaction_*.json` | Instances are built through google-genai's `Interaction` model and dumped; see the caveat below |

None of these call a paid API, and the test suite gains no dependency on any
vendor package: the fixtures are committed JSON and the tests only read them.

## Regenerating

Each script writes straight into `tests/fixtures/`, the directory the test suite
reads. Use a throwaway venv, never a repo venv — the point of the pattern is that
the test suite never depends on a vendor package.

```bash
python3.11 -m venv /tmp/capture-openai && /tmp/capture-openai/bin/pip install openai
/tmp/capture-openai/bin/python capture_openai.py            # add --check to re-validate
```

```bash
python3.11 -m venv /tmp/capture-bedrock && /tmp/capture-bedrock/bin/pip install boto3
/tmp/capture-bedrock/bin/python capture_bedrock.py
```

Both scripts use a fixed clock, so regenerating against the same vendor version
reproduces the fixtures byte for byte. Verified: the two OpenAI fixtures come
back identical, and the Bedrock one differs only in its `_capture.captured_at`
stamp. Anything else that moves is a real vendor change and deserves reading,
not a rubber-stamped commit.

Each fixture records the exact vendor version it was built from, in
`_provenance` (OpenAI) or `_capture` (Bedrock). Several findings below are
version-specific, so check that field before trusting an old fixture.

## Caveat: google-genai is more lenient than the openai models

The OpenAI guarantee is the strong one: pydantic rejects any field or literal the
API cannot return. `google-genai`'s Interactions models are looser, and
`capture_gemini.py` compensates rather than pretending otherwise:

- `_gaos.BaseModel` sets `extra="allow"`, so invented keys are accepted.
- `status`, `model`, `agent` and `service_tier` are open enums
  (`Union[Literal[...], UnrecognizedStr]`), so an unknown value passes. The
  `Literal` documents the vocabulary, it does not gate it.
- Steps parse through `parse_open_union(..., lenient=True)`. An unknown `type`
  degrades to `UnknownStep`, and a *known* `type` with an invalid body degrades
  to an unvalidated construct rather than raising.

So `Interaction.model_validate` alone would accept some malformed payloads. The
script's `--check` mode therefore does two passes: the SDK's real read path, plus
a strict per-step re-validation through each concrete step class, which does
raise, and an assertion that no step landed on `UnknownStep`.

## What a capture is allowed to conclude

If the adapter cannot parse a payload the vendor's own machinery produced, the
adapter is wrong. Do not adjust the payload to make the test pass — that is
precisely how the original defects were hidden. Mark the gap with a strict
`xfail` and record the evidence, so the marker fails loudly once the adapter is
fixed.

The strict xfails currently recorded in `test_adapters_openai_bedrock.py`:

- A real `CodeInterpreterToolCall` arrives inside a `tool_calls` step, but
  `_parse_tool_call` reads only `call["function"]`, so the code and its outputs
  are dropped entirely. `FileSearchToolCall` loses its data the same way.
- The Responses API reports `input_tokens`/`output_tokens`, but the usage helper
  reads only `prompt_tokens`/`completion_tokens`, so that token split is lost.
  The Assistants path is unaffected.
- `TracePart.eventTime` and the per-node `Metadata` start/end times are all
  discarded, so Bedrock child spans fall back to ingestion time and carry no
  end time.
- `ModelInvocationInput.foundationModel` is read past without being emitted, so
  no Bedrock span carries `gen_ai.request.model`.

## Adapters still without a real-payload fixture

- **`deep_agents.py`** — consumes LangGraph state checkpoints; capturing them
  needs a graph execution, so it belongs with the ADK-style live-run harness
  rather than this offline pattern.
- **`google_adk.py`** — live captures already exist under
  `docs/plans/adk-evidence/raw/`. They are deliberately not wired in here,
  because the adapter cannot consume raw ADK events at all; that is a redesign,
  not a fixture.
