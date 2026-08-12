#!/usr/bin/env python3
"""Capture REAL-shaped Gemini Interactions API payloads for the Pisama adapter.

Principle (copied verbatim from capture_openai.py): the fixtures are NOT
hand-written dicts. Every payload here is built by instantiating the
`google-genai` SDK's own pydantic models and serialising them with
`.model_dump(mode="json")`. Pydantic therefore validates the shape at
construction: a required field on the vendor model is a required field in the
fixture, a `Literal` discriminator can only take a value the vendor actually
defines, and a nested object can only be one of the union members the vendor
declares.

That is the whole point. If the adapter branches on a shape the SDK's types
cannot express, this script CANNOT produce it, and that branch is dead code.

No Gemini model is called. No API key is needed. Only the type layer runs.

Two caveats specific to google-genai, both reported by --check rather than
papered over:

  1. `_gaos` BaseModel sets `extra="allow"`, so EXTRA keys are not rejected the
     way `openai`'s models reject them. The guarantee here covers required
     fields, Literal discriminators and union membership, not key closure.
  2. `Step`/`Content`/`Tool` are OPEN unions parsed by `parse_open_union(...,
     lenient=True)`. On the read path an unknown `type` degrades to
     `UnknownStep`, and a KNOWN `type` whose body fails validation degrades to
     `construct_unvalidated(...)` instead of raising. So `Interaction.
     model_validate` alone is a weak check. `--check` therefore also
     re-validates every dumped step through its CONCRETE class, which does
     raise, and asserts no step landed on `UnknownStep`.

Usage:
    python capture_gemini.py            # writes gemini_interaction_*.json
    python capture_gemini.py --check    # also re-validates the round trip
"""

from __future__ import annotations

import json
import sys
import typing
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path

# --- Interactions API types -------------------------------------------------
# Defining modules, addressed exactly. `google.genai.interactions` re-exports
# every one of these names; --check proves the two paths are the same objects.
from google.genai._gaos.types.interactions.error import Error
from google.genai._gaos.types.interactions.function import Function as FunctionTool
from google.genai._gaos.types.interactions.functioncallstep import FunctionCallStep
from google.genai._gaos.types.interactions.functionresultstep import FunctionResultStep
from google.genai._gaos.types.interactions.generationconfig import GenerationConfig
from google.genai._gaos.types.interactions.groundingtoolcount import GroundingToolCount
from google.genai._gaos.types.interactions.interaction import Interaction
from google.genai._gaos.types.interactions.modalitytokens import ModalityTokens
from google.genai._gaos.types.interactions.modeloutputstep import ModelOutputStep
from google.genai._gaos.types.interactions.status import Status
from google.genai._gaos.types.interactions.step import UnknownStep
from google.genai._gaos.types.interactions.textcontent import TextContent
from google.genai._gaos.types.interactions.thoughtstep import ThoughtStep
from google.genai._gaos.types.interactions.usage import Usage
from google.genai._gaos.types.interactions.userinputstep import UserInputStep

OUT = Path(__file__).resolve().parent

GOOGLE_GENAI_VERSION = metadata.version("google-genai")

# Fixed clock so the fixtures are byte-stable across re-captures. The
# Interactions API carries `created`/`updated` as ISO 8601 strings, not epochs.
T0 = datetime(2026, 8, 11, 8, 0, 0, tzinfo=timezone.utc)


def ts(seconds: int) -> str:
    """ISO 8601 (YYYY-MM-DDThh:mm:ssZ), the format the vendor documents."""
    return (T0 + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


MODEL = "gemini-3.6-flash"

COMPLETED_ID = "interactions/01K9QF7ZC4M2XNVD8TRJ6BPHWA"
FAILED_ID = "interactions/01K9QG2M8RD5YTWQ3FNXB7VKHE"
PREVIOUS_ID = "interactions/01K9QF4A1JB6ZMSK9DHVT2CXPR"

SYSTEM_INSTRUCTION = (
    "You are a support agent. Look up the order, verify the shipping SLA, and "
    "issue a refund when the order shipped later than promised."
)

LOOKUP_ORDER_TOOL = FunctionTool(
    type="function",
    name="lookup_order",
    description="Fetch an order by id.",
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
)

ISSUE_REFUND_TOOL = FunctionTool(
    type="function",
    name="issue_refund",
    description="Refund an order.",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "amount_usd": {"type": "number"},
        },
        "required": ["order_id", "amount_usd"],
    },
)

GENERATION_CONFIG = GenerationConfig(
    max_output_tokens=2048,
    seed=7,
    thinking_level="low",
    thinking_summaries="auto",
)


# ---------------------------------------------------------------------------
# Fixture 1: a successful multi-step interaction.
# ---------------------------------------------------------------------------
def build_completed() -> Interaction:
    """Order-4417 late-shipment refund, start to finish.

    user_input -> thought -> function_call -> function_result -> model_output.
    """
    # Built first and passed into the constructor: pydantic v2 does not validate
    # on assignment, so appending to `interaction.steps` afterwards would skip
    # the open-union parse that is the point of this file.
    steps = [
        UserInputStep(
            type="user_input",
            content=[
                TextContent(
                    type="text",
                    text=(
                        "Order 4417 turned up six days after the date you promised. "
                        "Can you check it and refund me if it shipped late?"
                    ),
                )
            ],
        ),
        ThoughtStep(
            type="thought",
            signature="Qk9HVVMtVEhPVUdIVC1TSUdOQVRVUkUtNDQxNw==",
            summary=[
                TextContent(
                    type="text",
                    text=(
                        "I need the order record before I can judge the SLA. "
                        "Calling lookup_order with order_id 4417."
                    ),
                )
            ],
        ),
        FunctionCallStep(
            type="function_call",
            id="call_9QF7ZC4M2XNVD8TRJ6BPHW",
            name="lookup_order",
            arguments={"order_id": "4417"},
        ),
        FunctionResultStep(
            type="function_result",
            call_id="call_9QF7ZC4M2XNVD8TRJ6BPHW",
            name="lookup_order",
            is_error=False,
            # The `result` union's object arm. `FunctionResultStepResult` is an
            # empty model with extra="allow", so an arbitrary JSON object is a
            # shape the vendor genuinely accepts here.
            result={
                "order_id": "4417",
                "status": "delivered",
                "total_usd": 89.99,
                "promised_days": 3,
                "actual_days": 9,
            },
        ),
        ModelOutputStep(
            type="model_output",
            content=[
                TextContent(
                    type="text",
                    text=(
                        "Order 4417 was delivered 9 days out against a 3-day promise, "
                        "so it clears the 5-day threshold for a 50% refund. That is "
                        "$45.00 back to your original payment method, and it should "
                        "settle within 3 business days."
                    ),
                )
            ],
        ),
    ]

    return Interaction(
        status="completed",
        id=COMPLETED_ID,
        model=MODEL,
        created=ts(0),
        updated=ts(31),
        previous_interaction_id=PREVIOUS_ID,
        service_tier="standard",
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[LOOKUP_ORDER_TOOL, ISSUE_REFUND_TOOL],
        generation_config=GENERATION_CONFIG,
        labels={"channel": "support-web", "team": "billing"},
        input="Order 4417 turned up six days after the date you promised.",
        usage=Usage(
            total_input_tokens=1842,
            total_output_tokens=214,
            total_thought_tokens=96,
            total_cached_tokens=1280,
            total_tool_use_tokens=57,
            total_tokens=2209,
            input_tokens_by_modality=[ModalityTokens(modality="text", tokens=1842)],
            output_tokens_by_modality=[ModalityTokens(modality="text", tokens=214)],
            cached_tokens_by_modality=[ModalityTokens(modality="text", tokens=1280)],
            tool_use_tokens_by_modality=[ModalityTokens(modality="text", tokens=57)],
            grounding_tool_count=[GroundingToolCount(type="google_search", count=0)],
        ),
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Fixture 2: the same job, failing at the refund call.
# ---------------------------------------------------------------------------
def build_failed() -> Interaction:
    """Same dispute. The refund tool errors, the model output carries a
    `Status`, and the interaction ends `failed` with `errors[]` populated.
    """
    steps = [
        UserInputStep(
            type="user_input",
            content=[
                TextContent(
                    type="text",
                    text=(
                        "Order 4417 shipped late and support already approved the "
                        "refund. Please push it through."
                    ),
                )
            ],
        ),
        ThoughtStep(
            type="thought",
            signature="Qk9HVVMtVEhPVUdIVC1TSUdOQVRVUkUtRkFJTA==",
            summary=[
                TextContent(
                    type="text",
                    text=(
                        "The SLA breach is already established, so I can skip the "
                        "lookup and call issue_refund for $45.00 directly."
                    ),
                )
            ],
        ),
        FunctionCallStep(
            type="function_call",
            id="call_QG2M8RD5YTWQ3FNXB7VKHE",
            name="issue_refund",
            arguments={"order_id": "4417", "amount_usd": 45.0},
        ),
        FunctionResultStep(
            type="function_result",
            call_id="call_QG2M8RD5YTWQ3FNXB7VKHE",
            name="issue_refund",
            is_error=True,
            # The `result` union's subcontent arm: List[FunctionResultSubcontent],
            # whose members are TextContent / ImageContent.
            result=[
                TextContent(
                    type="text",
                    text=(
                        "upstream_timeout: the payments gateway did not respond "
                        "within 30s. No refund was created; the call is retryable."
                    ),
                )
            ],
        ),
        ModelOutputStep(
            type="model_output",
            content=[
                TextContent(
                    type="text",
                    text=(
                        "I tried to issue the $45.00 refund on order 4417, but the "
                        "payments gateway timed out before it settled. Nothing has "
                        "been credited yet."
                    ),
                )
            ],
            error=Status(
                code=4,
                message="Deadline exceeded while calling tool issue_refund.",
                details=[
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "TOOL_CALL_DEADLINE_EXCEEDED",
                        "domain": "generativelanguage.googleapis.com",
                        "metadata": {"tool": "issue_refund", "timeout_s": "30"},
                    }
                ],
            ),
        ),
    ]

    return Interaction(
        status="failed",
        id=FAILED_ID,
        model=MODEL,
        created=ts(120),
        updated=ts(163),
        previous_interaction_id=COMPLETED_ID,
        service_tier="standard",
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[LOOKUP_ORDER_TOOL, ISSUE_REFUND_TOOL],
        generation_config=GENERATION_CONFIG,
        labels={"channel": "support-web", "team": "billing"},
        input="Order 4417 shipped late and support already approved the refund.",
        errors=[
            Error(
                code="https://ai.google.dev/api/errors#TOOL_EXECUTION_FAILED",
                message=(
                    "Tool issue_refund failed: upstream_timeout from the payments "
                    "gateway after 30s."
                ),
            ),
            Error(
                code="https://ai.google.dev/api/errors#INTERACTION_FAILED",
                message="The interaction ended before the requested action completed.",
            ),
        ],
        usage=Usage(
            total_input_tokens=1104,
            total_output_tokens=142,
            total_thought_tokens=71,
            total_cached_tokens=896,
            total_tool_use_tokens=44,
            total_tokens=1361,
            input_tokens_by_modality=[ModalityTokens(modality="text", tokens=1104)],
            output_tokens_by_modality=[ModalityTokens(modality="text", tokens=142)],
        ),
        steps=steps,
    )


# ---------------------------------------------------------------------------
EXPECTED_STEP_CLASSES = [
    "UserInputStep",
    "ThoughtStep",
    "FunctionCallStep",
    "FunctionResultStep",
    "ModelOutputStep",
]


def assert_union_members(interaction: Interaction, label: str) -> None:
    """Prove each step survived the open union as its declared concrete type.

    `parse_open_union(..., lenient=True)` degrades silently, so this is the
    check that actually has teeth.
    """
    got = [type(s).__name__ for s in interaction.steps or []]
    assert got == EXPECTED_STEP_CLASSES, f"{label}: {got}"
    assert not any(isinstance(s, UnknownStep) for s in interaction.steps or []), label


def provenance(scenario: str) -> dict:
    return {
        "source": f"google-genai {GOOGLE_GENAI_VERSION} pydantic models",
        "generated": "2026-08-11",
        "how": (
            "Built by instantiating the SDK's own pydantic models "
            "(google.genai._gaos.types.interactions.Interaction and the Step "
            "union members) and serialising with model_dump(mode='json'). "
            "Pydantic validated every required field, Literal discriminator and "
            "open-union member at construction, so the shape is the vendor's, "
            "not the author's."
        ),
        "no_model_calls": (
            "No Gemini model was called and no API key is needed. Only the type "
            "layer ran, so this fixture adds no vendor dependency to the tests."
        ),
        "field_values_are_authored": (
            "The SHAPE is the vendor's; the VALUES (ids, text, timings, token "
            "counts) are authored to make a coherent scenario, exactly as the "
            "openai_*.json fixtures do."
        ),
        "shape_guarantee_limits": (
            "google-genai's _gaos BaseModel sets extra='allow', and Step is an "
            "OPEN union parsed leniently, so unknown keys are not rejected on "
            "the read path. The guarantee covers required fields, Literal "
            "discriminators and union membership at CONSTRUCTION time; "
            "capture_gemini.py --check adds a strict per-class re-validation "
            "because Interaction.model_validate alone would not raise."
        ),
        "scenario": scenario,
    }


COMPLETED_SCENARIO = (
    "Order-4417 late-shipment refund that succeeds. Steps run user_input -> "
    "thought -> function_call(lookup_order) -> function_result(is_error=False, "
    "object result) -> model_output; usage is fully populated including "
    "per-modality breakdowns, and the interaction ends `completed`."
)
FAILED_SCENARIO = (
    "The same refund, failing. Steps run user_input -> thought -> "
    "function_call(issue_refund) -> function_result(is_error=True, subcontent "
    "result) -> model_output carrying a google.rpc.Status error; the "
    "interaction ends `failed` with two entries in errors[]."
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    completed = build_completed()
    assert_union_members(completed, "completed")
    failed = build_failed()
    assert_union_members(failed, "failed")

    completed_json = {
        "_provenance": provenance(COMPLETED_SCENARIO),
        **completed.model_dump(mode="json"),
    }
    failed_json = {
        "_provenance": provenance(FAILED_SCENARIO),
        **failed.model_dump(mode="json"),
    }

    completed_path = OUT / "gemini_interaction_completed.json"
    failed_path = OUT / "gemini_interaction_failed.json"
    completed_path.write_text(json.dumps(completed_json, indent=2) + "\n")
    failed_path.write_text(json.dumps(failed_json, indent=2) + "\n")

    print(f"google-genai=={GOOGLE_GENAI_VERSION}")
    print(f"wrote {completed_path}")
    print(f"wrote {failed_path}")

    if "--check" in sys.argv:
        check_round_trip(completed_json, failed_json)
    return 0


def strip_provenance(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "_provenance"}


def check_round_trip(completed_json: dict, failed_json: dict) -> None:
    """Re-validate the dumped JSON back through the SDK models.

    Two passes, because one is not enough here:

      1. `Interaction.model_validate` — the SDK's own read path
         (`unmarshal_json_response` calls `unmarshal_json(body, Interaction)`).
         If it accepts, the JSON is a payload the SDK would have produced.
      2. Per-step validation through the CONCRETE class. Pass 1 routes steps
         through `parse_open_union(..., lenient=True)`, which swallows a bad
         body into `construct_unvalidated`. Pass 2 raises instead.
    """
    concrete = {
        "user_input": UserInputStep,
        "thought": ThoughtStep,
        "function_call": FunctionCallStep,
        "function_result": FunctionResultStep,
        "model_output": ModelOutputStep,
    }

    for label, payload in (("completed", completed_json), ("failed", failed_json)):
        raw = strip_provenance(payload)
        revalidated = Interaction.model_validate(raw)
        assert_union_members(revalidated, f"{label} (round trip)")
        assert revalidated.status == raw["status"], label
        for step in raw["steps"]:
            cls = concrete[step["type"]]
            cls.model_validate(step)  # strict: no lenient fallback in this path
        # The dump must be a fixed point: re-dumping the re-validated model
        # reproduces it byte for byte.
        assert revalidated.model_dump(mode="json") == raw, label
    print(
        "round-trip: Interaction.model_validate accepted both payloads, every "
        "step re-validated strictly through its concrete class, and both dumps "
        "are fixed points"
    )

    # --- Facts the adapter rewrite needs, answered by the vendor types --------
    from google.genai import interactions as public_interactions
    from google.genai._gaos.types.interactions import step as step_mod

    print("\n-- import paths --")
    print("  defining module: google.genai._gaos.types.interactions.interaction")
    print("  public re-export: google.genai.interactions")
    for name, obj in (
        ("Interaction", Interaction),
        ("UserInputStep", UserInputStep),
        ("ModelOutputStep", ModelOutputStep),
        ("ThoughtStep", ThoughtStep),
        ("FunctionCallStep", FunctionCallStep),
        ("FunctionResultStep", FunctionResultStep),
    ):
        same = getattr(public_interactions, name, None) is obj
        print(f"  google.genai.interactions.{name} is the same object: {same}")

    print("\n-- Step open union: discriminator -> class --")
    for disc, cls in sorted(step_mod._STEP_VARIANTS.items()):
        fields = ", ".join(cls.model_fields)
        print(f"  {disc:<24} {cls.__name__:<24} fields: {fields}")
    print(f"  (unknown discriminator falls back to {UnknownStep.__name__})")

    print("\n-- Interaction status Literal --")
    status_ann = Interaction.model_fields["status"].annotation
    print(f"  {typing.get_args(typing.get_args(status_ann)[0])}")

    print("\n-- serialization aliases (python attr -> JSON key) --")
    for cls in (
        Interaction,
        Usage,
        Error,
        Status,
        UserInputStep,
        ModelOutputStep,
        ThoughtStep,
        FunctionCallStep,
        FunctionResultStep,
        TextContent,
    ):
        renamed = {
            n: (f.serialization_alias or f.alias)
            for n, f in cls.model_fields.items()
            if (f.serialization_alias or f.alias) and (f.serialization_alias or f.alias) != n
        }
        print(f"  {cls.__name__:<20} {renamed or 'none (JSON key == attr name)'}")

    print("\n-- keys actually emitted --")
    top = strip_provenance(completed_json)
    print(f"  Interaction: {sorted(top)}")
    print(f"  usage:       {sorted(top['usage'])}")
    for step in top["steps"]:
        print(f"  step {step['type']:<18} {sorted(step)}")
    fail_top = strip_provenance(failed_json)
    print(f"  errors[0]:   {sorted(fail_top['errors'][0])}")
    fail_output = fail_top["steps"][-1]
    print(f"  model_output.error: {sorted(fail_output['error'])}")

    print("\n-- FunctionResultStep.result union arms --")
    for arm, value in (
        ("object", {"ok": True}),
        ("subcontent list", [{"type": "text", "text": "hi"}]),
        ("plain string", "hi"),
    ):
        parsed = FunctionResultStep.model_validate(
            {"type": "function_result", "call_id": "c1", "result": value}
        )
        print(f"  {arm:<16} -> {type(parsed.result).__name__}")

    guarantee_probes()


def guarantee_probes() -> None:
    """Show exactly where the vendor types bite and where they do not.

    The README claims a fixture's shape is the vendor's. For `openai` that
    holds unconditionally. For `google-genai` it holds for required fields,
    field TYPES and step discriminators, but NOT for extra keys or for the
    open string enums. Printing both columns keeps the claim honest.
    """
    print("\n-- what the vendor types reject (and what they wave through) --")

    def probe(label: str, fn) -> None:
        try:
            fn()
            print(f"  ACCEPTED  {label}")
        except Exception as exc:
            print(f"  REJECTED  {label}  [{type(exc).__name__}]")

    probe(
        "FunctionCallStep without required `id`",
        lambda: FunctionCallStep(type="function_call", name="x", arguments={}),
    )
    probe(
        "FunctionCallStep type='tool_call' (not the declared Literal)",
        lambda: FunctionCallStep(type="tool_call", id="a", name="x", arguments={}),
    )
    probe(
        "FunctionCallStep `arguments` as a JSON string, not a dict",
        lambda: FunctionCallStep(type="function_call", id="a", name="x", arguments='{"a": 1}'),
    )
    probe(
        "FunctionResultStep without required `result`",
        lambda: FunctionResultStep(type="function_result", call_id="c"),
    )
    probe(
        "TextContent without required `text`",
        lambda: TextContent(type="text"),
    )
    probe(
        "ThoughtStep `summary` as a plain str, not a list",
        lambda: ThoughtStep(type="thought", summary="hi"),
    )
    probe(
        "Interaction `created` as an epoch int (the API uses ISO 8601)",
        lambda: Interaction(status="completed", created=1754899200),
    )
    probe(
        "ModelOutputStep.error shaped like Error (str code), not Status (int code)",
        lambda: ModelOutputStep(type="model_output", error={"code": "http://x", "message": "m"}),
    )
    probe(
        "Interaction without required `status`",
        lambda: Interaction(id="x"),
    )
    # The three below ACCEPT. They are the limits of the guarantee, not bugs
    # in this script.
    probe(
        "ThoughtStep with an invented `text` field (BaseModel extra='allow')",
        lambda: ThoughtStep(type="thought", text="hi"),
    )
    probe(
        "Interaction status='succeeded' (open enum: UnrecognizedStr fallback)",
        lambda: Interaction(status="succeeded"),
    )
    probe(
        "Interaction with a top-level `error` key (no such field; extra='allow')",
        lambda: Interaction(status="failed", error={"code": "x"}),
    )

    print("\n-- `output_text` is DERIVED, never authored --")
    derived = Interaction(
        status="completed",
        output_text="AUTHORED VALUE",
        steps=[
            ModelOutputStep(
                type="model_output",
                content=[TextContent(type="text", text="DERIVED FROM THE STEP")],
            )
        ],
    )
    print(f"  passed 'AUTHORED VALUE', model holds {derived.output_text!r}")
    empty = Interaction(status="completed", output_text="AUTHORED VALUE", steps=[])
    print(f"  with no trailing model_output, model holds {empty.output_text!r}")

    print("\n-- why --check re-validates steps concretely --")
    lenient = Interaction.model_validate(
        {"status": "completed", "steps": [{"type": "function_call", "name": "x"}]}
    )
    print(
        "  Interaction.model_validate on a function_call step missing required "
        f"`id`/`arguments`: silently kept as {type(lenient.steps[0]).__name__}"
    )
    try:
        FunctionCallStep.model_validate({"type": "function_call", "name": "x"})
        print("  FunctionCallStep.model_validate on the same body: ACCEPTED")
    except Exception as exc:
        print(
            f"  FunctionCallStep.model_validate on the same body: REJECTED [{type(exc).__name__}]"
        )


if __name__ == "__main__":
    raise SystemExit(main())
