#!/usr/bin/env python3
"""Capture REAL-shaped OpenAI Assistants + Responses payloads for the Pisama adapter.

Principle (copied from tests/test_adapters_openai_agents.py): the fixtures are
NOT hand-written dicts. Every payload here is built by instantiating the
`openai` SDK's own pydantic models and serialising them with
`.model_dump(mode="json")`. Pydantic therefore validates the shape: a required
field on the vendor model is a required field in the fixture, a `Literal`
discriminator can only take a value the vendor actually defines, and a nested
object can only be one of the union members the vendor declares.

That is the whole point. If the adapter branches on a shape the SDK's types
cannot express, this script CANNOT produce it, and that branch is dead code.

No OpenAI model is called. No API key is needed. Only the type layer runs.

Usage:
    python capture_openai.py            # writes fixtures/*.json
    python capture_openai.py --check    # also re-validates the round trip
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import openai

# --- Assistants API (beta) types -------------------------------------------
from openai.types.beta.code_interpreter_tool import CodeInterpreterTool
from openai.types.beta.function_tool import FunctionTool as AssistantsFunctionTool
from openai.types.beta.threads.message import Message
from openai.types.beta.threads.run import LastError as RunLastError
from openai.types.beta.threads.run import Run
from openai.types.beta.threads.run import Usage as RunUsage
from openai.types.beta.threads.runs.code_interpreter_tool_call import (
    CodeInterpreter,
    CodeInterpreterOutputImage,
    CodeInterpreterOutputImageImage,
    CodeInterpreterOutputLogs,
    CodeInterpreterToolCall,
)
from openai.types.beta.threads.runs.function_tool_call import Function, FunctionToolCall
from openai.types.beta.threads.runs.message_creation_step_details import (
    MessageCreation,
    MessageCreationStepDetails,
)
from openai.types.beta.threads.runs.run_step import LastError as StepLastError
from openai.types.beta.threads.runs.run_step import RunStep
from openai.types.beta.threads.runs.run_step import Usage as StepUsage
from openai.types.beta.threads.runs.tool_calls_step_details import ToolCallsStepDetails
from openai.types.beta.threads.text import Text
from openai.types.beta.threads.text_content_block import TextContentBlock

# --- Responses API types ----------------------------------------------------
from openai.types.responses.function_tool import FunctionTool as ResponsesFunctionTool
from openai.types.responses.response import Response
from openai.types.responses.response_error import ResponseError
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary as ReasoningSummary
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from openai.types.shared.function_definition import FunctionDefinition

# Writes straight into tests/fixtures/, the location the test suite reads.
OUT = Path(__file__).resolve().parent.parent

# Fixed clock so the fixtures are byte-stable across re-captures.
T0 = 1_754_899_200  # 2026-08-11T08:00:00Z, Unix seconds (the unit OpenAI uses)

THREAD_ID = "thread_kV2p8QoR1sWnA4bXcTfLmZdY"
ASSISTANT_ID = "asst_9HxN3rQvBs7KtLpEeUwGmZaC"
RUN_ID = "run_7TqLm2XeRb9AvKdNpWzHsYcF"
USER_MSG_ID = "msg_1AqZxKr6PbTnVeLdWmSyHcUg"
ASSISTANT_MSG_ID = "msg_5BwYnJt3QcRmXeKdLpVaSzHo"


# ---------------------------------------------------------------------------
# Assistants API: refund-dispute run that fails on the final tool call.
# ---------------------------------------------------------------------------
def build_assistants_run() -> dict:
    """Order-4417 refund dispute. Lookup succeeds, code interpreter runs,
    the assistant posts an interim message, then `issue_refund` fails and the
    run ends `failed`.
    """
    run = Run(
        id=RUN_ID,
        object="thread.run",
        created_at=T0,
        assistant_id=ASSISTANT_ID,
        thread_id=THREAD_ID,
        status="failed",
        started_at=T0 + 1,
        failed_at=T0 + 37,
        last_error=RunLastError(
            code="server_error",
            message="The server had an error while processing your request.",
        ),
        model="gpt-4.1",
        instructions=(
            "You are a support agent. Look up the order, verify the shipping SLA, "
            "and issue a refund when the order shipped late."
        ),
        parallel_tool_calls=True,
        tools=[
            AssistantsFunctionTool(
                type="function",
                function=FunctionDefinition(
                    name="lookup_order",
                    description="Fetch an order by id.",
                    parameters={
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                ),
            ),
            AssistantsFunctionTool(
                type="function",
                function=FunctionDefinition(
                    name="issue_refund",
                    description="Refund an order.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "amount": {"type": "number"},
                        },
                        "required": ["order_id", "amount"],
                    },
                ),
            ),
            CodeInterpreterTool(type="code_interpreter"),
        ],
        tool_choice="auto",
        usage=RunUsage(prompt_tokens=1_842, completion_tokens=214, total_tokens=2_056),
        temperature=1.0,
        top_p=1.0,
        metadata={"channel": "support-web"},
    )

    steps = [
        # 1. Function tool call that succeeds.
        RunStep(
            id="step_2NcVpQr8XmLbKtEeAwZdHsYf",
            object="thread.run.step",
            created_at=T0 + 2,
            completed_at=T0 + 5,
            assistant_id=ASSISTANT_ID,
            thread_id=THREAD_ID,
            run_id=RUN_ID,
            status="completed",
            type="tool_calls",
            step_details=ToolCallsStepDetails(
                type="tool_calls",
                tool_calls=[
                    FunctionToolCall(
                        id="call_QmR4tLpXe9BvKcNdWzHsAy",
                        type="function",
                        function=Function(
                            name="lookup_order",
                            arguments='{"order_id": "4417"}',
                            output=(
                                '{"order_id": "4417", "status": "delivered", '
                                '"total": 89.99, "promised_days": 3, "actual_days": 9}'
                            ),
                        ),
                    )
                ],
            ),
            usage=StepUsage(prompt_tokens=612, completion_tokens=41, total_tokens=653),
        ),
        # 2. Code interpreter. NOTE: in the real SDK this is a TOOL CALL inside a
        #    `tool_calls` step, never a step `type` of its own. See the verdict
        #    printed by --check.
        RunStep(
            id="step_8FkWnJt5QcRmXbLdVpSaEz",
            object="thread.run.step",
            created_at=T0 + 6,
            completed_at=T0 + 19,
            assistant_id=ASSISTANT_ID,
            thread_id=THREAD_ID,
            run_id=RUN_ID,
            status="completed",
            type="tool_calls",
            step_details=ToolCallsStepDetails(
                type="tool_calls",
                tool_calls=[
                    CodeInterpreterToolCall(
                        id="call_Xr7NmKp2Vb8LtQcEeWdZsA",
                        type="code_interpreter",
                        code_interpreter=CodeInterpreter(
                            input=(
                                "promised, actual = 3, 9\n"
                                "sla_breach_days = actual - promised\n"
                                "refund = round(89.99 * 0.5, 2) "
                                "if sla_breach_days >= 5 else 0.0\n"
                                "print(sla_breach_days, refund)"
                            ),
                            outputs=[
                                CodeInterpreterOutputLogs(type="logs", logs="6 45.0\n"),
                                CodeInterpreterOutputImage(
                                    type="image",
                                    image=CodeInterpreterOutputImageImage(
                                        file_id="file-Lq3ZnVt7Rb5KcXmEeWpDsA"
                                    ),
                                ),
                            ],
                        ),
                    )
                ],
            ),
            usage=StepUsage(prompt_tokens=704, completion_tokens=98, total_tokens=802),
        ),
        # 3. Interim assistant message.
        RunStep(
            id="step_3JhTnQr9XmVbKpLdEeWzAc",
            object="thread.run.step",
            created_at=T0 + 20,
            completed_at=T0 + 24,
            assistant_id=ASSISTANT_ID,
            thread_id=THREAD_ID,
            run_id=RUN_ID,
            status="completed",
            type="message_creation",
            step_details=MessageCreationStepDetails(
                type="message_creation",
                message_creation=MessageCreation(message_id=ASSISTANT_MSG_ID),
            ),
            usage=StepUsage(prompt_tokens=418, completion_tokens=52, total_tokens=470),
        ),
        # 4. The failure: refund tool call never returns an output.
        RunStep(
            id="step_6PmYcKt4Rb8NvXqLdEeWzH",
            object="thread.run.step",
            created_at=T0 + 25,
            failed_at=T0 + 37,
            assistant_id=ASSISTANT_ID,
            thread_id=THREAD_ID,
            run_id=RUN_ID,
            status="failed",
            type="tool_calls",
            last_error=StepLastError(
                code="server_error",
                message="The server had an error while processing your request.",
            ),
            step_details=ToolCallsStepDetails(
                type="tool_calls",
                tool_calls=[
                    FunctionToolCall(
                        id="call_Ze8QmRt3Xp7LbKcNvWdHsY",
                        type="function",
                        function=Function(
                            name="issue_refund",
                            arguments='{"order_id": "4417", "amount": 45.0}',
                            output=None,  # never submitted: the call failed
                        ),
                    )
                ],
            ),
        ),
    ]

    thread_messages = [
        Message(
            id=USER_MSG_ID,
            object="thread.message",
            created_at=T0 - 12,
            thread_id=THREAD_ID,
            role="user",
            status="completed",
            content=[
                TextContentBlock(
                    type="text",
                    text=Text(
                        value=(
                            "Order 4417 arrived six days after the promised date. "
                            "Can you check it and refund me if it shipped late?"
                        ),
                        annotations=[],
                    ),
                )
            ],
        ),
        Message(
            id=ASSISTANT_MSG_ID,
            object="thread.message",
            created_at=T0 + 24,
            thread_id=THREAD_ID,
            role="assistant",
            assistant_id=ASSISTANT_ID,
            run_id=RUN_ID,
            status="completed",
            content=[
                TextContentBlock(
                    type="text",
                    text=Text(
                        value=(
                            "Order 4417 was delivered 6 days past the 3-day SLA, so a "
                            "50% refund of $45.00 applies. Processing that now."
                        ),
                        annotations=[],
                    ),
                )
            ],
        ),
    ]

    return {
        "run": run.model_dump(mode="json"),
        "steps": [s.model_dump(mode="json") for s in steps],
        "thread_messages": [m.model_dump(mode="json") for m in thread_messages],
    }


# ---------------------------------------------------------------------------
# Responses API: same dispute, same failure, expressed as a Response.
# ---------------------------------------------------------------------------
def build_response() -> dict:
    """Response covering every output-item type the adapter branches on:
    `reasoning`, `function_call`, `function_call_output`, `message`.
    """
    # Built first and passed into the constructor: pydantic v2 does not
    # validate on assignment, so appending to `response.output` afterwards
    # would skip the discriminated-union check that is the point of this file.
    output = [
        ResponseReasoningItem(
            id="rs_9WvKmQt4Xb7NpLcEeRdZsA",
            type="reasoning",
            status="completed",
            summary=[
                ReasoningSummary(
                    type="summary_text",
                    text=(
                        "The order missed its 3-day SLA by 6 days, which clears the "
                        "5-day threshold for a 50% refund. Calling issue_refund."
                    ),
                )
            ],
        ),
        ResponseFunctionToolCall(
            id="fc_2QmRt8Xp5LbKcNvWdHsYeZ",
            type="function_call",
            call_id="call_Ze8QmRt3Xp7LbKcNvWdHsY",
            name="issue_refund",
            arguments='{"order_id": "4417", "amount": 45.0}',
            status="completed",
        ),
        ResponseFunctionToolCallOutputItem(
            id="fco_7LbKcNvWdHsYeZ2QmRt8Xp",
            type="function_call_output",
            call_id="call_Ze8QmRt3Xp7LbKcNvWdHsY",
            status="incomplete",
            output=(
                '{"error": "upstream_timeout", "detail": "payments gateway did not '
                'respond within 30s", "retryable": true}'
            ),
        ),
        ResponseOutputMessage(
            id="msg_5NvWdHsYeZ2QmRt8Xp7LbK",
            type="message",
            role="assistant",
            status="incomplete",
            content=[
                ResponseOutputText(
                    type="output_text",
                    text=(
                        "I confirmed order 4417 shipped late and started a $45.00 "
                        "refund, but the payments gateway timed out before it "
                        "settled. Nothing has been charged back yet."
                    ),
                    annotations=[],
                )
            ],
        ),
    ]

    response = Response(
        id="resp_4XnQmKt7Rb2LvPcEeWdZsAyH",
        object="response",
        output=output,
        created_at=float(T0 + 100),
        completed_at=float(T0 + 131),
        model="gpt-4.1",
        status="failed",
        error=ResponseError(
            code="server_error",
            message="The server had an error while processing your request.",
        ),
        instructions="Refund an order when it shipped past the promised SLA.",
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[
            ResponsesFunctionTool(
                type="function",
                name="issue_refund",
                description="Refund an order.",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "amount": {"type": "number"},
                    },
                    "required": ["order_id", "amount"],
                },
                strict=True,
            )
        ],
        usage=ResponseUsage(
            input_tokens=1_204,
            input_tokens_details=InputTokensDetails(cached_tokens=896, cache_write_tokens=0),
            output_tokens=188,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=128),
            total_tokens=1_392,
        ),
        metadata={"channel": "support-web"},
    )

    # Prove the union members survived validation as their declared types
    # rather than being coerced into some other branch.
    assert [type(i).__name__ for i in response.output] == [
        "ResponseReasoningItem",
        "ResponseFunctionToolCall",
        "ResponseFunctionToolCallOutputItem",
        "ResponseOutputMessage",
    ], [type(i).__name__ for i in response.output]

    return response.model_dump(mode="json")


# ---------------------------------------------------------------------------
def provenance(scenario: str) -> dict:
    return {
        "source": f"openai {openai.__version__} pydantic models",
        "generated": "2026-08-11",
        "how": (
            "Built by instantiating the SDK's own pydantic models "
            "(openai.types.beta.threads.Run / RunStep / Message, "
            "openai.types.responses.Response) and serialising with "
            "model_dump(mode='json'). Pydantic validated every required field, "
            "Literal discriminator, and discriminated-union member, so the shape "
            "is the vendor's, not the author's."
        ),
        "no_model_calls": (
            "No OpenAI model was called and no API key is needed. Only the type "
            "layer ran, so this fixture adds no vendor dependency to the tests."
        ),
        "field_values_are_authored": (
            "The SHAPE is the vendor's; the VALUES (ids, text, timings) are "
            "authored to make a coherent scenario, exactly as the "
            "openai_agents_handoff_trace.json fixture does."
        ),
        "scenario": scenario,
    }


ASSISTANTS_SCENARIO = (
    "Order-4417 late-shipment refund. lookup_order succeeds, a code_interpreter "
    "tool call computes the 50% refund, the assistant posts an interim message, "
    "then issue_refund fails with a server_error and the run ends `failed`."
)
RESPONSES_SCENARIO = (
    "The same refund, via the Responses API. Output items cover reasoning, "
    "function_call, function_call_output (a gateway-timeout error payload) and "
    "message; the response itself ends `failed` with a top-level error."
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    assistants = build_assistants_run()
    assistants = {"_provenance": provenance(ASSISTANTS_SCENARIO), **assistants}
    (OUT / "openai_assistants_run.json").write_text(json.dumps(assistants, indent=2) + "\n")

    resp = build_response()
    resp = {"_provenance": provenance(RESPONSES_SCENARIO), **resp}
    (OUT / "openai_responses_api.json").write_text(json.dumps(resp, indent=2) + "\n")

    print(f"openai=={openai.__version__}")
    print(f"wrote {OUT / 'openai_assistants_run.json'}")
    print(f"wrote {OUT / 'openai_responses_api.json'}")

    if "--check" in sys.argv:
        check_round_trip(assistants, resp)
    return 0


def check_round_trip(assistants: dict, resp: dict) -> None:
    """Re-validate the dumped JSON back through the SDK models.

    If model_validate accepts it, the JSON is a payload the SDK itself would
    have produced from the wire.
    """
    Run.model_validate({k: v for k, v in assistants["run"].items()})
    for s in assistants["steps"]:
        RunStep.model_validate(s)
    for m in assistants["thread_messages"]:
        Message.model_validate(m)
    Response.model_validate({k: v for k, v in resp.items() if k != "_provenance"})
    print("round-trip: SDK models re-validated every dumped payload")

    # --- Suspected-bug probes, answered by the vendor types themselves ---
    import typing

    step_types = typing.get_args(RunStep.model_fields["type"].annotation)
    print(f"\nRunStep.type Literal = {step_types}")
    print(
        "  code_interpreter is a valid RunStep.type? "
        f"{'code_interpreter' in step_types}"
    )
    step_detail_members = [
        m.__name__ for m in typing.get_args(RunStep.model_fields["step_details"].annotation)
    ]
    print(f"  RunStep.step_details union = {step_detail_members}")

    # output: List[Annotated[Union[...], PropertyInfo]] -> unwrap List, then
    # Annotated, then Union.
    item_t = typing.get_args(Response.model_fields["output"].annotation)[0]
    if typing.get_origin(item_t) is not typing.Union:
        item_t = typing.get_args(item_t)[0]  # strip Annotated
    out_members = [getattr(m, "__name__", str(m)) for m in typing.get_args(item_t)]
    print(f"\nResponse.output declares {len(out_members)} item types")
    print(
        "  ResponseFunctionToolCallOutputItem present? "
        f"{'ResponseFunctionToolCallOutputItem' in out_members}"
    )
    print(
        "  its discriminator = "
        f"{typing.get_args(ResponseFunctionToolCallOutputItem.model_fields['type'].annotation)}"
    )

    print(
        "\nResponseUsage token keys = "
        f"{sorted(k for k in ResponseUsage.model_fields if 'token' in k)}"
    )
    print(f"Run.usage token keys      = {sorted(RunUsage.model_fields)}")


if __name__ == "__main__":
    raise SystemExit(main())
