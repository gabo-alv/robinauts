# SPDX-License-Identifier: Apache-2.0
# Copyright The Robinauts Authors

"""Both engines over OpenAI's Chat Completions: one behaviour, held of each.

What the two engines promise alike over the OpenAI kinds, written once and run
under each (the engine is in every test's id). The client half is asserted on
the constructed object; the turn half runs through each engine's **real**
client and real framework over a vendor the test writes
(``tests/chat_completions.py``), because what matters there is what the
framework really sends and really makes of a stream. Nothing reaches a
provider.

What is **not** here is what one engine does and the other does not, or what
only one engine's objects can be asked: those stay in the engine's own module
(``tests/unit/test_langgraph_engine.py``, ``tests/unit/test_pydantic_ai_engine.py``)
-- the client's own fields, the one shape of stream the two take differently (a
name before its id), what each makes of a reasoning field, a call the final
message holds that never streamed, and each framework's own tracing. The
request both send for one history is compared across the two in
``tests/unit/test_engine_swap.py``.

This module names both engines, as the swap fixtures do, and is one of them for
the discard test (``docs/layout.md``).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import openai
import pytest
import tiktoken

import chat_completions
from aio import asyncio_test
from chat_completions import (
    GATEWAY_ENDPOINT,
    GATEWAY_PROVIDER,
    GPT,
    KEY,
    OPENAI_PROVIDER,
    SYSTEM_PROMPT,
    call_delta,
)
from contracts.agents import SEARCH
from conversations import provenance
from engines import ADAPTERS, chat_completions_turn, over_chat_completions
from robinauts.adapters import ProviderKeys
from robinauts.core import check_engine_events
from robinauts.domain import (
    AnswerCompleted,
    AnswerReasoningDelta,
    AnswerStarted,
    AnswerTextDelta,
    ConfigError,
    Engine,
    ModelConfig,
    ModelProviderConfig,
    ProviderKind,
    TextPart,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallPart,
    ToolCallStarted,
    UnsupportedContentError,
    WaitingOnTools,
)

OPENAI_ENDPOINT = "https://api.openai.com/v1"
"""OpenAI's own endpoint, which both engines pin (each adapter's ``OPENAI_ENDPOINT``)."""

COMPATIBLE_ENDPOINT = "https://openrouter.ai/api"
"""An endpoint of the other protocol's, which an unbuilt kind might come with."""

ENGINES = pytest.mark.parametrize("engine", sorted(Engine), ids=lambda engine: engine.value)
"""Every test here, under each engine, with the engine in the test's id."""

KINDS = pytest.mark.parametrize(
    ("provider", "endpoint"),
    [(OPENAI_PROVIDER, OPENAI_ENDPOINT), (GATEWAY_PROVIDER, GATEWAY_ENDPOINT)],
    ids=["openai", "openai-compatible"],
)

NEVER_NAMED = {
    Engine.LANGGRAPH: "no tool call this engine announced",
    Engine.PYDANTIC_AI: "never gave its name",
}
"""How each engine words its refusal of a call that is never named."""

RENAMED = {
    Engine.LANGGRAPH: "two names",
    Engine.PYDANTIC_AI: "changed after the call was announced",
}
"""How each engine words its refusal of a call whose name changes."""


def streamed(*chunks: dict[str, Any]) -> chat_completions.Vendor:
    """A vendor answering every request with those chunks."""
    return chat_completions.Vendor(chat_completions.streamed(*chunks))


def answering(*pieces: str) -> chat_completions.Vendor:
    """A vendor answering with that text, in those pieces, and nothing else."""
    return streamed(*chat_completions.said(*pieces), *chat_completions.finished())


def calling(*deltas: dict[str, Any]) -> chat_completions.Vendor:
    """A vendor answering with those tool-call deltas, and ending on the calls."""
    return streamed(*deltas, *chat_completions.finished("tool_calls"))


def built(engine: Engine, provider: ModelProviderConfig = OPENAI_PROVIDER) -> Any:
    """What that engine's factory builds for the OpenAI model."""
    return ADAPTERS[engine].chat_model(chat_completions.gpt(provider), provider, KEY)


# --- the kinds the engine reaches --------------------------------------------


@ENGINES
def test_every_kind_the_platform_names_is_one_this_engine_offers(engine: Engine) -> None:
    # Two clients, two kinds each; a kind added to the vocabulary tomorrow is
    # a kind the engine does not offer until somebody writes its branch, and
    # this is the line that says so.
    adapter = ADAPTERS[engine]

    assert adapter.agent.kinds == frozenset(ProviderKind)
    assert adapter.anthropic_kinds | adapter.openai_kinds == adapter.agent.kinds
    assert not adapter.anthropic_kinds & adapter.openai_kinds


class _Unbuilt(StrEnum):
    """A kind the platform might name one day, which no engine has a client for."""

    BEDROCK = "bedrock"


@dataclass(frozen=True)
class _UnbuiltProvider:
    """A provider of that kind, built by hand the way no configuration can build one."""

    id: str
    kind: _Unbuilt
    base_url: str | None = None


@ENGINES
@pytest.mark.parametrize("base_url", [None, COMPATIBLE_ENDPOINT])
def test_a_kind_this_engine_does_not_reach_is_refused_rather_than_guessed_at(
    engine: Engine, base_url: str | None
) -> None:
    # The configuration is held to `kinds` at start-up, so nothing should ever
    # get here, and every kind there is today has a branch. If something does
    # -- a kind added to the vocabulary without one, a caller that built a
    # definition by hand -- it must be a refusal naming the provider and never
    # a turn sent to whatever endpoint happened to be nearest, whether or not
    # it came with an address; and the client is refused as the endpoint is,
    # rather than being the Anthropic one by default.
    adapter = ADAPTERS[engine]
    provider: Any = _UnbuiltProvider(id="vendor", kind=_Unbuilt.BEDROCK, base_url=base_url)
    refusal = [
        f"model_providers.vendor: this build of the {adapter.title} engine cannot reach"
        " a bedrock provider"
    ]

    with pytest.raises(ConfigError) as raised:
        adapter.endpoint_of(provider)
    assert list(raised.value.problems) == refusal
    with pytest.raises(ConfigError) as raised:
        adapter.chat_model(ModelConfig(id="m", provider="vendor", name="m"), provider, KEY)
    assert list(raised.value.problems) == refusal


# --- the client the configuration describes ----------------------------------


@ENGINES
@KINDS
def test_an_openai_client_is_the_configuration_s_and_never_the_environment_s(
    monkeypatch: pytest.MonkeyPatch, engine: Engine, provider: ModelProviderConfig, endpoint: str
) -> None:
    for name, value in chat_completions.REDIRECTING_VARIABLES.items():
        monkeypatch.setenv(name, value)
    adapter = ADAPTERS[engine]

    client = adapter.openai_client(built(engine, provider))

    assert adapter.endpoint_of(provider) == endpoint
    assert str(client.base_url).rstrip("/") == endpoint
    assert client.api_key == KEY


@ENGINES
def test_the_openai_key_header_is_the_configured_key_whatever_the_environment_injects(
    monkeypatch: pytest.MonkeyPatch, engine: Engine
) -> None:
    # The SDK drops an Authorization line of this variable only when the
    # caller passed one of its own -- and the engine always does.
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer sk-somebody-elses-key")

    sent = chat_completions.headers_of(ADAPTERS[engine].openai_client(built(engine)))

    assert sent["authorization"] == f"Bearer {KEY}"


@ENGINES
def test_what_no_argument_can_refuse_reaches_a_client_built_before_the_engine(
    monkeypatch: pytest.MonkeyPatch, engine: Engine
) -> None:
    # The half that shows why the variables have to go: without the engine
    # having been built, the client picks them up, headers and all.
    for name, value in chat_completions.UNCONFIGURED.items():
        monkeypatch.setenv(name, value)

    client = ADAPTERS[engine].openai_client(built(engine))
    sent = chat_completions.headers_of(client)

    assert sent["x-nobody-configured"] == "this"
    assert sent["openai-organization"] == "org-somebody-elses"
    assert sent["openai-project"] == "proj-somebody-elses"
    assert client.admin_api_key == "sk-admin-somebody-elses"


@ENGINES
def test_building_the_engine_keeps_every_unconfigured_openai_setting_off_a_turn(
    monkeypatch: pytest.MonkeyPatch, engine: Engine
) -> None:
    for name, value in chat_completions.UNCONFIGURED.items():
        monkeypatch.setenv(name, value)
    adapter = ADAPTERS[engine]

    adapter.build(chat_completions.openai_models(engine), ProviderKeys({"openai": KEY}))
    client = adapter.openai_client(built(engine))
    sent = chat_completions.headers_of(client)

    assert all(name not in os.environ for name in chat_completions.UNCONFIGURED)
    assert set(chat_completions.UNCONFIGURED) <= set(adapter.client_variables_removed)
    assert "x-nobody-configured" not in sent
    assert "openai-organization" not in sent
    assert "openai-project" not in sent
    assert sent["authorization"] == f"Bearer {KEY}"
    assert client.admin_api_key is None


# --- a turn --------------------------------------------------------------------


@ENGINES
@asyncio_test
async def test_an_openai_turn_streams_its_text_and_sends_the_configuration_s_request(
    monkeypatch: pytest.MonkeyPatch, engine: Engine
) -> None:
    # Nothing on a turn's path counts tokens: tiktoken fetches its encodings
    # from the network the first time it is asked -- or reads them from a warm
    # cache, which closed sockets would not notice -- and it is never asked.
    monkeypatch.setattr(tiktoken, "get_encoding", chat_completions.never_tokenised)
    monkeypatch.setattr(tiktoken, "encoding_for_model", chat_completions.never_tokenised)
    vendor = answering("Someone ", "who plays ", "fair.")
    agent = over_chat_completions(engine, vendor, timeout_seconds=17.0)

    seen = await chat_completions_turn(agent, engine)

    check_engine_events(seen)
    assert seen == [
        AnswerStarted(),
        AnswerTextDelta(text="Someone "),
        AnswerTextDelta(text="who plays "),
        AnswerTextDelta(text="fair."),
        AnswerCompleted(parts=(TextPart("Someone who plays fair."),)),
    ]
    assert agent.held == 0
    sent = vendor.request
    assert str(sent.url) == OPENAI_ENDPOINT + chat_completions.PATH
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    assert sent.headers["x-stainless-read-timeout"] == "17.0"
    assert vendor.body_sent == {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is a robinaut?"},
        ],
        "model": chat_completions.MODEL_NAME,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


@ENGINES
@asyncio_test
async def test_an_openai_compatible_turn_goes_to_the_configured_endpoint_and_nowhere_else(
    engine: Engine,
) -> None:
    vendor = answering("Hi.")

    await chat_completions_turn(over_chat_completions(engine, vendor, GATEWAY_PROVIDER), engine)

    assert str(vendor.request.url) == GATEWAY_ENDPOINT + chat_completions.PATH
    assert vendor.request.headers["authorization"] == f"Bearer {KEY}"


@ENGINES
@pytest.mark.parametrize(
    ("provider", "field", "not_field"),
    [
        (OPENAI_PROVIDER, "max_completion_tokens", "max_tokens"),
        (GATEWAY_PROVIDER, "max_tokens", "max_completion_tokens"),
    ],
    ids=["openai", "openai-compatible"],
)
@asyncio_test
async def test_a_configured_ceiling_is_sent_in_the_field_the_kind_takes(
    engine: Engine, provider: ModelProviderConfig, field: str, not_field: str
) -> None:
    # Chat Completions requires no ceiling, and on a reasoning model one picked
    # for the answer can be spent on the thinking, so none is sent unless it is
    # configured; one that is goes in the field the kind takes -- OpenAI's
    # current one for OpenAI, the older one OpenRouter and older compatible
    # servers know for a compatible endpoint.
    unbounded = answering("Hi.")
    bounded = chat_completions.Vendor(unbounded.body)

    await chat_completions_turn(over_chat_completions(engine, unbounded, provider), engine)
    await chat_completions_turn(
        over_chat_completions(engine, bounded, provider, max_output_tokens=1234), engine
    )

    assert field not in unbounded.body_sent and not_field not in unbounded.body_sent
    assert bounded.body_sent[field] == 1234
    assert not_field not in bounded.body_sent


@ENGINES
@asyncio_test
async def test_a_streamed_think_tag_is_text_and_is_kept_as_the_answer(engine: Engine) -> None:
    # What the vendor wrote is the answer. langchain-openai does not lift a
    # `<think>` out of the content, and Pydantic AI, which would, is given no
    # thinking tags (docs/specs/agents.md, "Known findings").
    vendor = answering("<think>", "hm", "</think>", "Hi.")

    seen = await chat_completions_turn(over_chat_completions(engine, vendor), engine)

    assert not [event for event in seen if isinstance(event, AnswerReasoningDelta)]
    assert seen[-1] == AnswerCompleted(parts=(TextPart("<think>hm</think>Hi."),))


# --- tools: a call made, and a call in the history -------------------------------


@ENGINES
@asyncio_test
async def test_an_openai_model_that_asks_for_tools_yields_the_calls_and_ends_the_turn_waiting(
    engine: Engine,
) -> None:
    vendor = streamed(
        *chat_completions.said("Let me look."),
        *chat_completions.calling("call_1", SEARCH.name, {"q": "robinauts"}),
        *chat_completions.calling("call_2", SEARCH.name, {"q": "fair play"}, index=1),
        *chat_completions.finished("tool_calls"),
    )

    seen = await chat_completions_turn(
        over_chat_completions(engine, vendor), engine, tools=(SEARCH,)
    )

    check_engine_events(seen)
    first = ToolCallPart("call_1", SEARCH.name, {"q": "robinauts"})
    second = ToolCallPart("call_2", SEARCH.name, {"q": "fair play"})
    # Each call announced, argued in the two pieces the vendor sent, and
    # completed, one after the other, between the text and the end.
    a_call = [ToolCallStarted, ToolCallArgumentsDelta, ToolCallArgumentsDelta, ToolCallCompleted]
    assert [type(event) for event in seen] == [
        AnswerStarted,
        AnswerTextDelta,
        *a_call,
        *a_call,
        AnswerCompleted,
        WaitingOnTools,
    ]
    assert seen[-2] == AnswerCompleted(parts=(TextPart("Let me look."), first, second))
    # The definition as the vendor takes it, as it stands: the server's schema,
    # no `strict` added, and the description sent.
    assert vendor.body_sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": SEARCH.name,
                "description": SEARCH.description,
                "parameters": dict(SEARCH.input_schema),
            },
        }
    ]


@ENGINES
@asyncio_test
async def test_a_turn_with_tools_in_its_history_is_translated_for_chat_completions(
    engine: Engine,
) -> None:
    vendor = answering("Found three.")
    history = chat_completions.history_with_a_call(SEARCH.name)

    await chat_completions_turn(
        over_chat_completions(engine, vendor), engine, history, tools=(SEARCH,)
    )

    # The Anthropic blocks in the answer's extras are not replayed: they were
    # made by another model, and Chat Completions has nowhere to put them. The
    # result's error flag has no field in this protocol, so the result is
    # written as an object saying so -- and the whole is spelt alike under both
    # engines, down to the keys' order and the JSON's separators.
    assert json.dumps(vendor.body_sent["messages"][1:]) == json.dumps(
        [
            {"role": "user", "content": history[0].text},
            {
                "role": "assistant",
                "content": "Let me look.",
                "tool_calls": [
                    {
                        "id": "toolu_01",
                        "type": "function",
                        "function": {"name": SEARCH.name, "arguments": '{"q":"robinauts"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_01", "content": '{"error":"found 3"}'},
        ]
    )


@ENGINES
@asyncio_test
async def test_no_thinking_is_ever_replayed_over_chat_completions(engine: Engine) -> None:
    # A model id the operator moved from an Anthropic provider to an OpenAI
    # one between restarts: the stored answer says it was made by this very
    # id, and still nothing in its extras is sent -- not Anthropic's signed
    # blocks, and not a block under OpenAI's own key, which Pydantic AI would
    # write between thinking tags its OpenAI models do not have.
    vendor = answering("Again.")
    asked, calling_, results = chat_completions.history_with_a_call(SEARCH.name)
    signed = {"thinking": [{"type": "thinking", "thinking": "hm", "signature": "SIG"}]}
    moved = dataclasses.replace(
        calling_,
        provenance=provenance(model=GPT),
        extras={**calling_.extras, "openai": signed},
    )

    await chat_completions_turn(
        over_chat_completions(engine, vendor), engine, (asked, moved, results), tools=(SEARCH,)
    )

    (sent,) = [m for m in vendor.body_sent["messages"] if m["role"] == "assistant"]
    assert sent["content"] == "Let me look."


# --- what a Chat Completions stream becomes, in shapes the vendors really send


@ENGINES
@pytest.mark.parametrize(
    "opening",
    [
        {"index": 0, "id": "call_1", "type": "function", "function": {}},
        {"index": 0, "id": "call_1", "type": "function", "function": {"arguments": ""}},
    ],
    ids=["empty-function", "empty-arguments"],
)
@asyncio_test
async def test_a_call_whose_id_and_name_arrive_in_separate_chunks_is_announced_once_both_have(
    engine: Engine, opening: dict[str, Any]
) -> None:
    vendor = calling(
        chat_completions.chunk({"tool_calls": [opening]}),
        call_delta(0, arguments='{"q":'),
        call_delta(0, name=SEARCH.name),
        call_delta(0, arguments='"x"}'),
    )

    seen = await chat_completions_turn(
        over_chat_completions(engine, vendor), engine, tools=(SEARCH,)
    )

    check_engine_events(seen)
    call = ToolCallPart("call_1", SEARCH.name, {"q": "x"})
    assert seen[:5] == [
        AnswerStarted(),
        ToolCallStarted(call_id="call_1", name=SEARCH.name),
        ToolCallArgumentsDelta(call_id="call_1", text='{"q":'),
        ToolCallArgumentsDelta(call_id="call_1", text='"x"}'),
        ToolCallCompleted(call=call),
    ]
    assert seen[-2] == AnswerCompleted(parts=(call,))
    assert isinstance(seen[-1], WaitingOnTools)


@ENGINES
@asyncio_test
async def test_a_call_whose_id_and_name_come_again_on_every_delta_is_one_call(
    engine: Engine,
) -> None:
    vendor = calling(
        call_delta(0, id="call_1", name=SEARCH.name, arguments='{"q":'),
        call_delta(0, id="call_1", name=SEARCH.name, arguments='"x"}'),
        call_delta(1, id="call_2", name=SEARCH.name, arguments='{"q":"y"}'),
    )

    seen = await chat_completions_turn(
        over_chat_completions(engine, vendor), engine, tools=(SEARCH,)
    )

    check_engine_events(seen)
    assert [event for event in seen if isinstance(event, ToolCallStarted)] == [
        ToolCallStarted(call_id="call_1", name=SEARCH.name),
        ToolCallStarted(call_id="call_2", name=SEARCH.name),
    ]
    assert seen[-2] == AnswerCompleted(
        parts=(
            ToolCallPart("call_1", SEARCH.name, {"q": "x"}),
            ToolCallPart("call_2", SEARCH.name, {"q": "y"}),
        )
    )


@ENGINES
@pytest.mark.parametrize(
    ("deltas", "calls"),
    [
        (
            [
                {"index": None, "id": "call_1", "name": SEARCH.name, "arguments": ""},
                {"index": None, "arguments": '{"q":'},
                {"index": None, "arguments": '"x"}'},
            ],
            [("call_1", {"q": "x"})],
        ),
        (
            [
                {"index": 0, "id": "call_1", "name": SEARCH.name, "arguments": '{"q":"x"}'},
                {"index": None, "id": "call_2", "name": SEARCH.name, "arguments": ""},
                {"index": None, "arguments": '{"q":"y"}'},
            ],
            [("call_1", {"q": "x"}), ("call_2", {"q": "y"})],
        ),
        (
            [
                {"index": None, "id": "call_1", "name": SEARCH.name, "arguments": '{"q":'},
                {"index": None, "id": "call_1", "name": SEARCH.name, "arguments": '"x"}'},
            ],
            [("call_1", {"q": "x"})],
        ),
    ],
    ids=["one-call", "after-an-indexed-call", "id-and-name-again"],
)
@asyncio_test
async def test_calls_whose_deltas_carry_no_index_are_taken_alike(
    engine: Engine, deltas: list[dict[str, Any]], calls: list[tuple[str, dict[str, Any]]]
) -> None:
    # OpenAI always sends an index; a compatible server need not. Both
    # engines take these shapes, and store the same calls
    # (docs/specs/agents.md, "How a streamed call may arrive").
    vendor = calling(*(call_delta(**dict(delta)) for delta in deltas))

    seen = await chat_completions_turn(
        over_chat_completions(engine, vendor), engine, tools=(SEARCH,)
    )

    check_engine_events(seen)
    assert seen[-2] == AnswerCompleted(
        parts=tuple(ToolCallPart(call_id, SEARCH.name, arguments) for call_id, arguments in calls)
    )
    assert isinstance(seen[-1], WaitingOnTools)


@ENGINES
@pytest.mark.parametrize("index", [0, None], ids=["indexed", "index-less"])
@asyncio_test
async def test_a_call_that_never_gets_a_name_is_refused_not_dropped(
    engine: Engine, index: int | None
) -> None:
    # Stored as it stands, it would be an answer that never waits for the call
    # it began: Pydantic AI leaves such a call out of its response altogether.
    vendor = calling(call_delta(index, id="call_1", arguments="{}"))

    with pytest.raises(UnsupportedContentError, match=NEVER_NAMED[engine]):
        await chat_completions_turn(over_chat_completions(engine, vendor), engine, tools=(SEARCH,))


@ENGINES
@asyncio_test
async def test_a_tool_name_streamed_in_pieces_is_refused_not_truncated(engine: Engine) -> None:
    # A name that changes after it is known is not one call's name: Pydantic AI
    # would announce the call at its first piece and store it under the rest.
    vendor = calling(
        call_delta(0, id="call_1", name="github__", arguments=""),
        call_delta(0, name="search", arguments="{}"),
    )

    with pytest.raises(UnsupportedContentError, match=RENAMED[engine]):
        await chat_completions_turn(over_chat_completions(engine, vendor), engine, tools=(SEARCH,))


# --- what the vendor refuses ----------------------------------------------------


@ENGINES
@pytest.mark.parametrize(
    ("status", "error", "raised"),
    [
        (401, {"code": "invalid_api_key", "message": "Bad key"}, openai.AuthenticationError),
        (404, {"code": "model_not_found", "message": "No such model"}, openai.NotFoundError),
        (429, {"code": "rate_limit_exceeded", "message": "Slow down"}, openai.RateLimitError),
        (400, {"code": "context_length_exceeded", "message": "Too long"}, openai.BadRequestError),
        (503, {"message": "Overloaded"}, openai.InternalServerError),
    ],
    ids=["key", "model", "rate", "context", "overloaded"],
)
@asyncio_test
async def test_what_the_openai_vendor_refuses_travels_out_of_the_turn_as_it_is(
    engine: Engine, status: int, error: dict[str, Any], raised: type[Exception]
) -> None:
    # The port's rule, over this protocol as over the other: an engine reports
    # a failure by raising, and the SDK's own exception is what is raised --
    # as it is under LangGraph, and as the cause of Pydantic AI's own
    # ``ModelHTTPError`` (``engines.Adapter.vendor_error``). The application
    # records it, and nothing is retried on the way (``MAX_RETRIES``): one
    # request, and the turn holds nothing after it.
    vendor = chat_completions.Vendor(status=status, error=error)
    agent = over_chat_completions(engine, vendor)

    with pytest.raises(Exception) as caught:  # noqa: B017 -- each engine's own
        await chat_completions_turn(agent, engine)

    assert getattr(caught.value, "status_code", None) == status
    said = ADAPTERS[engine].vendor_error(caught.value)
    assert isinstance(said, raised)
    assert isinstance(said, openai.APIStatusError) and said.status_code == status
    assert len(vendor.sent) == 1
    assert agent.held == 0
