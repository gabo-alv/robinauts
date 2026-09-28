# SPDX-License-Identifier: Apache-2.0
# Copyright The Robinauts Authors

"""The swap: one conversation, both engines, and nothing between them but the store.

**The seam the project exists to prove** (``docs/specs/agents.md``,
``docs/specs/core.md``). A conversation is begun on LangGraph, continued on
Pydantic AI and continued again on LangGraph, and what makes that possible is
that neither engine remembers anything: the conversation record is the whole of
the state, and a turn starts from it whichever engine runs it (ADR 0002).

Both engines are the **real** ones, over models the test scripts -- no network
and no key, exactly as each engine's own module runs them. The application is
the real ``Turns`` over the in-memory store, with the real executor and the
real signals, so what is asserted is what a deployment would store.

Three things, which are the whole of the claim:

1. **every answer says which engine produced it**: the conversation afterwards
   is a record of the swap, message by message (``provenance.engine``);
2. **each engine is given the same history** -- the stored messages, in order,
   with the roles right, the system prompt outside them, and no reasoning:
   what the second engine is told is what the first one's answer became in the
   store, not anything the first engine kept;
3. **the stored format is one format.** An answer written by one engine and an
   answer written by the other are the same document down to the keys and the
   format version, and differ in one value: ``provenance.engine``. That is
   what makes the swap a change of configuration rather than a migration.

And over OpenAI's Chat Completions, a fourth: **the two engines send the
same history, byte for byte** -- each through its real client and its real
framework, to a vendor the test writes (``tests/chat_completions.py``).

The other half of the swap -- doing it by editing the configuration file and
restarting -- is an integration test over the real PostgreSQL
(``tests/integration/test_create_app.py``).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import pytest

import chat_completions
from aio import asyncio_test
from conversations import AGENT, MODEL, OTHER_MODEL, agent_definition, answer, question
from engines import MODELS, Scripts, both_engines, scripts
from fakes import CountingIdSource, FakeClock, MemoryConversationStore
from robinauts.adapters import AsyncioRunExecutor, MemoryRunSignals, ProviderKeys
from robinauts.adapters.agents.langgraph import LangGraphAgent
from robinauts.adapters.agents.langgraph import chat_model as langgraph_chat_model
from robinauts.adapters.agents.pydantic_ai import PydanticAIAgent
from robinauts.adapters.agents.pydantic_ai import chat_model as pydantic_ai_chat_model
from robinauts.application import Turns
from robinauts.core import message_to_data
from robinauts.domain import (
    FORMAT_VERSION,
    Engine,
    Message,
    ModelConfig,
    ModelProviderConfig,
    ModelsConfig,
    ProviderKind,
    Role,
    Run,
    TextPart,
    ToolCallPart,
    ToolDefinition,
    ToolResultPart,
)
from turns import AUTHOR, NOW, stored_messages

SYSTEM_PROMPT = "Play fair."

ASKED = ("What is a robinaut?", "And a robin?", "Are you sure?")
"""One question per turn, so that each engine is given a longer history."""

ANSWERED = "Someone who plays fair."
"""What both engines are scripted to say, so that only the engine differs."""

THOUGHT = "let me think"
"""Streamed by one turn and stored by none: this version keeps no reasoning."""

BETWEEN_TURNS = 60.0
"""How far the clock moves between turns, so that they are told apart."""

TURN_SECONDS = 30.0
"""Far more than any turn here needs: nothing in this file is about a bound."""


def deployment(
    store: MemoryConversationStore,
    engine: Engine,
    said: Scripts,
    clock: FakeClock,
    ids: CountingIdSource,
) -> Turns:
    """The run lifecycle as a process holds it, with both engines wired.

    Built afresh per turn, which is what a **restart** is: the configuration is
    read once at start-up, and an agent's engine takes effect at the next turn
    after one (``docs/specs/agents.md``). The store, the clock and the source
    of ids are handed in and outlive it: the store because it is the state, and
    the other two because a restart does not put time back or start handing out
    ids somebody already has.
    """
    agent = agent_definition(engine=engine, system_prompt=SYSTEM_PROMPT)
    return Turns(
        store=store,
        clock=clock,
        ids=ids,
        agents={agent.id: agent},
        models=MODELS.models,
        engines=both_engines(said),
        executor=AsyncioRunExecutor(),
        signals=MemoryRunSignals(),
        turn_seconds=TURN_SECONDS,
    )


async def answered(turns: Turns, text: str, after: Message | None) -> Run:
    """One whole turn, begun the way a request begins one and run to its end.

    ``begin`` is the public path -- it starts the turn and hands the work to
    the executor -- and nothing sleeps afterwards: the loop is yielded to, and
    the turn is what runs.

    An **agent** begins a conversation; a **conversation id** and a parent
    continue one, which is what a client sends for the next turn
    (``docs/specs/wire.md``). Which of the two it is is what makes the second
    turn a continuation of the first rather than a second root beside it.
    """
    started = await turns.begin(
        AUTHOR,
        agent_id=AGENT if after is None else None,
        conversation_id=None if after is None else after.conversation_id,
        parent_id=None if after is None else after.id,
        text=text,
    )
    while started.run.id in turns.executing:
        await asyncio.sleep(0)
    return started.run


async def conversed(said: Scripts, *turns_of: tuple[Engine, str]) -> list[Message]:
    """One conversation, each turn on the engine the configuration then had.

    A deployment per turn over one store, which is the restart the swap takes
    -- nothing is carried over but what was written down.
    """
    store = MemoryConversationStore()
    clock, ids = FakeClock(now=NOW), CountingIdSource()
    messages: list[Message] = []
    for engine, asked in turns_of:
        turns = deployment(store, engine, said, clock, ids)
        run = await answered(turns, asked, messages[-1] if messages else None)
        # One turn, one moment: a clock that never moved would leave every
        # message of the conversation claiming the same one.
        clock.advance(BETWEEN_TURNS)
        messages = await stored_messages(store, run.conversation_id)
    return messages


THERE_AND_BACK = (
    (Engine.LANGGRAPH, ASKED[0]),
    (Engine.PYDANTIC_AI, ASKED[1]),
    (Engine.LANGGRAPH, ASKED[2]),
)
"""The swap, and the swap back: three turns of one conversation."""


@asyncio_test
async def test_a_conversation_moves_from_one_engine_to_the_other_and_back() -> None:
    said = scripts(ANSWERED, thinking=THOUGHT)

    messages = await conversed(said, *THERE_AND_BACK)

    # Question, answer, question, answer, question, answer: one conversation.
    assert [message.role for message in messages] == [Role.USER, Role.ASSISTANT] * 3
    assert [message.text for message in messages] == [
        ASKED[0],
        ANSWERED,
        ASKED[1],
        ANSWERED,
        ASKED[2],
        ANSWERED,
    ]
    # Every answer records the engine that produced it, so the conversation is
    # itself the record of the swap.
    assert [
        message.provenance.engine for message in messages if message.provenance is not None
    ] == [engine for engine, _asked in THERE_AND_BACK]
    # And the thinking one of them streamed is in no message at all.
    assert all(THOUGHT not in message.text for message in messages)


@asyncio_test
async def test_each_engine_is_given_the_stored_conversation_and_nothing_else() -> None:
    said = scripts(ANSWERED, thinking=THOUGHT)

    await conversed(said, *THERE_AND_BACK)

    user, assistant = Role.USER.value, Role.ASSISTANT.value
    first, third = said.langgraph.heard
    (second,) = said.pydantic_ai.heard
    assert first == ((user, ASKED[0]),)
    # The second engine is told what the first one's answer **became in the
    # store**: the same roles, in the same order, and no thinking anywhere --
    # neither a previous turn's nor its own, which it had not produced yet.
    assert second == ((user, ASKED[0]), (assistant, ANSWERED), (user, ASKED[1]))
    assert third == (
        (user, ASKED[0]),
        (assistant, ANSWERED),
        (user, ASKED[1]),
        (assistant, ANSWERED),
        (user, ASKED[2]),
    )
    # The system prompt is the agent's and is not one of the messages, under
    # either engine.
    assert said.langgraph.instructions == [SYSTEM_PROMPT, SYSTEM_PROMPT]
    assert said.pydantic_ai.instructions == [SYSTEM_PROMPT]


_ITS_OWN = frozenset({"id", "parent_id", "created_at"})
"""What one message has of its own, which two messages never share."""


@asyncio_test
async def test_an_answer_is_stored_the_same_way_whichever_engine_wrote_it() -> None:
    """The document of an answer, engine by engine: one difference, and it is named.

    Not "the same shape" but the same document: every key, the format version,
    the role, the parts and the provenance's own keys. What a message has of
    its own -- its id, its parent, when it was written, which run produced it
    -- is taken out, and what is left is identical but for one value. Both
    models are scripted to say the same words, so nothing but the engine can
    be what differs.
    """
    said = scripts(ANSWERED, thinking=THOUGHT)

    messages = await conversed(said, *THERE_AND_BACK[:2])

    written = [message_to_data(message) for message in messages if message.role is Role.ASSISTANT]
    by_langgraph, by_pydantic_ai = (_shared(document) for document in written)
    assert by_langgraph["provenance"]["engine"] == Engine.LANGGRAPH.value
    assert by_pydantic_ai["provenance"]["engine"] == Engine.PYDANTIC_AI.value
    assert _without_engine(by_langgraph) == _without_engine(by_pydantic_ai)
    # And it really is the whole document that was compared, not a fragment of
    # one: what was taken out is what one message has of its own, and nothing
    # else was.
    assert set(written[0]) == set(written[1]) == set(by_langgraph) | _ITS_OWN
    assert by_langgraph["parts"] == [{"kind": "text", "text": ANSWERED}]
    assert by_langgraph["format_version"] == FORMAT_VERSION


@asyncio_test
async def test_a_conversation_moved_to_another_model_runs_on_it_under_either_engine() -> None:
    # The engine is the agent's and the model is the conversation's, and they
    # change apart: each engine is handed the model the run was begun on, and
    # builds its client for that, not for the agent's default.
    said = scripts(ANSWERED)
    store = MemoryConversationStore()
    clock, ids = FakeClock(now=NOW), CountingIdSource()
    first = await answered(deployment(store, Engine.LANGGRAPH, said, clock, ids), ASKED[0], None)
    clock.advance(BETWEEN_TURNS)
    await deployment(store, Engine.LANGGRAPH, said, clock, ids).set_model(
        AUTHOR, first.conversation_id, OTHER_MODEL
    )
    for engine, asked in ((Engine.PYDANTIC_AI, ASKED[1]), (Engine.LANGGRAPH, ASKED[2])):
        messages = await stored_messages(store, first.conversation_id)
        await answered(deployment(store, engine, said, clock, ids), asked, messages[-1])
        clock.advance(BETWEEN_TURNS)

    messages = await stored_messages(store, first.conversation_id)
    assert said.built == [
        (Engine.LANGGRAPH, MODEL),
        (Engine.PYDANTIC_AI, OTHER_MODEL),
        (Engine.LANGGRAPH, OTHER_MODEL),
    ]
    # And every answer records the model that produced it.
    assert [
        (message.provenance.engine, message.provenance.model)
        for message in messages
        if message.provenance is not None
    ] == said.built


# --- the same request over Chat Completions ----------------------------------


GPT = "gpt"

SEARCH = ToolDefinition(
    name="github__search",
    description="",
    input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
)
"""A tool with no description, which the two frameworks would otherwise spell differently."""

READ = ToolDefinition(
    name="github__read_file",
    description="Read a file of a repository.",
    input_schema={
        "type": "object",
        "title": "Read",
        "properties": {"path": {"type": "string", "title": "Path"}},
        "required": ["path"],
    },
)
"""A described tool whose schema carries titles, which one framework would strip."""

SHARED_FIELDS = ("model", "messages", "tools", "stream", "stream_options")
"""The fields of the request both engines send, compared byte for byte.

What is left out is sent by one engine only: ``tool_choice: "auto"``, which
Pydantic AI adds and which is the protocol's default (``docs/specs/agents.md``,
"Known findings"). The ceiling is compared on its own, per kind.
"""

CEILING_FIELDS = ("max_completion_tokens", "max_tokens")


def _answered_calls(parent: Message, text: str, *calls: ToolCallPart) -> Message:
    return answer(parent, parts=(*((TextPart(text),) if text else ()), *calls))


def _results(parent: Message, *results: ToolResultPart) -> Message:
    return Message(
        id=uuid.uuid4(),
        conversation_id=parent.conversation_id,
        parent_id=parent.id,
        role=Role.TOOL,
        parts=results,
        created_at=parent.created_at,
    )


def _history() -> tuple[Message, ...]:
    """A history holding every shape an answer and a result take over this protocol.

    Text beside a call; a call with no text; a result that went wrong and one
    that did not, with text that is not ASCII; an answer that said nothing;
    and the question being answered last.
    """
    first = question("Find the robinauts.")
    looked = _answered_calls(
        first, "Let me look.", ToolCallPart("call_1", SEARCH.name, {"q": "robinauts", "n": 3})
    )
    failed = _results(looked, ToolResultPart("call_1", "rate limited", is_error=True))
    again = _answered_calls(failed, "", ToolCallPart("call_2", SEARCH.name, {"q": "café"}))
    found = _results(again, ToolResultPart("call_2", "found «3»"))
    told = answer(found, "Three.")
    second = question("And now?", parent=told)
    silent = answer(second, parts=(TextPart(""),))
    last = question("Well?", parent=silent)
    return (first, looked, failed, again, found, told, second, silent, last)


def _openai_models(engine: Engine, kind: ProviderKind) -> ModelsConfig:
    base_url = "https://gateway.example.test/v1" if kind is ProviderKind.OPENAI_COMPATIBLE else None
    return ModelsConfig(
        providers={
            "openai": ModelProviderConfig(
                id="openai", kind=kind, api_key_env="K", base_url=base_url
            )
        },
        models={
            GPT: ModelConfig(
                id=GPT, provider="openai", name=chat_completions.MODEL_NAME, max_output_tokens=1234
            )
        },
        agents={
            AGENT: agent_definition(id=AGENT, model=GPT, engine=engine, system_prompt=SYSTEM_PROMPT)
        },
    )


async def _sent_by(engine: Engine, kind: ProviderKind) -> dict[str, Any]:
    """The request the engine sends for the history, as the JSON it is."""
    vendor = chat_completions.Vendor(
        chat_completions.streamed(*chat_completions.said("Fine."), *chat_completions.finished())
    )
    models = _openai_models(engine, kind)
    keys = ProviderKeys({"openai": "not-a-real-key"})

    def plugged(model: ModelConfig, provider: ModelProviderConfig, key: str) -> Any:
        if engine is Engine.LANGGRAPH:
            built: Any = langgraph_chat_model(model, provider, key)
            vendor.plugged_into(built.root_async_client)
        else:
            built = pydantic_ai_chat_model(model, provider, key)
            vendor.plugged_into(built.client)
        return built

    agent: Any = (
        LangGraphAgent(models, keys, chat_model_for=plugged)
        if engine is Engine.LANGGRAPH
        else PydanticAIAgent(models, keys, model_for=plugged)
    )
    async for _ in agent.run_turn(models.agents[AGENT], _history(), (SEARCH, READ), model=GPT):
        pass
    body = json.loads(vendor.request.content)
    assert isinstance(body, dict)
    return body


def _bytes(body: dict[str, Any], *fields: str) -> bytes:
    """Those fields of a request, in the order they were sent, as JSON bytes."""
    return json.dumps(
        {key: value for key, value in body.items() if key in fields}, ensure_ascii=False
    ).encode()


@pytest.mark.parametrize(
    ("kind", "ceiling"),
    [
        (ProviderKind.OPENAI, "max_completion_tokens"),
        (ProviderKind.OPENAI_COMPATIBLE, "max_tokens"),
    ],
    ids=["openai", "openai-compatible"],
)
@asyncio_test
async def test_both_engines_send_the_same_request_over_chat_completions(
    kind: ProviderKind, ceiling: str
) -> None:
    """The history, the tools, the stream options and the ceiling, byte for byte.

    For the shapes written here: the arguments of a call are serialised by a
    different library under each engine (``json`` under LangGraph,
    ``pydantic_core`` under Pydantic AI), so what is proved is parity for
    these values -- nested objects, integers, text that is not ASCII -- and not
    for every value a model could write.
    """
    langgraph = await _sent_by(Engine.LANGGRAPH, kind)
    pydantic_ai = await _sent_by(Engine.PYDANTIC_AI, kind)

    assert _bytes(langgraph, *SHARED_FIELDS) == _bytes(pydantic_ai, *SHARED_FIELDS)
    assert _bytes(langgraph, *CEILING_FIELDS) == _bytes(pydantic_ai, *CEILING_FIELDS)
    assert langgraph[ceiling] == 1234
    assert not (set(CEILING_FIELDS) - {ceiling}) & set(langgraph)
    assert set(pydantic_ai) - set(langgraph) == {"tool_choice"}
    # And it is the request it should be, not merely the same one twice.
    sent = langgraph["messages"]
    assert [message["role"] for message in sent] == [
        "system", "user", "assistant", "tool", "assistant", "tool",
        "assistant", "user", "assistant", "user",
    ]  # fmt: skip
    assert sent[2]["tool_calls"][0]["function"]["arguments"] == '{"q":"robinauts","n":3}'
    assert sent[3]["content"] == '{"error":"rate limited"}'
    assert sent[4]["content"] == ""
    assert sent[4]["tool_calls"][0]["function"]["arguments"] == '{"q":"café"}'
    assert sent[5]["content"] == "found «3»"
    assert sent[8]["content"] == ""
    assert langgraph["tools"] == [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.input_schema),
            },
        }
        for tool in (SEARCH, READ)
    ]
    assert langgraph["stream_options"] == {"include_usage": True}


def _shared(document: dict[str, Any]) -> dict[str, Any]:
    """The document with what belongs to one message alone taken out."""
    shared = {key: value for key, value in document.items() if key not in _ITS_OWN}
    shared["provenance"] = {
        key: value for key, value in shared["provenance"].items() if key != "run_id"
    }
    return shared


def _without_engine(document: dict[str, Any]) -> dict[str, Any]:
    """The same again, with the one value that is allowed to differ taken out."""
    kept = dict(document)
    kept["provenance"] = {
        key: value for key, value in kept["provenance"].items() if key != "engine"
    }
    return kept
