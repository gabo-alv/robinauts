# SPDX-License-Identifier: Apache-2.0
# Copyright The Robinauts Authors

"""One turn on LangGraph: a graph compiled per turn, streamed, translated.

The first implementation of the ``Agent`` port (``robinauts.ports.agents``),
and the only module in the platform that may name LangGraph or LangChain
(``docs/layout.md``, enforced by the import contracts in
``backend/pyproject.toml``). Everything about the framework stops here: what
crosses the port is the platform's own history in and the platform's own
events out, and the **discard test** is that five places name this
sub-package and deleting it and its dependencies breaks those and nothing
else: its import in ``robinauts.app`` and its one entry in ``ENGINES``; the
contract exceptions in ``backend/pyproject.toml`` that name the sub-package;
this sub-package's own tests; and the **shared swap fixtures** under
``tests/`` (``tests/engines.py``, ``tests/unit/test_engine_swap.py``,
``tests/unit/test_engines_over_chat_completions.py`` and the configuration swap
in ``tests/integration/test_create_app.py``), which exist to name both engines
at once and cannot be written without both. The
composition tests (``tests/unit/test_app_composition.py``) fail too and name
no adapter: they say that *both* engines are wired, which is a claim about the
table and not about either sub-package (``docs/layout.md``).

**Stateless per turn** (ADR 0002). The graph is compiled for the turn, with
**no checkpointer**: the conversation record is the whole of the state and the
history handed in is where the turn starts from. Nothing is remembered between
calls, which is what lets the next turn of the same conversation run on the
other engine.

**A real graph, not a shortcut.** One node, which streams the chat model, and
two edges -- and **no ``ToolNode``**: the platform owns the tool loop
(``docs/specs/agents.md``, "Tools"). The tools a run has are bound to the
model (``bind_tools``), a call the model makes is yielded as events and the
turn ends there, waiting; the application runs the call, appends the result
and starts the next turn from the stored history, which this engine then sees
as a path ending in a tool message.

**The mapping**, which is the whole of the translation:

- LangGraph's ``messages`` stream carries what the model produced, chunk by
  chunk. The first chunk with anything in it opens the answer
  (``AnswerStarted``); text becomes ``AnswerTextDelta``, reasoning becomes
  ``AnswerReasoningDelta``, and a tool call becomes ``ToolCallStarted`` with
  its id and name, ``ToolCallArgumentsDelta`` for the JSON the model writes,
  and ``ToolCallCompleted`` with the platform's part for it;
- the node's own update carries the whole message, which is what an answer
  **that never streamed** completes with, where a call still open when the
  answer ends and that streamed no arguments takes them from, and where the
  vendor's signed thinking blocks are read off (``_Answer.complete``). A
  call closed by the next call's start with nothing streamed completes with
  no arguments, which is what this client's stream gives such a call anyway;
- **what was streamed is what is kept** (``docs/specs/agents.md``): an answer
  that yielded text deltas completes with exactly those deltas joined, never
  with whatever the framework made of the final message, and a call's
  arguments are what its deltas parse to;
- **reasoning is never in the completed parts.** ``domain.kept_parts`` would
  drop a ``ReasoningPart`` anyway; putting the model's thinking in the answer's
  *text* is the mistake that would survive that, so the text part is built
  from the text deltas alone. What **is** kept of the thinking is the vendor's
  signed blocks, in ``AnswerCompleted.extras`` under ``anthropic``, unread
  (``docs/specs/conversations.md``, "Reasoning"): the vendor requires them back
  with a tool call's results, and this adapter replays them -- to the model
  that made them, since they are bound to it -- when it translates the
  history (``_assistant``);
- **an answer that asked for tools ends the turn waiting** (``WaitingOnTools``)
  and the engine executes none of them.

Chunks are read through ``AIMessageChunk.content_blocks``, langchain-core's
own provider-neutral view of a message's content, for text and reasoning, and
through ``tool_call_chunks`` for calls, which is where every provider client
puts a streamed call.

**Its context policy is everything** (ADR 0004): the whole visible path goes
to the model, in order, with the system prompt in front. Trimming to a token
budget and cache breakpoints are this adapter's to add, and are not added yet.

**Two protocols, four kinds** (``LangGraphAgent.kinds``). ``ChatAnthropic``
speaks Anthropic's Messages API, to Anthropic or to an ``anthropic-compatible``
endpoint; ``ChatOpenAI`` speaks OpenAI's **Chat Completions**, to OpenAI or to
an ``openai-compatible`` endpoint (``chat_model``). Everything past the client
-- the graph, the stream, the mapping -- is one path for both: the chunks are
read through langchain-core's provider-neutral views, so a turn on either
protocol yields the same events for the same answer.

**Nothing phones home** (``docs/specs/core.md``). LangSmith is off, explicitly,
at construction and whatever the environment says (``force_tracing_off``), and
a vendor's client is built from the configuration rather than from the
environment -- the endpoint, the key, the proxy and the key's header
(``ANTHROPIC_ENDPOINT``, ``OPENAI_ENDPOINT``, ``clear_client_overrides``).

**And nothing is written down either.** The platform's logs never carry
conversation content: the vendor SDKs' loggers -- which on ``ANTHROPIC_LOG``
or ``OPENAI_LOG``, and on any root logger turned up afterwards, are switched on
for the whole process, and the first of them puts every request's messages
and system prompt on standard error -- are pinned at ``WARNING`` when the
engine is built (``quiet_client_logging``).

**Failure and cancellation** are the port's. Whatever the provider raises
travels out of the generator as it is; ``CancelledError`` is never swallowed;
and the ``finally`` closes the graph's stream, which is what lets go of the
model's stream, the HTTP response underneath it and the connection.

**What the ``finally`` does not do is close the vendor client.** A client is
built per turn (``chat_model``) and is released to the garbage collector with
the rest of the turn; its connection pool is closed by the HTTP client's own
finaliser and not by this adapter. Both engines are the same in this, and a
shared client held for the life of the process -- and closed by the lifespan,
as the identity provider's is -- is a change to both adapters and to the
composition root, which is a step of its own
(``docs/working-notes/poc-progress.md``).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import aclosing
from typing import Any

import langsmith
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import SystemMessage as ChatSystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from openai import AsyncOpenAI, OpenAI

from robinauts.adapters.config_file import ProviderKeys
from robinauts.domain import (
    NO_RESULT,
    AgentDefinition,
    AnswerCompleted,
    AnswerReasoningDelta,
    AnswerStarted,
    AnswerTextDelta,
    ConfigError,
    EngineEvent,
    InvalidValueError,
    Message,
    MessagePart,
    ModelConfig,
    ModelProviderConfig,
    ModelsConfig,
    ProviderKind,
    Role,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallPart,
    ToolCallStarted,
    ToolDefinition,
    UnsupportedContentError,
    WaitingOnTools,
    chain,
    checked_data,
    clean_text,
    text_parts,
    tools_for_request,
    unanswered_calls,
)
from robinauts.ports import Agent

_log = logging.getLogger(__name__)

ANTHROPIC_ENDPOINT = "https://api.anthropic.com"
"""Where Anthropic is, said here rather than left to the client to decide.

**A vendor's endpoint is not a thing to take from the environment.** Both the
Anthropic SDK and ``ChatAnthropic`` fall back to ``ANTHROPIC_BASE_URL`` /
``ANTHROPIC_API_URL`` when no base URL is passed, so one variable inherited
from a shell, a unit file or a container image would send every turn -- and
the operator's key with it -- to any host that variable named. The operator
says where a provider is in the configuration (``base_url``, and only for a
kind that names a protocol rather than a vendor), or it is this constant; there
is no third answer, and no way for the environment to be one
(``docs/specs/operations.md``). ``endpoint_of`` is where the choice between the
two is made, and it is the whole of it.
"""

OPENAI_ENDPOINT = "https://api.openai.com/v1"
"""Where OpenAI is, said here rather than left to the client to decide.

The same rule as ``ANTHROPIC_ENDPOINT``, for the other vendor: the OpenAI SDK
falls back to ``OPENAI_BASE_URL`` and ``ChatOpenAI`` to ``OPENAI_API_BASE``
before it, and either, inherited from a shell or an image, would send every
turn and the operator's key to the host it named. So the ``openai`` kind is
reached here and nowhere else, and it has no ``base_url`` to offer
(``domain.KINDS_WITH_BASE_URL``).

Spelt, unlike Anthropic's, **with** the ``/v1``: what OpenAI's client appends
to a base URL is ``/chat/completions`` alone, so the version is part of the
prefix -- which is also what an ``openai-compatible`` provider's ``base_url``
is (``endpoint_of``).
"""

ANTHROPIC_KINDS: frozenset[ProviderKind] = frozenset(
    {ProviderKind.ANTHROPIC, ProviderKind.ANTHROPIC_COMPATIBLE}
)
"""The kinds that speak Anthropic's Messages API, reached through ``ChatAnthropic``."""

OPENAI_KINDS: frozenset[ProviderKind] = frozenset(
    {ProviderKind.OPENAI, ProviderKind.OPENAI_COMPATIBLE}
)
"""The kinds that speak OpenAI's Chat Completions, reached through ``ChatOpenAI``.

Which client a turn is given is decided by these two sets and nothing else
(``chat_model``), and a kind in neither is refused rather than handed to
whichever client is nearer.
"""

CEILING_FIELDS: Mapping[ProviderKind, str] = {
    ProviderKind.OPENAI: "max_completion_tokens",
    ProviderKind.OPENAI_COMPATIBLE: "max_tokens",
}
"""The field a configured output ceiling is sent in, per OpenAI-protocol kind.

OpenAI's current models take ``max_completion_tokens`` and refuse the older
``max_tokens`` on its reasoning models; OpenRouter and many older compatible
servers take ``max_tokens`` only -- Pydantic AI's own OpenRouter profile says
so. A compatible endpoint is anybody's, so it is sent the field every one of
them knows, and OpenAI itself the field it asks for. The other engine sends
the same (``docs/specs/agents.md``).
"""

COMPATIBLE_KINDS: frozenset[ProviderKind] = frozenset(
    {ProviderKind.ANTHROPIC_COMPATIBLE, ProviderKind.OPENAI_COMPATIBLE}
)
"""The two kinds whose endpoint is the operator's ``base_url`` (``endpoint_of``).

Written out here rather than read off ``domain.KINDS_WITH_BASE_URL``, which is
the same two today: that set says which kinds the configuration gives an
address, and this one which addresses this engine has a client to send to. A
protocol kind added to the first tomorrow must not be sent anywhere by the
second before somebody has written its client.
"""

TRACING_VARIABLES_REMOVED = ("LANGCHAIN_TRACING", "LANGCHAIN_HANDLER")
"""The two variables ``force_tracing_off`` takes out of the environment.

They ask for LangChain's **version 1** tracer, which no longer exists.
langchain-core does not ignore them: when either is set and v2 tracing is off
-- which is what this adapter makes sure of -- it raises, so every turn of a
process that inherited one would fail. Turning tracing off must not be a way
of breaking a deployment, and the only honest way to say "no v1 tracing
either" is to unset them.
"""

CLIENT_VARIABLES_REMOVED = (
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_LOG",
    "OPENAI_CUSTOM_HEADERS",
    "OPENAI_LOG",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT_ID",
    "OPENAI_ADMIN_KEY",
)
"""The variables ``clear_client_overrides`` takes out of the environment.

The ones the two clients read that **no argument can override**; the endpoint,
the key and the proxy are arguments, and an argument wins.

``ANTHROPIC_CUSTOM_HEADERS`` because the SDK *merges* what it holds into
whatever the caller passed, so one line of it replaces the ``x-api-key``
header outright and a turn is spent on somebody else's key, or adds headers
nobody configured. An argument cannot say "and nothing else", so the variable
goes.

``ANTHROPIC_LOG`` because it is read at **import**, before any argument
exists: the Anthropic SDK under ``ChatAnthropic`` calls its own
``setup_logging()`` at the bottom of its ``__init__``, and ``debug`` or
``info`` there puts the ``anthropic`` and ``httpx2`` loggers at that level for
the whole process. What the first of them then writes is every request's
options, ``json_data`` included -- which is the system prompt and every
message of the conversation, in a log. Removing the variable is only half the
answer, since by construction time the import has already happened:
``quiet_client_logging`` is the other half, and it is the half that works.

The OpenAI SDK under ``ChatOpenAI`` has the same two, and four more of its
own. ``OPENAI_CUSTOM_HEADERS`` is merged exactly as Anthropic's is -- the key
travels in ``Authorization`` there, and an ``Authorization`` line in the
variable is dropped only when the caller passed one, which ``chat_model``
always does (``OPENAI_KEY_HEADER``); anything else in it would be sent on
every turn. ``OPENAI_LOG`` is read at import, as ``ANTHROPIC_LOG`` is, and on
``debug`` puts the ``openai`` logger at ``DEBUG`` and calls
``logging.basicConfig()``; this version of the SDK writes no request body there,
but the promise is not left to rest on what one version happens to log.
``OPENAI_ORG_ID`` and ``OPENAI_PROJECT_ID`` are read when the client is built
and not given, and ``None`` -- the only way to say "none" -- *is* not given,
so no argument can keep an ``OpenAI-Organization`` or ``OpenAI-Project``
header nobody configured off a turn; ``ChatOpenAI`` reads the first as
``OPENAI_ORGANIZATION`` too, with ``or``, so that an empty argument falls
through to it. And ``OPENAI_ADMIN_KEY`` is a second credential the client
picks up whenever it is not passed one, and holds for the life of the client:
it is not what a chat request is signed with, and it is still a key the
operator did not configure for this provider, in an object built for a turn.

Removed once, at construction, and never per turn: a process-wide edit made
while turns are running would be one turn changing another's environment.

What is **not** here, because an argument does say it or because nothing on a
turn's path reads it: ``OPENAI_API_KEY``, ``OPENAI_BASE_URL`` and
``OPENAI_API_BASE`` (the key and the endpoint are passed), ``OPENAI_PROXY``
(passed as ``None``), ``LC_OUTPUT_VERSION``,
``LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S`` and the
``LANGCHAIN_OPENAI_TCP_*`` socket settings (each passed, ``chat_model``),
``OPENAI_WEBHOOK_SECRET`` (read into a client, and used only to verify a
webhook, which the platform never receives), and the Azure and Bedrock
variables (``AZURE_OPENAI_*``, ``OPENAI_API_TYPE``, ``OPENAI_API_VERSION``,
``AWS_*``), which only the SDK's module-level client and its Azure and Bedrock
clients read, and none of those is ever built.
"""

QUIET_CLIENT_LOGGERS = (
    "anthropic",
    "anthropic._base_client",
    "openai",
    "openai._base_client",
    "httpx2",
    "httpcore2",
)
"""The loggers ``quiet_client_logging`` pins, and why these six.

``anthropic`` is the SDK's own, and ``anthropic._base_client`` is the module
that **emits** the record carrying a request's ``json_data`` -- the system
prompt and every message. The child is named as well as the parent because a
level set on a child is what a logger decides by: an ``anthropic._base_client``
entry in somebody's ``dictConfig`` would walk straight past a pin on
``anthropic``. ``httpx2`` is the HTTP client the SDK is built on -- a fork of
its own, so this is **not** the ``httpx`` the sign-in adapter uses and pinning
it silences nothing of ours -- and ``httpcore2`` is the connection layer under
that, which logs request lines and headers.

``openai`` and ``openai._base_client`` are the same pair in the other vendor's
SDK, which is built on the same ``httpx2``. This version of it writes no request
body at ``DEBUG`` -- its own comment says bodies "can contain private data" --
but ``OPENAI_LOG`` still turns the pair on for the whole process, and a pin is
cheaper than a promise that holds until an upgrade.

What this list is, and is not: the vendor loggers that write request bodies.
A logger an operator names at ``DEBUG`` in their own logging configuration
**after** start-up is their deliberate act on their own machine, and nothing
here fights it; what is answered is the state a process is *found* in.
"""

QUIET_CLIENT_LEVEL = logging.WARNING
"""The level those loggers are held at or above: no request is ever a record.

**The platform's logs never carry conversation content.** A message is content
of a conversation and belongs in the database (``docs/specs/conversations.md``);
a key is the operator's and is never logged (``docs/specs/agents.md``). Both
of those are in a vendor SDK's ``DEBUG`` records, so the honest guarantee is
that the records are never emitted -- not that nobody has attached a handler
that would catch them.
"""

ANTHROPIC_KEY_HEADER = "x-api-key"
"""The header Anthropic's key travels in, pinned by ``chat_model``.

Belt as well as braces, and per call rather than process-wide: a header the
caller passes wins over anything merged in from the environment, so the key
that is sent is the configured one whether or not the variable above was
there to be removed.
"""

OPENAI_KEY_HEADER = "Authorization"
"""The header OpenAI's key travels in, as ``Bearer <key>``, pinned by ``chat_model``.

The same belt as ``ANTHROPIC_KEY_HEADER``, and a little more than a belt: the
OpenAI SDK drops an ``Authorization`` line of ``OPENAI_CUSTOM_HEADERS`` only
when the caller passed an ``Authorization`` header of its own, so passing one is
what makes the configured key the only one that can be sent.
"""

OUTPUT_VERSION = "v0"
"""The shape langchain-core stores a chat model's answer in: the provider's own.

``BaseChatModel.output_version`` defaults to ``LC_OUTPUT_VERSION`` from the
environment, and ``v1`` there rewrites every answer's content into
langchain-core's standard blocks -- the thinking blocks this engine keeps as
the vendor's (``_extras``) and the ``tool_use`` blocks it checks for
(``_tool_use_ids``) would arrive under other names, and a turn would lose the
one and fail over the other. What the engine reads, it reads in the shape it
was written against, so the version is an argument and the variable is not
consulted.
"""

MAX_RETRIES = 0
"""How many times a failed model call is retried by the client: not at all.

The client's own retries are invisible to everything above -- they would
lengthen a turn silently and could send the same prompt twice after a timeout
the provider had already accepted. A turn is bounded by the application
(``robinauts.application.Turns``), a failure is reported by raising, and
retrying is sending the message again (``docs/specs/runs.md``).
"""

DEFAULT_ANTHROPIC_OUTPUT_TOKENS = 8192
"""What Anthropic is asked for when the model's configuration says nothing.

Its API requires a ceiling on every request, so an engine that sent none would
not work at all; the number is the client's business rather than the
platform's, which is why ``ModelConfig.max_output_tokens`` defaults to "leave
it to the engine" and this is where "the engine" answers.
"""

DEFAULT_OPENAI_OUTPUT_TOKENS: int | None = None
"""What an OpenAI-protocol provider is asked for when the configuration says nothing: no ceiling.

Chat Completions requires none, and the model's own limit is then the
ceiling. That is the engine's answer rather than a number of its own because
a number would be a worse one here than it is for Anthropic: on OpenAI's
reasoning models the ceiling covers the **reasoning** tokens as well as the
answer, so a ceiling picked to bound an answer's length can be spent entirely
on thinking and leave an empty answer. An operator who wants a bound writes
``max_output_tokens``, and it is sent in the field the kind takes
(``CEILING_FIELDS``). The other engine sends the same (``docs/specs/agents.md``).
"""

ANSWER_NODE = "answer"
"""The one node of the graph: the model, streamed."""

VENDOR = "anthropic"
"""The key the vendor's opaque blocks are kept under in a message's ``extras``.

One vendor's key for one vendor's blocks (``docs/specs/conversations.md``,
"Reasoning"): the two Anthropic kinds speak the Messages API, so what this
engine stores and what it replays are Anthropic's thinking blocks, under this
key. The two OpenAI kinds have none to keep: Chat Completions returns no signed
reasoning, so an answer produced over it carries no ``extras`` and nothing is
replayed to it -- the blocks are for the model that made them, and no model
reached over OpenAI's protocol made any.
"""

THINKING_BLOCKS = frozenset({"thinking", "redacted_thinking"})
"""The vendor's block types that carry signed reasoning.

What the vendor requires back, unchanged, when a tool call's results go back:
a ``thinking`` block with its ``signature``, or a ``redacted_thinking`` block
that is opaque throughout. These are what ``_extras`` keeps and ``_assistant``
replays, and only when they are signed (``_signed``); the reasoning a person
watched arrive is shown from the stream and kept in the run's events, and never
read out of these.
"""

BLOCKS_LEFT_OUT = (
    "the vendor's signed thinking blocks did not fit a message's extras and were left"
    " out; if the model asked for tools, the vendor may refuse the next round"
)
"""What the log says when an answer's blocks are bigger than ``extras`` may be.

The plan's open question, answered here for this iteration
(``docs/working-notes/mcp-plan.md``, "Open"): the answer is stored without
them rather than the turn failing over the size of the thinking, and the
line says what that may cost.
"""


ChatModelFactory = Callable[[ModelConfig, ModelProviderConfig, str], BaseChatModel]
"""How a chat model is built: the model, its provider, and the provider's key.

Injectable so that the tests run the engine over a chat model they script --
the graph, the streaming, the mapping and the releasing are then exercised for
real with no network and no key (``docs/layout.md``, "Testing strategy").
"""


def force_tracing_off() -> None:
    """Turn LangSmith off for this process, whatever the environment says.

    **Nothing phones home** (``docs/specs/core.md``, ``docs/specs/agents.md``).
    langchain-core decides whether to trace by asking langsmith
    (``langsmith.utils.tracing_is_enabled``), which answers from, in order: the
    tracing context, a run already in flight, a process-wide setting, and only
    then ``LANGSMITH_TRACING`` / ``LANGCHAIN_TRACING_V2`` in the environment.
    ``langsmith.configure(enabled=False)`` sets the process-wide setting *and*
    the context, so the environment is never reached and no ``LangChainTracer``
    is ever attached to a run -- which is the only thing that would open a
    client and send a conversation to a third party.

    It is deliberately **not** enough to pass no callbacks: langchain-core adds
    the tracer itself when it believes tracing is on, whatever a caller's
    ``config`` says. And it is deliberately process-wide: a stale export in a
    unit file must not turn tracing on for somebody else's runnable either.

    **It also unsets two variables**, which is the one thing here that reaches
    out of this adapter and into the process. An adapter may touch the
    environment -- it is the layer that may -- and this is why it must:
    ``LANGCHAIN_TRACING`` and ``LANGCHAIN_HANDLER`` ask for the version 1
    tracer, and langchain-core **raises** when one of them is set and v2
    tracing is off. Left alone, a deployment that inherited either would fail
    every turn as a *result* of tracing being turned off, which would make
    "nothing phones home" a way of breaking a server. Unsetting them says the
    same thing the switch above says, in the only words that half of
    langchain-core reads (``TRACING_VARIABLES_REMOVED``).
    """
    langsmith.configure(enabled=False)
    for name in TRACING_VARIABLES_REMOVED:
        os.environ.pop(name, None)


def clear_client_overrides() -> None:
    """Take out of the environment what no argument can override.

    The other half of "a vendor's client is built from the configuration and
    not from the environment" (``ANTHROPIC_ENDPOINT``, ``OPENAI_ENDPOINT``). An
    endpoint, a key and a proxy are arguments, and an argument wins; **headers
    are merged**, and an organisation, a project and an admin key are read
    whenever the argument is ``None`` -- which is the only way to pass
    "none" -- so the only way to say "and nothing else" is for the variable not
    to be there (``CLIENT_VARIABLES_REMOVED``).

    An adapter may touch the environment -- it is the layer that may -- and
    this does it once, when the engine is built at start-up, so that no turn
    ever edits the environment another turn is reading. ``ANTHROPIC_LOG`` and
    ``OPENAI_LOG`` go with them so that a subprocess this deployment starts
    does not import an SDK and turn its own logging on again; what protects
    **this** process, where the imports have already happened, is
    ``quiet_client_logging``.
    """
    for name in CLIENT_VARIABLES_REMOVED:
        os.environ.pop(name, None)


def quiet_client_logging() -> None:
    """Hold the vendor SDK's loggers at ``WARNING``: no request is ever a record.

    **The platform's logs never carry conversation content, and never a key.**
    The Anthropic SDK -- which ``langchain-anthropic`` is built on -- reads
    ``ANTHROPIC_LOG`` at *import*, before anything here exists (the OpenAI SDK
    under ``langchain-openai`` reads ``OPENAI_LOG`` the same way, and is pinned
    the same way), and on ``debug`` puts its own logger and its HTTP client's
    at ``DEBUG`` and calls
    ``logging.basicConfig()``, which attaches a handler to the root logger if
    nothing else has. What the SDK's logger then writes for every call is
    "Request options: ..." with the request's ``json_data`` in it: the agent's
    system prompt and every message of the conversation, on standard error.

    Removing the variable cannot undo that, because the import is what read it
    (``CLIENT_VARIABLES_REMOVED``). What does undo it is this: the vendor's
    loggers (``QUIET_CLIENT_LOGGERS``) are pinned at ``WARNING``
    (``QUIET_CLIENT_LEVEL``) when the engine is built, so the records are never
    emitted and it does not matter who has attached a handler -- which is the
    only form of the promise this adapter can keep, since the platform does not
    own every handler in the process and a deployment may add its own.

    **Each logger's own level is what is decided on, not its effective one.**
    An ordinary deployment has the variable unset and the root at ``WARNING``,
    so these loggers sit at ``NOTSET`` and their *effective* level is already
    ``WARNING`` -- and a pin that looked at that would do nothing at all,
    leaving them inheriting whatever the root becomes later. An operator who
    then turns their root logger up to ``DEBUG``, which is a thing an operator
    does, would get every message of every conversation on standard error from
    a switch that had nothing to do with the vendor. So ``NOTSET`` is treated
    as "not pinned yet" and set, and the promise holds whatever the root is
    moved to afterwards.

    A level already **stricter** than ``WARNING`` is left where it is: an
    operator who silenced the SDK altogether meant it.

    Spelt out again here rather than imported from the Pydantic AI adapter:
    the two adapters do not import each other (``docs/layout.md``), they reach
    the same SDK by different routes, and deleting either must leave the other
    whole.
    """
    for name in QUIET_CLIENT_LOGGERS:
        logger = logging.getLogger(name)
        # `NOTSET` is spelt out rather than left to be the zero it is: it means
        # "inherit", which is the case this exists for, and a reader should not
        # have to know its value to see that it is covered.
        if logger.level == logging.NOTSET or logger.level < QUIET_CLIENT_LEVEL:
            logger.setLevel(QUIET_CLIENT_LEVEL)


def endpoint_of(provider: ModelProviderConfig) -> str:
    """Where that provider is: the address the operator gave, or the vendor's own.

    The four kinds this engine reaches are two **vendors** and two
    **protocols** (``LangGraphAgent.kinds``). ``anthropic`` is Anthropic, at
    ``ANTHROPIC_ENDPOINT`` and nowhere else, and ``openai`` is OpenAI, at
    ``OPENAI_ENDPOINT``; neither has a ``base_url`` to offer.
    ``anthropic-compatible`` and ``openai-compatible`` are endpoints the
    operator names that speak the same protocol as the vendor -- OpenRouter
    serves both, a vLLM server or a gateway the second -- so the address is
    theirs, checked where every configured endpoint is
    (``domain.is_endpoint_url``: https, or http on the loopback interface, no
    query, no fragment and no credential written into it).

    **A base URL is a prefix the client appends the protocol's own path to**,
    and the two protocols' paths differ. Anthropic's client appends
    ``/v1/messages``, so an operator reaching OpenRouter that way writes
    ``https://openrouter.ai/api`` and the request goes to
    ``https://openrouter.ai/api/v1/messages``; OpenAI's appends
    ``/chat/completions``, so the version is the operator's to write, and the
    same OpenRouter over OpenAI's protocol is ``https://openrouter.ai/api/v1``,
    reached at ``https://openrouter.ai/api/v1/chat/completions``
    (``docs/specs/agents.md``).

    A provider of any other kind should not arrive here: the configuration was
    held to ``LangGraphAgent.kinds`` at start-up. One that does -- a kind added
    to the platform's vocabulary without a branch here, a caller that built a
    definition by hand -- gets a refusal naming it, and never a turn sent to
    whichever endpoint happened to be nearest. Every kind there is today has a
    branch, so the tests reach this one with a kind of their own, which is what
    keeps it a branch that is exercised rather than merely written.
    """
    if provider.kind is ProviderKind.ANTHROPIC:
        return ANTHROPIC_ENDPOINT
    if provider.kind is ProviderKind.OPENAI:
        return OPENAI_ENDPOINT
    if provider.kind in COMPATIBLE_KINDS and provider.base_url:
        return provider.base_url
    raise _unreachable(provider)


def _unreachable(provider: ModelProviderConfig) -> ConfigError:
    """The refusal for a provider this engine has no client for, naming it."""
    return ConfigError(
        [
            f"model_providers.{provider.id}: this build of the LangGraph engine cannot"
            f" reach a {provider.kind.value} provider"
        ]
    )


def chat_model(model: ModelConfig, provider: ModelProviderConfig, key: str) -> BaseChatModel:
    """The chat model that model's configuration describes.

    The key is passed in and not read here: what may touch the environment is
    ``robinauts.adapters.config_file``, and what reaches this is one key for
    one call (``docs/specs/agents.md``).

    **The provider's kind chooses the client, and nothing else does**: the two
    Anthropic kinds get ``ChatAnthropic``, described below, and the two OpenAI
    kinds ``ChatOpenAI`` (``_openai_chat_model``). A kind in neither set is
    refused, naming the provider.

    **Everything the client would otherwise take from the environment is
    passed.** The key, so that ``ANTHROPIC_API_KEY`` is never consulted and
    the key a turn spends is the one the operator configured for that
    provider; the endpoint (``endpoint_of``), so that ``ANTHROPIC_BASE_URL`` /
    ``ANTHROPIC_API_URL`` cannot redirect a turn and a key to another host --
    the configuration is the only thing that decides where a turn goes; and
    the proxy, as ``None``, so that
    ``ANTHROPIC_PROXY`` -- a variable only this one client would obey -- is
    not a second way to do the same thing. An operator who needs a proxy sets
    ``HTTPS_PROXY``, which is theirs and is how every other outbound call of
    this process is proxied. An explicit key and endpoint also keep the
    LangSmith gateway out of it: langchain-core reaches for it only when
    neither was given.

    Headers are the one thing an argument cannot settle on its own, because
    the SDK **merges** ``ANTHROPIC_CUSTOM_HEADERS`` into what the caller
    passed: the key is therefore pinned as a header too
    (``ANTHROPIC_KEY_HEADER``), where a caller's value wins, and the variable
    itself is taken out of the environment when the engine is built
    (``clear_client_overrides``). What is deliberately left alone: the SDK's
    credential auto-discovery, which is not consulted at all once a key is
    passed. The shape the answer is stored in is an argument too
    (``OUTPUT_VERSION``), so that ``LC_OUTPUT_VERSION`` cannot rename the
    blocks this engine reads.
    """
    if provider.kind in OPENAI_KINDS:
        return _openai_chat_model(model, provider, key)
    if provider.kind not in ANTHROPIC_KINDS:
        raise _unreachable(provider)
    return ChatAnthropic(
        model=model.name,  # type: ignore[call-arg]  # `model_name`'s alias
        api_key=key,  # type: ignore[call-arg]  # `anthropic_api_key`'s alias
        base_url=endpoint_of(provider),  # type: ignore[call-arg]
        anthropic_proxy=None,
        default_headers={ANTHROPIC_KEY_HEADER: key},
        timeout=model.timeout_seconds,  # type: ignore[call-arg]
        max_retries=MAX_RETRIES,
        max_tokens=model.max_output_tokens or DEFAULT_ANTHROPIC_OUTPUT_TOKENS,
        output_version=OUTPUT_VERSION,
    )


def _openai_chat_model(model: ModelConfig, provider: ModelProviderConfig, key: str) -> ChatOpenAI:
    """``ChatOpenAI`` over Chat Completions, with clients this adapter built itself.

    **The protocol is Chat Completions, for both kinds**, and never the
    Responses API: ``use_responses_api`` is ``False`` rather than left to
    ``ChatOpenAI``, which otherwise switches to Responses by itself for some
    model names and some arguments. An ``openai-compatible`` endpoint -- a
    gateway, vLLM, OpenRouter -- speaks Chat Completions, one protocol for both
    kinds keeps the two kinds and the two engines symmetric, and Pydantic AI's
    engine sends the same request through its own Chat Completions model
    (``docs/specs/agents.md``). The Responses API is a decision of its own.

    **The vendor's clients are built here and handed over**, as the other
    engine builds its own: an ``AsyncOpenAI`` for the turn, and an ``OpenAI``
    beside it only because ``ChatOpenAI`` builds a synchronous client of its
    own when it is not given one, and a client it built would be one this
    adapter had not pinned. Built here, each gets exactly what the
    configuration says -- the key, the endpoint (``endpoint_of``), the timeout,
    no retries (``MAX_RETRIES``) and the key pinned as its header
    (``OPENAI_KEY_HEADER``) -- and none of what ``ChatOpenAI`` would otherwise
    add on the way: an HTTP client **shared by every turn of the process** from
    a cache of its own, with TCP socket options read from
    ``LANGCHAIN_OPENAI_TCP_*``. What the SDK itself reads when an argument is
    ``None`` -- the organisation, the project, the admin key, the headers -- is
    out of the environment by the time a turn runs (``clear_client_overrides``).

    **And every environment fallback ``ChatOpenAI`` has is an argument.** The
    key and the endpoint, so that ``OPENAI_API_KEY``, ``OPENAI_API_BASE`` and
    the LangSmith gateway are never consulted; the proxy, as ``None``, so that
    ``OPENAI_PROXY`` -- a proxy only this client would obey -- is not a second
    ``HTTPS_PROXY``; ``stream_usage``, which it otherwise turns on or off
    according to whether ``OPENAI_BASE_URL`` is set, on, because the other
    engine always asks for the stream's usage and the two send the same
    request; ``stream_chunk_timeout``, off, because a turn's timeouts are the
    platform's -- the model call's (``timeout_seconds``) and the turn's -- and a
    third, set by ``LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S`` and found on
    neither the other client nor the other engine, would be a failure nobody
    configured; the socket options, as none; and the output shape
    (``OUTPUT_VERSION``).

    **The ceiling is sent only when the model's configuration has one**
    (``DEFAULT_OPENAI_OUTPUT_TOKENS``), and in the field the kind takes
    (``CEILING_FIELDS``). The history is written as the other engine writes
    it (``_ChatCompletions``).
    """
    endpoint = endpoint_of(provider)
    headers = {OPENAI_KEY_HEADER: f"Bearer {key}"}
    asynchronous = AsyncOpenAI(
        api_key=key,
        base_url=endpoint,
        timeout=model.timeout_seconds,
        max_retries=MAX_RETRIES,
        default_headers=headers,
    )
    synchronous = OpenAI(
        api_key=key,
        base_url=endpoint,
        timeout=model.timeout_seconds,
        max_retries=MAX_RETRIES,
        default_headers=headers,
    )
    return _ChatCompletions(
        ceiling_field=CEILING_FIELDS[provider.kind],
        model=model.name,
        api_key=key,  # type: ignore[arg-type]  # a str becomes the SecretStr
        base_url=endpoint,
        root_async_client=asynchronous,
        async_client=asynchronous.chat.completions,
        root_client=synchronous,
        client=synchronous.chat.completions,
        openai_proxy=None,
        timeout=model.timeout_seconds,
        max_retries=MAX_RETRIES,
        max_tokens=model.max_output_tokens or DEFAULT_OPENAI_OUTPUT_TOKENS,
        stream_usage=True,
        use_responses_api=False,
        output_version=OUTPUT_VERSION,
        stream_chunk_timeout=None,
        http_socket_options=(),
    )


class _ChatCompletions(ChatOpenAI):
    """``ChatOpenAI``, writing the request the way the other engine writes it.

    Two engines are handed one history, and over Chat Completions they are to
    send one request (``docs/specs/agents.md``). langchain-openai and Pydantic
    AI write the same messages in three different ways, and it is this side
    that moves, because this side has a single place to move it: the payload,
    after ``ChatOpenAI`` has built it and before the SDK sends it. What is
    changed is the **spelling**, never the meaning:

    - an assistant message with calls and no text has ``content`` ``""``, as
      Pydantic AI writes it, where langchain-openai writes ``null`` (OpenAI
      takes either);
    - a call's arguments are compact JSON -- no spaces after ``,`` and ``:``,
      and no ASCII escaping -- as Pydantic AI serialises them, where
      langchain-openai uses ``json.dumps``'s default separators;
    - each message's keys, and each call's, are in Pydantic AI's order, so
      that the messages are the same *bytes* and not merely equal objects.

    And one thing that is the kind's rather than the framework's: the output
    ceiling goes in the field the kind takes (``CEILING_FIELDS``), where
    ``ChatOpenAI`` always renames it to ``max_completion_tokens``.
    """

    ceiling_field: str = "max_completion_tokens"
    """Which field the ceiling is sent in: ``max_completion_tokens`` or ``max_tokens``."""

    def _get_request_payload(
        self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        payload: dict[str, Any] = super()._get_request_payload(input_, stop=stop, **kwargs)
        for field in ("max_completion_tokens", "max_tokens"):
            if field != self.ceiling_field and field in payload:
                payload[self.ceiling_field] = payload.pop(field)
        payload["messages"] = [_as_written(message) for message in payload.get("messages", [])]
        return payload


MESSAGE_KEYS = ("role", "tool_call_id", "content", "tool_calls")
"""The order Pydantic AI writes a Chat Completions message's keys in (``_as_written``)."""


def _as_written(message: dict[str, Any]) -> dict[str, Any]:
    """One Chat Completions message, spelt as the other engine spells it."""
    if message.get("role") == "assistant" and message.get("tool_calls"):
        if message.get("content") is None:
            message["content"] = ""
        message["tool_calls"] = [_call_as_written(call) for call in message["tool_calls"]]
    ordered = {key: message[key] for key in MESSAGE_KEYS if key in message}
    ordered.update((key, value) for key, value in message.items() if key not in ordered)
    return ordered


def _call_as_written(call: dict[str, Any]) -> dict[str, Any]:
    """One call of an assistant message: its keys in order, its arguments compact."""
    function = dict(call.get("function") or {})
    arguments = function.get("arguments")
    if isinstance(arguments, str) and arguments:
        function["arguments"] = _compact(json.loads(arguments))
    written = {"id": call.get("id"), "type": call.get("type", "function"), "function": function}
    written.update((key, value) for key, value in call.items() if key not in written)
    return written


def _compact(value: Any) -> str:
    """JSON with no whitespace and no ASCII escaping: how Pydantic AI writes it."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class LangGraphAgent(Agent):
    """The LangGraph engine: one turn, one graph, compiled and thrown away."""

    kinds: frozenset[ProviderKind] = ANTHROPIC_KINDS | OPENAI_KINDS
    """The provider kinds this build of the engine has a client for: all four.

    Two clients, two kinds each. ``ChatAnthropic`` reaches Anthropic itself and
    any endpoint that speaks Anthropic's Messages API at an address the
    operator gives; ``ChatOpenAI`` reaches OpenAI itself and any endpoint that
    speaks OpenAI's Chat Completions (``endpoint_of``, ``chat_model``).
    OpenRouter speaks both, so it may be configured as either kind -- the
    Messages API with ``https://openrouter.ai/api``, or Chat Completions with
    ``https://openrouter.ai/api/v1`` -- and they are two providers, not two
    spellings of one.

    A kind this engine does not build is not offered (``docs/specs/agents.md``),
    so the configuration would refuse it at start-up rather than the engine
    failing at the first turn; today there is no such kind. A new one is this
    set, a branch in ``endpoint_of`` and ``chat_model``, and its client's
    dependency -- nothing else.

    It is declared by the port (``robinauts.ports.Agent.kinds``) and answered
    here, so that the composition root asks the engine what it can reach
    instead of importing a second name from this sub-package: deleting the
    adapter must break the line that constructs it and nothing besides
    (``docs/layout.md``, the discard test).
    """

    def __init__(
        self,
        models: ModelsConfig,
        keys: ProviderKeys,
        *,
        chat_model_for: ChatModelFactory = chat_model,
    ) -> None:
        force_tracing_off()
        clear_client_overrides()
        quiet_client_logging()
        self._models = models
        """The models a turn may run on, and the provider each is reached through."""
        self._keys = keys
        """The providers' keys, as start-up read them. It prints nothing."""
        self._chat_model_for = chat_model_for
        self._open = 0

    @property
    def held(self) -> int:
        """How many turns of this engine still hold a stream open.

        Zero once every turn has ended or been closed, which is the promise a
        cancelled run depends on (``robinauts.ports.agents``). It counts the
        engine's own graph streams; the model's stream lives inside one and
        goes with it.

        **One engine, one count, however many turns it is running.** A
        deployment shares one ``LangGraphAgent`` between every conversation,
        so this says "is anything still open", not "is *that* turn still
        open": a caller watching one turn while another is in flight reads
        the other one's stream in this number. Nothing in the platform needs
        the finer answer -- the application releases a turn by closing its
        own stream and never asks -- and the contract suite reads it between
        turns, one at a time, which is when the two questions have the same
        answer.
        """
        return self._open

    def run_turn(
        self,
        agent: AgentDefinition,
        history: Sequence[Message],
        tools: Sequence[ToolDefinition],
        *,
        model: str,
    ) -> AsyncGenerator[EngineEvent, None]:
        """Answer ``history`` as ``agent`` on ``model``, streaming the events of the turn.

        Not a coroutine and nothing is done here: everything -- building the
        model, compiling the graph, opening the stream -- happens inside the
        generator, so that a provider that refuses a key is a failure of the
        turn, reported by raising where the caller is iterating, and not an
        exception thrown at whoever asked for the stream.
        """
        return self._turn(agent, history, tuple(tools), model)

    async def _turn(
        self,
        agent: AgentDefinition,
        history: Sequence[Message],
        tools: tuple[ToolDefinition, ...],
        model_id: str,
    ) -> AsyncGenerator[EngineEvent, None]:
        # The run's model, never the agent's default: the conversation may
        # have been moved to another (``robinauts.ports.agents``).
        model = self._models.model_by_id(model_id)
        provider = self._models.provider_for(model)
        chat = self._chat_model_for(model, provider, self._keys.key_for(provider.id))
        # The tools the run has, bound as the vendor's own definitions and
        # never executed here: a call is yielded and the turn ends. The port
        # is a seam a test double crosses too, so what comes over it is
        # checked to be the platform's definition and nothing that looks
        # like one.
        for tool in tools:
            if not isinstance(tool, ToolDefinition):
                raise InvalidValueError(f"a run's tools are ToolDefinitions, not {tool!r}")
        # With a stub for every name the history calls that the run lacks:
        # the vendor refuses tool blocks its request defines no tool for
        # (``core.tools_for_request``).
        defined = tools_for_request(tools, history)
        declared = _openai_tool if provider.kind in OPENAI_KINDS else _anthropic_tool
        bound: Runnable[Any, Any] = (
            chat.bind_tools([declared(tool) for tool in defined]) if defined else chat
        )
        graph = _compiled(bound)
        answer = _Answer()
        whole: AIMessage | None = None
        self._open += 1
        try:
            stream = graph.astream(
                {
                    "messages": _messages(
                        agent, history, model_id, chat_completions=provider.kind in OPENAI_KINDS
                    )
                },
                stream_mode=["messages", "updates"],
                # A fresh configuration each turn, carrying no callbacks: a
                # turn inherits nothing from whatever context it happens to
                # run in. What keeps a tracer away is `force_tracing_off`;
                # this keeps everything else away.
                config={"callbacks": []},
            )
            async with aclosing(stream):
                async for mode, payload in stream:
                    if mode == "messages":
                        chunk, _metadata = payload
                        for event in answer.events_of(chunk):
                            yield event
                    else:
                        whole = _reply(payload)
            for event in answer.complete(whole):
                yield event
        finally:
            # Reached when the turn ends, when it raises, and when the
            # iteration is closed -- which is what a cancellation does.
            self._open -= 1


def _compiled(chat: Runnable[Any, Any]) -> Any:
    """The turn's graph: one node that streams the model, and no checkpointer.

    Compiled per turn and thrown away with it. Cheap -- it is a handful of
    objects, not a client or a connection -- and the alternative would be a
    graph held between turns, which is the state ADR 0002 says an engine does
    not keep.

    The node streams rather than invokes: a model asked for the whole answer
    at once would arrive as one piece however well it streams, and what a
    person watches arrive is what the platform stores. ``chat`` is the model
    with the run's tools bound, or the model alone; there is no tool node,
    because the platform runs the tools (``docs/specs/agents.md``).
    """

    async def answer(state: MessagesState) -> dict[str, list[BaseMessage]]:
        reply: BaseMessage | None = None
        async for chunk in chat.astream(state["messages"]):
            reply = chunk if reply is None else reply + chunk  # type: ignore[operator]
        return {"messages": [reply] if reply is not None else []}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(MessagesState)
    graph.add_node(ANSWER_NODE, answer)
    graph.add_edge(START, ANSWER_NODE)
    graph.add_edge(ANSWER_NODE, END)
    return graph.compile()


def _anthropic_tool(tool: ToolDefinition) -> dict[str, Any]:
    """The definition as the vendor's client takes it, as it stands.

    Anthropic's own shape -- ``name``, ``description``, ``input_schema`` --
    which ``bind_tools`` passes through untouched. The name is the full name
    the platform gave it, and the schema is the server's, as plain data. An
    empty description is left out rather than sent as ``""``: MCP's is
    optional, the vendor takes a tool without one, and the other engine sends
    the same bytes for the same definition (``docs/specs/agents.md``).
    """
    definition: dict[str, Any] = {"name": tool.name, "input_schema": dict(tool.input_schema)}
    if tool.description:
        definition["description"] = tool.description
    return definition


def _openai_tool(tool: ToolDefinition) -> dict[str, Any]:
    """The definition as Chat Completions takes it, as it stands.

    OpenAI's own shape -- a ``function`` holding ``name``, ``description`` and
    ``parameters`` -- which ``bind_tools`` passes through untouched, as it does
    Anthropic's. The name and the schema are the platform's and the server's,
    exactly as for the other protocol, and no ``strict`` is added: strict mode
    would have the vendor hold the model to a schema rewritten to its rules,
    and the schema sent is the server's. **An empty description is sent as
    ``""``** here, where the Anthropic shape leaves it out, because that is what
    the other engine's client sends over this protocol and the two send the
    same bytes for the same definition (``docs/specs/agents.md``).
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": dict(tool.input_schema),
        },
    }


def _messages(
    agent: AgentDefinition,
    history: Sequence[Message],
    model_id: str,
    *,
    chat_completions: bool = False,
) -> list[BaseMessage]:
    """The history as the framework's messages, with the system prompt in front.

    The system prompt is the **agent's** and is not one of the messages
    (``docs/specs/conversations.md``), so it is put here, at every turn, from
    the definition as it stands now. An empty one is left out rather than sent
    as an empty system message, which some providers refuse.

    **Text, calls and results.** The reasoning a previous turn streamed is
    not carried back to the model as content: what is stored of it is the
    platform's record, not the vendor's, and the vendor's own signed blocks
    travel in ``extras`` and are replayed by ``_assistant`` on their own
    terms. A message with no text at all still
    becomes a message, empty, because dropping it *here* would be this engine
    deciding what a turn that said nothing means. What becomes of it is the
    vendor mapping's, and over Anthropic's protocol it is the same answer
    under both engines: the API refuses an empty content block, so
    langchain-anthropic drops an assistant message whose content came out empty
    and Pydantic AI leaves out the assistant message it emptied. The model is
    shown the same history either way, which is what the swap needs. Over
    OpenAI's protocol, which takes an empty assistant message, both frameworks
    send it as ``""``.

    An answer's tool calls travel as the framework's ``tool_calls``, and a
    **tool message** becomes one ``ToolMessage`` per result, each naming its
    call and whether it went wrong -- which langchain-anthropic folds into the
    one ``user`` turn of ``tool_result`` blocks the vendor wants back, and
    langchain-openai sends as one ``tool`` message per result. Chat
    Completions has no field for a result that went wrong, so over it such a
    result is written as ``{"error": <text>}``, which is how the other
    engine's framework writes it (``_tool_message``). An answer whose calls no
    tool message answers is followed by one error result per call saying no
    result of it was recorded (``domain.NO_RESULT``).

    **Over Chat Completions** (``chat_completions``) the history is the other
    engine's, byte for byte (``docs/specs/agents.md``): an answer is its text
    as a string beside its calls, with no Anthropic blocks -- Anthropic's
    signed blocks go back only to a model reached over Anthropic's protocol,
    since over OpenAI's there is nowhere to put them, and a model id the
    operator moved from one provider to another between restarts must not
    carry them across (``VENDOR``) -- and ``_ChatCompletions`` spells what
    langchain-openai then writes as Pydantic AI writes it.
    """
    replay_to = None if chat_completions else model_id
    messages: list[BaseMessage] = []
    if agent.system_prompt:
        messages.append(ChatSystemMessage(agent.system_prompt))
    unanswered = unanswered_calls(history)
    for message in history:
        if message.role is Role.USER:
            messages.append(HumanMessage(message.text))
        elif message.role is Role.ASSISTANT:
            messages.append(_assistant(message, replay_to, chat_completions=chat_completions))
            # A call no tool message answers is answered here with what the
            # record says of it (``domain.NO_RESULT``): the vendor refuses a
            # call with nothing answering it, and the record, which keeps the
            # call without a result, is not what is edited.
            messages.extend(
                _tool_message(call.call_id, NO_RESULT, is_error=True, wrapped=chat_completions)
                for call in unanswered.get(message.id, ())
            )
        else:
            messages.extend(
                _tool_message(
                    part.call_id, part.text, is_error=part.is_error, wrapped=chat_completions
                )
                for part in message.tool_results
            )
    return messages


def _tool_message(call_id: str, text: str, *, is_error: bool, wrapped: bool) -> ToolMessage:
    """One result as the framework's message, naming its call and how it went.

    ``wrapped`` is Chat Completions, whose ``tool`` message has no error flag:
    a result that went wrong is then written ``{"error": <text>}``, compact, as
    Pydantic AI writes it, so that the model is told and the two engines send
    the same bytes.
    """
    content = _compact({"error": text}) if is_error and wrapped else text
    return ToolMessage(
        content=content, tool_call_id=call_id, status="error" if is_error else "success"
    )


def _assistant(message: Message, model_id: str | None, *, chat_completions: bool) -> AIMessage:
    """An answer as the framework's message: its text, its calls, its signed blocks.

    **The vendor's signed thinking blocks are replayed to the model that made
    them** (``docs/specs/conversations.md``, "Reasoning"): they are bound to
    it, so an answer produced on another model is sent without them and
    nothing else is lost. They go in front of the text, where the vendor put
    them, exactly as they were stored; nothing here reads them. A plain answer
    -- no calls, no blocks -- is the one string it always was.
    """
    text = message.text
    calls = [
        {"name": part.name, "args": dict(part.arguments), "id": part.call_id, "type": "tool_call"}
        for part in message.tool_calls
    ]
    blocks = _replayed(message, model_id)
    if chat_completions:
        # The text as a string beside the calls, as the other engine writes it.
        return AIMessage(content=text, tool_calls=calls)
    if not calls and not blocks:
        return AIMessage(text)
    content: list[dict[str, Any]] = [*blocks]
    if text:
        content.append({"type": "text", "text": text})
    return AIMessage(content=content, tool_calls=calls)


def _replayed(message: Message, model_id: str | None) -> list[dict[str, Any]]:
    """The vendor's blocks stored on that answer, if they are for this model.

    ``None`` is a model no blocks are replayed to: one reached over OpenAI's
    protocol (``LangGraphAgent._turn``).
    """
    if model_id is None or message.provenance is None or message.provenance.model != model_id:
        return []
    kept = message.extras.get(VENDOR)
    if not isinstance(kept, Mapping):
        return []
    blocks = kept.get("thinking")
    if not isinstance(blocks, list):
        return []
    return [dict(block) for block in blocks if _signed(block)]


def _signed(block: object) -> bool:
    """Whether a block is one the vendor takes back: signed, or opaque throughout.

    A ``thinking`` block with a signature, or a ``redacted_thinking`` block
    with its data -- the same two shapes the other engine keeps and replays.
    A ``thinking`` block with no signature is what a model that signs nothing
    sends through an Anthropic-compatible endpoint (OpenRouter's GPT, for one),
    and the Messages API refuses a request that carries one back, so it is
    neither kept nor replayed: its text was streamed as reasoning already.
    Checked on the way back too, since an answer stored before this was
    checked may hold one.
    """
    if not isinstance(block, Mapping):
        return False
    if block.get("type") == "redacted_thinking":
        return isinstance(block.get("data"), str) and bool(block["data"])
    return (
        block.get("type") == "thinking"
        and isinstance(block.get("thinking"), str)
        and isinstance(block.get("signature"), str)
        and bool(block["signature"])
    )


class _Streaming:
    """One tool call while it streams: where it is in the answer, and what is known of it.

    A call is **keyed by its index** in the answer, which is how both clients
    say which call a chunk belongs to. Its id and its name need not arrive
    together, nor first: OpenAI-protocol servers send them on the first chunk,
    some compatible ones split them over two or send them again on every delta,
    and arguments may come before either. So a call is announced the moment
    both are known, and whatever arguments arrived before that are published
    then, in the order they came.
    """

    def __init__(self, index: int | None) -> None:
        self.index = index
        self.call_id: str | None = None
        self.name: str | None = None
        self.announced = False
        self.arguments: list[str] = []

    def continued_by(self, index: int | None, call_id: str | None) -> bool:
        """Whether a chunk with that index and id is more of this call."""
        if index is not None and self.index is not None:
            return index == self.index
        return call_id is None or self.call_id is None or call_id == self.call_id


class _Answer:
    """One answer as it streams: what was said, what was called, and the events.

    The order the port asks for (``robinauts.core.check_engine_events``) is
    kept here: the answer is announced on its first piece, a call is announced
    once the stream has given both its id and its name (``_Streaming``) and
    completed when the next one begins or the answer ends, and what the call
    completes with is what its deltas parse to.
    """

    def __init__(self) -> None:
        self.started = False
        self.streamed: list[str] = []
        self.calls: list[ToolCallPart] = []
        self._open: _Streaming | None = None
        """The call being made, while its arguments stream."""

    def events_of(self, chunk: Any) -> list[EngineEvent]:
        """The events one streamed chunk carries, in the order they came."""
        if not isinstance(chunk, AIMessageChunk):
            # A model that does not stream: LangGraph passes the whole
            # ``AIMessage`` through here unchanged, and the answer is built
            # from the node's update in ``complete`` instead.
            return []
        events: list[EngineEvent] = []
        for block in chunk.content_blocks:
            kind = block.get("type")
            if kind == "text" and block.get("text"):
                text = str(block["text"])
                self.streamed.append(text)
                events.append(AnswerTextDelta(text=text))
            elif kind == "reasoning" and block.get("reasoning"):
                events.append(AnswerReasoningDelta(text=str(block["reasoning"])))
        for piece in chunk.tool_call_chunks:
            events.extend(self._piece(piece))
        if events and not self.started:
            self.started = True
            events.insert(0, AnswerStarted())
        return events

    def complete(self, whole: AIMessage | None) -> list[EngineEvent]:
        """The events that end the answer, given the message the node left in the state.

        ``whole`` is what an answer that never streamed completes with, where a
        call still open here that streamed no arguments takes its arguments
        from, and where the signed blocks are read off. A call the framework's final message holds
        that was never announced -- a block the client did not lift off the
        stream -- is refused rather than dropped: an answer missing a call
        would be half an answer that looks whole.
        """
        events: list[EngineEvent] = list(self._closed(whole))
        if not self.started:
            # Nothing was streamed that this version carries: the answer is
            # whatever the node left in the state, announced and completed in
            # one breath -- an engine is never required to stream.
            events.append(AnswerStarted())
            self.started = True
            for call in _calls_of(whole):
                events.append(ToolCallStarted(call_id=call.call_id, name=call.name))
                events.append(ToolCallCompleted(call=call))
                self.calls.append(call)
        announced = {call.call_id for call in self.calls}
        if any(call_id not in announced for call_id in _tool_use_ids(whole)):
            raise UnsupportedContentError(
                "the model asked for a tool in a form this engine did not translate"
            )
        parts: list[MessagePart] = []
        text = clean_text("".join(self.streamed)) if self.streamed else _text_of(whole)
        if text or not self.calls:
            parts.extend(text_parts(text))
        parts.extend(self.calls)
        events.append(AnswerCompleted(parts=tuple(parts), extras=_extras(whole)))
        if self.calls:
            events.append(WaitingOnTools())
        return events

    def _piece(self, piece: Mapping[str, Any]) -> list[EngineEvent]:
        """The events one tool-call chunk carries: a call begun, announced, or argued."""
        index = piece.get("index")
        call_id = str(piece["id"]) if piece.get("id") else None
        name = str(piece["name"]) if piece.get("name") else None
        arguments = str(piece.get("args") or "")
        events: list[EngineEvent] = []
        streaming = self._open
        if streaming is None or not streaming.continued_by(index, call_id):
            # A new call: the one before it, if any, is whole.
            events.extend(self._closed(None))
            streaming = self._open = _Streaming(index if isinstance(index, int) else None)
        if call_id is not None:
            if streaming.call_id is None:
                streaming.call_id = call_id
            elif streaming.call_id != call_id:
                raise UnsupportedContentError("the model gave one tool call two ids")
        if name is not None:
            if streaming.name is None:
                streaming.name = name
            elif streaming.name != name:
                raise UnsupportedContentError("the model gave one tool call two names")
        if arguments:
            streaming.arguments.append(arguments)
        if streaming.announced:
            if arguments:
                events.append(
                    ToolCallArgumentsDelta(call_id=str(streaming.call_id), text=arguments)
                )
        elif streaming.call_id is not None and streaming.name is not None:
            streaming.announced = True
            events.append(ToolCallStarted(call_id=streaming.call_id, name=streaming.name))
            events.extend(
                ToolCallArgumentsDelta(call_id=streaming.call_id, text=held)
                for held in streaming.arguments
            )
        return events

    def _closed(self, whole: AIMessage | None) -> list[EngineEvent]:
        """Complete the call being made, if there is one, with what it streamed.

        A call the stream never gave both an id and a name was never announced,
        and cannot be completed: it is refused, as a call that did not arrive
        whole.
        """
        streaming = self._open
        if streaming is None:
            return []
        if not streaming.announced or streaming.call_id is None or streaming.name is None:
            raise UnsupportedContentError(
                "the model streamed arguments for no tool call this engine announced"
                if streaming.arguments
                else "the model streamed a tool call without its id or its name"
            )
        call_id = streaming.call_id
        joined = clean_text("".join(streaming.arguments))
        if joined.strip():
            arguments = _parsed(joined)
        else:
            arguments = next(
                (call.arguments for call in _calls_of(whole) if call.call_id == call_id), {}
            )
        call = ToolCallPart(call_id=call_id, name=streaming.name, arguments=arguments)
        self.calls.append(call)
        self._open = None
        return [ToolCallCompleted(call=call)]


def _parsed(arguments: str) -> dict[str, Any]:
    """The JSON the model wrote for a call, as the object it has to be."""
    try:
        parsed = json.loads(arguments)
    except (ValueError, RecursionError):
        raise UnsupportedContentError(
            "the model's arguments for a tool call were not JSON"
        ) from None
    if not isinstance(parsed, dict):
        raise UnsupportedContentError("the model's arguments for a tool call were not an object")
    return parsed


def _reply(payload: Any) -> AIMessage | None:
    """The message the node put in the state, if it put one there."""
    if not isinstance(payload, Mapping):  # pragma: no cover -- LangGraph yields these
        return None
    update = payload.get(ANSWER_NODE)
    if not isinstance(update, Mapping):  # pragma: no cover -- our own node's shape
        return None
    messages = update.get("messages") or []
    return next((message for message in messages if isinstance(message, AIMessage)), None)


def _calls_of(whole: AIMessage | None) -> list[ToolCallPart]:
    """The calls the framework's final message holds, as the platform's parts."""
    if whole is None:
        return []
    calls: list[ToolCallPart] = []
    for call in whole.tool_calls:
        arguments = call.get("args")
        calls.append(
            ToolCallPart(
                call_id=str(call.get("id") or ""),
                name=str(call.get("name") or ""),
                arguments=arguments if isinstance(arguments, Mapping) else {},
            )
        )
    return calls


def _tool_use_ids(whole: AIMessage | None) -> list[str]:
    """The ids of every call the final message holds, in any of the forms it holds one.

    Three places are read. Anthropic's own ``tool_use`` blocks in the content,
    which ``ChatAnthropic`` streams beside its tool-call chunks. The
    framework's own ``tool_calls``, which langchain-core builds from the
    streamed tool-call chunks merged by index -- over OpenAI's protocol, the
    one place a call is held, since ``langchain-openai`` 1.6.2 keeps no raw
    call on the chunks it streams. And OpenAI-shaped raw ``tool_calls`` in
    ``additional_kwargs``, which the streaming path of that version does not
    write and a client or a model that answers whole may; read so that a call
    carried only there is not passed over. A call in any of them that the
    stream never announced is refused by ``_Answer.complete`` rather than
    dropped, over either protocol.
    """
    if whole is None:
        return []
    ids: list[str] = []
    if not isinstance(whole.content, str):
        ids.extend(
            str(block.get("id"))
            for block in whole.content
            if isinstance(block, Mapping) and block.get("type") == "tool_use"
        )
    raw = whole.additional_kwargs.get("tool_calls")
    if isinstance(raw, list):
        ids.extend(str(call["id"]) for call in raw if isinstance(call, Mapping) and call.get("id"))
    ids.extend(str(call["id"]) for call in whole.tool_calls if call.get("id"))
    return ids


def _text_of(whole: AIMessage | None) -> str:
    """A framework message's text, and none of its reasoning.

    ``BaseMessage.text`` is the text blocks alone, which is exactly the line
    this version draws: thinking is shown as it arrives and is never part of
    what is stored (``docs/specs/conversations.md``).
    """
    if whole is None:
        return ""
    text = getattr(whole, "text", "")
    return clean_text(text) if isinstance(text, str) else ""


def _extras(whole: AIMessage | None) -> dict[str, Any]:
    """The vendor's signed blocks off the final message, keyed by vendor.

    Kept as they came, less the stream's own ``index``, and only those the
    vendor takes back (``_signed``). Bounded as every ``extras`` is: blocks
    that do not fit are left out with a line in the log (``BLOCKS_LEFT_OUT``)
    rather than failing the turn over the size of the thinking.
    """
    if whole is None or isinstance(whole.content, str):
        return {}
    blocks = [
        {key: value for key, value in block.items() if key != "index"}
        for block in whole.content
        if _signed(block)
    ]
    if not blocks:
        return {}
    extras = {VENDOR: {"thinking": blocks}}
    try:
        return checked_data(extras, "an answer's extras")
    except InvalidValueError as too_big:
        _log.warning("%s: %s", BLOCKS_LEFT_OUT, chain(too_big))
        return {}
