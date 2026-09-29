# SPDX-License-Identifier: Apache-2.0
# Copyright The Robinauts Authors

"""One turn on Pydantic AI: an agent built per turn, streamed, translated.

The second implementation of the ``Agent`` port (``robinauts.ports.agents``),
and the only module in the platform that may name Pydantic AI
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

**Stateless per turn** (ADR 0002). The framework's agent is built for the turn
and thrown away, and the whole of the conversation goes in as
``message_history``: nothing is remembered between calls, which is what lets
the next turn of the same conversation run on the other engine.

**The system prompt is ``instructions=``, not ``system_prompt=``.** The two
differ in exactly the way that matters here: a ``system_prompt`` becomes a
``SystemPromptPart`` *in the message history*, carried along with it and
replayed, while ``instructions`` are taken from the agent's definition at
every run and never enter the history. The platform's history holds messages
and no system prompt (``docs/specs/conversations.md``), and editing an agent
must take effect at the next turn of its existing conversations
(``docs/specs/agents.md``) -- both of which are what ``instructions`` means.

**One model call is the whole of a turn.** The graph is iterated until its
first model request node has been streamed, and then left. What that keeps out
is the framework's own tool loop and its retrying: a model asking for a tool
makes Pydantic AI run it and ask the model again, and an answer with no text
in it makes it send a *retry prompt* -- a second answer in one turn, charged
to the operator, that nobody asked for. The platform owns the tool loop
(``docs/specs/runs.md``, "Tools") and retrying is sending the message again,
so the turn ends where the model's first answer ends, whatever it asked for.

**Tools are declared and never run here.** The run's ``ToolDefinition``s go
to the model as an ``ExternalToolset`` -- the framework's name for tools
something outside it executes -- so the vendor is shown them as it would be
any tool and the framework holds nothing that could run one; the turn ends
before the framework would look for one anyway.

**The mapping**, which is the whole of the translation (``_Answer``):

- the node's stream yields ``PartStartEvent``s (a part, with the first of its
  content already in it), ``PartDeltaEvent``s (more of one) and
  ``PartEndEvent``s (the whole part, once it is). Text becomes
  ``AnswerTextDelta``, thinking becomes ``AnswerReasoningDelta``, and the
  first of anything opens the answer (``AnswerStarted``);
- a **tool call** starting is ``ToolCallStarted`` with the vendor's id and the
  tool's full name, each piece of its arguments is ``ToolCallArgumentsDelta``,
  and the call is completed -- when its part ends, when the next part begins
  or when the answer ends -- with what the pieces parse to (arguments the
  framework hands over whole are written out as one piece);
- ``PartEndEvent`` for text and thinking is passed over, as is every other
  event of the stream: repeating it would store the answer twice;
- **what was streamed is what is kept** (``docs/specs/agents.md``): the answer
  completes with exactly the text deltas, joined, and exactly the calls it
  announced. The response the framework assembled from the same stream is
  read for two things only, neither of them content a person watched arrive:
  the vendor's **signed thinking blocks**, kept in the answer's ``extras``
  under the vendor's key and replayed to the model that made them and to no
  other (``docs/specs/conversations.md``, "Reasoning"); and a call the
  framework holds that was never announced, which is refused rather than
  dropped;
- **reasoning is never in the completed parts.** ``domain.kept_parts`` would
  drop a ``ReasoningPart`` anyway; putting the model's thinking in the
  answer's *text* is the mistake that would survive that, so the text part is
  built from the text deltas alone -- and the signed blocks are never read
  as reasoning either: they are carried and replayed, as the vendor's data.

**Two protocols, four kinds** (``PydanticAIAgent.kinds``). ``AnthropicModel``
speaks Anthropic's Messages API, to Anthropic or to an ``anthropic-compatible``
endpoint; ``OpenAIChatModel`` speaks OpenAI's **Chat Completions** -- never the
Responses API, which is ``OpenAIResponsesModel`` and is not built here -- to
OpenAI or to an ``openai-compatible`` endpoint (``chat_model``). Everything
past the model is one path for both: the framework hands either protocol's
stream over as the same part events, and the mapping below reads those.

**Nothing phones home** (``docs/specs/core.md``). Pydantic AI's instrumentation
is turned off explicitly on every agent it builds (``force_tracing_off``,
``_runner``), so no tracer, no exporter and no Logfire client is ever made
whatever the environment says; and a vendor's client is built from the
configuration rather than from the environment -- the endpoint, the key and
the key's header (``ANTHROPIC_ENDPOINT``, ``OPENAI_ENDPOINT``,
``clear_client_overrides``).

**And nothing is written down either.** The platform's logs never carry
conversation content: the vendor SDKs' loggers -- which on ``ANTHROPIC_LOG``
or ``OPENAI_LOG``, and on any root logger turned up afterwards, are switched on
for the whole process, and the first of them puts every request's messages
and system prompt on standard error -- are pinned at ``WARNING`` when the
engine is built (``quiet_client_logging``).

**Failure and cancellation** are the port's. Whatever the provider raises
travels out of the generator as it is; ``CancelledError`` is never swallowed;
and the ``finally`` leaves the framework's two context managers, which is what
lets go of the model's stream, the HTTP response underneath it and the
connection.

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
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import pydantic_ai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types.chat import chat_completion_chunk
from pydantic_ai import Agent as FrameworkAgent
from pydantic_ai import ModelRequest, ModelResponse, ModelSettings
from pydantic_ai.messages import (
    BaseToolCallPart,
    ModelMessage,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.messages import TextPart as FrameworkTextPart
from pydantic_ai.messages import ToolCallPart as FrameworkToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIStreamedResponse
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import ToolDefinition as FrameworkToolDefinition
from pydantic_ai.toolsets import ExternalToolset

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
    ToolResultPart,
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

**A vendor's endpoint is not a thing to take from the environment.** The
Anthropic SDK falls back to ``ANTHROPIC_BASE_URL`` when no base URL is passed,
so one variable inherited from a shell, a unit file or a container image would
send every turn -- and the operator's key with it -- to any host that variable
named. The operator says where a provider is in the configuration
(``base_url``, and only for a kind that names a protocol rather than a vendor),
or it is this constant; there is no third answer, and no way for the
environment to be one (``docs/specs/operations.md``). ``endpoint_of`` is where
the choice between the two is made, and it is the whole of it.

Spelt out again here rather than imported from the LangGraph adapter: the two
adapters do not import each other (``docs/layout.md``), and deleting either
must leave the other whole.
"""

OPENAI_ENDPOINT = "https://api.openai.com/v1"
"""Where OpenAI is, said here rather than left to the client to decide.

The same rule as ``ANTHROPIC_ENDPOINT``, for the other vendor: the OpenAI SDK
falls back to ``OPENAI_BASE_URL`` when no base URL is passed, and a variable
inherited from a shell or an image would send every turn and the operator's
key to the host it named. So the ``openai`` kind is reached here and nowhere
else, and it has no ``base_url`` to offer (``domain.KINDS_WITH_BASE_URL``).

Spelt, unlike Anthropic's, **with** the ``/v1``: what OpenAI's client appends
to a base URL is ``/chat/completions`` alone, so the version is part of the
prefix -- which is also what an ``openai-compatible`` provider's ``base_url``
is (``endpoint_of``).
"""

ANTHROPIC_KINDS: frozenset[ProviderKind] = frozenset(
    {ProviderKind.ANTHROPIC, ProviderKind.ANTHROPIC_COMPATIBLE}
)
"""The kinds that speak Anthropic's Messages API, reached through ``AnthropicModel``."""

OPENAI_KINDS: frozenset[ProviderKind] = frozenset(
    {ProviderKind.OPENAI, ProviderKind.OPENAI_COMPATIBLE}
)
"""The kinds that speak OpenAI's Chat Completions, reached through ``OpenAIChatModel``.

Which model a turn is given is decided by these two sets and nothing else
(``chat_model``), and a kind in neither is refused rather than handed to
whichever client is nearer.
"""

COMPATIBLE_KINDS: frozenset[ProviderKind] = frozenset(
    {ProviderKind.ANTHROPIC_COMPATIBLE, ProviderKind.OPENAI_COMPATIBLE}
)
"""The two kinds whose endpoint is the operator's ``base_url`` (``endpoint_of``).

Written out here rather than read off ``domain.KINDS_WITH_BASE_URL``, which is
the same two today: that set says which kinds the configuration gives an
address, and this one which addresses this engine has a client to send to.
"""

TRACING_VARIABLES_REMOVED: tuple[str, ...] = ()
"""What ``force_tracing_off`` takes out of the environment: nothing.

Deliberately empty, and named all the same so that the claim is written down
rather than merely true today. Pydantic AI has **no environment switch for
instrumentation**: it traces when, and only when, an agent's ``instrument``
says so -- set by ``logfire.instrument_pydantic_ai()``, by
``Agent.instrument_all()`` or per agent -- and ``_runner`` sets it to ``False``
on every agent this engine builds, which beats both of the others. Nothing is
read from ``LOGFIRE_TOKEN``, ``LOGFIRE_SEND_TO_LOGFIRE``,
``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_TRACES_EXPORTER`` or any
``PYDANTIC_AI_*`` variable on the path a turn takes: an ``OTel`` tracer
provider is asked for only while *building* instrumentation settings, which
never happens. So there is nothing here that an argument cannot say, and
editing a process's environment for the sake of saying it again would be an
adapter reaching further than it needs to (compare the LangGraph adapter,
where a variable really is load-bearing).
"""

CLIENT_VARIABLES_REMOVED = (
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_LOG",
    "OPENAI_CUSTOM_HEADERS",
    "OPENAI_LOG",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "OPENAI_ADMIN_KEY",
)
"""The variables ``clear_client_overrides`` takes out of the environment.

The ones the two clients read that **no argument can override**; the endpoint
and the key are arguments, and an argument wins.

``ANTHROPIC_CUSTOM_HEADERS`` because the SDK *merges* what it holds into
whatever the caller passed, so one line of it replaces the ``x-api-key``
header outright and a turn is spent on somebody else's key, or adds headers
nobody configured. An argument cannot say "and nothing else", so the variable
goes.

``ANTHROPIC_LOG`` because it is read at **import**, before any argument
exists: ``anthropic`` calls its own ``setup_logging()`` at the bottom of its
``__init__``, and ``debug`` or ``info`` there puts the ``anthropic`` and
``httpx2`` loggers at that level for the whole process. What the first of them
then writes is every request's options, ``json_data`` included -- which is the
system prompt and every message of the conversation, in a log. Removing the
variable is only half the answer, since by construction time the import has
already happened: ``quiet_client_logging`` is the other half, and it is the
half that works.

The OpenAI SDK, which this engine builds its OpenAI client from, has the same
two and three more. ``OPENAI_CUSTOM_HEADERS`` is merged exactly as Anthropic's
is -- the key travels in ``Authorization`` there, and an ``Authorization`` line
in the variable is dropped only when the caller passed one, which
``chat_model`` always does (``OPENAI_KEY_HEADER``); anything else in it would
be sent on every turn. ``OPENAI_LOG`` is read at import, as ``ANTHROPIC_LOG``
is, and on ``debug`` puts the ``openai`` logger at ``DEBUG`` and calls
``logging.basicConfig()``; this version of the SDK writes no request body
there, but the promise is not left to rest on what one version happens to log.
``OPENAI_ORG_ID`` and ``OPENAI_PROJECT_ID`` are read when the client is built
and not given, and ``None`` -- the only way to say "none" -- *is* not given,
so no argument can keep an ``OpenAI-Organization`` or ``OpenAI-Project``
header nobody configured off a turn. And ``OPENAI_ADMIN_KEY`` is a second
credential the client picks up whenever it is not passed one, and holds for
the life of the client: not what a chat request is signed with, and still a
key the operator did not configure for this provider.

Removed once, at construction, and never per turn: a process-wide edit made
while turns are running would be one turn changing another's environment.

What is **not** here, because an argument does say it or because nothing on a
turn's path reads it: ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` (the key and
the endpoint are passed; ``OpenAIProvider`` looks at both only when it builds a
client, and it is handed one), ``OPENAI_WEBHOOK_SECRET`` (read into the client
and used only to verify a webhook, which the platform never receives), and the
Azure and Bedrock variables (``AZURE_OPENAI_*``, ``OPENAI_API_TYPE``,
``OPENAI_API_VERSION``, ``AWS_*``), which only the SDK's module-level client
and its Azure and Bedrock clients read, none of them ever built.
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
body at ``DEBUG``, but ``OPENAI_LOG`` still turns the pair on for the whole
process, and a pin is cheaper than a promise that holds until an upgrade.

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

Belt as well as braces, and per client rather than process-wide: a header the
caller passes wins over anything merged in from the environment, so the key
that is sent is the configured one whether or not the variable above was there
to be removed.
"""

OPENAI_KEY_HEADER = "Authorization"
"""The header OpenAI's key travels in, as ``Bearer <key>``, pinned by ``chat_model``.

The same belt as ``ANTHROPIC_KEY_HEADER``, and a little more than a belt: the
OpenAI SDK drops an ``Authorization`` line of ``OPENAI_CUSTOM_HEADERS`` only
when the caller passed an ``Authorization`` header of its own, so passing one is
what makes the configured key the only one that can be sent.
"""

MAX_RETRIES = 0
"""How many times a failed model call is retried by the client: not at all.

The client's own retries are invisible to everything above -- they would
lengthen a turn silently and could send the same prompt twice after a timeout
the provider had already accepted. A turn is bounded by the application
(``robinauts.application.Turns``), a failure is reported by raising, and
retrying is sending the message again (``docs/specs/runs.md``).

**One retry this does not switch off is the framework's own.** For the models
whose thinking blocks Anthropic binds to a conversation, Pydantic AI answers
the vendor's refusal of a replayed block ("bound to a different
conversation") by sending the request once more with the blocks marked to be
dropped, and says so through ``warnings``. That is the "ask the vendor to
drop what it can no longer match" of the plan's open question
(``docs/working-notes/mcp-plan.md``), happening on this engine and not on
the other, which fails that turn; left as the framework has it, and to be
seen in a live turn on such a model after the system prompt was edited.
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

Chat Completions requires none, and the model's own limit is then the ceiling.
On OpenAI's reasoning models the ceiling covers the reasoning tokens as well
as the answer, so a number picked to bound an answer can be spent entirely on
thinking and leave an empty one. An operator who wants a bound writes
``max_output_tokens``, and it is sent in the field the kind takes
(``_as_chat_completions``) -- which is what the other engine sends too
(``docs/specs/agents.md``).
"""


ChatModelFactory = Callable[[ModelConfig, ModelProviderConfig, str], Model]
"""How a model is built: the model, its provider, and the provider's key.

Injectable so that the tests run the engine over one of Pydantic AI's own test
models -- the graph, the streaming, the mapping and the releasing are then
exercised for real with no network and no key (``docs/layout.md``, "Testing
strategy").
"""


def force_tracing_off() -> None:
    """Turn Pydantic AI's hosted observability off, whatever else is running.

    **Nothing phones home** (``docs/specs/core.md``, ``docs/specs/agents.md``).
    Pydantic AI instruments a run only when instrumentation settings resolve to
    something, which happens when ``logfire.configure()`` plus
    ``logfire.instrument_pydantic_ai()`` has set the process-wide default
    (``Agent.instrument_all()``), when an agent's own ``instrument`` asks for
    it, or when the model handed in is already an ``InstrumentedModel``. Only
    then is an OpenTelemetry tracer provider asked for, which is the only thing
    that would open an exporter and send a conversation to a third party.

    Two of those three are answered where an agent is built: ``_runner`` sets
    ``instrument = False`` on every agent this engine makes, which **beats**
    the process-wide default, and the model comes from this adapter's own
    factory and is never an instrumented one. There is deliberately no
    environment variable to unset, because there is none to read
    (``TRACING_VARIABLES_REMOVED``).

    What is left, and what this function is for, is the **banner**: on its
    first run in a process Pydantic AI writes an advertisement for its hosted
    observability to standard error, which has no business in a server's log.
    ``BANNER_ENABLED`` is the package's own switch for it, and is set here,
    once, when the engine is built -- alongside the environment edit the
    LangGraph adapter makes for the same reason, and for the same kind of
    reason: it is the layer that may.
    """
    pydantic_ai.BANNER_ENABLED = False
    for name in TRACING_VARIABLES_REMOVED:  # pragma: no cover -- none, and said why
        os.environ.pop(name, None)


def clear_client_overrides() -> None:
    """Take out of the environment what no argument can override.

    The other half of "a vendor's client is built from the configuration and
    not from the environment" (``ANTHROPIC_ENDPOINT``, ``OPENAI_ENDPOINT``). An
    endpoint and a key are arguments, and an argument wins; **headers are
    merged**, and an organisation, a project and an admin key are read
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
    The Anthropic SDK reads ``ANTHROPIC_LOG`` at *import* -- before anything
    here exists, as the OpenAI SDK reads ``OPENAI_LOG``, whose loggers are
    pinned the same way -- and on ``debug`` puts its own logger and its HTTP client's
    at ``DEBUG`` and calls ``logging.basicConfig()``, which attaches a handler
    to the root logger if nothing else has. What the SDK's logger then writes
    for every call is "Request options: ..." with the request's ``json_data``
    in it: the agent's system prompt and every message of the conversation, on
    standard error.

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
    **protocols** (``PydanticAIAgent.kinds``). ``anthropic`` is Anthropic, at
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
    held to ``PydanticAIAgent.kinds`` at start-up. One that does -- a kind added
    to the platform's vocabulary without a branch here, a caller that built a
    definition by hand -- gets a refusal naming it, and never a turn sent to
    whichever endpoint happened to be nearest. Every kind there is today has a
    branch, so the tests reach this one with a kind of their own, which is what
    keeps it a branch that is exercised rather than merely written.

    Spelt out again here rather than imported from the LangGraph adapter: the
    two adapters do not import each other (``docs/layout.md``), and deleting
    either must leave the other whole.
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
            f"model_providers.{provider.id}: this build of the Pydantic AI engine cannot"
            f" reach a {provider.kind.value} provider"
        ]
    )


def chat_model(model: ModelConfig, provider: ModelProviderConfig, key: str) -> Model:
    """The model that model's configuration describes.

    The key is passed in and not read here: what may touch the environment is
    ``robinauts.adapters.config_file``, and what reaches this is one key for
    one call (``docs/specs/agents.md``).

    **The provider's kind chooses the model, and nothing else does**: the two
    Anthropic kinds get ``AnthropicModel``, described below, and the two
    OpenAI kinds ``OpenAIChatModel`` (``_openai_model``). A kind in neither set
    is refused, naming the provider.

    **The vendor's client is built here rather than by the provider**, because
    two of the things that must be pinned are the client's and not the
    provider's: ``max_retries`` (``MAX_RETRIES``) and the headers
    (``ANTHROPIC_KEY_HEADER``). ``AnthropicProvider(anthropic_client=...)``
    takes the finished client, so nothing is left for it to decide.

    **Everything the client would otherwise take from the environment is
    passed.** The key, so that ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_AUTH_TOKEN``
    are never consulted and the key a turn spends is the one the operator
    configured for that provider -- an explicit credential also switches the
    SDK's auto-discovery off outright, so a profile on disk, a federation
    token or an ``ANTHROPIC_PROFILE`` cannot supply one either; and the
    endpoint (``endpoint_of``), so that ``ANTHROPIC_BASE_URL`` cannot redirect
    a turn and a key to another host, nor a profile quietly fill one in -- the
    configuration is the only thing that decides where a turn goes. There is
    no vendor-specific proxy variable in this SDK to pin off: an
    operator's proxy is ``HTTPS_PROXY``, which is theirs and is how every other
    outbound call of this process is proxied.

    Headers are the one thing an argument cannot settle on its own, because
    the SDK **merges** ``ANTHROPIC_CUSTOM_HEADERS`` into what the caller
    passed: the key is therefore pinned as a header too
    (``ANTHROPIC_KEY_HEADER``), where a caller's value wins, and the variable
    itself is taken out of the environment when the engine is built
    (``clear_client_overrides``).

    The timeout and the output ceiling are **not** here: they travel with the
    request, as ``ModelSettings`` (``_settings``), which is where the model's
    configuration reaches a turn.
    """
    if provider.kind in OPENAI_KINDS:
        return _openai_model(model, provider, key)
    if provider.kind not in ANTHROPIC_KINDS:
        raise _unreachable(provider)
    client = AsyncAnthropic(
        api_key=key,
        base_url=endpoint_of(provider),
        max_retries=MAX_RETRIES,
        default_headers={ANTHROPIC_KEY_HEADER: key},
    )
    return AnthropicModel(
        model.name, provider=AnthropicProvider(anthropic_client=client), profile=_schemas_as_given
    )


def _openai_model(model: ModelConfig, provider: ModelProviderConfig, key: str) -> Model:
    """``OpenAIChatModel`` over a client this adapter built itself.

    **The protocol is Chat Completions, for both kinds**: the framework's
    ``OpenAIChatModel``, and never its ``OpenAIResponsesModel``. An
    ``openai-compatible`` endpoint -- a gateway, vLLM, OpenRouter -- speaks
    Chat Completions, one protocol for both kinds keeps the two kinds and the
    two engines symmetric, and the LangGraph engine sends the same request
    through ``ChatOpenAI`` (``docs/specs/agents.md``). The Responses API is a
    decision of its own.

    **The client is built here and handed to the provider**, for the reason
    Anthropic's is: what must be pinned is the client's. The key, so that
    ``OPENAI_API_KEY`` is never consulted -- ``OpenAIProvider`` looks at it,
    and at ``OPENAI_BASE_URL``, only when it builds a client of its own, which
    it is not asked to; the endpoint (``endpoint_of``); no retries
    (``MAX_RETRIES``); and the key pinned as its header
    (``OPENAI_KEY_HEADER``). What the SDK reads when an argument is ``None`` --
    the organisation, the project, the admin key, the headers -- is out of the
    environment by the time a turn runs (``clear_client_overrides``). There is
    no vendor-specific proxy variable in this SDK either.

    The timeout and the ceiling travel with the request (``_settings``), and
    the tool schemas are sent as the servers gave them, the ceiling in the
    field the kind takes and the answer's text as text
    (``_as_chat_completions``).
    """
    client = AsyncOpenAI(
        api_key=key,
        base_url=endpoint_of(provider),
        max_retries=MAX_RETRIES,
        default_headers={OPENAI_KEY_HEADER: f"Bearer {key}"},
    )
    compatible = provider.kind is ProviderKind.OPENAI_COMPATIBLE

    def profile(given: ModelProfile) -> ModelProfile:
        return _as_chat_completions(given, compatible=compatible)

    return _ChatCompletionsModel(
        model.name, provider=OpenAIProvider(openai_client=client), profile=profile
    )


@dataclass
class _ChatCompletionsStream(OpenAIStreamedResponse):
    """The framework's Chat Completions stream, holding a streamed call to what it is.

    Two things, both through the hook the framework documents for it
    (``_map_tool_call_delta``) and its public lookup of a part by the vendor's
    index (``get_part_by_vendor_id``):

    - **a name sent again is not more of the name.** The framework appends
      every ``name`` a call's deltas carry, so a server that repeats the id
      and the name on every delta -- some compatible ones do -- would have
      the call stored as ``github__searchgithub__search``. A name equal to
      the one the call already has is dropped before the framework sees it;
      a name that *differs* is left to arrive, and ``_Answer.complete``
      refuses the call as one whose name changed after it was announced;
    - **a call the stream began and never named is refused.** The framework
      holds such a call as a delta, announces nothing for it and leaves it
      out of the response it assembles, so the turn would be stored as an
      answer that asked for nothing. At the end of the stream every call a
      delta began is looked for among the whole calls, and one that is not
      there fails the turn, as it does under the other engine.

    **A delta may carry no ``index``**: OpenAI's always do, and a compatible
    server's need not, in which case the framework keeps the call under no
    vendor id at all -- a delta with a name begins a new call, and one
    without continues the latest. So a call is tracked by its index when it
    has one, and by its id when it has none: an index-less delta's call is
    "the latest call" for the name check, and is looked for among the whole
    calls by the id it gave.
    """

    _streamed_calls: set[int] = field(default_factory=set, init=False)
    """The indexes the stream's calls came under."""
    _unindexed_calls: set[str] = field(default_factory=set, init=False)
    """The ids of the calls whose deltas came with no index."""

    def _map_tool_call_delta(
        self, choice: chat_completion_chunk.Choice
    ) -> Iterable[ModelResponseStreamEvent]:
        for delta in choice.delta.tool_calls or []:
            index: int | None = delta.index
            if index is not None:
                self._streamed_calls.add(index)
                known: object = self._parts_manager.get_part_by_vendor_id(index)
            else:
                if delta.id:
                    self._unindexed_calls.add(delta.id)
                whole = self._parts_manager.get_parts()
                known = whole[-1] if whole else None
                if (
                    isinstance(known, FrameworkToolCallPart)
                    and delta.id
                    and delta.id != known.tool_call_id
                ):
                    known = None
            if delta.function is not None and delta.function.name:
                if isinstance(known, FrameworkToolCallPart) and (
                    known.tool_name == delta.function.name
                ):
                    delta.function.name = None
        yield from super()._map_tool_call_delta(choice)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for event in super()._get_event_iterator():
            yield event
        named = {
            part.tool_call_id
            for part in self._parts_manager.get_parts()
            if isinstance(part, FrameworkToolCallPart)
        }
        found = self._parts_manager.get_part_by_vendor_id
        unnamed = [
            index
            for index in self._streamed_calls
            if not isinstance(found(index), FrameworkToolCallPart)
        ]
        if unnamed or not self._unindexed_calls <= named:
            raise UnsupportedContentError("the model streamed a tool call and never gave its name")


class _ChatCompletionsModel(OpenAIChatModel):
    """The framework's Chat Completions model, streaming through ``_ChatCompletionsStream``.

    ``_streamed_response_cls`` is the framework's documented seam for a
    stream of one's own; nothing else about the model is changed.
    """

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        return _ChatCompletionsStream


def _as_chat_completions(profile: ModelProfile, *, compatible: bool) -> ModelProfile:
    """The framework's OpenAI profile, less what would make this engine's request its own.

    Three things, each to send what the other engine sends
    (``docs/specs/agents.md``):

    - the tool schemas as the servers gave them (``_schemas_as_given``);
    - **no thinking tags**: the framework otherwise lifts a streamed
      ``<think>`` delta, and what follows it up to ``</think>``, out of the
      answer's text and into thinking. ``langchain-openai`` does no such
      thing, and the platform stores what the vendor wrote as the answer, so
      the answer's text is text under both engines. (What the framework also
      reads, and the other does not, are the ``reasoning`` and
      ``reasoning_content`` fields some compatible servers stream beside the
      text; that is a field and not a guess about the text, it is shown as
      reasoning and never kept, and it is recorded as a difference in
      "Known findings".) With no tags there is nothing for a thinking part
      in the history to be written back as either, and none is: the thinking
      a turn streamed is not replayed (``_replayed``);
    - the ceiling in the field the kind takes: ``max_completion_tokens`` for
      OpenAI itself, ``max_tokens`` for an ``openai-compatible`` endpoint,
      which is the field OpenRouter and older servers know -- the
      framework's own OpenRouter profile makes the same choice.

    **Two constraints come with no tags**, because the framework unpacks the
    pair unguarded in two places this engine never goes: the non-streamed
    request path, which splits tags out of a whole answer -- the engine only
    ever streams (``PydanticAIAgent._turn``, and the tests assert that the
    request says ``"stream": true``); and writing a thinking part of the
    history back as text between tags -- the engine replays no thinking to a
    model reached over this protocol (``_replayed``, and its test). A change
    that took either path would fail on its first turn, in the tests, rather
    than in a deployment. Pydantic AI has no switch for "no tags" other than
    the profile's value, which is why it is ``None`` against the type.
    """
    written: dict[str, Any] = {**_schemas_as_given(profile), "thinking_tags": None}
    if compatible:
        written["openai_chat_supports_max_completion_tokens"] = False
    return cast(ModelProfile, written)


def _schemas_as_given(profile: ModelProfile) -> ModelProfile:
    """The framework's profile for the model, less its rewriting of a tool's schema.

    The framework's Anthropic profile runs every tool's input schema through
    a transformer that strips ``title`` and ``$schema`` at every depth, and
    its OpenAI profile through one that rewrites a schema towards OpenAI's
    strict mode and, where the result qualifies, marks the tool ``strict``.
    The platform promises the opposite: a tool's schema is the server's, handed
    to the vendor as it is, and two engines send byte-identical lists
    (``docs/specs/agents.md``, "Tools"; ``domain.ToolDefinition``) -- the
    other engine passes it through untouched, and this one now does too, over
    either protocol. With no transformer a tool's ``strict`` stays unset, so
    none is sent. Everything else the profile says about the model is kept.
    """
    return {**profile, "json_schema_transformer": None}


class PydanticAIAgent(Agent):
    """The Pydantic AI engine: one turn, one agent, built and thrown away."""

    kinds: frozenset[ProviderKind] = ANTHROPIC_KINDS | OPENAI_KINDS
    """The provider kinds this build of the engine has a client for: all four.

    Two clients, two kinds each. ``AsyncAnthropic`` reaches Anthropic itself
    and any endpoint that speaks Anthropic's Messages API at an address the
    operator gives; ``AsyncOpenAI`` reaches OpenAI itself and any endpoint that
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
        model_for: ChatModelFactory = chat_model,
    ) -> None:
        force_tracing_off()
        clear_client_overrides()
        quiet_client_logging()
        self._models = models
        """The models a turn may run on, and the provider each is reached through."""
        self._keys = keys
        """The providers' keys, as start-up read them. It prints nothing."""
        self._model_for = model_for
        self._open = 0

    @property
    def held(self) -> int:
        """How many turns of this engine still hold a stream open.

        Zero once every turn has ended or been closed, which is the promise a
        cancelled run depends on (``robinauts.ports.agents``). It counts the
        engine's own turns; the framework's run and the model's stream live
        inside one and go with it.

        **One engine, one count, however many turns it is running.** A
        deployment shares one ``PydanticAIAgent`` between every conversation,
        so this says "is anything still open", not "is *that* turn still
        open": a caller watching one turn while another is in flight reads the
        other one's stream in this number. Nothing in the platform needs the
        finer answer -- the application releases a turn by closing its own
        stream and never asks -- and the contract suite reads it between turns,
        one at a time, which is when the two questions have the same answer.
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
        model, building the framework's agent, opening the stream -- happens
        inside the generator, so that a provider that refuses a key is a
        failure of the turn, reported by raising where the caller is iterating,
        and not an exception thrown at whoever asked for the stream.
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
        client = self._model_for(model, provider, self._keys.key_for(provider.id))
        runner = _runner(agent, client, tools, history)
        settings = _settings(model, provider)
        # The vendor's own name for itself is the key its blocks are kept
        # under (``VENDOR`` for the two Anthropic kinds, ``openai`` for the
        # other two, which have no signed blocks to keep).
        # Over Chat Completions nothing is replayed, whatever is stored
        # (``_replayed``): ``None`` says so.
        replay_as = None if isinstance(client, OpenAIChatModel) else client.system
        messages = _messages(history, model_id, replay_as)
        answer = _Answer()
        response: ModelResponse | None = None
        self._open += 1
        try:
            async with runner.iter(message_history=messages, model_settings=settings) as run:
                async for node in run:
                    # The user prompt node and everything after the model's
                    # answer are passed over: the first is a question already
                    # in the history, and the rest is where the framework would
                    # run a tool or retry (see the module docstring).
                    if not FrameworkAgent.is_model_request_node(node):
                        continue
                    async with node.stream(run.ctx) as answering:
                        async for event in answering:
                            for made in answer.events_of(event):
                                yield made
                        response = answering.response
                    break
            # ``response`` is ``None`` only if no model request node was
            # reached, which the framework does not do with a history the
            # port guarantees -- one ending in the message being answered
            # (``robinauts.ports.agents``) -- and does not do with an
            # off-contract one either in this version, which asks the model
            # once regardless. Kept for a framework that would leave the loop
            # above with no iterations: that is a turn that produced an empty
            # answer rather than one that produced none, which would be a
            # failed run and is the application's to decide.
            for made in answer.complete(response, client.system):
                yield made
        finally:
            # Reached when the turn ends, when it raises, and when the
            # iteration is closed -- which is what a cancellation does. The two
            # context managers above are left on the way out, which is what
            # releases the model's stream and the response under it.
            self._open -= 1


def _runner(
    agent: AgentDefinition,
    model: Model,
    tools: tuple[ToolDefinition, ...],
    history: Sequence[Message],
) -> FrameworkAgent[None, str]:
    """The framework's agent for one turn: this model, this prompt, these tools.

    **The system prompt is ``instructions``** and not ``system_prompt``: it is
    the agent's, it is taken from the definition as it stands now, and it is
    not one of the messages (``docs/specs/conversations.md``). An empty one is
    left out rather than sent as an empty instruction.

    **The tools are declared, not given.** The run's definitions go in as an
    ``ExternalToolset``, the framework's own kind for tools it does not
    execute: the model is shown them as any tool, the framework holds no
    function to call, and the turn ends before it would try
    (``PydanticAIAgent._turn``). The port is a seam a test double crosses too,
    so what comes over it is checked to be the platform's definition and
    nothing that looks like one. **No output type**: the default output of a
    Pydantic AI agent is plain text, which declares no output tool.

    ``instrument`` is set to ``False`` rather than left alone, which is the
    strongest of the three switches: it beats ``Agent.instrument_all()``, so
    a process where something called ``logfire.instrument_pydantic_ai()``
    still traces nothing of ours (``force_tracing_off``).

    It is given a ``name``, which is also what keeps the framework from
    looking for one in the caller's stack frame at every turn.
    """
    for tool in tools:
        if not isinstance(tool, ToolDefinition):
            raise InvalidValueError(f"a run's tools are ToolDefinitions, not {tool!r}")
    # With a stub for every name the history calls that the run lacks: the
    # vendor refuses tool blocks its request defines no tool for
    # (``core.tools_for_request``).
    defined = tools_for_request(tools, history)
    runner: FrameworkAgent[None, str] = FrameworkAgent(
        model=model,
        name=agent.id,
        instructions=agent.system_prompt or None,
        toolsets=[ExternalToolset([_declared(tool) for tool in defined])] if defined else None,
    )
    runner.instrument = False
    return runner


def _declared(tool: ToolDefinition) -> FrameworkToolDefinition:
    """The definition as the framework takes one, as it stands.

    The name is the full name the platform gave it, the schema is the
    server's, as plain data, and an empty description is none -- MCP's is
    optional, and the vendors take a tool without one.
    """
    return FrameworkToolDefinition(
        name=tool.name,
        parameters_json_schema=dict(tool.input_schema),
        description=tool.description or None,
    )


def _settings(model: ModelConfig, provider: ModelProviderConfig) -> ModelSettings:
    """What the model's configuration says about one request.

    The timeout is per model call and is the configuration's
    (``domain.ModelConfig``). The ceiling is required by Anthropic on every
    request, so over Anthropic's protocol the engine sends one whether or not
    the operator set it (``DEFAULT_ANTHROPIC_OUTPUT_TOKENS``); over OpenAI's it
    is sent only when the operator set one (``DEFAULT_OPENAI_OUTPUT_TOKENS``),
    in the field the kind takes (``_as_chat_completions``). Both travel with
    the request rather than with the client, so that two models of one
    provider can differ.
    """
    default = (
        DEFAULT_OPENAI_OUTPUT_TOKENS
        if provider.kind in OPENAI_KINDS
        else DEFAULT_ANTHROPIC_OUTPUT_TOKENS
    )
    ceiling = model.max_output_tokens or default
    if ceiling is None:
        return ModelSettings(timeout=model.timeout_seconds)
    return ModelSettings(timeout=model.timeout_seconds, max_tokens=ceiling)


VENDOR = "anthropic"
"""The key the vendor's opaque blocks are kept under in a message's ``extras``.

One vendor's key for one vendor's blocks (``docs/specs/conversations.md``,
"Reasoning"), and the framework's own name for the vendor (``Model.system``),
which is what the engine reads at run time: the two Anthropic kinds speak the
Messages API through one client, so what it stores and what it replays for
them are Anthropic's thinking blocks under this key -- the same key and the
same blocks as the other engine's, which is what lets a conversation cross the
swap with its thinking (``docs/specs/conversations.md``, "What crosses a
swap"). The two OpenAI kinds read ``openai`` off their model instead, and keep
nothing under it: Chat Completions returns no signed reasoning, and the
reasoning some compatible endpoints stream is unsigned, which is shown and not
kept (``_extras``).
"""

REDACTED = "redacted_thinking"
"""The block the vendor sends in place of thinking it will not show.

Opaque throughout: the framework carries it as a ``ThinkingPart`` with this
``id``, no content and the block's data for a signature, and that is how it
is stored (``{"type": "redacted_thinking", "data": ...}``) and replayed.
"""

BLOCKS_LEFT_OUT = (
    "the vendor's signed thinking blocks did not fit a message's extras and were left"
    " out; if the model asked for tools, the vendor may refuse the next round"
)
"""What the log says when an answer's blocks are bigger than ``extras`` may be.

The plan's open question, answered here as the other engine answers it
(``docs/working-notes/mcp-plan.md``, "Open"): the answer is stored without
them rather than the turn failing over the size of the thinking, and the
line in the log is what an operator finds when the vendor then refuses.
"""

BLOCK_NOT_REPLAYED = (
    "a stored thinking block of the vendor's is not of a shape this engine replays, and"
    " was left out"
)
"""What the log says of a block in ``extras`` that is not one the vendor made."""


def _messages(history: Sequence[Message], model_id: str, vendor: str | None) -> list[ModelMessage]:
    """The history as the framework's messages. The system prompt is not here.

    It is the agent's, it is passed as ``instructions`` (``_runner``), and it
    is deliberately not a message: what goes in ``message_history`` is the
    conversation and nothing else, so the history the model is given is the
    history the platform stored (``docs/specs/conversations.md``).

    **Text, calls and results.** A question is one ``UserPromptPart``. An
    answer is its stored text, its calls as the framework's ``ToolCallPart``s
    -- the vendor's id and the tool's full name, the arguments as data -- and,
    in front of them, the vendor's signed thinking blocks kept in its
    ``extras``, **only when the run's model is the one that made them**
    (``_replayed``): a signature is the model's own, and a block sent to
    another model is refused by the vendor or, worse, turned into text by the
    framework. A tool message is one ``ToolReturnPart`` per result, naming
    the call it answers by the id and by the name the answer before it gave
    that id, and whether it went wrong -- which the framework folds into the
    one ``user`` turn of ``tool_result`` blocks the vendor wants back. An
    answer whose calls no tool message answers is followed by one error
    result per call saying no result of it was recorded (``domain.NO_RESULT``). The
    reasoning a previous turn streamed is not carried back: what is stored of
    it is the platform's record, not the vendor's.

    A message with no text at all still becomes a message, empty, because
    dropping it *here* would be this engine deciding what a turn that said
    nothing means. What becomes of it is the **vendor mapping's**, and it is
    the same answer under both engines: Anthropic's API refuses an empty
    content block, so Pydantic AI leaves an empty text part out and then
    leaves out the assistant message it emptied -- unless it has calls, which
    stand on their own -- and langchain-anthropic drops an assistant message
    whose content came out empty. The model is shown the same history either
    way, which is what the swap needs. (Their rules differ in one case --
    langchain keeps a *trailing* empty assistant message and Pydantic AI does
    not -- and that case cannot arise here: a history ends in the message
    being answered, ``robinauts.ports.agents``.) Over OpenAI's protocol, which
    takes an empty assistant message, the whole history goes out as the same
    bytes under both engines: a result that went wrong, for which Chat
    Completions has no field, is written ``{"error": <text>}`` by the framework
    here, and the LangGraph adapter writes it the same way, as it moves every
    other spelling of langchain-openai's to this framework's
    (``docs/specs/agents.md``, "Known findings"; the test that compares the two
    is in ``tests/unit/test_engine_swap.py``).

    So the line this adapter draws is that the framework decides, from the
    same input, rather than this adapter deciding for it; if a mapping ever
    changed, it would change under both engines together or be a finding about
    one of them.

    The last of them is the message being answered -- the question, or the
    tool message of a round -- which is what the port guarantees, so the turn
    is run with no separate prompt and the whole of the history in one place.
    """
    messages: list[ModelMessage] = []
    named: dict[str, str] = {}
    """The calls of the answer before, by id: what a result is named after."""
    unanswered = unanswered_calls(history)
    for message in history:
        if message.role is Role.USER:
            messages.append(ModelRequest(parts=[UserPromptPart(content=message.text)]))
        elif message.role is Role.ASSISTANT:
            messages.append(_response(message, model_id, vendor))
            named = {call.call_id: call.name for call in message.tool_calls}
            if message.id in unanswered:
                # A call no tool message answers is answered here with what
                # the record says of it (``domain.NO_RESULT``): the vendor
                # refuses a call with nothing answering it, and the record,
                # which keeps the call without a result, is not what is edited.
                messages.append(
                    ModelRequest(
                        parts=[
                            _returned(ToolResultPart(call.call_id, NO_RESULT, is_error=True), named)
                            for call in unanswered[message.id]
                        ]
                    )
                )
        else:
            # A tool message holds results and nothing else (``domain.Message``);
            # one that holds anything else is a fault of ours, refused here
            # rather than sent as an empty turn.
            if len(message.tool_results) != len(message.parts):
                raise InvalidValueError("a tool message holds tool results only")
            messages.append(
                ModelRequest(parts=[_returned(part, named) for part in message.tool_results])
            )
    return messages


def _response(message: Message, model_id: str, vendor: str | None) -> ModelResponse:
    """A stored answer as the framework's response: blocks, text, calls, in that order."""
    parts: list[ModelResponsePart] = list(_replayed(message, model_id, vendor))
    parts.append(FrameworkTextPart(content=message.text))
    parts.extend(
        FrameworkToolCallPart(
            tool_name=call.name, args=dict(call.arguments), tool_call_id=call.call_id
        )
        for call in message.tool_calls
    )
    return ModelResponse(parts=parts)


def _returned(part: ToolResultPart, named: Mapping[str, str]) -> ToolReturnPart:
    """One result as the framework's return part, named after the call it answers.

    The tree already holds a tool message to its parent's calls
    (``core.check_answers_calls``), so a result naming no call of the answer
    before it is a fault of ours and not the model's: ``InvalidValueError``.
    """
    name = named.get(part.call_id)
    if name is None:
        raise InvalidValueError(
            f"a tool result answers a call of the answer before it, not {part.call_id!r}"
        )
    return ToolReturnPart(
        tool_name=name,
        content=part.text,
        tool_call_id=part.call_id,
        outcome="failed" if part.is_error else "success",
    )


def _replayed(message: Message, model_id: str, vendor: str | None) -> list[ThinkingPart]:
    """The vendor's blocks stored on that answer, as the framework replays them.

    Only for the model that made them, and only the two shapes the vendor
    makes (``_extras``): a ``thinking`` block with its signature, or a
    ``redacted_thinking`` block whose data rides as the signature. The
    framework sends a signed part back as the vendor's block and would send
    an unsigned one as *text* between thinking tags, so a block of any other
    shape is left out with a line in the log rather than handed over.

    **Nothing is ever replayed to a model reached over Chat Completions**
    (``vendor`` is ``None`` for one, ``_messages``). There is nothing to
    replay -- Chat Completions returns no signed reasoning -- and there must
    never be anything: the OpenAI models this engine builds have no thinking
    tags (``_as_chat_completions``), and the framework writes a replayed
    thinking part to Chat Completions *between* those tags, unpacking them
    unguarded. So whatever ``extras`` an answer holds, nothing is handed over
    for it, and a test stores such a block and runs the turn.
    """
    if vendor is None:
        return []
    if message.provenance is None or message.provenance.model != model_id:
        return []
    kept = message.extras.get(vendor)
    blocks = kept.get("thinking") if isinstance(kept, Mapping) else None
    if not isinstance(blocks, list):
        return []
    parts: list[ThinkingPart] = []
    for block in blocks:
        part = _thinking_part(block, vendor)
        if part is None:
            _log.warning(BLOCK_NOT_REPLAYED)
        else:
            parts.append(part)
    return parts


def _thinking_part(block: object, vendor: str) -> ThinkingPart | None:
    """One stored block as the framework's part, or ``None`` for a shape it is not."""
    if not isinstance(block, Mapping):
        return None
    if block.get("type") == REDACTED and isinstance(block.get("data"), str) and block["data"]:
        return ThinkingPart(content="", id=REDACTED, signature=block["data"], provider_name=vendor)
    if (
        block.get("type") == "thinking"
        and isinstance(block.get("thinking"), str)
        and isinstance(block.get("signature"), str)
        and block["signature"]
    ):
        return ThinkingPart(
            content=block["thinking"], signature=block["signature"], provider_name=vendor
        )
    return None


class _Answer:
    """One answer as it streams: what was said, what was called, and the events.

    The order the port asks for (``robinauts.core.check_engine_events``) is
    kept here: the answer is announced on its first piece, a call is announced
    when its part starts and completed when that part ends, when the next
    begins or when the answer ends, and what the call completes with is what
    its pieces parse to.
    """

    def __init__(self) -> None:
        self.started = False
        self.streamed: list[str] = []
        self.calls: list[ToolCallPart] = []
        self._open: tuple[str, str] | None = None
        """The call being made -- its id and name -- while its arguments arrive."""
        self._pieces: list[str] = []
        """The pieces of JSON streamed for the open call."""

    def events_of(self, event: ModelResponseStreamEvent) -> list[EngineEvent]:
        """The events one stream event carries, in the order they came.

        A part **starting** arrives with the first of its content already in
        it, a **delta** is more of one, and a part **ending** carries the whole
        part: for a call, the moment it is complete; for anything else, a
        repeat, passed over. A part of a kind this version does not carry --
        a file, a citation -- is passed over too, because an engine that
        guessed at one would be inventing content; but **a call of a kind this
        engine did not declare** -- the vendor's own server-side tools, which
        the framework spells as another class -- is refused, because passing
        it over would finish the turn with an answer that looks whole and is
        half of one.
        """
        events: list[EngineEvent] = []
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, FrameworkToolCallPart):
                events.extend(self._closed())
                self._open, self._pieces = (part.tool_call_id, part.tool_name), []
                events.append(ToolCallStarted(call_id=part.tool_call_id, name=part.tool_name))
                events.extend(self._argued(part.args, part.tool_call_id))
            elif isinstance(part, BaseToolCallPart):
                raise UnsupportedContentError(
                    "the model asked for a tool of a kind this engine did not declare"
                )
            elif isinstance(part, FrameworkTextPart):
                events.extend(self._said(part.content))
            elif isinstance(part, ThinkingPart):
                events.extend(self._thought(part.content))
        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, ToolCallPartDelta):
                events.extend(self._argued(delta.args_delta, delta.tool_call_id))
            elif isinstance(delta, TextPartDelta):
                events.extend(self._said(delta.content_delta))
            elif isinstance(delta, ThinkingPartDelta):
                events.extend(self._thought(delta.content_delta or ""))
        elif isinstance(event, PartEndEvent):
            part = event.part
            if isinstance(part, FrameworkToolCallPart) and self._open is not None:
                if part.tool_call_id == self._open[0]:
                    events.extend(self._closed())
        if events and not self.started:
            self.started = True
            events.insert(0, AnswerStarted())
        return events

    def complete(self, response: ModelResponse | None, vendor: str) -> list[EngineEvent]:
        """The events that end the answer, given the response the framework assembled.

        ``response`` is read for the vendor's signed blocks and for the calls
        it holds, which have to be the calls that were announced, under the
        names they were announced with. One it holds that never started -- a
        part the framework made in some way this engine did not see -- is
        refused rather than dropped, because an answer missing a call would be
        half an answer that looks whole; so is one whose name grew after it
        was announced (a name streamed in pieces, which the framework announces
        at its first piece). A call the stream began and never named never
        reaches here: the framework would leave it out of the response
        altogether, and the OpenAI stream this engine builds refuses it first
        (``_ChatCompletionsStream``).
        """
        events: list[EngineEvent] = list(self._closed())
        if not self.started:
            # Nothing was streamed: the answer is announced and completed in
            # one breath -- an engine is never required to stream.
            events.append(AnswerStarted())
            self.started = True
        announced = {call.call_id: call.name for call in self.calls}
        held = [] if response is None else response.parts
        called = [part for part in held if isinstance(part, FrameworkToolCallPart)]
        if any(part.tool_call_id not in announced for part in called):
            raise UnsupportedContentError(
                "the model asked for a tool in a form this engine did not translate"
            )
        if any(part.tool_name != announced[part.tool_call_id] for part in called):
            raise UnsupportedContentError(
                "the model's name for a tool call changed after the call was announced"
            )
        parts: list[MessagePart] = []
        text = clean_text("".join(self.streamed))
        if text or not self.calls:
            parts.extend(text_parts(text))
        parts.extend(self.calls)
        events.append(AnswerCompleted(parts=tuple(parts), extras=_extras(held, vendor)))
        if self.calls:
            events.append(WaitingOnTools())
        return events

    def _said(self, text: str) -> list[EngineEvent]:
        if not text:
            return []
        self.streamed.append(text)
        return [AnswerTextDelta(text=text)]

    def _thought(self, text: str) -> list[EngineEvent]:
        return [AnswerReasoningDelta(text=text)] if text else []

    def _argued(self, arguments: object, call_id: str | None) -> list[EngineEvent]:
        """A piece of the open call's arguments, streamed as JSON.

        The framework hands a piece over as text -- the JSON as the model
        writes it -- or, from a model that does not stream its arguments, as
        the whole object at once; the second is written out as one piece, so
        that what was published parses to what the call completes with either
        way (``core.check_engine_events``), and a model that did both in one
        call has written something that is not JSON and fails the turn.
        """
        if arguments is None or arguments == "":
            return []
        if self._open is None or (call_id is not None and call_id != self._open[0]):
            raise UnsupportedContentError(
                "the model streamed arguments for no tool call this engine announced"
            )
        piece = json.dumps(arguments) if isinstance(arguments, Mapping) else str(arguments)
        self._pieces.append(piece)
        return [ToolCallArgumentsDelta(call_id=self._open[0], text=piece)]

    def _closed(self) -> list[EngineEvent]:
        """Complete the call being made, if there is one, with what it streamed."""
        if self._open is None:
            return []
        call_id, name = self._open
        joined = clean_text("".join(self._pieces))
        arguments = _parsed(joined) if joined.strip() else {}
        call = ToolCallPart(call_id=call_id, name=name, arguments=arguments)
        self.calls.append(call)
        self._open, self._pieces = None, []
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


def _extras(parts: Sequence[ModelResponsePart], vendor: str) -> dict[str, Any]:
    """The vendor's signed blocks off the response, keyed by vendor.

    The framework's signed ``ThinkingPart``s, put back into the vendor's own
    two shapes so that both engines store the same thing; a part with no
    signature -- an interrupted stream, a model that signs nothing -- is not
    a block the vendor would take back and is left out, its text having been
    streamed as reasoning already. Bounded as every ``extras`` is: blocks
    that do not fit are left out with a line in the log (``BLOCKS_LEFT_OUT``)
    rather than failing the turn over the size of the thinking.
    """
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, ThinkingPart) or part.provider_name != vendor or not part.signature:
            continue
        if part.id == REDACTED:
            blocks.append({"type": REDACTED, "data": part.signature})
        else:
            blocks.append(
                {"type": "thinking", "thinking": part.content, "signature": part.signature}
            )
    if not blocks:
        return {}
    extras = {vendor: {"thinking": blocks}}
    try:
        return checked_data(extras, "an answer's extras")
    except InvalidValueError as too_big:
        _log.warning("%s: %s", BLOCKS_LEFT_OUT, chain(too_big))
        return {}
