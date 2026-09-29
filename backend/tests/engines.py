# SPDX-License-Identifier: Apache-2.0
# Copyright The Robinauts Authors

"""Both real engines, each over a model a test writes the answer for.

What the two swap tests share (``tests/unit/test_engine_swap.py`` and the
configuration swap in ``tests/integration/test_create_app.py``), and what the
tests that hold both engines to one behaviour over OpenAI's Chat Completions
build them with (``tests/unit/test_engines_over_chat_completions.py``,
``Adapter``). Each engine's
own module scripts its own framework's model in its own terms; here they are
scripted **together**, and each records the same normalised view of what it was
told -- ``(role, text)`` pairs and the system prompt beside them -- so that a
test can ask both engines the same question about the history they received.

Nothing here reaches a provider: what is replaced is the one seam each adapter
has for its vendor (``chat_model_for``, ``model_for``), and the adapters, the
application and the store are the deployment's own.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from pydantic_ai import ModelRequest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import AgentInfo, DeltaThinkingPart, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel

import chat_completions
import robinauts.adapters.agents.langgraph as langgraph_adapter
import robinauts.adapters.agents.pydantic_ai as pydantic_ai_adapter
from conversations import AGENT, MODEL, OTHER_MODEL, agent_definition, question
from robinauts.adapters import ProviderKeys
from robinauts.adapters.agents.langgraph import LangGraphAgent
from robinauts.adapters.agents.pydantic_ai import PydanticAIAgent
from robinauts.domain import (
    Engine,
    EngineEvent,
    Message,
    ModelConfig,
    ModelProviderConfig,
    ModelsConfig,
    ProviderKind,
    Role,
    ToolDefinition,
)
from robinauts.ports import Agent

PROVIDER = "anthropic"
KEY_VARIABLE = "ROBINAUTS_ANTHROPIC_KEY"
KEY = "not-a-real-key"
"""What the engines are handed. Nothing here spends it, or could."""

MODELS = ModelsConfig(
    providers={
        PROVIDER: ModelProviderConfig(
            id=PROVIDER, kind=ProviderKind.ANTHROPIC, api_key_env=KEY_VARIABLE
        )
    },
    models={
        MODEL: ModelConfig(id=MODEL, provider=PROVIDER, name="claude-sonnet-5"),
        OTHER_MODEL: ModelConfig(id=OTHER_MODEL, provider=PROVIDER, name="claude-opus-5"),
    },
    agents={AGENT: agent_definition()},
)
"""One provider and two models, which both engines reach by the same ids.

An agent's engine can be swapped only if its model exists under both
(``docs/specs/agents.md``); this is that, as a deployment writes it. The
second is what a conversation moved off the agent's default runs on.
"""

Heard = tuple[tuple[str, str], ...]
"""One call's history as (role, text) pairs: what a model was really told."""

_ROLE_OF = {"human": Role.USER.value, "ai": Role.ASSISTANT.value}
"""LangChain's names for the two roles, as the platform spells them."""


class ScriptedChat(BaseChatModel):
    """The LangGraph half: a chat model that streams what the test wrote.

    A real ``BaseChatModel`` with a real ``_astream``, so the engine's graph,
    its streaming and its releasing are exercised as they would be against a
    provider.
    """

    said: str = ""
    heard: list[Heard] = []
    """Every call's history, normalised: what this engine handed the model."""
    instructions: list[str] = []
    """Every call's system prompt, which is not one of the messages."""

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover -- never asked
        raise NotImplementedError("this model only streams")

    async def _astream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.instructions.append(
            "".join(str(message.content) for message in messages if message.type == "system")
        )
        self.heard.append(
            tuple(
                (_ROLE_OF[message.type], str(message.content))
                for message in messages
                if message.type != "system"
            )
        )
        yield ChatGenerationChunk(message=AIMessageChunk(content=self.said))


class ScriptedStream:
    """The Pydantic AI half: a model that streams what the test wrote.

    A real ``FunctionModel``, for the same reason, recording the same
    normalised view of the history so that the two engines can be asked one
    question about it.
    """

    def __init__(self, said: str, *, thinking: str = "") -> None:
        self.said = said
        self.thinking = thinking
        self.heard: list[Heard] = []
        self.instructions: list[str] = []
        self.model = FunctionModel(stream_function=self._stream, model_name="scripted")

    async def _stream(self, messages: list[Any], info: AgentInfo) -> AsyncIterator[Any]:
        self.instructions.append(info.instructions or "")
        self.heard.append(
            tuple(
                (
                    Role.USER.value if isinstance(message, ModelRequest) else Role.ASSISTANT.value,
                    "".join(getattr(part, "content", "") for part in message.parts),
                )
                for message in messages
            )
        )
        if self.thinking:
            yield {0: DeltaThinkingPart(content=self.thinking)}
        yield self.said


@dataclass(frozen=True, slots=True)
class Scripts:
    """The two models one conversation is answered by, one per engine.

    Held together so that a test reads what **both** were told without knowing
    which engine ran which turn.
    """

    langgraph: ScriptedChat
    pydantic_ai: ScriptedStream
    built: list[tuple[Engine, str]] = field(default_factory=list)
    """Each turn's engine and the model it built its client for, in order."""


def scripts(said: str, *, thinking: str = "") -> Scripts:
    """Both models, saying the same words, one of them thinking first.

    The same answer under either engine on purpose: a test that compared what
    was stored would otherwise be told the two apart by their text rather than
    by the one field that is allowed to differ.
    """
    return Scripts(
        langgraph=ScriptedChat(said=said),
        pydantic_ai=ScriptedStream(said, thinking=thinking),
    )


def both_engines(said: Scripts) -> dict[Engine, Agent]:
    """Both real engines, wired as a deployment wires them, over those models."""
    keys = ProviderKeys({PROVIDER: KEY})

    def chat_for(model: ModelConfig, *_: object) -> ScriptedChat:
        said.built.append((Engine.LANGGRAPH, model.id))
        return said.langgraph

    def model_for(model: ModelConfig, *_: object) -> Any:
        said.built.append((Engine.PYDANTIC_AI, model.id))
        return said.pydantic_ai.model

    return {
        Engine.LANGGRAPH: LangGraphAgent(MODELS, keys, chat_model_for=chat_for),
        Engine.PYDANTIC_AI: PydanticAIAgent(MODELS, keys, model_for=model_for),
    }


# --- both engines over OpenAI's Chat Completions ------------------------------


def _langgraph_client(built: Any) -> openai.AsyncOpenAI:
    """The OpenAI client a turn of the LangGraph engine sends through."""
    assert isinstance(built, ChatOpenAI)
    client = built.root_async_client
    assert isinstance(client, openai.AsyncOpenAI)
    return client


def _pydantic_ai_client(built: Any) -> openai.AsyncOpenAI:
    """The OpenAI client a turn of the Pydantic AI engine sends through."""
    assert isinstance(built, OpenAIChatModel)
    client = built.client
    assert isinstance(client, openai.AsyncOpenAI)
    return client


def _raised_as_it_is(caught: BaseException) -> BaseException:
    """LangGraph lets the SDK's own exception out of the turn, unwrapped."""
    return caught


def _raised_in_model_http_error(caught: BaseException) -> BaseException:
    """Pydantic AI wraps the SDK's exception in ``ModelHTTPError``, as its cause."""
    assert isinstance(caught, ModelHTTPError)
    assert caught.__cause__ is not None
    return caught.__cause__


@dataclass(frozen=True, slots=True)
class Adapter:
    """One engine as the tests that hold both to one behaviour need it.

    Everything that differs between the two for such a test, named once: the
    agent class and the name of its seam for a vendor's client, the adapter's
    own functions and sets, where its built model keeps the OpenAI client, how
    a vendor's refusal reaches the caller, and the engine's name as its
    refusals spell it.
    """

    engine: Engine
    title: str
    """The engine's name as its own refusals spell it ("the LangGraph engine")."""
    agent: type[Agent]
    seam: str
    chat_model: Callable[[ModelConfig, ModelProviderConfig, str], Any]
    endpoint_of: Callable[[Any], str]
    anthropic_kinds: frozenset[ProviderKind]
    openai_kinds: frozenset[ProviderKind]
    client_variables_removed: tuple[str, ...]
    openai_client: Callable[[Any], openai.AsyncOpenAI]
    vendor_error: Callable[[BaseException], BaseException]
    """The SDK's own exception, from what the turn raised."""

    def build(self, models: ModelsConfig, keys: ProviderKeys, factory: Any = None) -> Agent:
        """The engine, over its real factory for a vendor's client or over ``factory``."""
        seam = {} if factory is None else {self.seam: factory}
        return self.agent(models, keys, **seam)  # type: ignore[call-arg]


ADAPTERS: dict[Engine, Adapter] = {
    Engine.LANGGRAPH: Adapter(
        engine=Engine.LANGGRAPH,
        title="LangGraph",
        agent=LangGraphAgent,
        seam="chat_model_for",
        chat_model=langgraph_adapter.chat_model,
        endpoint_of=langgraph_adapter.endpoint_of,
        anthropic_kinds=langgraph_adapter.ANTHROPIC_KINDS,
        openai_kinds=langgraph_adapter.OPENAI_KINDS,
        client_variables_removed=langgraph_adapter.CLIENT_VARIABLES_REMOVED,
        openai_client=_langgraph_client,
        vendor_error=_raised_as_it_is,
    ),
    Engine.PYDANTIC_AI: Adapter(
        engine=Engine.PYDANTIC_AI,
        title="Pydantic AI",
        agent=PydanticAIAgent,
        seam="model_for",
        chat_model=pydantic_ai_adapter.chat_model,
        endpoint_of=pydantic_ai_adapter.endpoint_of,
        anthropic_kinds=pydantic_ai_adapter.ANTHROPIC_KINDS,
        openai_kinds=pydantic_ai_adapter.OPENAI_KINDS,
        client_variables_removed=pydantic_ai_adapter.CLIENT_VARIABLES_REMOVED,
        openai_client=_pydantic_ai_client,
        vendor_error=_raised_in_model_http_error,
    ),
}
"""Both engines, by the engine the configuration names."""


def over_chat_completions(
    engine: Engine,
    vendor: chat_completions.Vendor,
    provider: ModelProviderConfig = chat_completions.OPENAI_PROVIDER,
    **changes: Any,
) -> Agent:
    """The engine, building its real client, with that client's transport the vendor's.

    What is replaced is the transport under the vendor's own SDK and nothing
    above it: the client is the one ``chat_model`` builds, and the framework,
    the adapter and the mapping are the deployment's.
    """
    adapter = ADAPTERS[engine]

    def plugged(model: ModelConfig, provider_: ModelProviderConfig, key: str) -> Any:
        built = adapter.chat_model(model, provider_, key)
        vendor.plugged_into(adapter.openai_client(built))
        return built

    return adapter.build(
        chat_completions.openai_models(engine, provider, **changes),
        ProviderKeys({provider.id: chat_completions.KEY}),
        plugged,
    )


async def chat_completions_turn(
    agent: Agent,
    engine: Engine,
    history: Sequence[Message] | None = None,
    tools: Sequence[ToolDefinition] = (),
) -> list[EngineEvent]:
    """Every event of one turn of that engine on the OpenAI model, run to its end."""
    seen: list[EngineEvent] = []
    asked = history if history is not None else (question("What is a robinaut?"),)
    async for event in agent.run_turn(
        chat_completions.gpt_agent(engine), asked, tools, model=chat_completions.GPT
    ):
        seen.append(event)
    return seen
