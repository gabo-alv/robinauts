# SPDX-License-Identifier: Apache-2.0
# Copyright The Robinauts Authors

"""One real turn against a real provider. Run by hand, never by CI.

Everything else about this engine is proved without a network
(``tests/unit/test_langgraph_engine.py``); what only this can show is that the
client is built the way the vendor expects and that a real stream maps onto
the platform's events. It costs money and needs a key, so:

- it is **not collected** by a plain ``pytest`` run, because ``tests/live`` is
  in ``norecursedirs`` (``backend/pyproject.toml``). CI runs
  ``scripts/check-tests.sh``, which is a plain run, so CI never sees it;
- it is run by naming it::

      ROBINAUTS_LIVE_ANTHROPIC_KEY=sk-... \\
          uv run pytest tests/live/test_langgraph_live.py

- and even then it **skips** without the key, so naming it by mistake costs
  nothing.

The key is read from a variable of its own rather than from the one a
deployment uses, so that running the suite on a machine that has a deployment
configured does not quietly start spending its key.

**The two vendors' own endpoints**, one turn each, each skipped without its
own key: Anthropic's with ``ROBINAUTS_LIVE_ANTHROPIC_KEY``, OpenAI's with
``ROBINAUTS_LIVE_OPENAI_KEY``::

    ROBINAUTS_LIVE_OPENAI_KEY=sk-... \\
        uv run pytest tests/live/test_langgraph_live.py

The engine also reaches the two ``-compatible`` kinds -- any endpoint that
speaks one of the two protocols at a configured ``base_url``, OpenRouter's
among them -- and those routes have a test of their own that needs no key at
all (``tests/live/test_vendor_routing.py``), so there is nothing here to
repeat.
"""

from __future__ import annotations

import os

import pytest

from aio import asyncio_test
from conversations import agent_definition, question
from robinauts.adapters import ProviderKeys
from robinauts.adapters.agents.langgraph import LangGraphAgent
from robinauts.core import check_engine_events
from robinauts.domain import (
    AnswerCompleted,
    AnswerStarted,
    AnswerTextDelta,
    Engine,
    EngineEvent,
    ModelConfig,
    ModelProviderConfig,
    ModelsConfig,
    ProviderKind,
    TextPart,
)

pytestmark = [pytest.mark.io, pytest.mark.live]

ANTHROPIC_KEY_VARIABLE = "ROBINAUTS_LIVE_ANTHROPIC_KEY"
"""The key this test spends, and nothing else in the repository reads."""

ANTHROPIC_MODEL_VARIABLE = "ROBINAUTS_LIVE_ANTHROPIC_MODEL"
"""Which model to ask, for when the default has been retired."""

DEFAULT_MODEL = "claude-haiku-4-5"
"""The cheapest model that can answer the question below."""

OPENAI_KEY_VARIABLE = "ROBINAUTS_LIVE_OPENAI_KEY"
"""The OpenAI key this test spends, and nothing else in the repository reads."""

OPENAI_MODEL_VARIABLE = "ROBINAUTS_LIVE_OPENAI_MODEL"
"""Which OpenAI model to ask, for when the default has been retired."""

DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"
"""A cheap model that answers the question below without reasoning first.

Chosen over a newer cheap one because it does not reason by default: on a
model that does, the ceiling below covers the reasoning as well, and could be
spent on it before a word of the answer.
"""

PROVIDER = "live"
MODEL = "live"
AGENT = "live"

ASKED = "Reply with the single word: robinaut"
"""Short, deterministic enough to assert on, and a handful of tokens."""

WANTED = "robinaut"

TURN_SECONDS = 60.0
"""Generous: a slow provider is not a failure of this test."""

MAX_TOKENS = 64
"""A ceiling, so that a model that misreads the question costs nothing much."""


def live_key(variable: str = ANTHROPIC_KEY_VARIABLE, vendor: str = "Anthropic") -> str:
    key = os.environ.get(variable)
    if not key:
        pytest.skip(f"set {variable} to run one real turn against {vendor}")
    return key


def live_models(
    kind: ProviderKind = ProviderKind.ANTHROPIC,
    variable: str = ANTHROPIC_KEY_VARIABLE,
    name: str | None = None,
) -> ModelsConfig:
    return ModelsConfig(
        providers={PROVIDER: ModelProviderConfig(id=PROVIDER, kind=kind, api_key_env=variable)},
        models={
            MODEL: ModelConfig(
                id=MODEL,
                provider=PROVIDER,
                name=name or os.environ.get(ANTHROPIC_MODEL_VARIABLE) or DEFAULT_MODEL,
                timeout_seconds=TURN_SECONDS,
                max_output_tokens=MAX_TOKENS,
            )
        },
        agents={
            AGENT: agent_definition(
                id=AGENT,
                model=MODEL,
                engine=Engine.LANGGRAPH,
                system_prompt="Answer in one word, in lower case, with no punctuation.",
            )
        },
    )


@asyncio_test
async def test_one_real_turn_against_anthropic_streams_and_completes() -> None:
    models = live_models()
    agent = LangGraphAgent(models, ProviderKeys({PROVIDER: live_key()}))
    await one_real_turn(agent, models)


@asyncio_test
async def test_one_real_turn_against_openai_streams_and_completes() -> None:
    """The same turn over Chat Completions, at OpenAI's own endpoint."""
    key = live_key(OPENAI_KEY_VARIABLE, "OpenAI")
    models = live_models(
        ProviderKind.OPENAI,
        OPENAI_KEY_VARIABLE,
        os.environ.get(OPENAI_MODEL_VARIABLE) or DEFAULT_OPENAI_MODEL,
    )
    agent = LangGraphAgent(models, ProviderKeys({PROVIDER: key}))
    await one_real_turn(agent, models)


async def one_real_turn(agent: LangGraphAgent, models: ModelsConfig) -> None:
    seen: list[EngineEvent] = []

    async for event in agent.run_turn(models.agents[AGENT], (question(ASKED),), (), model=MODEL):
        seen.append(event)

    check_engine_events(seen)
    assert isinstance(seen[0], AnswerStarted)
    assert [event for event in seen if isinstance(event, AnswerTextDelta)]
    completed = seen[-1]
    assert isinstance(completed, AnswerCompleted)
    said = "".join(part.text for part in completed.parts if isinstance(part, TextPart))
    assert WANTED in said.lower()
    assert agent.held == 0
