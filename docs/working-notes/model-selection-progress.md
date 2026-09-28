# Model selection progress

The plan is [model-selection-plan.md](model-selection-plan.md). The process
is the three-agent recipe (`recipes/three-agent-steps.md`, beside this
repository). The codebase map is "What exists" in
[poc-progress.md](poc-progress.md); this file records only what the model
selection adds to it.

## What exists

- `ModelConfig.title` (optional in `[models.*]`, the id when absent) and
  `ModelsConfig.model_by_id`, which raises `UnknownModelError` for a model
  the deployment does not offer. Since step 7 it is not a `NotFoundError`:
  422 for a model a request names (a new chat, the `PUT`), and its subclass
  `ModelNotOfferedError` is 409 for a conversation's own model at its next
  turn, checked after the owner. Both bodies are fixed sentences.
- `Conversation.model`, required: the agent's default is copied in when a
  conversation starts (`Turns._new_chat`). Stored in
  `conversations.model text NOT NULL`; `SCHEMA_VERSION` stays 1 and
  `SCHEMA_SHA256` is re-pinned with every edit of `schema.sql`.
- `ConversationStore.set_model(conversation_id, model, *, now)`, beside
  `rename_conversation` in both stores: checks the id's spelling, dates the
  conversation, returns the written record or `None`.
- `Turns` holds the offered models (`Mapping[str, ModelConfig]`, wired in
  `app.py`, which refuses at start-up an agent whose default is not
  offered). `begin(model_id=)` picks a new chat's model (`None` is the
  agent's default); `_new_run` takes `conversation.model` and raises
  `UnknownModelError` before any write for a model no longer offered;
  `Turns.set_model(user, conversation_id, model_id)` checks the model, then
  the owner, then writes. `_produce` hands the engine `run.model`: the agent
  port is `run_turn(agent, history, *, model)` and both engines look it up
  with `ModelsConfig.model_by_id` (`model_for` is gone).
- API: `GET /api/models` (`{items: [{id, title}]}`, configuration order,
  beside `/api/agents` in `api/agent_routes.py`); `PUT
  /api/conversations/{id}/model` with `{"model_id"}`, answering the
  `ConversationSummary` like the rename, 422 `UnknownModelError` for a
  model not offered (checked before the conversation, so it says nothing about it);
  `NewChatRequest.model_id` (absent or null is the agent's default, unknown
  is 422); `ConversationSummary.model`; `AgentSummary.model`.
- Frontend: `shell/ModelPicker.tsx` (`useModels`, `useChosenModel`, the
  `ModelPicker` select, `ConversationModel` which makes the PUT) and
  `shell/offered.ts` (`useOffered`, the fetch-once hook `useAgents` shares).
  The empty chat's model sits beside the agent; it follows the agent's
  default until picked, then is remembered (storage key `model`). The open
  conversation's model sits on the "with <agent>" line; changes are
  sent last-wins and the shell keeps the later of the PUT's answer and the
  chat's reads by `updated_at`. `ChatProps.modelId` and `onModelRefused`
  (the shell forgets a refused model). The chat branches on the error's
  name: a 409 `ModelNotOfferedError` or a new chat's 422 says the model is
  no longer offered and puts the text back (an edit's into its edit box); a
  new chat's 404 is its agent, and also puts the text back. The empty
  chat's pickers live in a `WelcomeSlot` context so a pick keeps focus.
  `scripts/fixture-server.mjs` serves the models (scene `conversation` has
  a retired one) but still speaks the pre-#18 branch API.
- Demo: `demo/robinauts.toml.in` declares three models through the one
  provider, ids and titles filled in by `demo/start.sh` per provider kind
  (OpenRouter: `claude-sonnet-5`, `gpt-5-5`, `gemini-3-8-flash`; Anthropic:
  `claude-sonnet-5`, `claude-opus-5-5`, `claude-haiku-4-5`), overridable by
  `ROBINAUTS_DEMO_MODEL`, `_2`, `_3` (an override keeps the default's id).
  `demo/config.py` takes `--model ID NAME TITLE` three times and checks what
  it wrote.

## Steps

### Step 1 — specs and domain   (feature/model-selection-1-domain)

Summary: `ModelConfig.title` and its parsing, `ModelsConfig.model_by_id`,
`UnknownModelError`, and `Conversation.model`, required and copied from the
agent's default at creation. The column moved here from step 2, because a
required field the Postgres store cannot read leaves its suite red; step 2
is now `set_model` alone. Specs: agents.md (the agent's model is a default;
a conversation's model changes at any point; a model no longer offered is
refused as not found; an agent's new default reaches new conversations
only), conversations.md, deployment.md.

Review: 1 round.
- High: 0
- Medium: 2 (1/1) — left: the specs describe the conversation's model as
  what a turn runs on, which is step 3's code.
- Low: 3 (3/0)

Checks: lint; unit suite 2716 passed; full suite against a throwaway
Postgres 2915 passed.
Not done / to watch: until step 3, a turn runs on the agent's current
default, not the conversation's stored model.
Design decisions: `UnknownModelError` is a `NotFoundError` (404 on the turn
routes, like a removed agent); the PUT's 422 is step 4's.

### Step 2 — storage: set_model   (feature/model-selection-2-storage)

Summary: `set_model` on the conversation store port, the Postgres store and
the fake, with the contract suite for both (sets, persists, refuses a bad id
without writing, `None` for a missing conversation, a race with an append)
and the lock-order test grown to six methods. A model change dates the
conversation, as a rename does. Whether the deployment offers the model is
the application's check, step 3.

Review: 1 round.
- High: 0
- Medium: 0
- Low: 4 (3/1) — left: a model-change-and-completion race twin, which the
  rename's covers for the same one-statement write.

Checks: lint; unit suite 2723 passed; full suite against a throwaway
Postgres 2928 passed (before the comment-only fixes).
Not done / to watch: nothing calls `set_model` yet.

### Step 3 — application and port   (feature/model-selection-3-application)

Summary: turns run on the conversation's model, copied onto the run; a new
chat takes `model_id` or the agent's default; `Turns.set_model` changes it,
owner only, allowed mid-run (the run keeps its model); a model the
deployment no longer offers refuses continue, edit and regenerate before
anything is written. The engines are handed the run's model. `set_model`
lives in `Turns`, not beside the rename, so that one service holds what a
turn accepts. Start-up now refuses an agent whose default model is not
offered even when agents and engines are handed in.

Review: 1 round.
- High: 0
- Medium: 0
- Low: 3 (3/0)

Checks: lint; unit suite 2745 passed; full suite against a throwaway
Postgres 2950 passed (before the docstring and one-test fixes).
Not done / to watch: nothing in the API reaches `model_id` or `set_model`
yet (step 4). `_produce` does not re-check `run.model` against `Turns`'
models; an engine's `model_by_id` fails such a turn.

### Step 4 — API   (feature/model-selection-4-api)

Summary: the two routes, the three schema fields and `model_id` on a new
chat, the error mapping (404 for an unknown model on the turn routes, 422
on the PUT), the OpenAPI snapshot and `wire.md`. The frontend changed only
in two typed test fixtures. No UI.

Review: 1 round.
- High: 0
- Medium: 0
- Low: 3 (1/2) — left: a badly shaped `model_id` on `POST /api/turns` is a
  422 that names no field, as a badly shaped `agent_id` already is; and the
  step 5 note below.

Checks: lint; unit suite 2763 passed; frontend check (374 tests, tsc,
eslint, build) passed; full suite against a throwaway Postgres 2967 passed
(before the docstring fix and one unit test).
Not done / to watch: for step 5, a turn in a conversation whose model was
removed is the same 404 as a missing conversation while opening it is 200;
the frontend explains it by comparing the conversation's `model` with
`GET /api/models`.

### Step 5 — frontend   (feature/model-selection-5-frontend)

Summary: the model picker beside the agent picker on the empty chat and on
the agent line of an open conversation; `model_id` on the first message
only; `PUT` on change, last-wins while one is in flight; a conversation on a
model no longer offered shows it marked, explains a refused turn and keeps
the typed text. frontend.md, the README and the fixture server follow. The
3e02925 agent picker was already on main (#16).

Review: 4 rounds.
- High: 0
- Medium: 3 (3/0)
- Low: 17 (12/5) — left: every 422 on the PUT read as "not offered" (no
  other 422 can happen today); the chat's read preferred over a newer panel
  row for the model across tabs; a "model gone" sentence kept after a
  change that lands before the turn's 404.

Checks: frontend check (prettier, eslint, tsc, 406 tests, build, audit);
screenshots through the fixture server with Playwright at 1280 and 390
wide.
Not done / to watch: the backend answers the same 404 for a removed agent
and a removed model on a new chat, so the first-message sentence names
both. The fixture server's branch fixtures still show a discarded answer.

### Step 6 — demo   (feature/model-selection-6-demo)

Summary: three models in the demo configuration, chosen by provider kind,
each with an id naming its default model and a title; both agents default
to Claude Sonnet 5. `config.py` refuses blank or over-long values and
placeholder-like input, and a template placeholder nothing fills. The
README says what the ids record, and that an override keeps the default's
id.

Review: 2 rounds.
- High: 0
- Medium: 2 (2/0)
- Low: 7 (3/4) — left: an argparse usage dump for a name starting with
  `-`, and an unchecked `--provider-id` (both older than this step, and
  `start.sh` passes only constants); OpenRouter serving GPT and Gemini
  through its Messages API, unverified; the progress entry (this one).

Checks: lint; unit suite 2764 passed; the template rendered for both kinds
and parsed by the backend's loader, `GET /api/models` served through
`create_app` in the local mode.
Not done / to watch: no real model call was made. That GPT-5.5 and Gemini
3.8 Flash answer through OpenRouter's Messages API, and the Anthropic ids
`claude-opus-5-5` and `claude-haiku-4-5`, are to be confirmed with a key.

### Step 7 — a model not offered has its own error   (feature/model-selection-7-model-not-offered)

Summary: replaces the plan's optional per-answer caption, which was judged
not needed. After an external review of #21, a model not offered stops
being a 404: `UnknownModelError` is 422 for a model a request names and
`ModelNotOfferedError` 409 for a conversation's model at its next turn,
both checked after the owner, so a stranger still gets the plain 404. The
frontend branches on the code, which removed the three-state `modelGone`,
the guessed sentences and the `unsaid` action. The same review's other
items: a refused edit goes back into its edit box, a model pick on the
empty chat keeps the keyboard focus, the fixture server checks the scene's
agents and answers the new codes, and `isOffered` is the one check. Specs:
agents.md, wire.md, frontend.md, deployment.md; decision 3 of the plan.

Review: 1 round.
- High: 0
- Medium: 1 (1/0)
- Low: 3 (3/0)

Checks: lint; unit suite 2770 passed; full suite against a throwaway
Postgres 2975 passed (before the frontend-only fixes); frontend check (407
tests, tsc, eslint, build, audit); screenshots through the fixture server.
Not done / to watch: a refusal's sentence stays until the next turn, even
after the model is switched; with the model list failed it still says to
pick another model above; text put back carries over to the empty chat
when `#/` is reached by a hash change rather than "New chat" (older).
