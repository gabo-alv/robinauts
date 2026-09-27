# Model selection progress

The plan is [model-selection-plan.md](model-selection-plan.md). The process
is the three-agent recipe (`recipes/three-agent-steps.md`, beside this
repository). The codebase map is "What exists" in
[poc-progress.md](poc-progress.md); this file records only what the model
selection adds to it.

## What exists

- `ModelConfig.title` (optional in `[models.*]`, the id when absent) and
  `ModelsConfig.model_by_id`, which raises `UnknownModelError` (a
  `NotFoundError`, 404, like `UnknownAgentError`) for a model the
  deployment does not offer.
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
