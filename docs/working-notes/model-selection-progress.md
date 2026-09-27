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
- Not yet: runs and engines still use the agent's default
  (`definition.model`, `model_for(agent)`) until step 3.

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
