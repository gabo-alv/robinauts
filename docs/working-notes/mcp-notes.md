# Notes towards tools over MCP

Written 2026-09-27 at the end of the POC session, for whoever plans the
tools work. Not a spec: a list of what already leaves room for tools, what
will have to change, and what this session learned the hard way. The
wanted behaviour is in [specs/runs.md](../specs/runs.md),
[specs/agents.md](../specs/agents.md) and
[specs/conversations.md](../specs/conversations.md) ("Tools", planned).

## What is already in place for tools

- **The agent port ends a turn one way today** (`AnswerCompleted`); the
  port's docstring says a tool call is "another engine event at the end of
  this stream" and the run lifecycle grows a `waiting` branch. `RunState`
  already has `WAITING`; the store's "one active run per conversation"
  index treats it as active. Nothing else has to change shape.
- **The conversation format has the place**: `Role.TOOL` exists and is
  refused everywhere today (`check_supported_role`, both engines'
  `_messages`, the wire's `SentRole`) — every refusal is a named spot to
  open. `MessagePart` kinds are a closed set with a version; a tool-call
  and a tool-result part are two new kinds behind `SUPPORTED_PART_KINDS`,
  and the format's upgrader registry is how an old row keeps reading.
- **Both engines refuse a tool call today on purpose** (`NO_TOOLS` in
  `adapters/agents/langgraph/engine.py` and `.../pydantic_ai/engine.py`),
  detected three ways in LangGraph (`tool_call_chunks`, a `tool_call`
  content block, a `non_standard` block whose value is `tool_use`) and via
  `BaseToolCallPart`/`ToolCallPartDelta` in Pydantic AI. Those are the
  exact seams where the engine will yield a tool-call event instead.
- **The Pydantic AI engine breaks after the first model node** so the
  framework cannot call the model twice or run a tool itself. With tools
  the platform, not the framework, must own the loop: the run suspends
  `waiting`, the tool result is appended to the conversation, the run is
  resumed from the stored history. Neither framework's tool loop or
  checkpointer is used (ADR 0002, `agents.md`).
- **The wire is AG-UI**, which already has `TOOL_CALL_START/ARGS/END` and
  `TOOL_CALL_RESULT`; `api/agui.py` maps our events to AG-UI in one place
  (`AguiMapper`), ids are derived deterministically from the answer's id
  and the position, and the "id on the last derived event" rule applies to
  any new bracketed event. The frontend's `events.ts` ignores unknown event
  types, so a backend that starts emitting tool events cannot break an
  older UI; the reducer (`state.ts`) has to learn them.
- **The vendored assistant-ui components include `tool-fallback` and
  `tool-group`** (kept in step 20 exactly so tools would not mean a fork of
  `thread.aui.tsx`); the runtime adapter has to hand them `tool-call`
  parts.
- **Long-running tools**: the run executes in the background, survives a
  dropped request, and a watcher re-attaches by position — a tool that
  takes minutes is already the ordinary case for the stream. What is
  missing is the `waiting` state and who resumes the run.

## Things to decide before building

- Where MCP servers are configured: the operator's TOML beside
  `[model_providers]`/`[models]`/`[agents]` (an agent names the servers it
  may use), secrets by environment-variable *name* like everything else,
  refused at start-up with every problem at once (`core/models_config.py`
  is the pattern). Which transports (stdio spawns a process on the backend
  host — a corporate-readiness question; streamable HTTP is remote).
- Who runs the tool loop: the application (`Turns._produce`) on a
  `waiting` run — call the MCP tool, append the tool-result message, start
  the next engine turn from the history — versus a per-turn adapter. The
  specs want it above the port so both engines behave the same.
- Approval: the vendored `tool-fallback` shows an approval affordance
  (`approval.dismissible` is the field read as optional in step 20); the
  runs spec's `waiting` state can carry "waiting for a person" as well as
  "waiting for a tool".
- Licences: the MCP Python SDK and any transport it brings must pass the
  gate; check the transitive tree (`scripts/check-licences.sh`) before
  adopting — `orjson` (MPL-2.0) came in that way with LangGraph and needed
  a restricted-table row. On the JavaScript side nothing is needed: the
  UI sees tools only as AG-UI events.

## Learned in this session, worth not relearning

- **Both engines pin the vendor client**: endpoint, key as
  `default_headers`, proxy `None`, `max_retries=0`, tracing forced off,
  vendor loggers held at WARNING. A tool adapter that talks HTTP must
  follow the same rules (no env-var fallbacks, nothing phones home, no
  request bodies in logs). `ANTHROPIC_LOG=debug` logged whole
  conversations before we pinned the loggers — MCP SDKs may have the same.
- **Every `schema.sql` edit re-pins the hash and leaves `SCHEMA_VERSION`
  at 1**, and there are no migrations before the first release: a tool-call
  table or new columns means the database is recreated.
- **Reviews find the most in three places**: refusals that reflect input,
  no-ops on re-attach (a `*_START` for an open id, a `*_END` for a closed
  one, a repeated terminal), and anything a hand-written parser or regex
  tries to do that a real parser already does. Tool arguments are
  attacker-influenced text going to the browser: render as data, never
  Markdown-with-HTML, never a URL without the CSP in mind.
- **The three-agent recipe** (`~/code/recipes/three-agent-steps.md`) held:
  one branch per step, fresh reviewer every round, ten-round cap. Step 17
  hit the cap on a scanner that never converged; the lesson is to prefer
  a structural rule or the real tool's own resolver over a growing regex.
- The demo (`demo/start.sh`) is the fastest way to see a change end to
  end with a real model through OpenRouter; `scripts/rehearse-deployment.sh`
  covers sign-in without real providers.
