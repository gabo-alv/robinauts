// SPDX-License-Identifier: Apache-2.0
// Copyright The Robinauts Authors

/**
 * The shell: the panel on the left, the chat on the right, and no top bar
 * (`docs/specs/frontend.md`).
 *
 * Two things about the panel are decided here rather than inside it, because
 * both are about the window and not about the panel's contents:
 *
 * - **the rail**, remembered per browser, which is what the panel looks like
 *   on a screen wide enough to keep it beside the page;
 * - **the drawer**, which is what it is on a screen that is not. Which of
 *   the two applies is a CSS breakpoint's decision, so the drawer's state is
 *   kept whatever the width is and simply has no effect above `md`.
 *
 * The shape is neorc's, written again for this project; recorded in
 * `docs/legal/ip-clearance.md`.
 */
import { Menu } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Chat } from "../chat";
import { conversationIn, useHistory } from "../history/history";
import { navigate, NEW_CHAT, useRoute } from "../router";
import type { Session } from "../session/session";
import { signOut as endSession } from "../session/session";
import { shownTitle } from "../conversation/conversation";
import type { Conversation } from "../conversation/conversation";
import {
  AgentPicker,
  type Agents,
  useAgents,
  useChosenAgent,
} from "./AgentPicker";
import { LocalModeBanner } from "./LocalModeBanner";
import { PANEL_ID, Panel } from "./Panel";
import { remember, remembered } from "./storage";

export const PANEL_KEY = "panel";

/**
 * What to call the agent a conversation is with (`docs/specs/agents.md`).
 *
 * The conversation names its agent by id, and the list the picker was given
 * has the title. Until that list has arrived there is nothing to call it by,
 * so nothing is said rather than an id that a moment later turns into a
 * name. An id the list does not have -- an agent the operator has removed,
 * or a list that never came -- is shown as it is: a conversation with an
 * agent nobody can name any more still had one.
 */
export function agentTitle(agents: Agents, id: string): string | null {
  if (agents.status === "loading") return null;
  if (agents.status === "failed") return id;
  return agents.items.find((agent) => agent.id === id)?.title ?? id;
}

/** The rail, as this browser last left it. */
function usePanelCollapsed(): [boolean, (collapsed: boolean) => void] {
  const [collapsed, setCollapsed] = useState(
    () => remembered(PANEL_KEY) === "rail",
  );
  return [
    collapsed,
    (next: boolean) => {
      setCollapsed(next);
      remember(PANEL_KEY, next ? "rail" : "open");
    },
  ];
}

export function Shell({
  session,
  signOut = endSession,
}: {
  session: Session;
  signOut?: () => Promise<void>;
}) {
  const [collapsed, setCollapsed] = usePanelCollapsed();
  const [drawer, setDrawer] = useState(false);
  // Where the interface is: `#/` or `#/c/<id>` (`src/router.ts`).
  const route = useRoute();
  const history = useHistory();
  // "New chat" pressed while the empty chat is already up changes no hash
  // and so re-renders nothing; the count is what remounts the area, so
  // nothing typed into it carries over into the next one.
  const [chat, setChat] = useState(0);
  // Asked for here, not inside the empty chat: that is remounted on every
  // "New chat", and the agents do not change while the server is running.
  // The choice is held here too, because the chat needs it to begin a
  // conversation and the picker is what the chat draws above its box.
  const agents = useAgents();
  const [agentId, chooseAgent] = useChosenAgent(agents);
  const opener = useRef<HTMLButtonElement>(null);
  const closer = useRef<HTMLButtonElement>(null);

  // Closing puts the focus back on the button that opened it, so a keyboard
  // is never left on an element that has slid off the screen -- or, worse,
  // on <body>, which is where the focus falls when what held it goes away.
  const close = useCallback(() => {
    if (drawer) opener.current?.focus();
    setDrawer(false);
  }, [drawer]);

  // Escape closes the drawer, wherever the focus is inside it, and it is a
  // close like any other: the same `close`, so the focus comes back.
  useEffect(() => {
    if (!drawer) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
    };
  }, [drawer, close]);

  // The focus follows the drawer into it when it opens, so a keyboard
  // carries on where the eye is rather than behind the overlay.
  useEffect(() => {
    if (drawer) closer.current?.focus();
  }, [drawer]);

  const current = route.kind === "conversation" ? route.id : null;
  const listed = conversationIn(history.items, current);
  // What the chat's own read said about the conversation on the screen,
  // which the panel's pages need not hold; the title is the panel's first,
  // since renaming happens there.
  const [opened, setOpened] = useState<{
    id: string;
    agent: string;
    title: string | null;
  } | null>(null);
  const about = current !== null && opened?.id === current ? opened : null;
  const title = listed?.title ?? about?.title ?? null;
  const conversationAgent = listed?.agent ?? about?.agent ?? null;
  const withAgent =
    conversationAgent === null ? null : agentTitle(agents, conversationAgent);
  const conversationOpened = useCallback((conversation: Conversation) => {
    setOpened({
      id: conversation.id,
      agent: conversation.agent,
      title: conversation.title,
    });
  }, []);
  // Held across renders: the chat memoises what it is given, so that a
  // keystroke in the message box does not remount the picker under it.
  const welcome = useMemo(
    () => (
      <AgentPicker agents={agents} chosen={agentId} onChoose={chooseAgent} />
    ),
    // `chooseAgent` is made afresh on every render and does the same thing
    // each time; what the picker draws is the two values above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [agents, agentId],
  );
  const startedConversation = useCallback(
    (id: string) => {
      navigate({ kind: "conversation", id });
      // A conversation that has just been created is not in the panel's
      // list, and its title is the beginning of the message that created it
      // (`docs/specs/conversations.md`). Its agent is the one just chosen.
      if (agentId !== null) setOpened({ id, agent: agentId, title: null });
      history.refresh();
    },
    [history, agentId],
  );

  return (
    <div className="flex h-screen bg-ground text-ink">
      <Panel
        user={session.user ?? null}
        history={history}
        current={current}
        collapsed={collapsed}
        onCollapse={setCollapsed}
        drawer={drawer}
        onCloseDrawer={close}
        onNewChat={() => {
          navigate(NEW_CHAT);
          setChat((count) => count + 1);
          close();
        }}
        signOut={signOut}
        signInConfigured={session.sign_in}
        closeRef={closer}
      />
      {drawer && (
        <div
          data-backdrop=""
          aria-hidden="true"
          onClick={close}
          className="fixed inset-0 z-10 bg-scrim md:hidden"
        />
      )}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <LocalModeBanner local={session.local_development ?? false} />
        {/* The one thing above the chat, and only where the panel is a
            drawer: `docs/specs/frontend.md` says there is no top bar, so
            above the breakpoint this is gone and nothing replaces it. */}
        <button
          type="button"
          ref={opener}
          title="Open the panel"
          aria-label="Open the panel"
          aria-controls={PANEL_ID}
          aria-expanded={drawer}
          onClick={() => {
            setDrawer(true);
          }}
          className="m-2 self-start rounded-ui p-1.5 text-muted-foreground hover:bg-hover hover:text-ink md:hidden"
        >
          <Menu size={16} aria-hidden="true" />
        </button>
        {/* The heading is the first line of the page, because
            `docs/specs/frontend.md` leaves no top bar to put it in. It is the
            shell's and not the chat's: the title comes from the panel's list,
            which is where renaming happens.

            **The chat is not keyed by the conversation.** A first message
            creates the conversation and the route follows it, and remounting
            on that would throw away the stream that is arriving. Which
            conversation is on the screen is a prop it handles itself; the
            `chat` count is what "New chat" remounts it by, so that nothing
            typed into one carries over into the next. */}
        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          <h1 className="mx-auto w-full max-w-3xl px-6 pt-4 text-xl font-semibold">
            {route.kind === "conversation"
              ? title === null
                ? "…"
                : shownTitle(title)
              : "New chat"}
          </h1>
          {/* Who the conversation is with, under its title. The picker is
              gone once a conversation exists (`./AgentPicker.tsx`), and
              with it the only thing on the screen that said so; this is
              the fact it left behind, told as a line rather than a control
              that could not be changed. */}
          {withAgent !== null && (
            <p
              data-agent=""
              className="mx-auto w-full max-w-3xl px-6 pt-1 text-sm text-muted-foreground"
            >
              with {withAgent}
            </p>
          )}
          <Chat
            key={chat}
            conversationId={current}
            agentId={agentId}
            onConversationStarted={startedConversation}
            onConversationOpened={conversationOpened}
            onTurnEnded={history.refresh}
            welcome={welcome}
          />
        </main>
      </div>
    </div>
  );
}
