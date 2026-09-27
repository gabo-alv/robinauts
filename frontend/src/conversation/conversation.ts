// SPDX-License-Identifier: Apache-2.0
// Copyright The Robinauts Authors

/**
 * What the rest of the application calls a conversation and a message.
 *
 * `GET /api/conversations/{id}` answers **one moment** of a conversation
 * (`docs/specs/conversations.md`, "The visible thread"): the one thread it
 * shows, oldest first, and the run in flight or the way the last one ended.
 * The chat is what reads it (`src/chat/assistant-ui/runtime.tsx`); what is
 * here is the shapes, the title rule, and the one call that is not a turn.
 */
import { request } from "../api/client";
import type { components } from "../api/schema";
import type { ConversationId } from "../chat";

export type Conversation = components["schemas"]["ConversationSummary"];
export type Message = components["schemas"]["MessageView"];
export type Resume = components["schemas"]["ResumeView"];
export type EndedBadly = components["schemas"]["EndedBadlyView"];

/** What the panel and the view call a conversation nobody has named. */
export const UNTITLED = "Untitled";

/**
 * The title to show, which is never nothing.
 *
 * A conversation's title is the beginning of its first message and a message
 * with no text in it gives none (`docs/specs/conversations.md`, "Titles"), so
 * an empty title is a state the API really has. A blank line in the panel
 * would be a conversation nobody could aim at.
 */
export function shownTitle(title: string): string {
  return title.trim() === "" ? UNTITLED : title;
}

/** What a message says, as one string: its text parts, joined. */
export function textOf(message: Message): string {
  return message.parts
    .filter((part) => part.kind === "text")
    .map((part) => part.text)
    .join("");
}

/** Stop the run that is in flight. */
export async function cancelRun(
  id: ConversationId,
  runId: string,
): Promise<void> {
  await request(
    "post",
    "/api/conversations/{conversation_id}/runs/{run_id}/cancel",
    { path: { conversation_id: id, run_id: runId } },
  );
}
