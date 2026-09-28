// SPDX-License-Identifier: Apache-2.0
// Copyright The Robinauts Authors

/**
 * What the deployment offers to choose from: its agents, its models.
 *
 * Both are a list the operator configured, asked for once and not again,
 * because they change only when somebody edits the configuration and
 * restarts the server (`docs/specs/agents.md`). One hook for the two, so
 * that asking, dropping an answer nobody is waiting for and saying why a
 * list did not come are written once.
 */
import { useEffect, useState } from "react";

/** A list, as far as the one call for it has got. */
export type Offered<T> =
  | { status: "loading" }
  | { status: "ready"; items: T[] }
  | { status: "failed"; detail: string };

/**
 * The list `ask` answers, asked for once -- and again whenever `round`
 * changes, which is how a caller that has just been told the list is stale
 * asks for it afresh. The list it had stays shown until the new one comes.
 *
 * `ask` must be the same function on every render -- a module's own, not one
 * written inline -- or every render would be another call.
 */
export function useOffered<T>(
  ask: (signal: AbortSignal) => Promise<{ items: T[] }>,
  round = 0,
): Offered<T> {
  const [offered, setOffered] = useState<Offered<T>>({ status: "loading" });
  useEffect(() => {
    const dropped = new AbortController();
    ask(dropped.signal).then(
      (answer) => {
        setOffered({ status: "ready", items: answer.items });
      },
      (failure: unknown) => {
        // An abort is this component going away, or a newer round, not a
        // failure to show.
        if (dropped.signal.aborted) return;
        setOffered({
          status: "failed",
          detail: failure instanceof Error ? failure.message : String(failure),
        });
      },
    );
    return () => {
      dropped.abort();
    };
  }, [ask, round]);
  return offered;
}
