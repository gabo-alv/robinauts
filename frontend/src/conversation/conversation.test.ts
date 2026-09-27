// SPDX-License-Identifier: Apache-2.0
// Copyright The Robinauts Authors
import { expect, test } from "vitest";

import { message } from "../test/conversations";
import { shownTitle, textOf } from "./conversation";

test("a message says its text parts, joined", () => {
  expect(textOf(message("m", "user", "Hello"))).toBe("Hello");
  expect(
    textOf({
      ...message("m", "assistant", ""),
      parts: [
        { kind: "text", text: "A long " },
        { kind: "text", text: "answer." },
      ],
    }),
  ).toBe("A long answer.");
});

test("a conversation nobody named is shown as Untitled", () => {
  expect(shownTitle("Named")).toBe("Named");
  expect(shownTitle("")).toBe("Untitled");
  expect(shownTitle("   ")).toBe("Untitled");
});
