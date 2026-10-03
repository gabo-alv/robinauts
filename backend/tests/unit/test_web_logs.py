# SPDX-License-Identifier: Apache-2.0
# Copyright The Robinauts Authors

from __future__ import annotations

from robinauts.web.logs import loggable


def test_a_plain_value_is_kept() -> None:
    assert loggable("okta") == "okta"


def test_a_line_break_cannot_forge_a_line() -> None:
    assert loggable("okta\r\nINFO user 1 signed in") == "okta??INFO user 1 signed in"


def test_other_control_characters_are_replaced() -> None:
    assert loggable("a\x00b\x1b[31mc\x7f") == "a?b?[31mc?"


def test_a_long_value_is_cut() -> None:
    assert loggable("x" * 500) == "x" * 200
