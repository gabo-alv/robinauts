# SPDX-License-Identifier: Apache-2.0
# Copyright The Robinauts Authors

"""Request values made safe to put in a log line.

A value from a request may carry a line break or another control character, which would let
it forge a log line of its own (CWE-117). Every such value goes through ``loggable`` on its way
to the log, validated or not, so that no log line depends on a check made elsewhere.
"""

from __future__ import annotations

import re

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_LONGEST = 200


def loggable(value: str) -> str:
    """``value`` with its control characters replaced by ``?`` and cut to 200 characters."""
    return _CONTROL.sub("?", value[:_LONGEST])
