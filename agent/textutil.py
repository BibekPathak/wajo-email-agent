"""Small text helpers shared by the understanding and risk layers."""

from __future__ import annotations

import re

_BOUNDARY = r"(?<!\w)"


def contains_phrase(text: str, phrase: str) -> bool:
    """True when ``phrase`` appears in ``text`` as a standalone word/phrase.

    Uses word boundaries so e.g. "move to" does not match inside "move
    tomorrow" and "pay" does not match inside "payment".
    """
    if not phrase:
        return False
    return re.search(_BOUNDARY + re.escape(phrase) + r"(?!\w)", text) is not None


def contains_any(text: str, phrases) -> bool:
    return any(contains_phrase(text, phrase) for phrase in phrases)


def count_mentions(text: str, phrases) -> int:
    return sum(1 for phrase in phrases if contains_phrase(text, phrase))
