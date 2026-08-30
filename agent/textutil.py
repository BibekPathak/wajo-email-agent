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


def normalize_filename(filename: str) -> str:
    """Normalize a filename into searchable tokens.

    ``customer_list_2024.xlsx`` becomes ``customer list 2024 xlsx`` so phrase
    matching (e.g. "customer list") works on real attachment names.
    """
    base = filename.rsplit("/", 1)[-1]
    base = re.sub(r"\.[A-Za-z0-9]+$", "", base)
    base = re.sub(r"[_.\-+]+", " ", base)
    return base.lower().strip()


def attachment_text(attachments) -> str:
    """Join attachment names/content types into a searchable string."""
    parts = []
    for attachment in attachments:
        name = getattr(attachment, "filename", "") or ""
        content_type = getattr(attachment, "content_type", "") or ""
        parts.append(normalize_filename(name))
        if content_type:
            parts.append(content_type.lower().replace("/", " "))
    return " ".join(parts)
