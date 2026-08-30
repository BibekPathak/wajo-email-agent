"""Email understanding layer.

Separates WHAT THE EMAIL MEANS from what the user prefers and whether acting is
safe. Two implementations of the same protocol:

* :class:`MockUnderstanding` -- deterministic rule-based classifier (default).
  Runs fully offline and keeps the evaluation harness deterministic.
* :class:`OpenAIUnderstanding` -- OpenAI-compatible chat completions with JSON
  output, selected via ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
  ``OPENAI_MODEL``.

The mock and the LLM output the same :class:`AnalysisResult`, so the rest of
the pipeline never cares which backend produced the analysis.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from .models import AnalysisResult, EmailSituation
from .prompts import build_analysis_messages
from .textutil import attachment_text, contains_any, contains_phrase


class UnderstandingError(RuntimeError):
    """Raised when the LLM understanding backend cannot run."""


class EmailUnderstanding(ABC):
    """Protocol for the semantic understanding layer."""

    @abstractmethod
    def analyze(self, email: EmailSituation) -> AnalysisResult:
        """Turn an email into a structured analysis."""


def make_understanding(backend: Optional[str] = None, **kwargs) -> EmailUnderstanding:
    """Factory: ``"mock"`` (default) or ``"openai"``.

    The backend can also be chosen with the ``UNDERSTANDING_BACKEND``
    environment variable.
    """
    backend = (backend or os.environ.get("UNDERSTANDING_BACKEND", "mock")).lower()
    if backend == "openai":
        return OpenAIUnderstanding(**kwargs)
    return MockUnderstanding(**kwargs)


# ---------------------------------------------------------------------------
# Deterministic mock understanding
# ---------------------------------------------------------------------------

# intent -> (requested tool action or None, [(phrase, weight), ...])
_INTENT_RULES: dict[str, tuple[Optional[str], list[tuple[str, float]]]] = {
    "pay_invoice": ("make_payment", [
        ("pay the invoice", 4.0), ("pay invoice", 4.0), ("payment", 2.0),
        ("pay", 1.5), ("invoice", 1.0), ("remit", 2.0), ("settle", 2.0),
    ]),
    "transfer_money": ("make_payment", [
        ("bank transfer", 4.0), ("send money", 4.0), ("transfer money", 4.0),
        ("transfer", 3.0), ("wire", 3.0), ("send usd", 3.0), ("send eur", 3.0),
    ]),
    "approve_invoice": ("make_payment", [
        ("approve the invoice", 4.0), ("approve invoice", 4.0),
        ("approve the payment", 4.0), ("approve payment", 4.0),
        ("authorize payment", 3.0), ("sign off on", 2.0),
    ]),
    "make_purchase": ("make_payment", [
        ("place an order", 3.0), ("place order", 3.0), ("purchase", 3.0),
        ("buy", 2.0), ("make a purchase", 3.0),
    ]),
    "delete_email": ("delete_email", [
        ("delete the email", 4.0), ("delete this email", 4.0),
        ("permanently delete", 4.0), ("remove permanently", 4.0),
        ("delete", 2.0), ("erase", 3.0), ("purge", 3.0),
    ]),
    "cancel_account": (None, [
        ("cancel my account", 4.0), ("close my account", 4.0),
        ("cancel account", 3.0), ("close account", 3.0),
        ("cancel service", 3.0), ("cancel subscription", 2.5),
        ("terminate", 2.0),
    ]),
    "disclose_data": ("disclose_data", [
        ("send the customer list", 4.0), ("share the customer list", 4.0),
        ("share customer list", 4.0), ("send customer data", 4.0),
        ("share personal data", 4.0), ("send personal data", 4.0),
        ("disclose", 3.0), ("hand over", 3.0),
        ("share the", 2.0), ("share a", 2.0), ("share our", 2.0), ("share", 1.5),
    ]),
    "share_confidential": ("disclose_data", [
        ("api key", 4.0), ("private key", 4.0), ("password", 3.0),
        ("credentials", 3.0), ("secret", 2.0),
    ]),
    "reschedule_meeting": ("schedule_meeting", [
        ("move tomorrow's meeting", 3.5), ("reschedule the meeting", 3.5),
        ("move tomorrow's", 2.5), ("move tomorrows", 2.5),
        ("reschedule", 3.0), ("move the meeting", 3.0), ("push the meeting", 3.0),
        ("move our meeting", 3.0), ("change the meeting", 2.5),
        ("move it to", 2.0),
    ]),
    "schedule_meeting": ("schedule_meeting", [
        ("schedule a meeting", 3.0), ("set up a meeting", 3.0),
        ("book a meeting", 3.0), ("arrange a meeting", 3.0),
        ("find time", 2.5), ("schedule", 1.5), ("calendar", 1.0),
    ]),
    "cancel_meeting": ("cancel_meeting", [
        ("cancel the meeting", 3.5), ("cancel our meeting", 3.5),
        ("cancel tomorrow's meeting", 3.5), ("cancel meeting", 3.0),
    ]),
    "accept_invitation": ("schedule_meeting", [
        ("accept the invitation", 3.5), ("accept invitation", 3.0),
        ("accept the invite", 3.0), ("rsvp", 2.5),
    ]),
    "schedule_internal_reminder": ("schedule_meeting", [
        ("remind me", 3.0), ("set a reminder", 3.0), ("reminder", 2.0),
    ]),
    "forward_email": ("forward_email", [
        ("forward this email", 3.0), ("forward the email", 3.0),
        ("forward it to", 3.0), ("forward", 2.0), ("send it to", 2.0),
    ]),
    "send_external_email": ("send_email", [
        ("send an email", 3.0), ("send a message", 3.0), ("send the email", 3.0),
        ("get back to", 3.0), ("reach out", 3.0), ("email them", 3.0),
        ("reply", 2.0), ("respond", 2.0), ("contact", 2.0), ("send", 1.0),
    ]),
    "confirm_terms": ("send_email", [
        ("confirm the terms", 3.5), ("confirm terms", 3.0),
        ("confirm the contract", 3.0), ("approve the contract", 3.0),
        ("confirm", 1.5),
    ]),
    "share_info": ("send_email", [
        ("share the info", 3.0), ("share the information", 3.0),
        ("share it with", 2.5), ("send them", 2.0), ("share with", 2.0),
    ]),
    "archive_email": ("archive_email", [
        ("archive", 2.0), ("file it", 2.0), ("file this", 2.0), ("store it", 2.0),
    ]),
    "archive_newsletter": ("archive_email", [
        ("archive this newsletter", 4.0), ("unsubscribe", 3.0),
        ("weekly digest", 3.0), ("newsletter", 2.5), ("weekly update", 2.5),
        ("digest", 2.0),
    ]),
    "label_email": ("label_email", [
        ("label", 2.0), ("tag", 2.0),
    ]),
    "move_email": ("move_email", [
        ("move the email", 3.0), ("move this email", 3.0), ("move to", 2.0),
        ("move into", 2.0), ("move this", 2.0), ("file under", 2.5),
    ]),
    "classify_email": (None, [
        ("classify", 2.5), ("categorize", 2.5), ("triage", 2.0), ("sort", 1.5),
    ]),
    "deduplicate": (None, [
        ("deduplicate", 3.0), ("dedupe", 3.0), ("duplicate emails", 3.0),
    ]),
    "create_draft": ("create_draft", [
        ("draft a reply", 3.0), ("draft an email", 3.0), ("prepare a draft", 3.0),
        ("draft", 2.0),
    ]),
}

# Pairs of phrases that indicate genuinely conflicting requests.
_CONFLICT_PAIRS: list[tuple[str, str]] = [
    ("delete", "archive"),
    ("delete", "keep"),
    ("delete", "retain"),
    ("pay", "cancel"),
]

# Language that suggests the email itself is ambiguous / needs human judgment.
_AMBIGUITY_MARKERS = [
    "not sure", "i don't know", "what should i", "what do you think",
    "handle this", "decide for me", "confused", "ambiguous", "either way",
    "maybe", "up to you", "figure out", "you decide", "what to do",
]

_NEWSLETTER_SENDER_MARKERS = [
    "newsletter", "no-reply", "noreply", "mailer", "updates", "mailing",
    "digest", "substack", "weekly", "unsubscribe", "sendy",
]
_NEWSLETTER_SUBJECT_MARKERS = [
    "newsletter", "weekly digest", "weekly update", "digest", "what's new",
]

_AMOUNT_RE = re.compile(r"\$\s?([0-9][\d,]*(?:\.\d{2})?)")
_TIME_RE = re.compile(r"\b([0-9]{1,2}:[0-9]{2}\s*(?:am|pm)|[0-9]{1,2}\s*(?:am|pm))\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")


def _classify_sender(email: EmailSituation) -> str:
    sender = email.sender.lower()
    text = (f"{email.subject} {email.body}").lower()
    if contains_any(sender, _NEWSLETTER_SENDER_MARKERS) or contains_any(
        text, _NEWSLETTER_SUBJECT_MARKERS
    ):
        return "newsletter"
    if contains_any(sender, ("billing", "invoice", "vendor", "supplier", "store", "shop", "support")):
        return "vendor"
    if contains_any(sender, ("customer", "client")):
        return "customer"
    return "external"


def _score_intents(text: str) -> list[tuple[str, Optional[str], float]]:
    scored: list[tuple[str, Optional[str], float]] = []
    for intent, (action, markers) in _INTENT_RULES.items():
        score = sum(w for phrase, w in markers if contains_phrase(text, phrase))
        if score > 0:
            scored.append((intent, action, score))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored


def _resolve(text: str, scored: list[tuple[str, Optional[str], float]]) -> tuple[str, Optional[str], float]:
    """Pick intent/action/confidence from scored rules.

    Conflicting request pairs and closely-matched alternative intents are
    marked ambiguous (low confidence, no action) so the risk layer escalates.
    Emails with no action signal at all are treated as background/informational
    (classify_email -> SILENT), unless they contain ambiguous language.
    """
    if any(contains_phrase(text, a) and contains_phrase(text, b) for a, b in _CONFLICT_PAIRS):
        return "unknown", None, 0.3

    strong = [s for s in scored if s[2] >= 1.0]
    if not strong:
        if any(contains_phrase(text, m) for m in _AMBIGUITY_MARKERS):
            return "unknown", None, 0.3
        return "classify_email", None, 0.75

    top_intent, top_action, top_score = strong[0]
    ambiguous = False
    if len(strong) >= 2:
        _, second_action, second_score = strong[1]
        if second_action != top_action and top_score - second_score < 1.0:
            ambiguous = True
    confidence = 0.35 if ambiguous else min(0.95, 0.55 + 0.1 * top_score)
    return top_intent, top_action, confidence


def extract_entities(text: str) -> dict[str, Any]:
    """Pull a few structured facts out of the text (used as tool arguments)."""
    entities: dict[str, Any] = {}
    amount = _AMOUNT_RE.search(text)
    if amount:
        entities["amount"] = amount.group(1).replace(",", "")
    match = _TIME_RE.search(text)
    if match:
        entities["time"] = match.group(1)
    emails = _EMAIL_RE.findall(text)
    if emails:
        entities["recipient"] = emails[0]
    return entities


class MockUnderstanding(EmailUnderstanding):
    """Deterministic, offline, rule-based email understanding (default)."""

    def analyze(self, email: EmailSituation) -> AnalysisResult:
        text = (f"{email.subject}\n{email.body}").lower()
        attachment_hint = attachment_text(email.attachments)
        if attachment_hint:
            text = f"{text}\n{attachment_hint}"
        sender_category = _classify_sender(email)
        intent, action, confidence = _resolve(text, _score_intents(text))

        # External messages from customers are customer replies.
        if (
            action == "send_email"
            and sender_category == "customer"
            and intent in ("send_external_email", "confirm_terms", "share_info")
        ):
            intent = "reply_customer"

        entities = extract_entities(text)
        return AnalysisResult(
            intent=intent,
            requested_action=action,
            sender_category=sender_category,
            confidence=confidence,
            entities=entities,
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible LLM understanding
# ---------------------------------------------------------------------------


class OpenAIUnderstanding(EmailUnderstanding):
    """LLM-based understanding via the OpenAI-compatible chat API.

    Configured through ``OPENAI_API_KEY``, ``OPENAI_BASE_URL`` and
    ``OPENAI_MODEL`` (or explicit constructor arguments). Lazy-imports the
    ``openai`` package so the rest of the system runs offline without it.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 30,
    ) -> None:
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._temperature = temperature
        self._timeout = timeout

    def analyze(self, email: EmailSituation) -> AnalysisResult:
        if not self._api_key:
            raise UnderstandingError(
                "OPENAI_API_KEY is not set; use the 'mock' backend for offline runs."
            )
        from openai import OpenAI

        client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )
        response = client.chat.completions.create(
            model=self._model,
            messages=build_analysis_messages(email),
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise UnderstandingError("LLM returned an empty response.")
        data = json.loads(content)
        return AnalysisResult(
            intent=str(data.get("intent", "unknown")),
            requested_action=data.get("requested_action"),
            sender_category=str(data.get("sender_category", "unknown")),
            confidence=float(data.get("confidence", 0.5)),
            entities=dict(data.get("entities", {})),
        )
