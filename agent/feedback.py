"""Feedback and the learning loop.

Takes user feedback about a decision and updates the user's preference memory.

Feedback can be structured (``signal="positive"``/``"negative"``) or free text
interpreted by a deterministic keyword classifier. Explicit feedback (the user
directly responding to a decision) is weighted more strongly than inferred
feedback, and explicit negative feedback reduces autonomy fastest.

The learning loop never weakens the hard safety floor: preferences only affect
decisions the safety floor allows. An aggressive preference for "always act
automatically" cannot turn a payment, a deletion, or a sensitive-data
disclosure into an autonomous action.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from .models import (
    AutonomyDecision,
    DecisionResult,
    Preference,
    PreferenceKey,
)
from .preferences import PreferenceStore

# ---------------------------------------------------------------------------
# Signal interpretation
# ---------------------------------------------------------------------------


class FeedbackSignal(str, Enum):
    """Direction of a feedback signal about a decision."""

    POSITIVE = "positive"  # user is happy with autonomous handling
    NEGATIVE = "negative"  # user wanted to be consulted / handled differently
    NEUTRAL = "neutral"    # no actionable signal


# Phrase -> score. Positive markers raise the score, negative markers lower it.
_POSITIVE_MARKERS: list[tuple[str, float]] = [
    ("handle automatically", 3.0),
    ("handle these automatically", 3.0),
    ("handle this automatically", 3.0),
    ("handle it automatically", 3.0),
    ("handle them automatically", 3.0),
    ("handle silently", 3.0),
    ("handle these silently", 3.0),
    ("handle it silently", 3.0),
    ("should have handled", 3.0),
    ("you should do that", 2.0),
    ("approved", 2.0),
    ("correct", 2.0),
    ("go ahead", 2.0),
    ("great", 2.0),
    ("perfect", 2.0),
    ("well done", 2.0),
    ("that's right", 2.0),
    ("fine with me", 1.5),
    ("yes", 1.5),
    ("good", 1.0),
    ("fine", 1.0),
    ("thanks", 1.0),
    ("thank you", 1.0),
]

_NEGATIVE_MARKERS: list[tuple[str, float]] = [
    ("don't do that", 3.0),
    ("do not do that", 3.0),
    ("ask me first", 3.0),
    ("ask me", 3.0),
    ("without asking", 3.0),
    ("should have asked", 3.0),
    ("shouldn't have", 3.0),
    ("should not have", 3.0),
    ("don't handle", 3.0),
    ("stop", 2.0),
    ("wrong", 2.0),
    ("no", 1.5),
    ("don't", 1.5),
    ("not", 1.0),
]

_POSITIVE_THRESHOLD = 0.75
_NEGATIVE_THRESHOLD = -0.75


def parse_feedback(text: str) -> Optional[FeedbackSignal]:
    """Interpret free-text feedback as a positive/negative/neutral signal.

    Deterministic keyword scoring. Empty or ambiguous text returns None or
    NEUTRAL so the learning loop makes no update when the signal is unclear.
    """
    if not text or not text.strip():
        return None
    lowered = text.lower()
    score = 0.0
    for marker, weight in _POSITIVE_MARKERS:
        if marker in lowered:
            score += weight
    for marker, weight in _NEGATIVE_MARKERS:
        if marker in lowered:
            score -= weight
    if score > _POSITIVE_THRESHOLD:
        return FeedbackSignal.POSITIVE
    if score < _NEGATIVE_THRESHOLD:
        return FeedbackSignal.NEGATIVE
    return FeedbackSignal.NEUTRAL


# ---------------------------------------------------------------------------
# Feedback engine
# ---------------------------------------------------------------------------


class FeedbackEngine:
    """Applies feedback to preference memory (the learning loop)."""

    def __init__(self, store: PreferenceStore) -> None:
        self._store = store

    @property
    def store(self) -> PreferenceStore:
        return self._store

    def apply(
        self,
        user_id: str,
        decision: DecisionResult,
        feedback_text: str = "",
        *,
        signal: Optional[FeedbackSignal] = None,
        explicit: Optional[bool] = None,
    ) -> Optional[Preference]:
        """Record feedback about a decision and update preferences.

        ``signal`` takes precedence when provided; otherwise ``feedback_text``
        is interpreted by :func:`parse_feedback`. Feedback is treated as
        *explicit* when the user directly responded (i.e. provided text), and
        *inferred* otherwise; this can be overridden with ``explicit``.

        Returns the updated preference, or None when the feedback is neutral.
        """
        parsed = parse_feedback(feedback_text) if signal is None else FeedbackSignal(signal)
        if parsed is None or parsed is FeedbackSignal.NEUTRAL:
            return None

        is_explicit = explicit if explicit is not None else bool(feedback_text.strip())

        # The preference is keyed by the action the user gave feedback about.
        action_type = decision.recommended_action or decision.policy_trace.intent
        key = PreferenceKey(action_type=action_type, sender_category="*", context="")

        # When the user approves an ASK decision, the target autonomous level
        # for a generic action is ACT_NOTIFY (keeps SILENT for backgrounds).
        default_decision = (
            decision.decision
            if decision.decision.is_autonomous
            else AutonomyDecision.ACT_NOTIFY
        )

        return self._store.record_signal(
            user_id,
            key,
            positive=(parsed is FeedbackSignal.POSITIVE),
            explicit=is_explicit,
            default_decision=default_decision,
        )

    def record(
        self,
        user_id: str,
        *,
        action_type: str,
        signal: FeedbackSignal,
        explicit: bool = True,
        sender_category: str = "*",
        default_decision: Optional[AutonomyDecision] = None,
    ) -> Preference:
        """Low-level structured feedback without a decision object.

        Used by the evaluation harness to simulate batch user feedback.
        ``sender_category`` scopes the preference (e.g. ``"newsletter"``) so
        feedback about newsletters does not leak onto other senders.
        """
        return self._store.record_signal(
            user_id,
            PreferenceKey(
                action_type=action_type,
                sender_category=sender_category,
                context="",
            ),
            positive=(signal is FeedbackSignal.POSITIVE),
            explicit=explicit,
            default_decision=default_decision,
        )
