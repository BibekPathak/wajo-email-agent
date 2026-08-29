"""Unit tests for the SQLite-backed preference memory."""

from agent.models import AutonomyDecision, Preference, PreferenceKey
from agent.preferences import (
    EXPLICIT_NEGATIVE_WEIGHT,
    INFERRED_WEIGHT,
    PreferenceStore,
    compute_confidence,
    derive_preferred_decision,
)


def make_key(action_type="schedule_meeting", sender_category="*"):
    return PreferenceKey(action_type=action_type, sender_category=sender_category)


# --- Basic persistence --------------------------------------------------------


def test_new_preference_starts_with_no_evidence():
    store = PreferenceStore(":memory:")
    assert store.get("u1", make_key()) is None
    assert store.lookup("u1", "schedule_meeting", "alice@example.com") is None


def test_get_and_upsert_roundtrip():
    store = PreferenceStore(":memory:")
    key = make_key()
    preference = Preference(
        key=key,
        preferred_decision=AutonomyDecision.ACT_NOTIFY,
        positive_count=8,
        negative_count=1,
        explicit_positive_count=8,
        explicit_negative_count=1,
        confidence=0.9,
    )
    store.upsert("u1", preference)
    stored = store.get("u1", key)
    assert stored == preference


def test_get_all_returns_saved_preferences():
    store = PreferenceStore(":memory:")
    store.record_signal("u1", make_key(action_type="a"), positive=True)
    store.record_signal("u1", make_key(action_type="b"), positive=True)
    store.record_signal("u2", make_key(action_type="a"), positive=True)
    assert [p.key.action_type for p in store.get_all("u1")] == ["a", "b"]


def test_file_backed_store_persists_across_instances(tmp_path):
    db_path = str(tmp_path / "prefs.db")
    store = PreferenceStore(db_path)
    store.record_signal(
        "u1",
        make_key(),
        positive=True,
        explicit=True,
        default_decision=AutonomyDecision.ACT_NOTIFY,
    )
    store.close()

    reopened = PreferenceStore(db_path)
    preference = reopened.get("u1", make_key())
    assert preference is not None
    assert preference.positive_count == 1
    reopened.close()


# --- Learning rules -----------------------------------------------------------


def test_repeated_consistent_feedback_increases_confidence():
    store = PreferenceStore(":memory:")
    key = make_key()
    confidences = []
    for _ in range(10):
        p = store.record_signal(
            "u1",
            key,
            positive=True,
            explicit=True,
            default_decision=AutonomyDecision.ACT_NOTIFY,
        )
        confidences.append(p.confidence)
    assert confidences == sorted(confidences)
    assert confidences[-1] > confidences[0]
    assert p.preferred_decision == AutonomyDecision.ACT_NOTIFY
    assert p.positive_count == 10
    assert p.negative_count == 0


def test_ten_consistent_approvals_reach_upgrade_confidence():
    store = PreferenceStore(":memory:")
    key = make_key()
    for _ in range(10):
        p = store.record_signal(
            "u1",
            key,
            positive=True,
            explicit=True,
            default_decision=AutonomyDecision.ACT_NOTIFY,
        )
    assert p.confidence >= 0.8


def test_new_preference_starts_with_low_confidence():
    store = PreferenceStore(":memory:")
    p = store.record_signal(
        "u1",
        make_key(),
        positive=True,
        explicit=True,
        default_decision=AutonomyDecision.ACT_NOTIFY,
    )
    assert p.confidence < 0.5


def test_negative_feedback_pushes_toward_ask():
    store = PreferenceStore(":memory:")
    key = make_key()
    p = store.record_signal(
        "u1",
        key,
        positive=False,
        explicit=True,
        default_decision=AutonomyDecision.ACT_NOTIFY,
    )
    assert p.preferred_decision == AutonomyDecision.ASK
    assert p.negative_count == 1


def test_negative_feedback_quickly_reduces_autonomy_after_approvals():
    store = PreferenceStore(":memory:")
    key = make_key()
    for _ in range(10):
        p = store.record_signal(
            "u1",
            key,
            positive=True,
            explicit=True,
            default_decision=AutonomyDecision.ACT_NOTIFY,
        )
    assert p.confidence >= 0.8
    p = store.record_signal("u1", key, positive=False, explicit=True)
    assert p.confidence < 0.8  # one explicit "ask me" drops it below upgrade


def test_conflicting_feedback_lowers_confidence():
    store = PreferenceStore(":memory:")
    key = make_key()
    for _ in range(10):
        p = store.record_signal("u1", key, positive=True, explicit=True)
    high = p.confidence
    for _ in range(3):
        p = store.record_signal("u1", key, positive=False, explicit=True)
    assert p.confidence < high


def test_explicit_feedback_stronger_than_inferred():
    explicit = Preference(
        key=make_key(),
        preferred_decision=AutonomyDecision.ACT_NOTIFY,
        positive_count=10,
        negative_count=0,
        explicit_positive_count=10,
        explicit_negative_count=0,
    )
    inferred = Preference(
        key=make_key(),
        preferred_decision=AutonomyDecision.ACT_NOTIFY,
        positive_count=10,
        negative_count=0,
        explicit_positive_count=0,
        explicit_negative_count=0,
    )
    assert compute_confidence(explicit) > compute_confidence(inferred)
    assert INFERRED_WEIGHT < 1.0


def test_explicit_negative_stronger_than_inferred_negative():
    explicit = Preference(
        key=make_key(),
        preferred_decision=AutonomyDecision.ASK,
        positive_count=0,
        negative_count=10,
        explicit_positive_count=0,
        explicit_negative_count=10,
    )
    inferred = Preference(
        key=make_key(),
        preferred_decision=AutonomyDecision.ASK,
        positive_count=0,
        negative_count=10,
        explicit_positive_count=0,
        explicit_negative_count=0,
    )
    assert compute_confidence(explicit) > compute_confidence(inferred)
    assert EXPLICIT_NEGATIVE_WEIGHT > 1.0


def test_default_decision_kept_when_already_autonomous():
    pref = Preference(
        key=make_key(action_type="archive_newsletter"),
        preferred_decision=AutonomyDecision.SILENT,
        positive_count=1,
        negative_count=0,
        explicit_positive_count=1,
        explicit_negative_count=0,
    )
    assert derive_preferred_decision(pref) == AutonomyDecision.SILENT


def test_positive_feedback_without_default_falls_back_to_act_notify():
    store = PreferenceStore(":memory:")
    p = store.record_signal("u1", make_key(), positive=True, explicit=True)
    assert p.preferred_decision == AutonomyDecision.ACT_NOTIFY


# --- Lookup -------------------------------------------------------------------


def test_lookup_falls_back_to_wildcard_sender():
    store = PreferenceStore(":memory:")
    store.record_signal(
        "u1",
        make_key(sender_category="*"),
        positive=True,
        explicit=True,
        default_decision=AutonomyDecision.ACT_NOTIFY,
    )
    found = store.lookup("u1", "schedule_meeting", "alice@example.com")
    assert found is not None
    assert found.key.sender_category == "*"


def test_lookup_prefers_exact_sender_over_wildcard():
    store = PreferenceStore(":memory:")
    store.record_signal(
        "u1",
        make_key(sender_category="alice@example.com"),
        positive=True,
        explicit=True,
    )
    store.record_signal(
        "u1",
        make_key(sender_category="*"),
        positive=True,
        explicit=True,
    )
    found = store.lookup("u1", "schedule_meeting", "alice@example.com")
    assert found.key.sender_category == "alice@example.com"


def test_lookup_returns_none_when_no_preference():
    store = PreferenceStore(":memory:")
    assert store.lookup("u1", "pay_invoice", "vendor@example.com") is None
