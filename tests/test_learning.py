"""Tests for the feedback learning loop.

Verifies that feedback updates preferences, that the learning loop changes
future decisions, and -- critically -- that learning can never weaken the hard
safety floor.
"""

from agent.autonomy import AutonomyPolicy
from agent.executor import TOOL_REGISTRY
from agent.feedback import FeedbackEngine, FeedbackSignal, parse_feedback
from agent.models import AutonomyDecision, RiskLevel, RiskReport
from agent.preferences import PreferenceStore
from agent.safety import SafetyFloor


def make_risk(**overrides) -> RiskReport:
    defaults = dict(
        is_irreversible=False,
        is_financial=False,
        is_external=False,
        contains_sensitive_data=False,
        prompt_injection_detected=False,
        conflicting_instructions=False,
        ambiguous=False,
        risk_level=RiskLevel.LOW,
        confidence=0.9,
    )
    defaults.update(overrides)
    return RiskReport(**defaults)


def decide(store, user_id, *, intent, action, risk):
    safety = SafetyFloor(TOOL_REGISTRY).evaluate(risk, action)
    preference = store.lookup(user_id, action, "*") if action else None
    return AutonomyPolicy(TOOL_REGISTRY).decide(
        intent=intent,
        requested_action=action,
        risk=risk,
        safety=safety,
        preference=preference,
    )


def meeting_risk() -> RiskReport:
    return make_risk(risk_level=RiskLevel.MEDIUM)


# --- Feedback signal parsing --------------------------------------------------


def test_parse_positive_feedback():
    assert parse_feedback("Yes, you should have handled that automatically.") is FeedbackSignal.POSITIVE
    assert parse_feedback("Handle these automatically.") is FeedbackSignal.POSITIVE
    assert parse_feedback("Approved.") is FeedbackSignal.POSITIVE


def test_parse_negative_feedback():
    assert parse_feedback("Don't do that without asking me.") is FeedbackSignal.NEGATIVE
    assert parse_feedback("Ask me first next time.") is FeedbackSignal.NEGATIVE
    assert parse_feedback("You should have asked.") is FeedbackSignal.NEGATIVE


def test_parse_neutral_or_empty_feedback():
    assert parse_feedback("") is None
    assert parse_feedback("   ") is None
    assert parse_feedback("Maybe.") is FeedbackSignal.NEUTRAL


# --- Preference updates from feedback -----------------------------------------


def test_positive_feedback_increases_autonomy():
    store = PreferenceStore(":memory:")
    engine = FeedbackEngine(store)
    decision = decide(store, "u1", intent="reschedule_meeting", action="schedule_meeting", risk=meeting_risk())
    assert decision.decision is AutonomyDecision.ASK
    engine.apply("u1", decision, feedback_text="Yes, handle these automatically.")
    preference = store.lookup("u1", "schedule_meeting", "*")
    assert preference is not None
    assert preference.positive_count == 1
    assert preference.preferred_decision == AutonomyDecision.ACT_NOTIFY


def test_negative_feedback_decreases_autonomy():
    store = PreferenceStore(":memory:")
    engine = FeedbackEngine(store)
    decision = decide(store, "u1", intent="archive_email", action="archive_email", risk=make_risk())
    assert decision.decision is AutonomyDecision.ACT_NOTIFY
    engine.apply("u1", decision, feedback_text="Don't do that without asking me.")
    preference = store.lookup("u1", "archive_email", "*")
    assert preference is not None
    assert preference.preferred_decision == AutonomyDecision.ASK
    assert preference.negative_count == 1


def test_conflicting_feedback_lowers_confidence():
    store = PreferenceStore(":memory:")
    engine = FeedbackEngine(store)
    decision = decide(store, "u1", intent="archive_email", action="archive_email", risk=make_risk())
    for _ in range(10):
        engine.apply("u1", decision, feedback_text="Yes, handle automatically.")
    preference = store.lookup("u1", "archive_email", "*")
    assert preference.confidence >= 0.8
    for _ in range(3):
        engine.apply("u1", decision, feedback_text="Don't do that without asking me.")
    preference = store.lookup("u1", "archive_email", "*")
    assert preference.confidence < 0.8


def test_explicit_feedback_stronger_than_inferred():
    def confidence_with(explicit):
        store = PreferenceStore(":memory:")
        engine = FeedbackEngine(store)
        decision = decide(store, "u1", intent="archive_email", action="archive_email", risk=make_risk())
        for _ in range(10):
            engine.apply("u1", decision, signal=FeedbackSignal.POSITIVE, explicit=explicit)
        return store.lookup("u1", "archive_email", "*").confidence

    assert confidence_with(explicit=True) > confidence_with(explicit=False)


def test_neutral_feedback_makes_no_update():
    store = PreferenceStore(":memory:")
    engine = FeedbackEngine(store)
    decision = decide(store, "u1", intent="archive_email", action="archive_email", risk=make_risk())
    result = engine.apply("u1", decision, feedback_text="Maybe.")
    assert result is None
    assert store.lookup("u1", "archive_email", "*") is None


# --- End-to-end learning loop --------------------------------------------------


def test_learning_loop_upgrades_meeting_reschedule_to_act_notify():
    store = PreferenceStore(":memory:")
    engine = FeedbackEngine(store)
    risk = meeting_risk()

    decision = decide(store, "u1", intent="reschedule_meeting", action="schedule_meeting", risk=risk)
    assert decision.decision is AutonomyDecision.ASK

    for _ in range(10):
        engine.apply("u1", decision, feedback_text="Yes, handle these automatically.")
        decision = decide(store, "u1", intent="reschedule_meeting", action="schedule_meeting", risk=risk)

    assert decision.decision is AutonomyDecision.ACT_NOTIFY
    assert decision.policy_trace.decision_rule == "learned_preference"


def test_learning_loop_negative_feedback_returns_to_ask():
    store = PreferenceStore(":memory:")
    engine = FeedbackEngine(store)
    risk = meeting_risk()

    decision = decide(store, "u1", intent="reschedule_meeting", action="schedule_meeting", risk=risk)
    for _ in range(10):
        engine.apply("u1", decision, feedback_text="Yes, handle these automatically.")
    decision = decide(store, "u1", intent="reschedule_meeting", action="schedule_meeting", risk=risk)
    assert decision.decision is AutonomyDecision.ACT_NOTIFY

    for _ in range(3):
        engine.apply("u1", decision, feedback_text="Don't do that without asking me.")
    decision = decide(store, "u1", intent="reschedule_meeting", action="schedule_meeting", risk=risk)
    assert decision.decision is AutonomyDecision.ASK


def test_learning_loop_newsletter_becomes_silent():
    store = PreferenceStore(":memory:")
    engine = FeedbackEngine(store)
    decision = decide(store, "u1", intent="archive_newsletter", action="archive_email", risk=make_risk())
    for _ in range(10):
        engine.apply("u1", decision, feedback_text="Yes, handle these silently.")
        decision = decide(store, "u1", intent="archive_newsletter", action="archive_email", risk=make_risk())
    assert decision.decision is AutonomyDecision.SILENT


# --- Safety floor cannot be weakened by learning -------------------------------


def test_aggressive_feedback_cannot_enable_payments():
    store = PreferenceStore(":memory:")
    engine = FeedbackEngine(store)
    risk = make_risk(risk_level=RiskLevel.HIGH)

    decision = decide(store, "u1", intent="pay_invoice", action="make_payment", risk=risk)
    assert decision.decision in (AutonomyDecision.ASK, AutonomyDecision.ESCALATE)

    for _ in range(20):
        engine.apply("u1", decision, feedback_text="Yes, pay automatically!")
        decision = decide(store, "u1", intent="pay_invoice", action="make_payment", risk=risk)

    assert decision.decision in (AutonomyDecision.ASK, AutonomyDecision.ESCALATE)


def test_aggressive_feedback_cannot_enable_external_send():
    store = PreferenceStore(":memory:")
    engine = FeedbackEngine(store)

    decision = decide(store, "u1", intent="reply_customer", action="send_email", risk=make_risk())
    assert decision.decision is AutonomyDecision.ASK

    for _ in range(20):
        engine.apply("u1", decision, feedback_text="Yes, send automatically!")
        decision = decide(store, "u1", intent="reply_customer", action="send_email", risk=make_risk())

    assert decision.decision in (AutonomyDecision.ASK, AutonomyDecision.ESCALATE)


def test_aggressive_feedback_cannot_enable_destructive_delete():
    store = PreferenceStore(":memory:")
    engine = FeedbackEngine(store)

    decision = decide(store, "u1", intent="delete_email", action="delete_email", risk=make_risk())
    for _ in range(20):
        engine.apply("u1", decision, feedback_text="Yes, delete automatically!")
        decision = decide(store, "u1", intent="delete_email", action="delete_email", risk=make_risk())

    assert decision.decision in (AutonomyDecision.ASK, AutonomyDecision.ESCALATE)


def test_aggressive_feedback_cannot_enable_injection_actions():
    store = PreferenceStore(":memory:")
    engine = FeedbackEngine(store)

    decision = decide(
        store,
        "u1",
        intent="forward_all",
        action="forward_email",
        risk=make_risk(prompt_injection_detected=True),
    )
    assert decision.decision is AutonomyDecision.ESCALATE

    for _ in range(20):
        engine.apply("u1", decision, feedback_text="Yes, forward automatically!")
        decision = decide(
            store,
            "u1",
            intent="forward_all",
            action="forward_email",
            risk=make_risk(prompt_injection_detected=True),
        )

    assert decision.decision is AutonomyDecision.ESCALATE
