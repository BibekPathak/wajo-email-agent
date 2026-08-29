"""Unit tests for the basic autonomy policy.

These tests verify the four-way decision mapping and -- critically -- that
preferences can only select within the safety floor's allowed set.
"""

from agent.autonomy import AutonomyPolicy
from agent.executor import TOOL_REGISTRY
from agent.models import AutonomyDecision, Preference, PreferenceKey, RiskLevel, RiskReport
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


def make_decision(*, intent="generic", action=None, risk=None, preference=None):
    risk = risk if risk is not None else make_risk()
    safety = SafetyFloor(TOOL_REGISTRY).evaluate(risk, action)
    return AutonomyPolicy(TOOL_REGISTRY).decide(
        intent=intent,
        requested_action=action,
        risk=risk,
        safety=safety,
        preference=preference,
    )


# --- The four decisions -----------------------------------------------------


def test_no_action_is_silent():
    result = make_decision(intent="classify_email", action=None)
    assert result.decision is AutonomyDecision.SILENT
    assert result.recommended_action is None


def test_low_risk_internal_action_is_act_notify():
    result = make_decision(intent="archive_email", action="archive_email")
    assert result.decision is AutonomyDecision.ACT_NOTIFY
    assert result.recommended_action == "archive_email"


def test_newsletter_archive_is_silent():
    result = make_decision(intent="archive_newsletter", action="archive_email")
    assert result.decision is AutonomyDecision.SILENT


def test_meeting_reschedule_asks_by_default():
    risk = make_risk(risk_level=RiskLevel.MEDIUM)
    result = make_decision(intent="reschedule_meeting", action="schedule_meeting", risk=risk)
    assert result.decision is AutonomyDecision.ASK
    assert result.recommended_action == "schedule_meeting"


def test_external_send_asks():
    result = make_decision(intent="reply_customer", action="send_email")
    assert result.decision is AutonomyDecision.ASK
    assert result.recommended_action == "send_email"


def test_payment_is_never_autonomous():
    result = make_decision(
        intent="pay_invoice",
        action="make_payment",
        risk=make_risk(risk_level=RiskLevel.HIGH),
    )
    assert result.decision in (AutonomyDecision.ASK, AutonomyDecision.ESCALATE)


def test_prompt_injection_escalates():
    result = make_decision(
        intent="forward_all",
        action="forward_email",
        risk=make_risk(prompt_injection_detected=True),
    )
    assert result.decision is AutonomyDecision.ESCALATE


# --- Safety can never be overridden by preferences --------------------------


def test_aggressive_preference_cannot_override_payment():
    preference = Preference(
        key=PreferenceKey(action_type="make_payment"),
        preferred_decision=AutonomyDecision.ACT_NOTIFY,
        positive_count=100,
        negative_count=0,
        confidence=0.99,
    )
    result = make_decision(
        intent="pay_invoice",
        action="make_payment",
        risk=make_risk(risk_level=RiskLevel.HIGH),
        preference=preference,
    )
    assert result.decision in (AutonomyDecision.ASK, AutonomyDecision.ESCALATE)


def test_aggressive_preference_cannot_override_external_send():
    preference = Preference(
        key=PreferenceKey(action_type="send_email"),
        preferred_decision=AutonomyDecision.SILENT,
        positive_count=100,
        negative_count=0,
        confidence=0.99,
    )
    result = make_decision(
        intent="reply_customer",
        action="send_email",
        preference=preference,
    )
    assert result.decision in (AutonomyDecision.ASK, AutonomyDecision.ESCALATE)


def test_low_confidence_preference_does_not_upgrade():
    preference = Preference(
        key=PreferenceKey(action_type="schedule_meeting"),
        preferred_decision=AutonomyDecision.ACT_NOTIFY,
        positive_count=1,
        negative_count=0,
        confidence=0.5,
    )
    result = make_decision(
        intent="reschedule_meeting",
        action="schedule_meeting",
        risk=make_risk(risk_level=RiskLevel.MEDIUM),
        preference=preference,
    )
    assert result.decision is AutonomyDecision.ASK


def test_confident_preference_can_upgrade_safe_internal_action():
    preference = Preference(
        key=PreferenceKey(action_type="schedule_meeting"),
        preferred_decision=AutonomyDecision.ACT_NOTIFY,
        positive_count=8,
        negative_count=1,
        confidence=0.9,
    )
    result = make_decision(
        intent="reschedule_meeting",
        action="schedule_meeting",
        risk=make_risk(risk_level=RiskLevel.MEDIUM),
        preference=preference,
    )
    assert result.decision is AutonomyDecision.ACT_NOTIFY


# --- Policy trace -------------------------------------------------------------


def test_decision_carries_explainable_policy_trace():
    result = make_decision(intent="pay_invoice", action="make_payment")
    trace = result.policy_trace
    assert trace.intent == "pay_invoice"
    assert trace.risk_checks["financial"] is False
    assert trace.safety_floor["passed"] is False
    assert trace.safety_floor["allowed_decisions"] == ["ask", "escalate"]
    assert trace.final_decision == result.decision.value
    assert trace.reason
