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


# --- Phase 2: intent-category defaults ----------------------------------------


def test_unknown_intent_escalates():
    result = make_decision(intent="unrecognized_intent", action=None)
    assert result.decision is AutonomyDecision.ESCALATE
    assert result.policy_trace.decision_rule == "ambiguous"


def test_ambiguous_risk_flag_escalates():
    result = make_decision(
        intent="reply_customer",
        action=None,
        risk=make_risk(ambiguous=True),
    )
    assert result.decision is AutonomyDecision.ESCALATE
    assert result.policy_trace.decision_rule == "ambiguous"


def test_high_risk_escalates_even_for_safe_internal_action():
    result = make_decision(
        intent="archive_email",
        action="archive_email",
        risk=make_risk(risk_level=RiskLevel.HIGH),
    )
    assert result.decision is AutonomyDecision.ESCALATE


def test_financial_category_asks_even_without_risk_flag():
    # Category-driven: even if the risk layer missed the flag, the intent
    # category alone forces ASK.
    result = make_decision(intent="pay_invoice", action=None, risk=make_risk())
    assert result.decision is AutonomyDecision.ASK
    assert result.policy_trace.decision_rule == "financial"


def test_sensitive_category_escalates():
    result = make_decision(intent="disclose_data", action=None)
    assert result.decision is AutonomyDecision.ESCALATE
    assert result.policy_trace.decision_rule == "sensitive"


def test_external_category_asks_by_default():
    result = make_decision(intent="confirm_terms", action="send_email")
    assert result.decision is AutonomyDecision.ASK
    assert result.policy_trace.decision_rule == "safety_floor"


def test_medium_risk_internal_asks_by_default():
    result = make_decision(
        intent="cancel_meeting",
        action="cancel_meeting",
        risk=make_risk(risk_level=RiskLevel.MEDIUM),
    )
    assert result.decision is AutonomyDecision.ASK


# --- Phase 2: preference upgrade gating ---------------------------------------


def test_preference_cannot_upgrade_external_category():
    preference = Preference(
        key=PreferenceKey(action_type="confirm_terms"),
        preferred_decision=AutonomyDecision.ACT_NOTIFY,
        positive_count=20,
        negative_count=0,
        confidence=0.99,
    )
    result = make_decision(
        intent="confirm_terms",
        action="send_email",
        preference=preference,
    )
    assert result.decision in (AutonomyDecision.ASK, AutonomyDecision.ESCALATE)


def test_preference_cannot_upgrade_financial_category():
    preference = Preference(
        key=PreferenceKey(action_type="pay_invoice"),
        preferred_decision=AutonomyDecision.ACT_NOTIFY,
        positive_count=50,
        negative_count=0,
        confidence=0.99,
    )
    result = make_decision(intent="pay_invoice", action=None, preference=preference)
    assert result.decision is AutonomyDecision.ASK


def test_preference_can_increase_oversight_on_no_op():
    # A user may prefer to be asked even about background actions.
    preference = Preference(
        key=PreferenceKey(action_type="archive_newsletter"),
        preferred_decision=AutonomyDecision.ASK,
        positive_count=9,
        negative_count=1,
        confidence=0.85,
    )
    result = make_decision(
        intent="archive_newsletter",
        action="archive_email",
        preference=preference,
    )
    assert result.decision is AutonomyDecision.ASK


def test_preference_can_make_low_risk_internal_silent():
    preference = Preference(
        key=PreferenceKey(action_type="archive_email"),
        preferred_decision=AutonomyDecision.SILENT,
        positive_count=10,
        negative_count=0,
        confidence=0.95,
    )
    result = make_decision(
        intent="archive_email",
        action="archive_email",
        preference=preference,
    )
    assert result.decision is AutonomyDecision.SILENT
    assert result.policy_trace.decision_rule == "learned_preference"


def test_learned_preference_rule_recorded_in_trace():
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
    assert result.policy_trace.decision_rule == "learned_preference"


def test_safety_block_has_high_confidence():
    result = make_decision(
        intent="pay_invoice",
        action="make_payment",
        risk=make_risk(confidence=0.4),
    )
    assert result.confidence >= 0.9


# --- Phase 2: all four decisions are reachable --------------------------------


def test_all_four_decisions_reachable():
    silent = make_decision(intent="classify_email", action=None)
    act_notify = make_decision(intent="archive_email", action="archive_email")
    ask = make_decision(intent="reply_customer", action="send_email")
    escalate = make_decision(intent="disclose_data", action=None)
    assert {silent.decision, act_notify.decision, ask.decision, escalate.decision} == set(
        AutonomyDecision
    )
