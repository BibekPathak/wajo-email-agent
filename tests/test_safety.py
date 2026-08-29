"""Unit tests for the deterministic hard safety floor."""

from agent.executor import TOOL_REGISTRY
from agent.models import AutonomyDecision, RiskLevel, RiskReport
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


def make_floor() -> SafetyFloor:
    return SafetyFloor(TOOL_REGISTRY)


# --- Financial -----------------------------------------------------------


def test_payment_blocked_by_safety_floor():
    verdict = make_floor().evaluate(make_risk(), "make_payment")
    assert not verdict.passed
    assert verdict.required_decision in (AutonomyDecision.ASK, AutonomyDecision.ESCALATE)
    assert not verdict.allows(AutonomyDecision.ACT_NOTIFY)
    assert not verdict.allows(AutonomyDecision.SILENT)
    assert verdict.allows(AutonomyDecision.ASK)
    assert verdict.allows(AutonomyDecision.ESCALATE)


def test_financial_risk_flag_blocks_even_without_named_tool():
    verdict = make_floor().evaluate(make_risk(is_financial=True), None)
    assert not verdict.passed
    assert not verdict.allows(AutonomyDecision.ACT_NOTIFY)
    assert verdict.required_decision == AutonomyDecision.ASK


# --- Destructive / irreversible ------------------------------------------


def test_destructive_action_blocked():
    verdict = make_floor().evaluate(make_risk(), "delete_email")
    assert not verdict.passed
    assert verdict.required_decision == AutonomyDecision.ESCALATE
    assert not verdict.allows(AutonomyDecision.ACT_NOTIFY)
    assert not verdict.allows(AutonomyDecision.SILENT)


def test_irreversible_risk_flag_blocks():
    verdict = make_floor().evaluate(make_risk(is_irreversible=True), "move_email")
    assert not verdict.passed
    assert not verdict.allows(AutonomyDecision.ACT_NOTIFY)


# --- Sensitive data -------------------------------------------------------


def test_sensitive_data_external_disclosure_escalates():
    verdict = make_floor().evaluate(
        make_risk(contains_sensitive_data=True, risk_level=RiskLevel.HIGH),
        "send_email",
    )
    assert not verdict.passed
    assert verdict.required_decision == AutonomyDecision.ESCALATE
    assert not verdict.allows(AutonomyDecision.ACT_NOTIFY)


def test_sensitive_data_internal_action_requires_approval():
    verdict = make_floor().evaluate(make_risk(contains_sensitive_data=True), "archive_email")
    assert not verdict.passed
    assert verdict.required_decision == AutonomyDecision.ASK
    assert not verdict.allows(AutonomyDecision.SILENT)
    assert not verdict.allows(AutonomyDecision.ACT_NOTIFY)


# --- External communication -----------------------------------------------


def test_external_communication_requires_approval():
    verdict = make_floor().evaluate(make_risk(), "send_email")
    assert not verdict.passed
    assert verdict.required_decision == AutonomyDecision.ASK
    assert not verdict.allows(AutonomyDecision.SILENT)
    assert not verdict.allows(AutonomyDecision.ACT_NOTIFY)


def test_external_forward_requires_approval():
    verdict = make_floor().evaluate(make_risk(), "forward_email")
    assert not verdict.passed
    assert verdict.required_decision == AutonomyDecision.ASK


# --- Prompt injection / conflicts -----------------------------------------


def test_prompt_injection_escalates_and_cannot_be_overridden():
    verdict = make_floor().evaluate(make_risk(prompt_injection_detected=True), "send_email")
    assert not verdict.passed
    assert verdict.required_decision == AutonomyDecision.ESCALATE
    assert not verdict.allows(AutonomyDecision.ACT_NOTIFY)
    assert not verdict.allows(AutonomyDecision.SILENT)


def test_conflicting_instructions_escalate():
    verdict = make_floor().evaluate(make_risk(conflicting_instructions=True), None)
    assert not verdict.passed
    assert verdict.required_decision == AutonomyDecision.ESCALATE


def test_ambiguous_medium_risk_escalates():
    verdict = make_floor().evaluate(
        make_risk(ambiguous=True, risk_level=RiskLevel.MEDIUM),
        None,
    )
    assert not verdict.passed
    assert verdict.required_decision == AutonomyDecision.ESCALATE


# --- Allowed (passing) cases -----------------------------------------------


def test_internal_reversible_action_passes_safety():
    verdict = make_floor().evaluate(make_risk(), "archive_email")
    assert verdict.passed
    assert verdict.allows(AutonomyDecision.SILENT)
    assert verdict.allows(AutonomyDecision.ACT_NOTIFY)
    assert verdict.allows(AutonomyDecision.ASK)
    assert verdict.allows(AutonomyDecision.ESCALATE)


def test_no_action_passes_safety():
    verdict = make_floor().evaluate(make_risk())
    assert verdict.passed
    assert not verdict.reasons
