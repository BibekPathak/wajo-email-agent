"""Tests for the simulated tool executor and tool permissions."""

import pytest

from agent.executor import (
    TOOL_ARGUMENTS,
    TOOL_REGISTRY,
    Executor,
    UnknownToolError,
)
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


def test_execute_records_call_with_all_fields():
    executor = Executor()
    call = executor.execute(
        "send_email",
        {"recipient": "alice@example.com", "subject": "Hi"},
        safety_allowed=True,
    )
    assert call.tool == "send_email"
    assert call.arguments["recipient"] == "alice@example.com"
    assert call.safety_allowed is True
    assert call.timestamp is not None
    assert call.result.startswith("Simulated send_email")
    assert executor.calls == [call]


def test_unknown_tool_raises():
    executor = Executor()
    with pytest.raises(UnknownToolError):
        executor.execute("fly_to_the_moon")


def test_unsafe_calls_detection():
    executor = Executor()
    executor.execute("archive_email", safety_allowed=True)
    executor.execute("delete_email", safety_allowed=False)  # gate refused
    executor.execute("make_payment", safety_allowed=False)
    assert len(executor.unsafe_calls()) == 2
    assert executor.is_safe is False


def test_is_safe_when_all_calls_allowed():
    executor = Executor()
    executor.execute("archive_email", safety_allowed=True)
    executor.execute("label_email", safety_allowed=True)
    assert executor.is_safe is True


def test_tools_executed_and_clear():
    executor = Executor()
    executor.execute("archive_email", safety_allowed=True)
    executor.execute("label_email", safety_allowed=True)
    assert executor.tools_executed() == ["archive_email", "label_email"]
    executor.clear()
    assert executor.calls == []


def test_audit_log_persists_to_jsonl(tmp_path):
    path = tmp_path / "audit.jsonl"
    executor = Executor(audit_path=str(path))
    executor.execute("archive_email", {"message_id": "msg-1"}, safety_allowed=True)
    executor.execute("send_email", {"recipient": "bob@x.com"}, safety_allowed=True)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    import json

    first = json.loads(lines[0])
    assert first["tool"] == "archive_email"
    assert first["safety_allowed"] is True
    assert first["arguments"]["message_id"] == "msg-1"


def test_simulate_flags_missing_required_arguments():
    executor = Executor()
    call = executor.execute("make_payment", {"amount": "100"}, safety_allowed=True)
    assert "missing" in call.result
    assert "currency" in call.result


def test_every_tool_declares_full_policy():
    for name, policy in TOOL_REGISTRY.items():
        assert policy.name == name
        assert policy.reversible in (True, False)
        assert policy.external in (True, False)
        assert policy.financial in (True, False)
        assert policy.destructive in (True, False)
        assert policy.requires_approval in (True, False)
        assert policy.description


@pytest.mark.parametrize(
    "tool, expect_autonomous",
    [
        ("archive_email", True),
        ("label_email", True),
        ("move_email", True),
        ("create_draft", True),
        ("schedule_meeting", True),
        ("cancel_meeting", True),
        ("delete_email", False),
        ("send_email", False),
        ("forward_email", False),
        ("disclose_data", False),
        ("make_payment", False),
    ],
)
def test_tool_policy_autonomy_permissions(tool, expect_autonomous):
    """Safety composability: each tool's policy decides whether autonomous
    execution is ever allowed, independent of any preference."""
    verdict = SafetyFloor(TOOL_REGISTRY).evaluate(make_risk(), tool)
    if expect_autonomous:
        assert verdict.allows(AutonomyDecision.ACT_NOTIFY)
    else:
        assert not verdict.allows(AutonomyDecision.ACT_NOTIFY)
        assert not verdict.allows(AutonomyDecision.SILENT)


def test_financial_and_external_tools_require_approval():
    for name in ("send_email", "forward_email", "disclose_data", "make_payment"):
        assert TOOL_REGISTRY[name].requires_approval is True


def test_all_argument_specs_are_nonempty():
    for name, args in TOOL_ARGUMENTS.items():
        assert name in TOOL_REGISTRY
        assert args, f"{name} has no documented arguments"
