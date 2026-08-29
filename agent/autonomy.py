"""Autonomy policy.

Phase 1 ships a deterministic, preference-aware policy. It may only choose a
decision the safety floor allows. Learned preferences can select within the
safety floor's allowed set -- for example upgrading ASK to ACT_NOTIFY for a
low-risk reversible internal action once confidence is high enough -- but they
can never expand the allowed set.

The invariant enforced everywhere is:

    if not safety.allows(decision):
        decision = safety.required_decision or ASK
"""

from __future__ import annotations

from typing import Optional

from .models import (
    AutonomyDecision,
    DecisionResult,
    PolicyTrace,
    Preference,
    RiskLevel,
    RiskReport,
    SafetyVerdict,
    ToolPolicy,
)

# Intents that require no user-visible action and are handled in the background.
SILENT_INTENTS = frozenset({"classify_email", "archive_newsletter", "deduplicate"})

# Minimum preference confidence before a learned preference may upgrade a
# default ASK decision into an autonomous one.
PREFERENCE_UPGRADE_THRESHOLD = 0.8


class AutonomyPolicy:
    """Decides how much autonomy to take for a given situation.

    The policy must only ever pick a decision the :class:`SafetyFloor` allows.
    """

    def __init__(self, tool_registry: Optional[dict[str, ToolPolicy]] = None) -> None:
        self._tools = tool_registry if tool_registry is not None else {}

    def decide(
        self,
        *,
        intent: str,
        requested_action: Optional[str],
        risk: RiskReport,
        safety: SafetyVerdict,
        preference: Optional[Preference] = None,
    ) -> DecisionResult:
        tool = self._tools.get(requested_action or "")

        if not safety.passed:
            decision = safety.required_decision or AutonomyDecision.ASK
            reason = "; ".join(safety.reasons) or "Blocked by the hard safety floor."
            confidence = max(0.9, risk.confidence)
        elif risk.prompt_injection_detected or risk.conflicting_instructions:
            decision = AutonomyDecision.ESCALATE
            reason = "Email content is untrusted; refusing instructions embedded in the email."
            confidence = max(0.9, risk.confidence)
        elif requested_action is None or intent in SILENT_INTENTS:
            decision = AutonomyDecision.SILENT
            reason = "No user-visible action required; handled in the background."
            confidence = risk.confidence
        else:
            decision, reason, confidence = self._default_for(risk, tool)
            if preference is not None and preference.preferred_decision is not None:
                if preference.confidence >= PREFERENCE_UPGRADE_THRESHOLD:
                    proposed = preference.preferred_decision
                    if safety.allows(proposed):
                        decision = proposed
                        reason = "Learned user preference supports this level of autonomy."
                        confidence = preference.confidence

        # The safety floor can never be overridden by anything upstream.
        if not safety.allows(decision):
            decision = safety.required_decision or AutonomyDecision.ASK

        recommended = (
            requested_action
            if decision in (AutonomyDecision.ACT_NOTIFY, AutonomyDecision.ASK)
            else None
        )

        trace = PolicyTrace(
            intent=intent,
            risk_checks=risk.risk_checks(),
            safety_floor={
                "passed": safety.passed,
                "required_decision": (
                    safety.required_decision.value if safety.required_decision else None
                ),
                "allowed_decisions": [d.value for d in safety.allowed_decisions],
            },
            preference=preference.model_dump(mode="json") if preference is not None else None,
            final_decision=decision.value,
            reason=reason,
        )

        return DecisionResult(
            decision=decision,
            reason=reason,
            confidence=confidence,
            safety_constraints=safety.constraints,
            preference_match=preference,
            recommended_action=recommended,
            policy_trace=trace,
        )

    def _default_for(
        self, risk: RiskReport, tool: Optional[ToolPolicy]
    ) -> tuple[AutonomyDecision, str, float]:
        """Default decision before preferences are consulted.

        Conservative by design: anything that is not clearly low-risk and
        reversible defaults to ASK, so autonomy only grows through learning.
        """
        if risk.ambiguous or risk.risk_level is RiskLevel.HIGH:
            return (
                AutonomyDecision.ESCALATE,
                "Situation is ambiguous or high-risk; requires human judgment.",
                risk.confidence,
            )
        if tool is None:
            return (
                AutonomyDecision.SILENT,
                "No tool action requested.",
                risk.confidence,
            )
        if tool.reversible and not tool.external and not tool.requires_approval:
            if risk.risk_level is RiskLevel.LOW:
                return (
                    AutonomyDecision.ACT_NOTIFY,
                    "Low-risk reversible internal action.",
                    risk.confidence,
                )
            return (
                AutonomyDecision.ASK,
                "Reversible internal action that may still need user approval.",
                risk.confidence,
            )
        return (
            AutonomyDecision.ASK,
            "Action requires user approval or is not fully covered by policy.",
            risk.confidence,
        )
