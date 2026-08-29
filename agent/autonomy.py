"""Autonomy engine.

Phase 2 ships the full four-way autonomy decision procedure with explicit,
explainable decision semantics:

    SILENT      no user-visible action; handled in the background
    ACT_NOTIFY  safe, reversible action; act and notify afterward
    ASK         action with external consequences; ask first
    ESCALATE    cannot resolve safely; hand to a human

The engine is a deterministic rule cascade:

    1. safety floor block          -> required decision (never overridable)
    2. prompt injection / conflict -> ESCALATE
    3. ambiguity                   -> ESCALATE
    4. intent category             -> default decision
    5. confident learned preference -> upgrade autonomy, but ONLY within the
       safety floor's allowed set and only for autonomy-eligible categories
    6. fallback                    -> ASK

The safety invariant enforced everywhere:

    if not safety.allows(decision):
        decision = safety.required_decision or ASK

Learned preferences may select within the safety floor's allowed set; they can
never expand it.
"""

from __future__ import annotations

from enum import Enum
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

# ---------------------------------------------------------------------------
# Intent categories
# ---------------------------------------------------------------------------


class IntentCategory(str, Enum):
    """Broad semantic class of an email intent.

    The autonomy engine maps intents to categories and categories to default
    autonomy levels. Unknown intents are treated as ambiguous (conservative):
    the agent asks or escalates rather than guessing.
    """

    NO_OP = "no_op"
    INTERNAL_LOW = "internal_low"
    INTERNAL_MEDIUM = "internal_medium"
    EXTERNAL = "external"
    FINANCIAL = "financial"
    SENSITIVE = "sensitive"
    AMBIGUOUS = "ambiguous"


INTENT_CATEGORIES: dict[str, IntentCategory] = {
    # Background / informational -- no user-visible action.
    "classify_email": IntentCategory.NO_OP,
    "deduplicate": IntentCategory.NO_OP,
    "archive_newsletter": IntentCategory.NO_OP,
    "label_newsletter": IntentCategory.NO_OP,
    # Low-risk reversible internal actions.
    "archive_email": IntentCategory.INTERNAL_LOW,
    "label_email": IntentCategory.INTERNAL_LOW,
    "move_email": IntentCategory.INTERNAL_LOW,
    "create_draft": IntentCategory.INTERNAL_LOW,
    "schedule_internal_reminder": IntentCategory.INTERNAL_LOW,
    # Reversible internal actions that usually need a judgment call.
    "schedule_meeting": IntentCategory.INTERNAL_MEDIUM,
    "reschedule_meeting": IntentCategory.INTERNAL_MEDIUM,
    "cancel_meeting": IntentCategory.INTERNAL_MEDIUM,
    "accept_invitation": IntentCategory.INTERNAL_MEDIUM,
    # External communication -- ask first.
    "reply_customer": IntentCategory.EXTERNAL,
    "send_external_email": IntentCategory.EXTERNAL,
    "confirm_terms": IntentCategory.EXTERNAL,
    "forward_email": IntentCategory.EXTERNAL,
    "share_info": IntentCategory.EXTERNAL,
    # Financial -- protected by the hard safety floor.
    "pay_invoice": IntentCategory.FINANCIAL,
    "transfer_money": IntentCategory.FINANCIAL,
    "approve_invoice": IntentCategory.FINANCIAL,
    "make_purchase": IntentCategory.FINANCIAL,
    # Sensitive data handling.
    "disclose_data": IntentCategory.SENSITIVE,
    "share_confidential": IntentCategory.SENSITIVE,
}


def classify_intent(intent: str) -> IntentCategory:
    """Classify an intent string into a category. Unknown -> AMBIGUOUS."""
    return INTENT_CATEGORIES.get(intent, IntentCategory.AMBIGUOUS)


# Categories whose base default may be autonomous (subject to the safety floor).
# Learned preferences are allowed to upgrade autonomy ONLY within these.
AUTONOMY_ELIGIBLE_CATEGORIES = frozenset(
    {
        IntentCategory.NO_OP,
        IntentCategory.INTERNAL_LOW,
        IntentCategory.INTERNAL_MEDIUM,
    }
)

# Minimum preference confidence before a learned preference may upgrade a
# default ASK into an autonomous decision.
PREFERENCE_UPGRADE_THRESHOLD = 0.8

# Confidence floor for deterministic safety blocks: a blocked action is blocked
# with high confidence because the rule, not the semantic analysis, decides.
SAFETY_BLOCK_CONFIDENCE = 0.9


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
        """Produce a decision for a fully analyzed email situation."""
        category = classify_intent(intent)
        tool = self._tools.get(requested_action or "") if requested_action else None

        # Rule 1: the hard safety floor wins outright.
        if not safety.passed:
            decision = safety.required_decision or AutonomyDecision.ASK
            rule = "safety_floor"
            reason = "; ".join(safety.reasons) or "Blocked by the hard safety floor."
            confidence = max(SAFETY_BLOCK_CONFIDENCE, risk.confidence)
        # Rule 2: untrusted email content can never steer the agent.
        elif risk.prompt_injection_detected or risk.conflicting_instructions:
            decision, rule, reason, confidence = (
                AutonomyDecision.ESCALATE,
                "untrusted_content",
                "Email content is untrusted; refusing instructions embedded in the email.",
                max(SAFETY_BLOCK_CONFIDENCE, risk.confidence),
            )
        # Rule 3: ambiguity requires a human.
        elif category is IntentCategory.AMBIGUOUS or risk.ambiguous:
            decision, rule, reason, confidence = (
                AutonomyDecision.ESCALATE,
                "ambiguous",
                "Cannot determine intent safely; requires human judgment.",
                risk.confidence,
            )
        else:
            # Rule 4: default by intent category (conservative).
            decision, rule, reason, confidence = self._default_for_category(
                category, risk, tool
            )

            # Rule 5: a confident learned preference may upgrade autonomy, but
            # only for autonomy-eligible categories and only within the safety
            # floor's allowed set. It can never turn an external, financial, or
            # sensitive action autonomous.
            if (
                preference is not None
                and preference.preferred_decision is not None
                and category in AUTONOMY_ELIGIBLE_CATEGORIES
                and preference.confidence >= PREFERENCE_UPGRADE_THRESHOLD
                and safety.allows(preference.preferred_decision)
            ):
                decision = preference.preferred_decision
                rule = "learned_preference"
                reason = "Learned user preference supports this level of autonomy."
                confidence = preference.confidence

        # Safety invariant: nothing upstream can ever expand the allowed set.
        if not safety.allows(decision):
            decision = safety.required_decision or AutonomyDecision.ASK
            rule = "safety_floor"

        return self._emit(
            intent=intent,
            requested_action=requested_action,
            risk=risk,
            safety=safety,
            preference=preference,
            decision=decision,
            rule=rule,
            reason=reason,
            confidence=confidence,
        )

    # -----------------------------------------------------------------------

    def _default_for_category(
        self,
        category: IntentCategory,
        risk: RiskReport,
        tool: Optional[ToolPolicy],
    ) -> tuple[AutonomyDecision, str, str, float]:
        """Default (pre-learning) decision for a category.

        Conservative by design: anything not clearly low-risk and reversible
        defaults to ASK, so autonomy only grows through learning.
        """
        if risk.risk_level is RiskLevel.HIGH:
            return (
                AutonomyDecision.ESCALATE,
                "high_risk",
                "High-risk situation; requires human judgment.",
                risk.confidence,
            )

        if category is IntentCategory.NO_OP:
            return (
                AutonomyDecision.SILENT,
                "no_op",
                "No user-visible action required; handled in the background.",
                risk.confidence,
            )

        if category is IntentCategory.INTERNAL_LOW:
            if (
                tool is not None
                and tool.reversible
                and not tool.external
                and not tool.requires_approval
            ):
                return (
                    AutonomyDecision.ACT_NOTIFY,
                    "internal_low_risk",
                    "Low-risk reversible internal action.",
                    risk.confidence,
                )
            return (
                AutonomyDecision.ASK,
                "internal_unclear",
                "Internal action not clearly covered by policy.",
                risk.confidence,
            )

        if category is IntentCategory.INTERNAL_MEDIUM:
            return (
                AutonomyDecision.ASK,
                "internal_medium_risk",
                "Reversible internal action that may still need user approval.",
                risk.confidence,
            )

        if category is IntentCategory.FINANCIAL:
            return (
                AutonomyDecision.ASK,
                "financial",
                "Financial action requires explicit approval.",
                risk.confidence,
            )

        if category is IntentCategory.SENSITIVE:
            return (
                AutonomyDecision.ESCALATE,
                "sensitive",
                "Sensitive data must not be handled autonomously.",
                risk.confidence,
            )

        # EXTERNAL (and any unexpected fallback).
        return (
            AutonomyDecision.ASK,
            "external",
            "External communication requires user approval.",
            risk.confidence,
        )

    def _emit(
        self,
        *,
        intent: str,
        requested_action: Optional[str],
        risk: RiskReport,
        safety: SafetyVerdict,
        preference: Optional[Preference],
        decision: AutonomyDecision,
        rule: str,
        reason: str,
        confidence: float,
    ) -> DecisionResult:
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
            decision_rule=rule,
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
