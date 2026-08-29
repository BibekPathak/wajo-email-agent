"""Deterministic hard safety floor.

The safety floor is the single most important component of the agent. It is:

* deterministic (no LLM involved),
* independent of learned user preferences,
* never overridden by the learned policy.

For a given risk report and requested tool action it computes the set of
autonomy decisions that are *ever* allowed. Learned preferences may select
within that set; they can never expand it. Conceptually:

    final_decision = safety_floor(learned_policy(email, preferences))

not:

    final_decision = learned_policy(safety_floor(...))
"""

from __future__ import annotations

from typing import Optional

from .models import (
    AutonomyDecision,
    RiskLevel,
    RiskReport,
    SafetyConstraint,
    SafetyVerdict,
    ToolPolicy,
)


class SafetyFloor:
    """Deterministic, preference-independent hard safety floor."""

    def __init__(self, tool_registry: Optional[dict[str, ToolPolicy]] = None) -> None:
        self._tools = tool_registry if tool_registry is not None else {}

    def tool_policy(self, action: Optional[str]) -> Optional[ToolPolicy]:
        if not action:
            return None
        return self._tools.get(action)

    def evaluate(self, risk: RiskReport, action: Optional[str] = None) -> SafetyVerdict:
        """Return the safety verdict for a risk report and requested action.

        The verdict carries the set of allowed decisions. When the floor blocks
        an action it returns the decision the agent must use instead (ASK or
        ESCALATE), which the learned policy can never override.
        """
        constraints: list[SafetyConstraint] = []
        reasons: list[str] = []
        blocked = False
        required: Optional[AutonomyDecision] = None

        def require(rule: str, decision: AutonomyDecision, action_name: str) -> None:
            nonlocal blocked, required
            blocked = True
            # Most restrictive decision wins (ESCALATE > ASK).
            if required is None or (
                decision is AutonomyDecision.ESCALATE
                and required is not AutonomyDecision.ESCALATE
            ):
                required = decision
            constraints.append(
                SafetyConstraint(
                    action=action_name,
                    allowed=False,
                    required_decision=decision,
                    reasons=[rule],
                )
            )
            reasons.append(rule)

        policy = self.tool_policy(action)

        # --- Untrusted content checks (injection / conflicts / ambiguity) ----
        if risk.prompt_injection_detected:
            require(
                "Untrusted email content attempted to manipulate agent behavior.",
                AutonomyDecision.ESCALATE,
                action or "",
            )
        if risk.conflicting_instructions:
            require(
                "Email contains conflicting instructions; cannot resolve safely.",
                AutonomyDecision.ESCALATE,
                action or "",
            )
        if risk.ambiguous and risk.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH):
            require(
                "Situation is too ambiguous to determine a safe autonomous action.",
                AutonomyDecision.ESCALATE,
                action or "",
            )

        # --- Risk-level checks (independent of tool mapping) ------------------
        if risk.is_financial:
            require(
                "Financial actions are never executed autonomously.",
                AutonomyDecision.ASK,
                action or "",
            )
        if risk.is_irreversible:
            require(
                "Irreversible actions require explicit approval.",
                AutonomyDecision.ESCALATE,
                action or "",
            )
        if risk.contains_sensitive_data and action:
            escalate = policy is not None and policy.external
            require(
                "Sensitive data is never acted upon autonomously.",
                AutonomyDecision.ESCALATE if escalate else AutonomyDecision.ASK,
                action or "",
            )

        # --- Tool-level checks -------------------------------------------------
        if policy is None:
            if not blocked:
                constraints.append(
                    SafetyConstraint(
                        action="",
                        allowed=True,
                        reasons=["No tool action requested."],
                    )
                )
        else:
            if policy.financial:
                require(
                    "Financial actions are never executed autonomously.",
                    AutonomyDecision.ASK,
                    action or "",
                )
            if policy.destructive:
                require(
                    "Irreversible/destructive actions require explicit approval.",
                    AutonomyDecision.ESCALATE,
                    action or "",
                )
            if policy.external and policy.requires_approval:
                require(
                    "External communication requires explicit user approval.",
                    AutonomyDecision.ASK,
                    action or "",
                )
            if not policy.external and policy.requires_approval:
                require(
                    "Tool policy requires explicit approval.",
                    AutonomyDecision.ASK,
                    action or "",
                )
            if not blocked:
                constraints.append(
                    SafetyConstraint(
                        action=action or "",
                        allowed=True,
                        reasons=["All safety checks passed."],
                    )
                )

        if blocked:
            allowed = [AutonomyDecision.ASK, AutonomyDecision.ESCALATE]
        else:
            allowed = sorted(AutonomyDecision, key=lambda d: d.value)

        return SafetyVerdict(
            passed=not blocked,
            required_decision=required,
            constraints=constraints,
            reasons=reasons,
            allowed_decisions=allowed,
        )
