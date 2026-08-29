"""Orchestrator: wires the full pipeline for a single email.

    email -> understanding -> risk -> safety floor -> preference -> autonomy
         -> executor (simulated) -> feedback -> updated preference

The orchestrator itself contains no business logic; it composes the modular
pieces. The pipeline order is fixed and matches the design invariant:

    final_decision = safety_floor(learned_policy(email, preferences))
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from .autonomy import AutonomyPolicy
from .executor import TOOL_REGISTRY, Executor
from .feedback import FeedbackEngine
from .models import (
    AutonomyDecision,
    DecisionResult,
    EmailSituation,
    Preference,
)
from .preferences import PreferenceStore
from .risk import RiskAnalyzer
from .safety import SafetyFloor
from .understanding import EmailUnderstanding, MockUnderstanding

# How many recent decisions to remember so the HTTP API can apply feedback
# by decision_id.
MAX_RECENT_DECISIONS = 1000


class EmailAgent:
    """A ready-to-use agent instance for one mailbox (per user)."""

    def __init__(
        self,
        *,
        understanding: Optional[EmailUnderstanding] = None,
        risk: Optional[RiskAnalyzer] = None,
        safety: Optional[SafetyFloor] = None,
        autonomy: Optional[AutonomyPolicy] = None,
        preferences: Optional[PreferenceStore] = None,
        feedback: Optional[FeedbackEngine] = None,
        executor: Optional[Executor] = None,
        tool_registry: Optional[dict] = None,
        default_user: str = "default",
    ) -> None:
        tools = tool_registry if tool_registry is not None else TOOL_REGISTRY
        self.understanding = understanding or MockUnderstanding()
        self.risk = risk or RiskAnalyzer(tools)
        self.safety = safety or SafetyFloor(tools)
        self.autonomy = autonomy or AutonomyPolicy(tools)
        self.preferences = preferences or PreferenceStore()
        self.feedback = feedback or FeedbackEngine(self.preferences)
        self.executor = executor or Executor()
        self._default_user = default_user
        self._recent_decisions: "OrderedDict[str, DecisionResult]" = OrderedDict()

    # --- main entry ---------------------------------------------------------

    def process_email(
        self,
        email: EmailSituation,
        user_id: Optional[str] = None,
    ) -> DecisionResult:
        """Run one email through the full pipeline and return the decision."""
        user_id = user_id or self._default_user

        analysis = self.understanding.analyze(email)
        risk_report = self.risk.assess(analysis, email)
        safety = self.safety.evaluate(risk_report, analysis.requested_action)

        action_type = analysis.requested_action or analysis.intent
        preference = None
        if action_type:
            preference = self.preferences.lookup(
                user_id, action_type, analysis.sender_category
            )

        decision = self.autonomy.decide(
            intent=analysis.intent,
            requested_action=analysis.requested_action,
            risk=risk_report,
            safety=safety,
            preference=preference,
        )

        self._execute(decision, analysis, email, safety)

        self._recent_decisions[decision.decision_id] = decision
        self._recent_decisions.move_to_end(decision.decision_id)
        while len(self._recent_decisions) > MAX_RECENT_DECISIONS:
            self._recent_decisions.popitem(last=False)

        return decision

    # --- feedback -----------------------------------------------------------

    def apply_feedback(
        self,
        user_id: Optional[str],
        decision: DecisionResult,
        feedback_text: str = "",
        *,
        signal=None,
        explicit: Optional[bool] = None,
    ) -> Optional[Preference]:
        """Record user feedback about a decision (the learning loop)."""
        user_id = user_id or self._default_user
        return self.feedback.apply(
            user_id,
            decision,
            feedback_text,
            signal=signal,
            explicit=explicit,
        )

    def preferences_for(self, user_id: Optional[str] = None) -> list[Preference]:
        return self.preferences.get_all(user_id or self._default_user)

    def decision_by_id(self, decision_id: str) -> Optional[DecisionResult]:
        """Look up a recent decision by its decision_id."""
        return self._recent_decisions.get(decision_id)

    def apply_feedback_by_id(
        self,
        user_id: Optional[str],
        decision_id: str,
        feedback_text: str = "",
        *,
        signal=None,
        explicit: Optional[bool] = None,
    ) -> Optional[Preference]:
        """Record feedback about a recent decision, looked up by id.

        Raises ``KeyError`` when the decision_id is unknown (e.g. the server
        restarted or the decision aged out of the recent-decision registry).
        """
        decision = self.decision_by_id(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        return self.apply_feedback(
            user_id, decision, feedback_text, signal=signal, explicit=explicit
        )

    # --- internals ------------------------------------------------------------

    def _execute(self, decision: DecisionResult, analysis, email, safety) -> None:
        """Simulate tool execution for autonomous decisions.

        SILENT/ACT_NOTIFY are executed (recorded) when the action passed the
        safety floor. ASK/ESCALATE are never executed; they wait on the user.
        """
        tool = decision.recommended_action
        if tool is None:
            return
        if decision.decision in (AutonomyDecision.SILENT, AutonomyDecision.ACT_NOTIFY):
            allowed = safety.allows(decision.decision)
            arguments = dict(analysis.entities)
            arguments.setdefault("message_id", email.id)
            arguments.setdefault("recipient", email.sender)
            self.executor.execute(tool, arguments, safety_allowed=allowed)
