"""Wajo proactive email agent.

The agent receives incoming email situations and decides how much autonomy it
should take: SILENT, ACT_NOTIFY, ASK, or ESCALATE. A deterministic hard safety
floor constrains the decisions a learned preference policy is ever allowed to
make.
"""

from .models import AutonomyDecision, DecisionResult, EmailSituation, RiskReport

__all__ = ["AutonomyDecision", "DecisionResult", "EmailSituation", "RiskReport"]
