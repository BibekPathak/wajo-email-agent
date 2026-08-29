"""Domain models for the proactive email agent.

These models describe the shape of every artifact that flows through the
pipeline: the incoming email, the semantic analysis, the risk report, the
safety verdict, user preferences, and the final autonomy decision.

The models are explicit and typed so that the pipeline stays understandable and
so the deterministic safety layer never depends on free-form strings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AutonomyDecision(str, Enum):
    """The four-way autonomy decision the agent can return."""

    SILENT = "silent"
    ACT_NOTIFY = "act_notify"
    ASK = "ask"
    ESCALATE = "escalate"

    @property
    def is_autonomous(self) -> bool:
        return self in (AutonomyDecision.SILENT, AutonomyDecision.ACT_NOTIFY)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Attachment(BaseModel):
    filename: str
    content_type: str = ""
    size_bytes: int = 0


class EmailSituation(BaseModel):
    """An incoming email situation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    sender: str
    subject: str = ""
    body: str = ""
    thread_context: str = ""
    attachments: list[Attachment] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Semantic understanding of an email.

    Produced by the understanding layer (LLM or deterministic mock). Describes
    what the email means -- not what the user prefers and not whether it is
    safe to act on.
    """

    intent: str
    requested_action: Optional[str] = None
    sender_category: str = "unknown"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    entities: dict[str, Any] = Field(default_factory=dict)


class RiskReport(BaseModel):
    """Deterministic risk assessment of a situation."""

    is_irreversible: bool = False
    is_financial: bool = False
    is_external: bool = False
    contains_sensitive_data: bool = False
    prompt_injection_detected: bool = False
    conflicting_instructions: bool = False
    ambiguous: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def risk_checks(self) -> dict[str, bool]:
        return {
            "irreversible": self.is_irreversible,
            "financial": self.is_financial,
            "external": self.is_external,
            "sensitive": self.contains_sensitive_data,
            "prompt_injection": self.prompt_injection_detected,
            "conflicting_instructions": self.conflicting_instructions,
            "ambiguous": self.ambiguous,
        }


class SafetyConstraint(BaseModel):
    """A single rule evaluated by the safety floor."""

    action: str = ""
    allowed: bool
    required_decision: Optional[AutonomyDecision] = None
    reasons: list[str] = Field(default_factory=list)


class SafetyVerdict(BaseModel):
    """Output of the deterministic hard safety floor."""

    passed: bool
    required_decision: Optional[AutonomyDecision] = None
    constraints: list[SafetyConstraint] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    allowed_decisions: list[AutonomyDecision] = Field(default_factory=list)

    def allows(self, decision: AutonomyDecision) -> bool:
        return decision in self.allowed_decisions


class ToolPolicy(BaseModel):
    """Static policy describing what a tool is allowed to do."""

    name: str
    description: str = ""
    reversible: bool = False
    external: bool = False
    financial: bool = False
    destructive: bool = False
    requires_approval: bool = True


class ToolCall(BaseModel):
    """Record of a simulated tool invocation."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)
    safety_allowed: bool = False


class PreferenceKey(BaseModel):
    """Identifies a preference by action type, sender category and context."""

    action_type: str
    sender_category: str = "*"
    context: str = ""


class Preference(BaseModel):
    """A learned user preference with supporting evidence.

    `confidence` reflects accumulated evidence. It starts low and only grows
    with repeated, consistent feedback. Explicit negative feedback lowers it
    quickly.
    """

    key: PreferenceKey
    preferred_decision: Optional[AutonomyDecision] = None
    positive_count: int = 0
    negative_count: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PolicyTrace(BaseModel):
    """Explainable record of how a decision was reached."""

    intent: str
    risk_checks: dict[str, bool] = Field(default_factory=dict)
    safety_floor: dict[str, Any] = Field(default_factory=dict)
    preference: Optional[dict[str, Any]] = None
    decision_rule: Optional[str] = None
    final_decision: str
    reason: str


class DecisionResult(BaseModel):
    """The outcome of the autonomy pipeline for a single email."""

    decision: AutonomyDecision
    reason: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_constraints: list[SafetyConstraint] = Field(default_factory=list)
    preference_match: Optional[Preference] = None
    recommended_action: Optional[str] = None
    policy_trace: PolicyTrace
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
