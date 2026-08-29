"""Deterministic risk analysis.

Transforms an :class:`AnalysisResult` plus the raw email into a
:class:`RiskReport`. Every check is a deterministic heuristic over the analysis
and the email text; the LLM is never asked to judge risk. Prompt injection,
fake authorization, authority spoofing, and preference poisoning are detected
here so the hard safety floor can block them before any preference is
consulted.
"""

from __future__ import annotations

from typing import Optional

from .autonomy import IntentCategory, classify_intent
from .executor import TOOL_REGISTRY
from .models import (
    AnalysisResult,
    EmailSituation,
    RiskLevel,
    RiskReport,
    ToolPolicy,
)
from .textutil import contains_any, contains_phrase, count_mentions

FINANCIAL_KEYWORDS = [
    "pay", "payment", "invoice", "transfer", "wire", "money", "usd", "eur",
    "purchase", "buy", "bill", "charge", "remit", "settle",
]
DESTRUCTIVE_KEYWORDS = [
    "delete", "permanently", "erase", "purge", "cancel my account",
    "close account", "cancel service", "cancel subscription", "terminate",
]
SENSITIVE_KEYWORDS = [
    "password", "api key", "credentials", "secret", "confidential",
    "customer list", "personal data", "personal information", "private",
    "credit card", "social security", "passport", "bank account",
    "proprietary", "health record", "medical record",
]

# Untrusted-content markers. Any of these is treated as an attempt to
# manipulate the agent and routes through the safety floor to ESCALATE.
INJECTION_PATTERNS = [
    "ignore all previous instructions", "ignore previous instructions",
    "ignore your instructions", "ignore all prior instructions",
    "disregard all previous", "forget all previous instructions",
    "forget your instructions", "you are now authorized",
    "you are authorized to", "you have full authorization",
    "system prompt", "override your instructions", "new system prompt",
    "forward all messages", "forward every email", "forward all emails",
    "send all messages", "send every email", "secret instruction",
    "hidden instruction", "you are an ai", "as an ai agent",
]
FAKE_AUTHORIZATION_PATTERNS = [
    "already approved", "user already approved", "approval has been given",
    "already authorized", "ceo approved", "manager approved",
    "you already have permission", "no need to ask",
]
AUTHORITY_SPOOFING_PATTERNS = [
    "system admin", "system administrator", "admin:",
]
PREFERENCE_POISONING_PATTERNS = [
    "remember this forever", "always act automatically",
    "always do this from now on", "never ask me", "from now on always",
    "update my preferences", "set this preference",
]


class RiskAnalyzer:
    """Deterministic mapping from analysis + email text to a risk report."""

    def __init__(self, tool_registry: Optional[dict[str, ToolPolicy]] = None) -> None:
        self._tools = tool_registry if tool_registry is not None else TOOL_REGISTRY

    def assess(self, analysis: AnalysisResult, email: EmailSituation) -> RiskReport:
        text = (f"{email.subject}\n{email.body}\n{email.thread_context}").lower()
        intent = analysis.intent
        action = analysis.requested_action
        category = classify_intent(intent)
        tool = self._tools.get(action or "") if action else None

        is_financial = (
            category is IntentCategory.FINANCIAL
            or (tool is not None and tool.financial)
            or contains_any(text, FINANCIAL_KEYWORDS)
        )
        is_irreversible = (
            (tool is not None and tool.destructive)
            or contains_any(text, DESTRUCTIVE_KEYWORDS)
        )
        is_external = (
            category is IntentCategory.EXTERNAL
            or (tool is not None and tool.external)
        )
        contains_sensitive_data = (
            category is IntentCategory.SENSITIVE
            or contains_any(text, SENSITIVE_KEYWORDS)
        )

        prompt_injection_detected = contains_any(
            text,
            (
                *INJECTION_PATTERNS,
                *FAKE_AUTHORIZATION_PATTERNS,
                *AUTHORITY_SPOOFING_PATTERNS,
                *PREFERENCE_POISONING_PATTERNS,
            ),
        )

        ignore_mentions = count_mentions(text, INJECTION_PATTERNS)
        conflicting_instructions = (
            ignore_mentions > 1
            or (
                contains_phrase(text, "delete")
                and contains_any(text, ("archive", "keep", "retain"))
            )
        )

        ambiguous = analysis.confidence < 0.5

        if (
            is_financial
            or prompt_injection_detected
            or contains_sensitive_data
            or is_irreversible
            or conflicting_instructions
        ):
            risk_level = RiskLevel.HIGH
        elif is_external or ambiguous:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return RiskReport(
            is_irreversible=is_irreversible,
            is_financial=is_financial,
            is_external=is_external,
            contains_sensitive_data=contains_sensitive_data,
            prompt_injection_detected=prompt_injection_detected,
            conflicting_instructions=conflicting_instructions,
            ambiguous=ambiguous,
            risk_level=risk_level,
            confidence=analysis.confidence,
        )
