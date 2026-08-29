"""FastAPI application exposing the agent as a thin HTTP service.

Endpoints
---------
* ``POST /v1/agent/decide``   -- run one email through the agent.
* ``POST /v1/agent/feedback`` -- record user feedback about a decision.
* ``GET  /v1/agent/preferences/{user_id}`` -- inspect learned preferences.

Run with ``uvicorn api.server:app`` (or ``python -m api``).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import EmailAgent
from agent.feedback import FeedbackSignal
from agent.models import EmailSituation


# --- request/response models -------------------------------------------------


class AttachmentIn(BaseModel):
    filename: str
    content_type: str = ""
    size_bytes: int = 0


class EmailIn(BaseModel):
    sender: str
    subject: str = ""
    body: str = ""
    thread_context: str = ""
    attachments: list[AttachmentIn] = Field(default_factory=list)


class DecideRequest(BaseModel):
    email: EmailIn
    user_id: str = "default"


class FeedbackRequest(BaseModel):
    user_id: str
    decision_id: str
    feedback: str = ""
    signal: Optional[str] = None
    explicit: Optional[bool] = None


class DecideResponse(BaseModel):
    decision: str
    confidence: float
    reason: str
    risk: dict[str, bool]
    policy_trace: dict[str, Any]
    decision_id: str


# --- application factory ------------------------------------------------------


def create_app(agent: Optional[EmailAgent] = None) -> FastAPI:
    """Build the API, delegating everything to an orchestrator."""
    agent = agent or EmailAgent()

    app = FastAPI(title="Wajo Proactive Email Agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/agent/decide", response_model=DecideResponse)
    def decide(request: DecideRequest) -> DecideResponse:
        email = EmailSituation(**request.email.model_dump())
        result = agent.process_email(email, user_id=request.user_id)
        trace = result.policy_trace.model_dump(mode="json")
        return DecideResponse(
            decision=result.decision.value,
            confidence=result.confidence,
            reason=result.reason,
            risk=trace["risk_checks"],
            policy_trace=trace,
            decision_id=result.decision_id,
        )

    @app.post("/v1/agent/feedback")
    def feedback(request: FeedbackRequest) -> dict:
        signal = None
        if request.signal:
            try:
                signal = FeedbackSignal(request.signal)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="signal must be 'positive', 'negative', or 'neutral'",
                ) from exc
        try:
            preference = agent.apply_feedback_by_id(
                request.user_id,
                request.decision_id,
                request.feedback,
                signal=signal,
                explicit=request.explicit,
            )
        except KeyError:
            raise HTTPException(
                status_code=404, detail="unknown decision_id"
            ) from None
        return {
            "updated": preference is not None,
            "preference": preference.model_dump(mode="json") if preference else None,
        }

    @app.get("/v1/agent/preferences/{user_id}")
    def preferences(user_id: str) -> dict:
        return {
            "user_id": user_id,
            "preferences": [
                preference.model_dump(mode="json")
                for preference in agent.preferences_for(user_id)
            ],
        }

    return app


app = create_app()
