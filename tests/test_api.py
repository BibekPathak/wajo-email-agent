"""Tests for the thin FastAPI service.

Verifies the API stays thin: it only serializes/deserializes and delegates to
the orchestrator, and the full learning loop works through HTTP.
"""

import pytest
from fastapi.testclient import TestClient

from agent import EmailAgent
from api.server import create_app


@pytest.fixture()
def client():
    app = create_app(EmailAgent())
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_decide_external_returns_ask(client):
    resp = client.post(
        "/v1/agent/decide",
        json={
            "email": {
                "sender": "vendor@acme.com",
                "subject": "Terms",
                "body": "Can you confirm our contract terms?",
            },
            "user_id": "u1",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "ask"
    assert body["decision_id"]
    assert body["risk"]["external"] is True
    assert body["policy_trace"]["final_decision"] == "ask"
    assert body["reason"]


def test_decide_newsletter_silent(client):
    resp = client.post(
        "/v1/agent/decide",
        json={
            "email": {
                "sender": "newsletter@notion.so",
                "subject": "Weekly digest",
                "body": "Here is the latest roundup.",
            },
            "user_id": "u1",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["decision"] == "silent"


def test_decide_validation_error(client):
    resp = client.post("/v1/agent/decide", json={"email": {}, "user_id": "u1"})
    assert resp.status_code == 422


def test_feedback_updates_preference_via_decision_id(client):
    decide = client.post(
        "/v1/agent/decide",
        json={
            "email": {
                "sender": "alice@company.com",
                "subject": "Re: tomorrow",
                "body": "Can we move tomorrow's internal meeting to 3pm?",
            },
            "user_id": "u1",
        },
    ).json()
    assert decide["decision"] == "ask"

    for _ in range(10):
        resp = client.post(
            "/v1/agent/feedback",
            json={
                "user_id": "u1",
                "decision_id": decide["decision_id"],
                "feedback": "Yes, handle these automatically.",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["updated"] is True

    prefs = client.get("/v1/agent/preferences/u1").json()["preferences"]
    assert len(prefs) == 1
    assert prefs[0]["preferred_decision"] == "act_notify"
    assert prefs[0]["confidence"] >= 0.8

    # The same email now triggers the learned preference.
    again = client.post(
        "/v1/agent/decide",
        json={
            "email": {
                "sender": "alice@company.com",
                "subject": "Re: tomorrow",
                "body": "Can we move tomorrow's internal meeting to 3pm?",
            },
            "user_id": "u1",
        },
    ).json()
    assert again["decision"] == "act_notify"


def test_feedback_unknown_decision_404(client):
    resp = client.post(
        "/v1/agent/feedback",
        json={"user_id": "u1", "decision_id": "does-not-exist", "feedback": "approved"},
    )
    assert resp.status_code == 404


def test_feedback_invalid_signal_422(client):
    resp = client.post(
        "/v1/agent/feedback",
        json={
            "user_id": "u1",
            "decision_id": "x",
            "signal": "banana",
        },
    )
    assert resp.status_code == 422


def test_preferences_empty_for_new_user(client):
    assert client.get("/v1/agent/preferences/nobody").json()["preferences"] == []


def test_feedback_never_weakens_safety_floor(client):
    """Even through the API, learning cannot authorize a payment."""
    decide = client.post(
        "/v1/agent/decide",
        json={
            "email": {
                "sender": "billing@acme.com",
                "subject": "Invoice",
                "body": "Please pay the attached invoice for $4,200.",
            },
            "user_id": "u1",
        },
    ).json()
    assert decide["decision"] in ("ask", "escalate")

    for _ in range(20):
        client.post(
            "/v1/agent/feedback",
            json={
                "user_id": "u1",
                "decision_id": decide["decision_id"],
                "signal": "positive",
            },
        )

    again = client.post(
        "/v1/agent/decide",
        json={
            "email": {
                "sender": "billing@acme.com",
                "subject": "Invoice",
                "body": "Please pay the attached invoice for $4,200.",
            },
            "user_id": "u1",
        },
    ).json()
    assert again["decision"] in ("ask", "escalate")
