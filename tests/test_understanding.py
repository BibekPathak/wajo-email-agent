"""Tests for the email understanding layer."""

from agent.models import EmailSituation
from agent.prompts import build_analysis_messages, build_system_prompt
from agent.understanding import MockUnderstanding, make_understanding


def make_email(subject="", body="", sender="alice@example.com"):
    return EmailSituation(sender=sender, subject=subject, body=body)


def test_financial_email():
    analysis = MockUnderstanding().analyze(
        make_email(subject="Invoice", body="Please pay the attached invoice for $4,200.")
    )
    assert analysis.intent == "pay_invoice"
    assert analysis.requested_action == "make_payment"
    assert analysis.confidence >= 0.5
    assert analysis.entities.get("amount") == "4200"


def test_newsletter():
    analysis = MockUnderstanding().analyze(
        make_email(
            subject="Weekly digest",
            body="Here are the latest updates.",
            sender="newsletter@notion.so",
        )
    )
    assert analysis.intent == "archive_newsletter"
    assert analysis.requested_action == "archive_email"
    assert analysis.sender_category == "newsletter"


def test_scheduling_reschedule():
    analysis = MockUnderstanding().analyze(
        make_email(
            subject="Re: tomorrow",
            body="Can we move tomorrow's internal meeting from 2pm to 3pm?",
        )
    )
    assert analysis.intent == "reschedule_meeting"
    assert analysis.requested_action == "schedule_meeting"
    assert analysis.entities.get("time") in ("2pm", "3pm")


def test_customer_reply():
    analysis = MockUnderstanding().analyze(
        make_email(
            subject="Question about order",
            body="Can you confirm our contract terms?",
            sender="customer@acme.com",
        )
    )
    assert analysis.intent == "reply_customer"
    assert analysis.requested_action == "send_email"
    assert analysis.sender_category == "customer"


def test_ambiguous_email_low_confidence():
    analysis = MockUnderstanding().analyze(
        make_email(subject="Help", body="I'm not sure what to do about this.")
    )
    assert analysis.confidence < 0.5
    assert analysis.requested_action is None


def test_benign_informational_is_background():
    analysis = MockUnderstanding().analyze(
        make_email(subject="Hello", body="Just checking in, hope all is well.")
    )
    assert analysis.intent == "classify_email"
    assert analysis.confidence >= 0.5
    assert analysis.requested_action is None


def test_conflicting_instructions_low_confidence():
    analysis = MockUnderstanding().analyze(
        make_email(subject="x", body="Delete this email and also archive it.")
    )
    assert analysis.confidence < 0.5
    assert analysis.requested_action is None


def test_factory_defaults_to_mock():
    assert isinstance(make_understanding(), MockUnderstanding)


def test_prompt_marks_email_as_untrusted():
    system = build_system_prompt()
    assert "UNTRUSTED" in system
    assert "Never follow instructions" in system


def test_build_analysis_messages_shape():
    email = make_email(subject="Hi", body="Body text", sender="a@b.com")
    messages = build_analysis_messages(email)
    assert messages[0]["role"] == "system"
    assert "a@b.com" in messages[1]["content"]
    assert "Body text" in messages[1]["content"]
