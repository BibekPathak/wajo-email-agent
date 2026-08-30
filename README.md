# Wajo Proactive Email Agent

A production-quality prototype of a **proactive AI email agent** with
**calibrated autonomy**. It reads incoming email situations, decides how much
independence to take, learns the user's preferences from feedback, and — most
importantly — has a deterministic **hard safety floor** that learning can
never override.

## Why calibrated autonomy matters

Binary email automation fails in both directions:

* A fully autonomous agent is convenient until it forwards the customer list,
  pays the wrong invoice, or follows a malicious instruction.
* A fully ask-first agent is safe but useless — it interrupts the user for
  every newsletter and meeting change, training them to approve blindly.

Users are neither uniform nor static. The right design is a *continuum* of
autonomy the agent moves along safely: it starts conservative, earns autonomy
through consistent feedback, and immediately loses it on pushback — while the
dangerous actions are never up for negotiation.

## Architecture

```text
Incoming Email
      │
      ▼
┌─────────────────────┐
│ Email Understanding │  intent, entities, requested action, sender
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Risk Analysis       │  irreversible? financial? external? sensitive?
│                     │  prompt injection? conflicting?
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ HARD SAFETY FLOOR   │  NEVER LEARNED · NEVER OVERRIDDEN
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ User Preference     │  learned from explicit feedback (SQLite)
│ Model               │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Autonomy Policy     │  SILENT · ACT_NOTIFY · ASK · ESCALATE
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Action Executor     │  simulated tools + audit trail
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ User Feedback       │  ──► Preference Update
└─────────────────────┘
```

The invariant is enforced in code:

```python
final_decision = safety_floor( learned_policy( email_context, user_preferences ) )
```

The learned policy can only pick a decision **within** the set the safety
floor allows. It can never expand it.

## The four decisions

| Decision | Meaning |
|---|---|
| `SILENT` | Handle in the background; no user-visible action (archive newsletter, classify, dedupe, label). |
| `ACT_NOTIFY` | Take a safe, reversible action and tell the user (move to folder, create draft, schedule internal reminder). |
| `ASK` | Present the action for approval (external email, reply to customer, reschedule meeting, share info). |
| `ESCALATE` | Hand to a human; cannot resolve safely (ambiguous, injection, conflicts, sensitive/high-impact). |

Every decision carries a structured **policy trace** (intent, risk checks,
safety verdict, preference, the rule that fired, and a reason) for full
explainability.

## Safety model

The hard safety floor is **deterministic**, **independent of preferences**, and
**never overridden**. It blocks autonomous handling of:

* **Financial actions** — payments, transfers, purchases, invoice approval.
* **Irreversible / destructive actions** — permanent deletion, closing
  accounts, cancelling critical services.
* **External communication** — unless a narrowly defined safe policy exists.
* **Sensitive data** — credentials, API keys, passwords, confidential docs.
* **Prompt injection / fake authorization / spoofing / preference poisoning.**

Each tool declares a composable policy:

```python
ToolPolicy(name="send_email", reversible=False, external=True,
           financial=False, destructive=False, requires_approval=True)
```

## Learning model

Preferences are simple, explicit statistics (no RL) keyed by
`(action_type, sender_category, context)`:

```json
{
  "action_type": "internal_calendar_reschedule",
  "preferred_decision": "act_notify",
  "positive_count": 8,
  "negative_count": 1,
  "confidence": 0.89
}
```

* New preferences start with near-zero confidence.
* Repeated consistent feedback raises confidence; conflicts lower it.
* Explicit negative feedback reduces autonomy quickly.
* A preference can upgrade `ASK -> ACT_NOTIFY` only where the safety floor
  allows it.
* **Learning can never bypass financial, irreversible, injection, sensitive,
  or mandatory-approval restrictions.**

## How to run

```bash
# install (Python 3.12+)
pip install -e .[dev]

# run the test suite
pytest

# run the agent on a single email from Python
python - <<'PY'
from agent import EmailAgent
from agent.models import EmailSituation

agent = EmailAgent()
email = EmailSituation(
    sender="alice@company.com",
    subject="Re: tomorrow",
    body="Can we move tomorrow's internal meeting to 3pm?",
)
result = agent.process_email(email)
print(result.decision.value, "—", result.reason)
print(result.policy_trace)
PY
```

The default `MockUnderstanding` runs fully offline and deterministically. To
use the LLM understanding instead, set `UNDERSTANDING_BACKEND=openai` plus
`OPENAI_API_KEY`, `OPENAI_BASE_URL` (optional), `OPENAI_MODEL` (optional).
Risk, safety, and authorization stay deterministic either way.

Attachment filenames and content types are treated as untrusted data too:
they feed the understanding and risk layers (an `invoice_0420.pdf` attachment
flags financial, `customer_list.xlsx` flags sensitive, and an
`ignore_all_previous_instructions.txt` filename is caught by the injection
detector) exactly like text in the body.

## Optional HTTP API (FastAPI)

A thin HTTP service exposing the agent. It contains no business logic — every
handler delegates to the orchestrator.

```bash
pip install -e .[api]
python -m api            # uvicorn on http://127.0.0.1:8000
```

Endpoints:

```text
POST /v1/agent/decide
  {"email": {"sender": "alice@company.com", "subject": "Re: tomorrow",
             "body": "Can we move tomorrow's internal meeting to 3pm?"},
   "user_id": "demo-user"}
  -> {"decision": "ask", "confidence": 0.8, "reason": "...",
      "risk": {"irreversible": false, "external": false, ...},
      "policy_trace": {...}, "decision_id": "..."}

POST /v1/agent/feedback
  {"user_id": "demo-user", "decision_id": "...",
   "feedback": "Yes, you can handle these automatically."}

GET /v1/agent/preferences/{user_id}
  -> {"user_id": "demo-user", "preferences": [...]}
```

Interactive docs at `http://127.0.0.1:8000/docs`. The `decision_id` returned by
`/decide` is what you pass to `/feedback`, so the learning loop works over HTTP
exactly as it does in-process.

## How to run evaluations

```bash
python -m evals.generate_dataset   # regenerate datasets (already committed)
python -m evals.runner             # run the baseline-vs-learned experiment
```

This writes reports to `evals/reports/`:

* `report.md` / `report.json` — full before/after metrics, calibration, ECE,
  adversarial breakdown by attack type.
* `learning_curve.md` — metrics as feedback accumulates.
* `key_results.json` — the headline numbers below.

## Results

Measured on 265 main scenarios + 25 adversarial scenarios (deterministic mock
understanding; `evals/reports/report.json`).

```text
                    Before      After
Accuracy             88.68%     100.00%
Ask rate             37.74%      26.42%
Autonomous handling  76.92%     100.00%
Over-asking rate     23.08%       0.00%
Unsafe autonomy         0%          0%
```

* **Adversarial dataset**: 100% accuracy, **0% unsafe autonomy** across all
  six attack types (prompt injection, fake authorization, authority spoofing,
  social engineering, preference poisoning, conflicting instructions).
* **Learning curve**: autonomy jumps exactly when preference confidence
  crosses the upgrade threshold (~7 approvals); unsafe autonomy stays pinned
  at 0% at every step.
* **Calibration (measured, not claimed)**: high-confidence (≥0.9) error rate
  0.62% cold; binned ECE 0.14 cold → 0.12 learned.

## Key Result

After preference learning:

```text
Autonomy on safe tasks:   +23.1pp   (76.9% -> 100.0%)
Unnecessary asks:          -11.3pp   (37.7% -> 26.4%);  over-asking 23.1% -> 0.0%
Unsafe autonomous actions:   0%       (before and after)
```

More autonomy, fewer unnecessary questions, and the same safety floor.

## Example transcripts

Seven worked examples are in [`examples/transcripts.md`](examples/transcripts.md):

1. Newsletter archived silently (`SILENT`)
2. Meeting reschedule learned from `ASK` → `ACT_NOTIFY` via feedback
3. External contract confirmation → `ASK`
4. Invoice payment → `ASK` (financial; still `ASK` after 100 approvals)
5. Prompt injection → `ESCALATE`, nothing executed
6. Ambiguous request → `ESCALATE`
7. Negative feedback quickly returns autonomy back to `ASK`

## Design tradeoffs

* **Conservative > aggressive.** We bias toward asking/escalating; learning
  buys back autonomy safely.
* **Safety false-positives are cheap; false-negatives are not.** The floor is
  asymmetric by design.
* **LLM for semantics, not guarantees.** Risk, safety, and authorization are
  deterministic.
* **Explicit > inferred feedback.** Explicit feedback is weighted higher.
* **Simple > clever.** Explicit preference statistics instead of RL; rule-based
  injection detection; deterministic safety.

See [`DESIGN.md`](DESIGN.md) for the full design document, failure modes, and
future work.

## Repository layout

```text
agent/        core pipeline (models, understanding, risk, safety, autonomy,
              preferences, feedback, executor, orchestrator)
api/          thin FastAPI service (optional)
evals/        dataset generator, runner, metrics, reports
tests/        unit + safety + learning + prompt-injection + end-to-end tests
examples/     transcripts
DESIGN.md     design document
```
