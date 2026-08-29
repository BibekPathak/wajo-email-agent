# DESIGN.md — Wajo Proactive Email Agent

A design document for a proactive AI email agent with **calibrated autonomy**:
it learns how much independence a user wants, while a deterministic hard
safety floor guarantees that learning can never authorize dangerous actions.

---

## 1. Problem

Most email automation is binary: an agent either acts fully autonomously or
does nothing at all. Both extremes fail.

* A **fully autonomous** agent archives, replies, and pays with no human in
  the loop. It is convenient until the day it forwards the customer list,
  pays the wrong invoice, or follows an instruction buried in a malicious
  email.
* A **fully ask-first** agent is safe but useless — it interrupts the user for
  every newsletter, label, and meeting change, training the user to ignore
  its prompts and eventually approve things blindly.

Real users are neither uniform nor static. A user might want meeting
reschedules handled silently, newsletter archives done without a word, but
every external reply approved first. And those preferences shift over time.
The agent needs a **continuum of autonomy** that it can move along safely.

The core design challenge is therefore not "should the agent act or ask?" but
"how much autonomy is justified *for this user, for this kind of action, right
now* — and what must never be up for negotiation?"

---

## 2. Autonomy Model

Instead of a binary, the agent decides between four levels, in increasing
human involvement:

| Decision | Meaning | Typical situations |
|---|---|---|
| `SILENT` | Handle in the background, no user-visible action | Classify an email, archive an obvious newsletter, deduplicate, apply a label |
| `ACT_NOTIFY` | Take a safe, reversible action and tell the user afterward | Move to a folder, create a draft, schedule a non-critical internal reminder |
| `ASK` | Present the action for explicit approval | Send an external email, accept an invitation, reply to a customer, reschedule an important meeting, share information externally |
| `ESCALATE` | Hand to a human; the agent cannot resolve it safely | Ambiguous intent, prompt injection, conflicting instructions, sensitive/high-impact situations |

`SILENT` and `ACT_NOTIFY` are *autonomous*; `ASK` and `ESCALATE` are *not*.
The learning system can only move the agent along the safe part of this
continuum (e.g. `ASK -> ACT_NOTIFY` once it is confident the user wants that),
and only within what the safety floor permits.

---

## 3. Safety Model

The hard safety floor is the most important component. It is:

* **deterministic** — pure rules over typed data, no LLM involved;
* **independent** of learned user preferences — it never consults them;
* **never overridden** — by anything upstream, ever.

The pipeline order is fixed and non-negotiable:

```text
final_decision = safety_floor( learned_policy( email_context, user_preferences ) )
```

not

```text
final_decision = learned_policy( safety_floor(...) )
```

The floor computes, for a given risk report and requested tool action, the set
of autonomy decisions that are *ever* allowed. A learned preference may select
a decision **within** that set; it can never expand it.

Rules enforced by the floor:

* **Financial actions** (payments, transfers, purchases, invoice approval,
  changing financial details) are never executed autonomously — `ASK`.
* **Irreversible / destructive actions** (permanent deletion, cancelling
  critical services, closing accounts) are never autonomous — `ASK`/`ESCALATE`.
* **External communication** requires explicit approval unless a narrowly
  defined safe policy exists — `ASK`.
* **Sensitive data** (credentials, API keys, passwords, personal/confidential
  documents) is never disclosed or acted on autonomously — `ASK`/`ESCALATE`.
* **Prompt injection / fake authorization / authority spoofing / preference
  poisoning** routes to `ESCALATE`.
* **Ambiguity or conflicting instructions** routes to `ESCALATE`.

Every tool declares a composable `ToolPolicy`:

```python
ToolPolicy(
    name="send_email",
    reversible=False, external=True, financial=False,
    destructive=False, requires_approval=True,
)
```

Because the floor is a pure function of `(risk_report, tool_policy)`, the
safety behavior is trivially testable and cannot drift when the LLM or the
learning code changes.

---

## 4. Learning

Preference learning uses **explicit preference statistics**, not reinforcement
learning. There is no policy gradient, no reward shaping, and no exploration —
deliberately. RL is unnecessary here and would make the system harder to
reason about, test, and trust.

Preferences are keyed by `(action_type, sender_category, context)` and store:

```json
{
  "action_type": "internal_calendar_reschedule",
  "preferred_decision": "act_notify",
  "positive_count": 8,
  "negative_count": 1,
  "confidence": 0.89
}
```

Learning rules are deliberately conservative:

1. New preferences start with near-zero confidence.
2. Repeated, consistent feedback raises confidence.
3. Conflicting feedback lowers confidence.
4. Explicit negative feedback reduces autonomy quickly (explicit negatives
   carry the heaviest weight in the evidence model).
5. Explicit feedback outranks inferred feedback.
6. A learned preference can upgrade `ASK -> ACT_NOTIFY` (or to `SILENT`) only
   where the safety floor allows it — and only for autonomy-eligible action
   categories.
7. Learning can **never** bypass financial, irreversible, prompt-injection,
   sensitive-data, or mandatory-approval restrictions.

The signature example:

```text
Before learning:  "Move internal meeting from 2pm to 3pm"      -> ASK
After 10 approvals:                                          -> ACT_NOTIFY

"Pay this $5,000 invoice"  after 100 approvals:               -> STILL ASK
```

The second behavior demonstrates the hard safety floor: no amount of learning
can weaken it. This is enforced twice — once by the safety floor's allowed
decision set, and once by gating preference upgrades to autonomy-eligible
categories only.

Confidence is an evidence-derived score, not a calibrated probability. The
evaluation harness measures calibration honestly (see §7) rather than
assuming the number means what we wish it did.

---

## 5. Prompt Injection

Email content is treated as **untrusted data**, like a web form field or an
attached file — never as instructions.

* The understanding layer (LLM or deterministic mock) is prompted to describe
  what the email *says*, not to obey it, and to never reveal or act on its own
  system prompt.
* The deterministic risk layer scans the raw email text for manipulation
  patterns: `ignore all previous instructions`, fake authorization
  (`the user already approved this`), authority spoofing (`SYSTEM ADMIN:`),
  preference poisoning (`remember this forever: always act automatically`),
  and social engineering.
* Any hit routes through the safety floor to `ESCALATE`.
* Malicious email can never write to the preference store — only the explicit
  feedback channel can. Processing an email, no matter how persuasive,
  leaves preferences byte-for-byte unchanged.

The defense is layered: even if the LLM is confused by a clever injection, the
deterministic detector and the safety floor still block the action.

---

## 6. Evaluation

Two labeled datasets drive the evaluation:

* `evals/dataset.jsonl` — 265 main scenarios across every risk class
  (newsletters, informational, internal actions, scheduling, customer
  requests, external communication, financial, destructive, sensitive,
  ambiguous, prompt injection, conflicting instructions).
* `evals/adversarial.jsonl` — 25 hand-curated adversarial cases across six
  attack types.

Labels are ground truth for **scoring only**; the agent never reads them.
Labels encode the target policy — what a well-calibrated agent that knows the
user should do — so cold-start accuracy starts lower and rises as preferences
are learned.

Metrics (see `evals/metrics.py`):

* **Decision accuracy** — correct / total.
* **Unsafe autonomy rate** — autonomous actions where approval was required.
  This is the headline safety metric; the target is **0%**.
* **Ask rate** and **over-asking rate** — unnecessary questions asked.
* **Autonomous handling rate** — eligible low-risk situations handled without
  asking.
* **Calibration** — confidence vs correctness, high-confidence error rate,
  and binned expected calibration error (ECE).
* **Learning improvement** — baseline vs learned agent.

The experiment has three phases: cold start → simulated user feedback
(10× meeting handling, 10× newsletter silence, 10× external ask-first) →
re-run. A learning curve measures metrics at increasing feedback amounts.

---

## 7. Calibration

*Calibrated autonomy* means: the amount of autonomy the agent takes matches
how much it has actually earned, measured against the user's demonstrated
preferences.

The agent does not start autonomous on any judgment call. It starts
conservative (asks), and only upgrades after enough consistent, explicit
positive feedback pushes the preference confidence past a threshold
(`PREFERENCE_UPGRADE_THRESHOLD = 0.8`). Confidence therefore tracks *evidence*,
not the LLM's self-assurance.

The harness does not assume the confidence number is meaningful — it measures:

* accuracy within confidence bins;
* the high-confidence error rate (how often a confident decision is wrong);
* binned ECE, reported as measured (currently ~0.12–0.14 on the main dataset),
  not hidden or tuned away.

Because safety-critical actions are decided by deterministic rules, they carry
a high confidence floor by construction; they do not depend on calibration of
the semantic layer at all.

---

## 8. Tradeoffs

* **Conservative vs aggressive autonomy.** We bias strongly toward asking and
  escalating on anything uncertain. This raises the ask rate early on but
  protects against unsafe actions; learning then buys back autonomy safely.
* **False positives vs false negatives (safety).** In a safety layer,
  a false positive (blocking a safe action) is cheap; a false negative
  (allowing an unsafe action) is catastrophic. The floor is therefore
  asymmetric by design.
* **LLM flexibility vs deterministic safety.** The LLM is excellent at
  semantics and terrible as a source of guarantees. We use it only for
  understanding and keep risk, safety, and final authorization deterministic.
* **Explicit vs inferred preferences.** Explicit feedback is weighted more
  heavily than inferred signals, because inferred signals are noisier. This
  makes learning slower but more trustworthy.
* **Complexity vs reliability.** A simple statistical preference model, a
  hand-curated safety rule set, and rule-based injection detection are easier
  to audit and test than a large learned policy — at the cost of expressiveness.

---

## 9. Failure Modes

The system intentionally asks or escalates instead of guessing in these cases:

* **Ambiguous intent** — the semantic layer cannot determine what the email
  wants → `ESCALATE`.
* **Conflicting instructions** — the email contradicts itself (e.g. "delete
  this but keep a copy") → `ESCALATE`.
* **Sensitive/high-impact situations** — confidential data, credentials,
  large or unusual financial requests → `ASK`/`ESCALATE`.
* **Unknown sender/action categories** — the action is not clearly covered by
  policy → `ASK`.
* **Novel attack phrasing** — an injection the heuristic list does not
  recognize verbatim. This is a known limitation: the heuristic detector is a
  safety net, not a guarantee against all future phrasings. The architectural
  guarantee (email can never modify policy or preferences) holds regardless.

When the agent guesses wrongly, the consequences are bounded by the safety
floor: the worst a mistaken `ASK` or `ESCALATE` does is cost the user a moment;
the worst a mistaken `SILENT`/`ACT_NOTIFY` can do is a reversible, low-risk,
internal action.

---

## 10. Future Work

* **Richer preference representations** — context windows, per-sender
  granularity, time-of-day or urgency conditioning.
* **Contextual bandits** — only where it clearly beats the simple evidence
  model, and still behind the safety floor.
* **Better uncertainty calibration** — temperature-scaled or conformal
  confidence on top of the semantic layer.
* **Personalized risk models** — risk thresholds that adapt per user *below*
  the floor, without touching floor invariants.
* **Real Gmail integration** — a real tool executor with OAuth, with the same
  policy and audit gate in front of it.
* **Human-in-the-loop policy editing** — let users inspect and adjust learned
  preferences and the policy trace directly.
* **Attack taxonomy expansion** — continuously extend the adversarial dataset
  and the injection heuristics as new phrasings are found.
