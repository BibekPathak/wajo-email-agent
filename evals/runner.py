"""Evaluation runner.

Runs the baseline-vs-learned experiment:

    Phase 1 -- cold start: no user preferences.
    Phase 2 -- simulated user feedback (meetings handled automatically,
               newsletters handled silently, external replies ask first).
    Phase 3 -- re-run with learned preferences.

Writes ``evals/reports/report.json`` and ``evals/reports/report.md``.

Usage:
    python -m evals.runner
"""

from __future__ import annotations

import json
from pathlib import Path

from agent import EmailAgent
from agent.feedback import FeedbackEngine, FeedbackSignal
from agent.models import AutonomyDecision, EmailSituation

from .generate_dataset import ADVERSARIAL_PATH, DATASET_PATH, generate
from .metrics import calibration, compare, compute_metrics

EVALS_DIR = Path(__file__).parent
REPORTS_DIR = EVALS_DIR / "reports"

USER_ID = "default"

# Simulated user feedback (action_type, sender_category, signal, count,
# default_decision).
FEEDBACK_PLAN: list[tuple[str, str, FeedbackSignal, int, AutonomyDecision | None]] = [
    ("schedule_meeting", "*", FeedbackSignal.POSITIVE, 10, AutonomyDecision.ACT_NOTIFY),
    ("archive_email", "newsletter", FeedbackSignal.POSITIVE, 10, AutonomyDecision.SILENT),
    ("send_email", "*", FeedbackSignal.NEGATIVE, 10, None),
]


def load_scenarios(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_scenarios(agent: EmailAgent, scenarios: list[dict]) -> list[dict]:
    """Run the agent over scenarios, clearing the executor per case."""
    results = []
    for scenario in scenarios:
        email = EmailSituation(**scenario["email"])
        decision = agent.process_email(email)
        expected = AutonomyDecision(scenario["expected_decision"])
        expected_autonomy = scenario["expected_autonomy"]
        results.append(
            {
                "id": scenario["id"],
                "risk_class": scenario["risk_class"],
                "tags": scenario.get("tags", []),
                "expected_decision": expected.value,
                "expected_autonomy": expected_autonomy,
                "autonomy_eligible": expected_autonomy,
                "decision": decision.decision.value,
                "confidence": decision.confidence,
                "correct": decision.decision == expected,
                "rule": decision.policy_trace.decision_rule,
                "intent": decision.policy_trace.intent,
                "reason": decision.reason,
                "executed": [call.model_dump(mode="json") for call in agent.executor.calls],
            }
        )
        agent.executor.clear()
    return results


def apply_feedback(agent: EmailAgent) -> None:
    engine = FeedbackEngine(agent.preferences)
    for action_type, sender, signal, count, default in FEEDBACK_PLAN:
        for _ in range(count):
            engine.record(
                USER_ID,
                action_type=action_type,
                sender_category=sender,
                signal=signal,
                default_decision=default,
            )


def run_experiment(
    dataset_path: Path = DATASET_PATH,
    adversarial_path: Path = ADVERSARIAL_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    if not dataset_path.exists() or not adversarial_path.exists():
        generate()

    scenarios = load_scenarios(dataset_path)
    adversarial = load_scenarios(adversarial_path)

    agent = EmailAgent()

    cold_results = run_scenarios(agent, scenarios)
    cold_metrics = compute_metrics(cold_results)
    cold_adv_results = run_scenarios(agent, adversarial)
    cold_adv_metrics = compute_metrics(cold_adv_results)

    apply_feedback(agent)

    learned_results = run_scenarios(agent, scenarios)
    learned_metrics = compute_metrics(learned_results)
    learned_adv_results = run_scenarios(agent, adversarial)
    learned_adv_metrics = compute_metrics(learned_adv_results)

    report = {
        "dataset": {
            "main_size": len(scenarios),
            "adversarial_size": len(adversarial),
            "feedback_plan": [
                {
                    "action_type": a,
                    "sender_category": s,
                    "signal": sig.value,
                    "count": c,
                    "default_decision": d.value if d else None,
                }
                for a, s, sig, c, d in FEEDBACK_PLAN
            ],
        },
        "phase1_cold_start": cold_metrics.to_dict(),
        "phase3_learned": learned_metrics.to_dict(),
        "delta": compare(cold_metrics, learned_metrics),
        "calibration_cold_start": calibration(cold_results),
        "calibration_learned": calibration(learned_results),
        "adversarial": {
            "cold_start": cold_adv_metrics.to_dict(),
            "learned": learned_adv_metrics.to_dict(),
        },
        "unsafe_autonomy_rate": {
            "cold_start": cold_metrics.unsafe_autonomy_rate,
            "learned": learned_metrics.unsafe_autonomy_rate,
        },
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "report.md").write_text(
        format_markdown(report), encoding="utf-8"
    )
    return report


def format_markdown(report: dict) -> str:
    cold = report["phase1_cold_start"]
    learned = report["phase3_learned"]
    delta = report["delta"]
    adv = report["adversarial"]

    def pct(x: float) -> str:
        return f"{x:.2%}"

    lines = [
        "# Wajo proactive agent -- evaluation report",
        "",
        f"- Main dataset: {report['dataset']['main_size']} scenarios",
        f"- Adversarial dataset: {report['dataset']['adversarial_size']} scenarios",
        "",
        "## Baseline vs learned policy",
        "",
        "| Metric | Cold start | After learning | Delta |",
        "|---|---|---|---|",
        f"| Accuracy | {pct(cold['accuracy'])} | {pct(learned['accuracy'])} | {pct(delta['accuracy_delta'])} |",
        f"| Ask rate | {pct(cold['ask_rate'])} | {pct(learned['ask_rate'])} | {pct(delta['ask_rate_delta'])} |",
        f"| Autonomous handling rate | {pct(cold['autonomous_handling_rate'])} | {pct(learned['autonomous_handling_rate'])} | {pct(delta['autonomous_handling_delta'])} |",
        f"| Over-asking rate | {pct(cold['over_asking_rate'])} | {pct(learned['over_asking_rate'])} | {pct(delta['over_asking_delta'])} |",
        f"| **Unsafe autonomy rate** | **{pct(cold['unsafe_autonomy_rate'])}** | **{pct(learned['unsafe_autonomy_rate'])}** | -- |",
        "",
        "## Adversarial dataset",
        "",
        "| Metric | Cold start | After learning |",
        "|---|---|---|",
        f"| Accuracy | {pct(adv['cold_start']['accuracy'])} | {pct(adv['learned']['accuracy'])} |",
        f"| Unsafe autonomy rate | {pct(adv['cold_start']['unsafe_autonomy_rate'])} | {pct(adv['learned']['unsafe_autonomy_rate'])} |",
        "",
        "## Calibration (confidence vs accuracy)",
        "",
        "### Cold start",
        "",
        _calibration_table(report["calibration_cold_start"]),
        "",
        "### After learning",
        "",
        _calibration_table(report["calibration_learned"]),
        "",
        "## Feedback plan",
        "",
        "| Action type | Sender | Signal | Count | Default decision |",
        "|---|---|---|---|---|",
    ]
    for entry in report["dataset"]["feedback_plan"]:
        lines.append(
            f"| {entry['action_type']} | {entry['sender_category']} | "
            f"{entry['signal']} | {entry['count']} | {entry['default_decision']} |"
        )
    return "\n".join(lines) + "\n"


def _calibration_table(rows: list[dict]) -> str:
    if not rows:
        return "_No predictions in these confidence bins._"
    header = "| Confidence bin | Count | Accuracy |"
    sep = "|---|---|---|"
    body = [f"| {r['confidence_bin']} | {r['count']} | {r['accuracy']:.2%} |" for r in rows]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    report = run_experiment()
    cold = report["phase1_cold_start"]
    learned = report["phase3_learned"]
    delta = report["delta"]
    print(f"dataset: {report['dataset']['main_size']} + "
          f"{report['dataset']['adversarial_size']} adversarial")
    print(f"accuracy       cold={cold['accuracy']:.2%} learned={learned['accuracy']:.2%} "
          f"({delta['accuracy_delta']:+.2%})")
    print(f"ask rate       cold={cold['ask_rate']:.2%} learned={learned['ask_rate']:.2%} "
          f"({delta['ask_rate_delta']:+.2%})")
    print(f"auto handling  cold={cold['autonomous_handling_rate']:.2%} "
          f"learned={learned['autonomous_handling_rate']:.2%} "
          f"({delta['autonomous_handling_delta']:+.2%})")
    print(f"unsafe autonomy cold={cold['unsafe_autonomy_rate']:.2%} "
          f"learned={learned['unsafe_autonomy_rate']:.2%}")
    print(f"reports written to {REPORTS_DIR}")
