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
from .metrics import (
    calibration,
    compare,
    compute_metrics,
    confidence_summary,
    expected_calibration_error,
)

EVALS_DIR = Path(__file__).parent
REPORTS_DIR = EVALS_DIR / "reports"

USER_ID = "default"

# Feedback levels for the learning-curve experiment: how many of each
# feedback group have been given (0 = cold start, 10 = fully trained).
LEARNING_CURVE_LEVELS = (0, 3, 5, 6, 7, 8, 10)

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
                "attack_type": scenario.get("attack_type", ""),
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


def apply_feedback(agent: EmailAgent, level: int = 10) -> None:
    """Record ``level`` examples from each feedback group.

    ``level=0`` is the cold start; ``level=10`` is the fully-trained plan.
    """
    engine = FeedbackEngine(agent.preferences)
    for action_type, sender, signal, count, default in FEEDBACK_PLAN:
        for _ in range(min(count, level)):
            engine.record(
                USER_ID,
                action_type=action_type,
                sender_category=sender,
                signal=signal,
                default_decision=default,
            )


def run_learning_curve(
    dataset_path: Path = DATASET_PATH,
    adversarial_path: Path = ADVERSARIAL_PATH,
    levels: tuple[int, ...] = LEARNING_CURVE_LEVELS,
) -> list[dict]:
    """Measure metrics as feedback accumulates (0 = cold start).

    Demonstrates the calibrated-autonomy story: autonomy grows and asks fall
    as the agent learns the user's preferences, while the unsafe autonomy
    rate stays pinned at zero.
    """
    scenarios = load_scenarios(dataset_path)
    adversarial = load_scenarios(adversarial_path)
    points = []
    for level in levels:
        agent = EmailAgent()
        apply_feedback(agent, level)
        metrics = compute_metrics(run_scenarios(agent, scenarios))
        adv_metrics = compute_metrics(run_scenarios(agent, adversarial))
        points.append(
            {
                "feedback_examples_per_group": level,
                "accuracy": metrics.accuracy,
                "ask_rate": metrics.ask_rate,
                "autonomous_handling_rate": metrics.autonomous_handling_rate,
                "over_asking_rate": metrics.over_asking_rate,
                "unsafe_autonomy_rate": metrics.unsafe_autonomy_rate,
                "adversarial_unsafe_autonomy_rate": adv_metrics.unsafe_autonomy_rate,
            }
        )
    return points


def key_results(cold: dict, learned: dict, dataset: dict) -> dict:
    """The headline numbers for the README (measured, not claimed)."""
    return {
        "dataset_size": dataset["main_size"],
        "autonomy_on_safe_tasks": {
            "before": round(cold["autonomous_handling_rate"], 4),
            "after": round(learned["autonomous_handling_rate"], 4),
            "delta": round(
                learned["autonomous_handling_rate"] - cold["autonomous_handling_rate"], 4
            ),
        },
        "unnecessary_asks": {
            "before": round(cold["ask_rate"], 4),
            "after": round(learned["ask_rate"], 4),
            "delta": round(learned["ask_rate"] - cold["ask_rate"], 4),
        },
        "unsafe_autonomous_actions": {
            "before": round(cold["unsafe_autonomy_rate"], 4),
            "after": round(learned["unsafe_autonomy_rate"], 4),
        },
    }


def adversarial_breakdown(results: list[dict]) -> list[dict]:
    """Per-attack-type stats for the adversarial dataset.

    Reports, for each attack category, how many cases were blocked
    (non-autonomous) and how many slipped through as autonomous.
    """
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r.get("attack_type") or r.get("risk_class", "adversarial")].append(r)

    rows = []
    for attack_type, cases in sorted(by_type.items()):
        non_autonomous = sum(1 for r in cases if r["decision"] != "escalate" and r["decision"] not in ("silent", "act_notify"))
        rows.append(
            {
                "attack_type": attack_type,
                "count": len(cases),
                "autonomous_violations": sum(
                    1 for r in cases if r["decision"] in ("silent", "act_notify")
                ),
                "escalated": sum(1 for r in cases if r["decision"] == "escalate"),
            }
        )
    return rows


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
        "confidence": {
            "cold_start": confidence_summary(cold_results),
            "learned": confidence_summary(learned_results),
        },
        "ece": {
            "cold_start": expected_calibration_error(cold_results),
            "learned": expected_calibration_error(learned_results),
        },
        "learning_curve": run_learning_curve(),
        "adversarial": {
            "cold_start": cold_adv_metrics.to_dict(),
            "learned": learned_adv_metrics.to_dict(),
            "breakdown": adversarial_breakdown(learned_adv_results),
        },
        "unsafe_autonomy_rate": {
            "cold_start": cold_metrics.unsafe_autonomy_rate,
            "learned": learned_metrics.unsafe_autonomy_rate,
        },
        "key_results": key_results(
            cold_metrics.to_dict(), learned_metrics.to_dict(), {"main_size": len(scenarios)}
        ),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "report.md").write_text(
        format_markdown(report), encoding="utf-8"
    )
    (reports_dir / "key_results.json").write_text(
        json.dumps(report["key_results"], indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "learning_curve.md").write_text(
        format_learning_curve(report["learning_curve"]), encoding="utf-8"
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
        "### Breakdown by attack type (learned policy)",
        "",
        "| Attack type | Cases | Escalated | Autonomous violations |",
        "|---|---|---|---|",
    ]
    for row in adv.get("breakdown", []):
        lines.append(
            f"| {row['attack_type']} | {row['count']} | {row['escalated']} | "
            f"{row['autonomous_violations']} |"
        )
    lines += ["", "## Calibration (confidence vs accuracy)", "", "### Cold start", "", _calibration_table(report["calibration_cold_start"]), "", "### After learning", "", _calibration_table(report["calibration_learned"]), ""]

    conf_cold = report["confidence"]["cold_start"]
    conf_learned = report["confidence"]["learned"]
    lines += [
        "## Confidence summary",
        "",
        "| Metric | Cold start | After learning |",
        "|---|---|---|",
        f"| Mean confidence | {conf_cold['mean_confidence']:.3f} | {conf_learned['mean_confidence']:.3f} |",
        f"| High-confidence predictions (>=0.9) | {conf_cold['high_confidence_predictions']} | {conf_learned['high_confidence_predictions']} |",
        f"| High-confidence error rate | {conf_cold['high_confidence_error_rate']:.2%} | {conf_learned['high_confidence_error_rate']:.2%} |",
        f"| ECE (expected calibration error) | {report['ece']['cold_start']:.3f} | {report['ece']['learned']:.3f} |",
        "",
        "## Learning curve",
        "",
    ]
    lines += format_learning_curve(report["learning_curve"]).splitlines()
    lines += ["", "## Feedback plan", "", "| Action type | Sender | Signal | Count | Default decision |", "|---|---|---|---|---|", ]
    for entry in report["dataset"]["feedback_plan"]:
        lines.append(
            f"| {entry['action_type']} | {entry['sender_category']} | "
            f"{entry['signal']} | {entry['count']} | {entry['default_decision']} |"
        )
    return "\n".join(lines) + "\n"


def format_learning_curve(points: list[dict]) -> str:
    """Markdown table for the learning-curve experiment."""
    def pct(x: float) -> str:
        return f"{x:.2%}"

    lines = [
        "| Feedback examples | Accuracy | Ask rate | Autonomous handling | Unsafe autonomy |",
        "|---|---|---|---|---|",
    ]
    for p in points:
        lines.append(
            f"| {p['feedback_examples_per_group']} | {pct(p['accuracy'])} | "
            f"{pct(p['ask_rate'])} | {pct(p['autonomous_handling_rate'])} | "
            f"{pct(p['unsafe_autonomy_rate'])} |"
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
    print("---")
    print("learning curve (feedback examples per group -> metrics):")
    for point in report["learning_curve"]:
        print(
            f"  {point['feedback_examples_per_group']:>2} -> acc={point['accuracy']:.2%} "
            f"ask={point['ask_rate']:.2%} auto={point['autonomous_handling_rate']:.2%} "
            f"unsafe={point['unsafe_autonomy_rate']:.2%}"
        )
    print("---")
    print("key results:", json.dumps(report["key_results"]))
    print(f"reports written to {REPORTS_DIR}")
