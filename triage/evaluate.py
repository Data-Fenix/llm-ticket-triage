"""Evaluation against the labelled dataset, with release gates.

Metrics
- schema_valid_rate: fraction of tickets that produced a valid TicketTriage
- category accuracy and macro-F1, with a confusion matrix
- priority accuracy and "within one level" accuracy (an urgent/high swap is
  less harmful than urgent/low)
- sentiment accuracy
- cost and token totals for the run

Gates are read from settings; a failed gate returns exit code 1 so the eval
can block a merge or a prompt rollout in CI.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from .config import settings
from .schemas import Category, Priority, Ticket, TriageRecord

PRIORITY_RANK = {Priority.low: 0, Priority.medium: 1, Priority.high: 2, Priority.urgent: 3}


def load_tickets(path: Path | None = None) -> list[Ticket]:
    path = path or settings.data_file
    return [Ticket.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def macro_f1(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def evaluate(tickets: list[Ticket], records: list[TriageRecord]) -> dict:
    by_id = {r.ticket_id: r for r in records}
    n = len(tickets)
    valid = [t for t in tickets if by_id.get(t.id) and by_id[t.id].result is not None]
    cat_pairs, pri_pairs, sent_pairs = [], [], []
    confusion: dict[str, Counter] = defaultdict(Counter)
    mistakes = []
    for t in valid:
        r = by_id[t.id].result
        assert r is not None
        if t.category:
            cat_pairs.append((t.category.value, r.category.value))
            confusion[t.category.value][r.category.value] += 1
            if t.category != r.category:
                mistakes.append({"id": t.id, "field": "category", "gold": t.category.value, "pred": r.category.value, "subject": t.subject})
        if t.priority:
            pri_pairs.append((t.priority, r.priority))
        if t.sentiment:
            sent_pairs.append((t.sentiment.value, r.sentiment.value))

    cat_acc = sum(g == p for g, p in cat_pairs) / len(cat_pairs) if cat_pairs else 0.0
    pri_acc = sum(g == p for g, p in pri_pairs) / len(pri_pairs) if pri_pairs else 0.0
    pri_within1 = (
        sum(abs(PRIORITY_RANK[g] - PRIORITY_RANK[p]) <= 1 for g, p in pri_pairs) / len(pri_pairs) if pri_pairs else 0.0
    )
    sent_acc = sum(g == p for g, p in sent_pairs) / len(sent_pairs) if sent_pairs else 0.0
    errors = Counter((by_id[t.id].error or "ok").split(":")[0] for t in tickets if t.id in by_id)

    report = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": records[0].model if records else settings.model,
        "prompt_version": records[0].prompt_version if records else settings.prompt_version,
        "n": n,
        "schema_valid_rate": len(valid) / n if n else 0.0,
        "category_accuracy": cat_acc,
        "category_macro_f1": macro_f1(cat_pairs, [c.value for c in Category]),
        "priority_accuracy": pri_acc,
        "priority_within_one_level": pri_within1,
        "sentiment_accuracy": sent_acc,
        "confusion": {g: dict(c) for g, c in confusion.items()},
        "errors": dict(errors),
        "input_tokens": sum(r.input_tokens for r in records),
        "output_tokens": sum(r.output_tokens for r in records),
        "cache_read_tokens": sum(r.cache_read_tokens for r in records),
        "cost_usd": round(sum(r.cost_usd for r in records), 4),
        "mistakes": mistakes,
    }
    report["gates"] = {
        "schema_valid_rate": report["schema_valid_rate"] >= settings.gate_schema_valid_rate,
        "category_accuracy": cat_acc >= settings.gate_category_accuracy,
        "priority_accuracy": pri_acc >= settings.gate_priority_accuracy,
    }
    report["passed"] = all(report["gates"].values())
    return report


def format_report(report: dict) -> str:
    lines = [
        f"model={report['model']} prompt={report['prompt_version']} n={report['n']}",
        f"schema valid      {report['schema_valid_rate']:.3f}  (gate {settings.gate_schema_valid_rate})",
        f"category accuracy {report['category_accuracy']:.3f}  (gate {settings.gate_category_accuracy})   macro-F1 {report['category_macro_f1']:.3f}",
        f"priority accuracy {report['priority_accuracy']:.3f}  (gate {settings.gate_priority_accuracy})   within-1 {report['priority_within_one_level']:.3f}",
        f"sentiment accuracy {report['sentiment_accuracy']:.3f}",
        f"tokens in/out/cached {report['input_tokens']}/{report['output_tokens']}/{report['cache_read_tokens']}   cost ${report['cost_usd']}",
        "errors " + json.dumps(report["errors"]),
        "RESULT: " + ("PASS" if report["passed"] else "FAIL " + str([k for k, v in report["gates"].items() if not v])),
    ]
    if report["mistakes"]:
        lines.append("category mistakes:")
        lines += [f"  {m['id']}: gold={m['gold']} pred={m['pred']}  ({m['subject']})" for m in report["mistakes"][:15]]
    return "\n".join(lines)


def save_report(report: dict, name: str) -> Path:
    settings.reports_dir.mkdir(exist_ok=True)
    path = settings.reports_dir / f"{name}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
