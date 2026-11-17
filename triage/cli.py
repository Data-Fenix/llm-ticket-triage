"""Command-line interface.

    python -m triage.cli classify "Subject...\n\nBody..."
    python -m triage.cli run   --input data/tickets_labeled.jsonl --output reports/predictions.jsonl [--batch]
    python -m triage.cli eval  [--batch] [--prompt v1] [--limit 20]
    python -m triage.cli compare v1 v2 [--batch]
"""

from __future__ import annotations

import argparse
import json
import sys

from .batch import run_batch, write_jsonl
from .classifier import Classifier
from .config import settings
from .evaluate import evaluate, format_report, load_tickets, save_report
from .prompts import list_versions, load_prompt
from .schemas import Ticket


def _records(tickets, prompt_version, batch):
    prompt = load_prompt(prompt_version)
    if batch:
        return run_batch(tickets, prompt=prompt, on_status=lambda s: print(s, file=sys.stderr))
    return Classifier(prompt=prompt).classify_many(tickets)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="triage")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify", help="Classify one ticket from the command line")
    c.add_argument("text")
    c.add_argument("--prompt", default=settings.prompt_version)

    r = sub.add_parser("run", help="Classify a JSONL file of tickets")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--batch", action="store_true")
    r.add_argument("--prompt", default=settings.prompt_version)

    e = sub.add_parser("eval", help="Evaluate against the labelled dataset and apply gates")
    e.add_argument("--batch", action="store_true")
    e.add_argument("--prompt", default=settings.prompt_version)
    e.add_argument("--limit", type=int)

    cmp_ = sub.add_parser("compare", help="Evaluate two prompt versions side by side")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--batch", action="store_true")

    sub.add_parser("prompts", help="List prompt versions")

    a = p.parse_args(argv)

    if a.cmd == "prompts":
        for v in list_versions():
            pr = load_prompt(v)
            print(f"{pr.label}  {pr.notes}")
        return 0

    if a.cmd == "classify":
        subject, _, body = a.text.partition("\n\n")
        ticket = Ticket(id="cli", subject=subject.strip() or "(no subject)", body=body.strip() or subject)
        rec = Classifier(prompt=load_prompt(a.prompt)).classify(ticket)
        print(json.dumps(rec.model_dump(mode="json"), indent=2))
        return 0 if rec.result else 1

    if a.cmd == "run":
        tickets = load_tickets(a.input)
        records = _records(tickets, a.prompt, a.batch)
        write_jsonl(records, a.output)
        ok = sum(1 for x in records if x.result)
        print(f"{ok}/{len(records)} classified, cost ${sum(x.cost_usd for x in records):.4f} -> {a.output}")
        return 0

    if a.cmd == "eval":
        tickets = load_tickets()
        if a.limit:
            tickets = tickets[: a.limit]
        records = _records(tickets, a.prompt, a.batch)
        report = evaluate(tickets, records)
        print(format_report(report))
        path = save_report(report, f"eval_{a.prompt}")
        print(f"report -> {path}")
        return 0 if report["passed"] else 1

    if a.cmd == "compare":
        tickets = load_tickets()
        for v in (a.a, a.b):
            print(f"\n=== {v} ===")
            report = evaluate(tickets, _records(tickets, v, a.batch))
            print(format_report(report))
            save_report(report, f"eval_{v}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
