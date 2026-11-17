import json

from tests.conftest import FakeClient, batch_item, triage
from triage import batch as batch_mod
from triage.batch import build_batch_requests, run_batch
from triage.prompts import load_prompt
from triage.schemas import Ticket

TICKETS = [
    Ticket(id="a", subject="Charged twice", body="Two charges."),
    Ticket(id="b", subject="Dark mode", body="Please add dark mode."),
    Ticket(id="c", subject="Parcel lost", body="Where is it?"),
]


def test_batch_requests_carry_custom_id_schema_and_cache(monkeypatch):
    reqs = build_batch_requests(TICKETS, load_prompt("v2"))
    assert [r["custom_id"] for r in reqs] == ["a", "b", "c"]
    p = reqs[0]["params"]
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert p["output_config"]["effort"] == "low"
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_run_batch_joins_results_by_id_and_handles_failures(monkeypatch):
    monkeypatch.setattr(batch_mod.time, "sleep", lambda s: None)
    client = FakeClient()
    good = triage().model_dump_json()
    client.messages.batches.results_items = [
        batch_item("c", text=json.dumps({"category": "shipping", "priority": "medium", "sentiment": "neutral", "product": None,
                                         "requested_action": "Find it.", "summary": "Lost parcel.", "needs_human": False, "confidence": 0.7})),
        batch_item("b", text='{"category": "unknown_cat"}'),  # schema invalid
        batch_item("a", text=good),
    ]
    statuses = []
    records = run_batch(TICKETS, client=client, prompt=load_prompt("v2"), on_status=statuses.append)
    # input order preserved, joined by custom_id not position
    assert [r.ticket_id for r in records] == ["a", "b", "c"]
    assert records[0].result.category.value == "billing"
    assert records[1].result is None and records[1].error.startswith("schema_invalid")
    assert records[2].result.category.value == "shipping"
    assert records[0].cost_usd > 0 and statuses and "ended" in statuses[-1]


def test_run_batch_reports_errored_and_missing(monkeypatch):
    monkeypatch.setattr(batch_mod.time, "sleep", lambda s: None)
    client = FakeClient()
    client.messages.batches.results_items = [batch_item("a", result_type="errored")]
    records = run_batch(TICKETS, client=client, prompt=load_prompt("v1"))
    assert records[0].error.startswith("errored")
    assert records[1].error == "missing_from_batch" and records[2].error == "missing_from_batch"
