from tests.conftest import triage
from triage.evaluate import evaluate, format_report, load_tickets, macro_f1
from triage.schemas import TriageRecord


def rec(t, **kw):
    return TriageRecord(ticket_id=t.id, result=triage(**kw), prompt_version="v2@abc", model="claude-opus-5-5", cost_usd=0.001)


def test_macro_f1_perfect_and_partial():
    assert macro_f1([("a", "a"), ("b", "b")], ["a", "b"]) == 1.0
    assert 0 < macro_f1([("a", "a"), ("b", "a")], ["a", "b"]) < 1


def test_perfect_predictions_pass_gates():
    tickets = load_tickets()
    records = [rec(t, category=t.category.value, priority=t.priority.value, sentiment=t.sentiment.value) for t in tickets]
    r = evaluate(tickets, records)
    assert r["category_accuracy"] == 1.0 and r["priority_accuracy"] == 1.0 and r["schema_valid_rate"] == 1.0
    assert r["passed"] and r["cost_usd"] > 0
    assert "PASS" in format_report(r)


def test_bad_predictions_fail_gates_and_list_mistakes():
    tickets = load_tickets()[:10]
    records = [rec(t, category="other", priority="low", sentiment="neutral") for t in tickets]
    records[0] = TriageRecord(ticket_id=tickets[0].id, result=None, error="refusal", prompt_version="v2@abc", model="m")
    r = evaluate(tickets, records)
    assert not r["passed"]
    assert r["schema_valid_rate"] == 0.9 and r["errors"]["refusal"] == 1
    assert r["mistakes"] and r["mistakes"][0]["field"] == "category"
    assert "FAIL" in format_report(r)
