import pytest
from pydantic import TypeAdapter, ValidationError

from triage.batch import schema_format
from triage.evaluate import load_tickets
from triage.schemas import Category, Priority, TicketTriage


def test_schema_rejects_unknown_category_and_out_of_range_confidence():
    good = {
        "category": "billing", "priority": "high", "sentiment": "neutral", "product": None,
        "requested_action": "Refund.", "summary": "Dup charge.", "needs_human": True, "confidence": 0.8,
    }
    TicketTriage.model_validate(good)
    with pytest.raises(ValidationError):
        TicketTriage.model_validate({**good, "category": "refunds"})
    with pytest.raises(ValidationError):
        TicketTriage.model_validate({**good, "confidence": 1.5})


def test_json_schema_lists_all_enum_values():
    schema = TypeAdapter(TicketTriage).json_schema()
    defs = schema.get("$defs", {})
    assert set(defs["Category"]["enum"]) == {c.value for c in Category}
    assert set(defs["Priority"]["enum"]) == {p.value for p in Priority}


def test_schema_format_for_batch_is_json_schema_type():
    fmt = schema_format()
    assert fmt["type"] == "json_schema"
    assert "properties" in fmt["schema"] and "category" in fmt["schema"]["properties"]


def test_labelled_dataset_is_valid_and_balanced():
    tickets = load_tickets()
    assert len(tickets) >= 40
    assert all(t.category and t.priority and t.sentiment for t in tickets)
    per_cat = {c: sum(1 for t in tickets if t.category == c) for c in Category}
    assert min(per_cat.values()) >= 4, per_cat
    assert len({t.id for t in tickets}) == len(tickets)
