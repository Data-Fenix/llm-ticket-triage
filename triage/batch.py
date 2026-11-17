"""Bulk classification with the Message Batches API.

Batches run asynchronously at 50% of the per-token price, which is the right
trade for nightly backfills and evaluation runs. Results arrive in any order,
so every request carries a ``custom_id`` equal to the ticket id and results
are joined by that id, never by position.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import anthropic
from pydantic import TypeAdapter, ValidationError

from .classifier import build_request
from .config import settings
from .cost import estimate
from .prompts import Prompt, load_prompt
from .schemas import Ticket, TicketTriage, TriageRecord


def schema_format() -> dict:
    """The ``output_config.format`` block for TicketTriage, as messages.parse would send it."""
    schema = TypeAdapter(TicketTriage).json_schema()
    try:
        from anthropic.lib._parse._transform import transform_schema

        schema = transform_schema(schema)
    except ImportError:  # SDK layout changed: send the plain schema
        pass
    return {"type": "json_schema", "schema": schema}


def build_batch_requests(tickets: list[Ticket], prompt: Prompt, model: str | None = None) -> list[dict]:
    fmt = schema_format()
    reqs = []
    for t in tickets:
        params = build_request(t, prompt, model)
        params["output_config"] = {**params["output_config"], "format": fmt}
        reqs.append({"custom_id": t.id, "params": params})
    return reqs


def _parse_result(ticket_id: str, item, prompt: Prompt, model: str) -> TriageRecord:
    base = {"ticket_id": ticket_id, "prompt_version": prompt.label, "model": model}
    r = item.result
    if r.type != "succeeded":
        detail = getattr(getattr(r, "error", None), "message", "") if r.type == "errored" else ""
        return TriageRecord(**base, result=None, error=f"{r.type}: {detail}".strip(": "))
    msg = r.message
    record = TriageRecord(
        **base,
        result=None,
        input_tokens=msg.usage.input_tokens,
        output_tokens=msg.usage.output_tokens,
        cache_read_tokens=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
        cost_usd=estimate(model, msg.usage, batch=True),
    )
    if msg.stop_reason == "refusal":
        record.error = "refusal"
        return record
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        record.result = TicketTriage.model_validate_json(text)
    except ValidationError as exc:
        record.error = f"schema_invalid: {exc.errors()[0]['msg']}"
    return record


def run_batch(
    tickets: list[Ticket],
    client: anthropic.Anthropic | None = None,
    prompt: Prompt | None = None,
    model: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> list[TriageRecord]:
    client = client or anthropic.Anthropic()
    prompt = prompt or load_prompt()
    model = model or settings.model
    by_id = {t.id: t for t in tickets}

    batch = client.messages.batches.create(requests=build_batch_requests(tickets, prompt, model))
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        c = batch.request_counts
        if on_status:
            on_status(f"batch {batch.id}: {batch.processing_status} (ok={c.succeeded} err={c.errored} pending={c.processing})")
        if batch.processing_status == "ended":
            break
        time.sleep(settings.batch_poll_seconds)

    records: dict[str, TriageRecord] = {}
    for item in client.messages.batches.results(batch.id):
        records[item.custom_id] = _parse_result(item.custom_id, item, prompt, model)
    # Preserve input order and surface anything the batch did not return.
    return [
        records.get(
            t.id, TriageRecord(ticket_id=t.id, result=None, error="missing_from_batch", prompt_version=prompt.label, model=model)
        )
        for t in tickets
        if t.id in by_id
    ]


def write_jsonl(records: list[TriageRecord], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(mode="json")) + "\n")
