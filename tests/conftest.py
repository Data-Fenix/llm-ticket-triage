"""Fakes for the Anthropic client. No network in tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from triage.schemas import Category, Priority, Sentiment, TicketTriage


def usage(inp=300, out=80, cache_read=0):
    return SimpleNamespace(input_tokens=inp, output_tokens=out, cache_read_input_tokens=cache_read, cache_creation_input_tokens=0)


def triage(category="billing", priority="high", sentiment="neutral", **kw) -> TicketTriage:
    return TicketTriage(
        category=Category(category),
        priority=Priority(priority),
        sentiment=Sentiment(sentiment),
        product=kw.get("product"),
        requested_action=kw.get("requested_action", "Refund the duplicate charge."),
        summary=kw.get("summary", "Customer reports a duplicate charge."),
        needs_human=kw.get("needs_human", True),
        confidence=kw.get("confidence", 0.9),
    )


def parsed_message(result: TicketTriage | None, stop_reason="end_turn", u=None):
    return SimpleNamespace(parsed_output=result, stop_reason=stop_reason, usage=u or usage())


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.batches = FakeBatches()

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeBatches:
    """Scripted batch lifecycle: create -> retrieve (in_progress, ended) -> results."""

    def __init__(self):
        self.created = None
        self.results_items = []
        self._retrieves = 0

    def create(self, requests):
        self.created = requests
        return SimpleNamespace(id="msgbatch_test", processing_status="in_progress", request_counts=counts(0, 0, len(requests)))

    def retrieve(self, batch_id):
        self._retrieves += 1
        done = self._retrieves >= 2
        return SimpleNamespace(
            id=batch_id,
            processing_status="ended" if done else "in_progress",
            request_counts=counts(len(self.results_items), 0, 0 if done else len(self.results_items)),
        )

    def results(self, batch_id):
        return iter(self.results_items)


def counts(succeeded, errored, processing):
    return SimpleNamespace(succeeded=succeeded, errored=errored, processing=processing, canceled=0, expired=0)


def batch_item(custom_id, text=None, result_type="succeeded", stop_reason="end_turn"):
    if result_type != "succeeded":
        return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type=result_type, error=SimpleNamespace(message="boom")))
    msg = SimpleNamespace(stop_reason=stop_reason, usage=usage(), content=[SimpleNamespace(type="text", text=text or "")])
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="succeeded", message=msg))


class FakeClient:
    def __init__(self, responses=()):
        self.messages = FakeMessages(responses)


@pytest.fixture
def fake_client():
    return FakeClient
