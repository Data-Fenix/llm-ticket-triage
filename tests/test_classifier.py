import anthropic
import httpx

from tests.conftest import FakeClient, parsed_message, triage, usage
from triage.classifier import Classifier
from triage.prompts import load_prompt
from triage.schemas import Ticket, TicketTriage

T = Ticket(id="t1", subject="Charged twice", body="Two charges for order 1.")


def test_classify_returns_validated_record_with_cost():
    client = FakeClient([parsed_message(triage(), u=usage(inp=500, out=90, cache_read=1200))])
    rec = Classifier(client=client, prompt=load_prompt("v2")).classify(T)
    assert rec.result and rec.result.category.value == "billing"
    assert rec.cache_read_tokens == 1200 and rec.cost_usd > 0
    assert rec.prompt_version.startswith("v2@") and rec.error is None


def test_request_uses_structured_outputs_cache_and_low_effort():
    client = FakeClient([parsed_message(triage())])
    Classifier(client=client, prompt=load_prompt("v2")).classify(T)
    req = client.messages.calls[0]
    assert req["output_format"] is TicketTriage
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["output_config"]["effort"] == "low"
    assert "<ticket>" in req["messages"][0]["content"] and "Charged twice" in req["messages"][0]["content"]


def test_refusal_and_truncation_are_recorded_not_raised():
    client = FakeClient([parsed_message(None, stop_reason="refusal"), parsed_message(None, stop_reason="max_tokens")])
    c = Classifier(client=client, prompt=load_prompt("v1"))
    assert c.classify(T).error == "refusal"
    assert c.classify(T).error.startswith("max_tokens")


def test_api_errors_become_error_records():
    resp = httpx.Response(429, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    err = anthropic.RateLimitError("slow down", response=resp, body=None)
    client = FakeClient([err])
    rec = Classifier(client=client, prompt=load_prompt("v1")).classify(T)
    assert rec.result is None and rec.error.startswith("rate_limited")
