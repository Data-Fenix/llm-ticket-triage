"""Synchronous classification with structured outputs.

``client.messages.parse`` sends the Pydantic schema as ``output_config.format``
so the API constrains generation to valid JSON for that schema, then validates
the response into a ``TicketTriage`` instance. The system prompt carries a
cache breakpoint, so across a run only the ticket text is billed at full price.
"""

from __future__ import annotations

import anthropic

from .config import settings
from .cost import estimate
from .prompts import Prompt, load_prompt
from .schemas import Ticket, TicketTriage, TriageRecord


def build_request(ticket: Ticket, prompt: Prompt, model: str | None = None) -> dict:
    """The request body shared by the sync path and the batch path (minus the schema)."""
    return {
        "model": model or settings.model,
        "max_tokens": settings.max_tokens,
        "system": [{"type": "text", "text": prompt.system, "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": settings.effort},
        "messages": [{"role": "user", "content": f"<ticket>\n{ticket.text}\n</ticket>"}],
    }


class Classifier:
    def __init__(self, client: anthropic.Anthropic | None = None, prompt: Prompt | None = None, model: str | None = None):
        self.client = client or anthropic.Anthropic()
        self.prompt = prompt or load_prompt()
        self.model = model or settings.model

    def classify(self, ticket: Ticket) -> TriageRecord:
        req = build_request(ticket, self.prompt, self.model)
        base = {"ticket_id": ticket.id, "prompt_version": self.prompt.label, "model": self.model}
        try:
            msg = self.client.messages.parse(output_format=TicketTriage, **req)
        except anthropic.RateLimitError as exc:
            return TriageRecord(**base, result=None, error=f"rate_limited: {exc.message}")
        except anthropic.APIStatusError as exc:
            return TriageRecord(**base, result=None, error=f"api_error {exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError as exc:
            return TriageRecord(**base, result=None, error=f"connection_error: {exc}")

        usage = msg.usage
        record = TriageRecord(
            **base,
            result=None,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cost_usd=estimate(self.model, usage),
        )
        if msg.stop_reason == "refusal":
            record.error = "refusal"
        elif msg.stop_reason == "max_tokens":
            record.error = "max_tokens: output truncated"
        elif msg.parsed_output is None:
            record.error = "no structured output in response"
        else:
            record.result = msg.parsed_output
        return record

    def classify_many(self, tickets: list[Ticket]) -> list[TriageRecord]:
        return [self.classify(t) for t in tickets]
