"""Pydantic schemas: the single source of truth for what the model must return.

The same model class drives three things:
1. the JSON schema sent to the API as ``output_config.format`` (structured outputs),
2. validation of every response (``messages.parse`` and batch results),
3. the typed objects the rest of the pipeline works with.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Category(StrEnum):
    billing = "billing"
    technical_issue = "technical_issue"
    account_access = "account_access"
    feature_request = "feature_request"
    shipping = "shipping"
    cancellation = "cancellation"
    other = "other"


class Priority(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class Sentiment(StrEnum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"


class TicketTriage(BaseModel):
    """Structured triage result for one support ticket."""

    category: Category = Field(description="Primary issue category.")
    priority: Priority = Field(
        description=(
            "urgent: service down, data loss, security, or legal threat; high: customer blocked from core use; "
            "medium: degraded experience or money question; low: cosmetic, informational, feature idea."
        )
    )
    sentiment: Sentiment
    product: str | None = Field(description="Product or plan mentioned, verbatim, or null if none.")
    requested_action: str = Field(description="What the customer wants to happen, in one short sentence.")
    summary: str = Field(description="One-sentence neutral summary for the support queue, max 25 words.")
    needs_human: bool = Field(description="True if an agent must reply (refund, legal, security, angry customer).")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-assessed confidence in the category, 0 to 1.")


class Ticket(BaseModel):
    id: str
    subject: str
    body: str
    # Optional gold labels for evaluation
    category: Category | None = None
    priority: Priority | None = None
    sentiment: Sentiment | None = None

    @property
    def text(self) -> str:
        return f"Subject: {self.subject}\n\n{self.body}"


class TriageRecord(BaseModel):
    """One pipeline output row: the ticket id, the result (or failure) and usage."""

    ticket_id: str
    result: TicketTriage | None
    error: str | None = None
    prompt_version: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
