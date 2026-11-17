"""Token pricing and cost estimation."""

from __future__ import annotations

# USD per million tokens. Batch runs are billed at 50% of these rates.
PRICING = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_write": 2.5, "cache_read": 0.10},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
}
BATCH_DISCOUNT = 0.5


def estimate(model: str, usage, batch: bool = False) -> float:
    p = PRICING.get(model)
    if not p or usage is None:
        return 0.0
    cost = (
        (getattr(usage, "input_tokens", 0) or 0) * p["input"]
        + (getattr(usage, "output_tokens", 0) or 0) * p["output"]
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * p["cache_write"]
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * p["cache_read"]
    ) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost
