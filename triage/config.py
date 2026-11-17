"""Environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("MODEL", "claude-opus-5-5")
    effort: str = os.getenv("EFFORT", "low")  # classification is a low-effort task
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))
    prompt_version: str = os.getenv("PROMPT_VERSION", "v2")
    prompts_dir: Path = ROOT / "triage" / "prompts"
    data_file: Path = ROOT / "data" / "tickets_labeled.jsonl"
    reports_dir: Path = ROOT / "reports"
    batch_poll_seconds: int = int(os.getenv("BATCH_POLL_SECONDS", "15"))
    # Eval gates: a prompt or model change ships only if these hold on the labelled set.
    gate_category_accuracy: float = float(os.getenv("GATE_CATEGORY_ACCURACY", "0.85"))
    gate_priority_accuracy: float = float(os.getenv("GATE_PRIORITY_ACCURACY", "0.70"))
    gate_schema_valid_rate: float = float(os.getenv("GATE_SCHEMA_VALID_RATE", "1.0"))


settings = Settings()
