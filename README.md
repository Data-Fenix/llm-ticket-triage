# LLM Ticket Triage Pipeline

A production-style LLMOps pipeline that turns free-text support tickets into
typed triage records: category, priority, sentiment, product, requested action,
summary, whether a human is needed, and a confidence score.

Built on the Anthropic Claude API with structured outputs, so every record is
schema-valid JSON, and operated the way an ML model would be: versioned
prompts, a labelled dataset, accuracy gates in CI, cost tracking and a cheap
batch mode for bulk runs.

```
tickets.jsonl ──► prompt registry (v1, v2, …; content-hashed)
                       │
                       ▼
            Claude structured outputs ── Pydantic schema sent as output_config.format
             sync (messages.parse)  │  batch (Message Batches API, 50% cost)
                       │
                       ▼
            TriageRecord per ticket ── result | error, tokens, cache hits, cost
                       │
                       ▼
            evaluate ── accuracy, macro-F1, confusion matrix, gates ── exit 1 on failure
```

## GenAI / LLMOps practices demonstrated

| Practice | Where |
|---|---|
| Structured outputs: one Pydantic model drives the API schema, validation and types | `triage/schemas.py`, `triage/classifier.py` |
| Prompt registry: prompts as versioned, content-hashed files; version pinned in every output record | `triage/prompts/` |
| Eval-gated releases: labelled set, per-field metrics, confusion matrix, thresholds, non-zero exit in CI | `triage/evaluate.py`, `.github/workflows/ci.yml` |
| A/B prompt comparison before rollout | `python -m triage.cli compare v1 v2` |
| Message Batches for bulk and eval runs at half price; results joined by `custom_id`, never by position | `triage/batch.py` |
| Prompt caching on the system prompt; low effort for a classification task | `triage/classifier.py` |
| Stop-reason handling (`refusal`, `max_tokens`), typed error handling, errors kept as records not exceptions | `triage/classifier.py`, `triage/batch.py` |
| Ticket text wrapped as untrusted input; injection example in the dataset (`t049`) | `triage/prompts/v2.md`, `data/` |
| Cost and token accounting per record and per run | `triage/cost.py` |
| Review UI for humans in the loop | `app/streamlit_app.py` |

## Quick start

```bash
git clone https://github.com/Data-Fenix/llm-ticket-triage
cd llm-ticket-triage
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                             # add ANTHROPIC_API_KEY

# one ticket
python -m triage.cli classify "Charged twice for order 8841

I see two identical charges of 1,299 EUR. Please refund one."

# evaluate the current prompt against the labelled set (50 tickets)
python -m triage.cli eval                 # synchronous
python -m triage.cli eval --batch         # Message Batches, half the cost, a few minutes

# compare prompt versions
python -m triage.cli compare v1 v2 --batch

# bulk run on your own file
python -m triage.cli run --input data/tickets_labeled.jsonl --output reports/predictions.jsonl --batch

# review UI
streamlit run app/streamlit_app.py
```

Input format: JSON lines with `id`, `subject`, `body` and optional gold labels
`category`, `priority`, `sentiment`.

## Tests

```bash
pytest          # 18 tests, no network: fake client for parse and batches
ruff check .
```

## Changing the prompt

1. Copy `triage/prompts/v2.md` to `v3.md`, edit, update the frontmatter notes.
2. Run `python -m triage.cli compare v2 v3 --batch` and look at accuracy,
   macro-F1 and the confusion matrix in `reports/`.
3. If v3 clears the gates, set `PROMPT_VERSION=v3` in `.env`. The CI eval gate
   re-checks it on `main`.

## Why these gates

Category accuracy drives routing, so it has the strictest gate. Priority is
subjective between adjacent levels, so the gate is looser and the report also
shows "within one level" accuracy. Schema validity must be 100%: structured
outputs make that the API's job, and the gate catches regressions if the
schema and the prompt drift apart.

## Cost

A 50-ticket eval costs a few cents on `claude-opus-5-5` at `low` effort with
caching, and half that through the Batch API. Set `MODEL=claude-haiku-5-5` for
high-volume routing once the eval shows it clears the gates.
