"""Streamlit review UI: classify a ticket, inspect the structured result, browse eval reports.

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.classifier import Classifier  # noqa: E402
from triage.config import settings  # noqa: E402
from triage.prompts import list_versions, load_prompt  # noqa: E402
from triage.schemas import Ticket  # noqa: E402

st.set_page_config(page_title="LLM Ticket Triage", page_icon="🗂️", layout="wide")
st.title("LLM Ticket Triage")
st.caption("Schema-enforced classification with Claude structured outputs, versioned prompts and eval gates.")

tab_classify, tab_reports = st.tabs(["Classify", "Eval reports"])

with tab_classify:
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        version = st.selectbox("Prompt version", list_versions(), index=len(list_versions()) - 1)
        subject = st.text_input("Subject", "Charged twice for order 8841")
        body = st.text_area(
            "Body",
            "Hi, I see two identical charges of 1,299 EUR on my card for order 8841. Please refund one of them, I need the money back this week.",
            height=180,
        )
        go = st.button("Triage ticket", type="primary")
    with col2:
        if go:
            with st.spinner("Calling model…"):
                rec = Classifier(prompt=load_prompt(version)).classify(Ticket(id="ui", subject=subject, body=body))
            if rec.result is None:
                st.error(f"No result: {rec.error}")
            else:
                r = rec.result
                a, b, c = st.columns(3)
                a.metric("Category", r.category.value)
                b.metric("Priority", r.priority.value)
                c.metric("Sentiment", r.sentiment.value)
                st.write(f"**Summary:** {r.summary}")
                st.write(f"**Requested action:** {r.requested_action}")
                st.write(f"**Product:** {r.product or '—'}   |   **Needs human:** {r.needs_human}   |   **Confidence:** {r.confidence:.2f}")
                st.json(rec.model_dump(mode="json"))
                st.caption(
                    f"{rec.model} · prompt {rec.prompt_version} · {rec.input_tokens} in / {rec.output_tokens} out · "
                    f"cache read {rec.cache_read_tokens} · ~${rec.cost_usd:.5f}"
                )

with tab_reports:
    reports = sorted(settings.reports_dir.glob("eval_*.json")) if settings.reports_dir.exists() else []
    if not reports:
        st.info("No reports yet. Run `python -m triage.cli eval` first.")
    for path in reports:
        rep = json.loads(path.read_text(encoding="utf-8"))
        with st.expander(f"{path.stem} · {'PASS' if rep['passed'] else 'FAIL'} · {rep['run_at']}", expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Schema valid", f"{rep['schema_valid_rate']:.2f}")
            m2.metric("Category acc", f"{rep['category_accuracy']:.2f}")
            m3.metric("Priority acc", f"{rep['priority_accuracy']:.2f}")
            m4.metric("Cost", f"${rep['cost_usd']}")
            st.write("Confusion matrix (rows = gold, columns = predicted)")
            st.dataframe(rep["confusion"])
            if rep["mistakes"]:
                st.dataframe(rep["mistakes"])
