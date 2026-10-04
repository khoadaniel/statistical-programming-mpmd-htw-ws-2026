"""A small dashboard for the service: suggest headings for a description of goods and show the metadata.

Two ways to run it:

    # 1. on its own, with the model loaded in the dashboard (uv sync --extra dashboard)
    MODEL_DIR=models uv run streamlit run dashboard/streamlit_app.py

    # 2. as the front end of the API: the dashboard sends each description to the back end
    API_URL=http://localhost:8000 uv run streamlit run dashboard/streamlit_app.py

compose.yaml uses the second way: front end, back end and database each run in their own container.
On Streamlit Community Cloud, point the app to this file; in the first way the model files must then be in the
repository or be downloaded at start-up (see README, "Publishing a dashboard").
"""

import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL")  # set: front end of the API; unset: load the model here


def call_api(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(f"{API_URL}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


@st.cache_resource  # load once per server process, not on every interaction
def get_model():
    from tariff_service.model import load_model  # only needed without an API

    return load_model(Path(os.environ.get("MODEL_DIR", "models")))


def suggest(text: str) -> tuple[pd.DataFrame, dict]:
    if API_URL:
        out = call_api("/predict", {"description": text})
        rows = [{"heading": t["heading"], "score": t["score"], "text": t["heading_description"]} for t in out["top"]]
        return pd.DataFrame(rows), call_api("/metadata")
    from tariff_service.model import decision_text, top_k

    pipe, meta = get_model()
    ranked = top_k(pipe, [decision_text(text)], k=3)[0]
    rows = [{"heading": h, "score": s, "text": meta.get("headings", {}).get(h, "")} for h, s in ranked]
    return pd.DataFrame(rows), {k: v for k, v in meta.items() if k != "headings"}


st.title("Tariff heading suggestion")
st.caption("A suggestion for a customs officer, not a decision" + (f" · back end: {API_URL}" if API_URL else ""))
text = st.text_area("Description of goods (any EU language)", "Damenstiefel mit Oberteil aus Rindleder")
if text.strip():
    table, meta = suggest(text)
    st.caption(f"Model {meta['model_version']} · validation accuracy {meta.get('validation', {}).get('accuracy')}")
    st.dataframe(table)
    with st.expander("Model metadata"):
        st.json(meta)
