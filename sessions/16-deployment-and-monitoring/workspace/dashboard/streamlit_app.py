"""A small dashboard for the service: suggest headings for a description of goods and show the metadata.

    uv sync --extra dashboard
    MODEL_DIR=models uv run streamlit run dashboard/streamlit_app.py

On Streamlit Community Cloud, point the app to this file; the model files must then be in the repository
or be downloaded at start-up (see README, "Publishing a dashboard").
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from tariff_service.model import decision_text, load_model, top_k


@st.cache_resource  # load once per server process, not on every interaction
def get_model():
    return load_model(Path(os.environ.get("MODEL_DIR", "models")))


pipe, meta = get_model()
st.title("Tariff heading suggestion")
st.caption(f"Model {meta['model_version']} · validation accuracy {meta['validation']['accuracy']} · "
           "a suggestion for a customs officer, not a decision")
text = st.text_area("Description of goods (any EU language)", "Damenstiefel mit Oberteil aus Rindleder")
if text.strip():
    ranked = top_k(pipe, [decision_text(text)], k=3)[0]
    st.dataframe(pd.DataFrame([{"heading": h, "score": s, "text": meta.get("headings", {}).get(h, "")}
                               for h, s in ranked]))
with st.expander("Model metadata"):
    st.json({k: v for k, v in meta.items() if k != "headings"})
