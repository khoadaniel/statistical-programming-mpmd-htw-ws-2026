"""A small dashboard for the service: classify a review and show the model metadata.

    uv sync --extra dashboard
    MODEL_DIR=models uv run streamlit run dashboard/streamlit_app.py

On Streamlit Community Cloud, point the app to this file; the model files must then be in the repository
or be downloaded at start-up (see README, "Publishing a dashboard").
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment_service.model import load_model, review_text


@st.cache_resource  # load once per server process, not on every interaction
def get_model():
    return load_model(Path(os.environ.get("MODEL_DIR", "models")))


pipe, meta = get_model()
st.title("Review sentiment")
st.caption(f"Model {meta['model_version']} · validation macro-F1 {meta['validation']['macro_f1']}")
title = st.text_input("Title", "Stopped working")
text = st.text_area("Review", "Broke after two days. Waste of money.")
if text.strip():
    proba = pipe.predict_proba([review_text(title, text)])[0]
    st.bar_chart(pd.DataFrame({"probability": proba}, index=pipe.classes_))
    st.write("Predicted label:", pipe.classes_[proba.argmax()])
with st.expander("Model metadata"):
    st.json(meta)
