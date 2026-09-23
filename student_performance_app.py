"""
app.py
Streamlit app for the Student Performance Index predictor.

Run locally with:
    streamlit run app.py

Requires model.pkl (and optionally model_metadata.json) produced by train.py
to be present in the same directory.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "model.pkl"
METADATA_PATH = "model_metadata.json"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH) as f:
        return json.load(f)


model = load_model()
metadata = load_metadata()

st.title("🎓 Student Performance Index Predictor")


if model is None:
    st.error(
        "No trained model found (`model.pkl`). "
        "Run `python train.py` first to download the dataset, "
        "train the model, and generate the model file."
    )
    st.stop()

st.divider()
st.subheader("Enter student details")

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.slider("Hours Studied (daily average)", 0, 12, 5)
    previous_scores = st.slider("Previous Scores", 0, 100, 70)
    sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

with col2:
    sample_papers = st.slider("Sample Question Papers Practiced", 0, 15, 3)
    extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)

extracurricular_val = 1 if extracurricular == "Yes" else 0

input_df = pd.DataFrame(
    [[hours_studied, previous_scores, extracurricular_val, sleep_hours, sample_papers]],
    columns=[
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ],
)

st.divider()

if st.button("Predict Performance Index", type="primary"):
    prediction = model.predict(input_df)[0]
    prediction = float(np.clip(prediction, 10, 100))

    st.metric("Predicted Performance Index", f"{prediction:.1f}")
    st.progress(min(max(prediction / 100, 0.0), 1.0))

    if prediction >= 80:
        st.success("Strong predicted performance! 🎉")
    elif prediction >= 50:
        st.info("Average predicted performance.")
    else:
        st.warning("Below-average predicted performance — more study time or practice may help.")

if metadata:
    with st.expander("Model details"):
        st.write("**Test set metrics**")
        st.json(metadata.get("metrics", {}))
        st.write("**Coefficients**")
        st.json(metadata.get("coefficients", {}))
