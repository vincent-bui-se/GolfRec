"""Streamlit web interface for the AI Golf Club Recommendation System."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocess import GOALS, SHOT_SHAPES, IRON_MISSES, IRON_FEELS, INPUT_COLUMNS, load_equipment_catalog
from recommend import GolferInput, recommend_clubs, recommend_irons
from train import LABEL_COLUMNS, train_from_csv


PROJECT_ROOT = Path(__file__).resolve().parent
EQUIPMENT_DIR = PROJECT_ROOT / "data" / "equipment" / "data"
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_PATH = PROJECT_ROOT / "data" / "golfers.csv"


@st.cache_resource
def load_models() -> dict[str, object]:
    missing = [label for label in LABEL_COLUMNS if not (MODEL_DIR / f"{label}_model.joblib").exists()]
    if missing:
        train_from_csv(DATASET_PATH, MODEL_DIR)
    return {label: joblib.load(MODEL_DIR / f"{label}_model.joblib") for label in LABEL_COLUMNS}


@st.cache_data
def load_catalog():
    return load_equipment_catalog(EQUIPMENT_DIR)


def predict_specs(models: dict[str, object], golfer_row: pd.DataFrame) -> dict[str, str]:
    return {label: str(models[label].predict(golfer_row[INPUT_COLUMNS])[0]) for label in LABEL_COLUMNS}


st.set_page_config(page_title="AI Golf Club Recommendation System", layout="wide")
st.title("AI Golf Club Recommendation System")

with st.sidebar:
    st.header("Golfer Profile")
    handicap = st.slider("Handicap", 0.0, 36.0, 16.0, 0.5)
    swing_speed = st.slider("Driver swing speed (mph)", 60.0, 120.0, 92.0, 1.0)
    carry = st.slider("Driver carry distance (yards)", 130, 315, 225, 1)
    shot_shape = st.selectbox("Typical shot shape", SHOT_SHAPES, index=0)
    goal = st.selectbox("Primary goal", GOALS, index=2)
    iron_miss = st.selectbox("Typical iron miss", IRON_MISSES, index=0)
    iron_feel = st.selectbox("Preferred iron feel/look", IRON_FEELS, index=2)
    submitted = st.button("Recommend Clubs", type="primary")

if submitted:
    models = load_models()
    catalog = load_catalog()
    golfer = GolferInput(
        handicap=handicap,
        swing_speed=swing_speed,
        driver_carry=carry,
        shot_shape=shot_shape,
        goal=goal,
        iron_miss=iron_miss,
        iron_feel=iron_feel,
    )
    golfer_row = pd.DataFrame(
        [
            {
                "handicap": handicap,
                "swing_speed": swing_speed,
                "driver_carry": carry,
                "shot_shape": shot_shape,
                "goal": goal,
                "iron_miss": iron_miss,
                "iron_feel": iron_feel,
            }
        ]
    )
    specs = predict_specs(models, golfer_row)
    recommendations = recommend_clubs(
        catalog=catalog,
        golfer=golfer,
        predicted_loft=specs["driver_loft"],
        predicted_iron_category=specs["iron_category"],
        top_n=5,
    )
    iron_recs = recommend_irons(
        catalog=catalog,
        golfer=golfer,
        predicted_iron_category=specs["iron_category"],
        top_n=5,
    )

    st.subheader("AI Predicted Fitting Specs")
    cols = st.columns(3)
    cols[0].metric("Driver loft", f"{specs['driver_loft']} deg")
    cols[1].metric("Shaft flex", specs["shaft_flex"])
    cols[2].metric("Iron type", specs["iron_category"].replace("-", " ").title())

    tab1, tab2 = st.tabs(["Driver Recommendations", "Iron Set Recommendations"])

    with tab1:
        for rec in recommendations:
            with st.container(border=True):
                left, right = st.columns([3, 1])
                left.markdown(f"### {rec.name}")
                right.metric("Match Score", f"{rec.score}%")
                if rec.msrp is not None:
                    left.write(f"MSRP: ${rec.msrp:,.0f}")
                left.markdown("**Reason:**")
                for reason in rec.reasons:
                    left.write(f"- {reason}")

    with tab2:
        for rec in iron_recs:
            with st.container(border=True):
                left, right = st.columns([3, 1])
                left.markdown(f"### {rec.name}")
                right.metric("Match Score", f"{rec.score}%")
                if rec.msrp is not None:
                    left.write(f"MSRP: ${rec.msrp:,.0f}")
                left.markdown("**Reason:**")
                for reason in rec.reasons:
                    left.write(f"- {reason}")
else:
    st.info("Enter a golfer profile in the sidebar and click Recommend Clubs.")
