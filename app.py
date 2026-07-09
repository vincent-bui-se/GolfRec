"""Streamlit web interface for the AI Golf Club Recommendation System."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocess import (
    GOALS,
    INPUT_COLUMNS,
    IRON_FEELS,
    IRON_MISSES,
    SHOPPING_TARGETS,
    SHOT_SHAPES,
    TRAJECTORIES,
    estimate_handicap_from_average_score,
    load_equipment_catalog,
)
from recommend import (
    GolferInput,
    filter_recommendations_by_budget,
    recommend_clubs,
    recommend_irons,
)
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


def estimate_swing_speed_from_handicap(handicap: float) -> float:
    return max(60.0, min(120.0, 108.0 - 1.15 * handicap))


def build_model_row(
    *,
    handicap: float,
    swing_speed: float,
    driver_carry: float,
    shot_shape: str,
    goal: str,
    iron_miss: str,
    iron_feel: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "handicap": handicap,
                "swing_speed": swing_speed,
                "driver_carry": driver_carry,
                "shot_shape": shot_shape,
                "goal": goal,
                "iron_miss": iron_miss,
                "iron_feel": iron_feel,
            }
        ]
    )


def render_recommendation_card(rec) -> None:
    meta_parts = [f"Match Score: {rec.score}%"]
    if rec.year is not None:
        meta_parts.append(f"Model year: {rec.year}")
    if rec.msrp is not None:
        meta_parts.append(f"MSRP: ${rec.msrp:,.0f}")
    reasons = "".join(f"<li>{reason}</li>" for reason in rec.reasons[:3])
    st.markdown(
        f"""
        <div class="rec-card">
            <div class="rec-card-header">
                <div class="rec-title">{rec.name}</div>
                <div class="rec-score">{rec.score}%</div>
            </div>
            <div class="rec-meta">{" | ".join(meta_parts)}</div>
            <ul class="rec-reasons">{reasons}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="GolfRec", layout="wide")
st.title("GolfRec")
st.header("AI Golf Recommendation System")
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.55rem 0.7rem;
        border-radius: 0.45rem;
    }
    .rec-card {
        --card-bg: #ffffff;
        --card-border: #d7dde5;
        --title-color: #111827;
        --muted-color: #4b5563;
        --score-bg: #e8f5ee;
        --score-color: #11633a;
        border: 1px solid var(--card-border);
        border-radius: 8px;
        padding: 0.6rem 0.75rem 0.5rem;
        margin: 0.35rem 0;
        background: var(--card-bg);
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
    }
    .rec-card-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.75rem;
    }
    .rec-title {
        font-weight: 700;
        font-size: 1rem;
        line-height: 1.2;
        color: var(--title-color);
    }
    .rec-score {
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--score-color);
        background: var(--score-bg);
        border-radius: 999px;
        padding: 0.12rem 0.48rem;
        white-space: nowrap;
    }
    .rec-meta {
        color: var(--muted-color);
        font-size: 0.82rem;
        margin-top: 0.15rem;
    }
    .rec-reasons {
        margin: 0.28rem 0 0 1rem;
        padding: 0;
        font-size: 0.88rem;
        line-height: 1.25;
        color: var(--title-color);
    }
    .rec-reasons li {
        margin: 0.08rem 0;
    }
    @media (prefers-color-scheme: dark) {
        .rec-card {
            --card-bg: #111827;
            --card-border: #374151;
            --title-color: #f9fafb;
            --muted-color: #cbd5e1;
            --score-bg: #123524;
            --score-color: #8ee0b3;
            box-shadow: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = None

with st.sidebar:
    st.header("Golfer Profile")
    shopping_for = st.selectbox("What are you shopping for?", SHOPPING_TARGETS, index=2)
    score_mode = st.radio("What's your handicap or average score?", ["Handicap", "Average score"], horizontal=True)
    if score_mode == "Handicap":
        handicap = st.slider("Handicap", 0.0, 36.0, 16.0, 0.5)
    else:
        average_score = st.slider("Average 18-hole score", 68, 112, 90, 1)
        handicap = float(estimate_handicap_from_average_score(average_score))
        st.caption(f"Estimated handicap: {handicap:.0f}")

    wants_driver = shopping_for in {"Driver", "Both"}
    wants_irons = shopping_for in {"Irons", "Both"}

    driver_shot_shape = "Straight"
    driver_goal = "Accuracy"
    driver_trajectory = "About right"
    carry = round(estimate_swing_speed_from_handicap(handicap) * 2.35)
    swing_speed = estimate_swing_speed_from_handicap(handicap)

    if wants_driver:
        st.subheader("Driver")
        speed_mode = st.radio("What's your swing speed or distance?", ["Swing speed", "Carry distance"], horizontal=True)
        if speed_mode == "Swing speed":
            swing_speed = st.slider("Driver swing speed (mph)", 60.0, 120.0, float(round(swing_speed)), 1.0)
            carry = int(round(swing_speed * 2.35))
        else:
            carry = st.slider("Driver carry distance (yards)", 130, 315, int(carry), 1)
            swing_speed = round(carry / 2.35, 1)
            st.caption(f"Estimated swing speed: {swing_speed:.1f} mph")
        driver_shot_shape = st.selectbox("What is your driver shot shape?", SHOT_SHAPES, index=2)
        driver_trajectory = st.selectbox("Do you hit driver too high, too low, or about right?", TRAJECTORIES, index=1)
        driver_goal = st.selectbox("Primary driver goal", GOALS, index=1)

    iron_shot_shape = driver_shot_shape
    iron_goal = driver_goal
    iron_trajectory = "About right"
    iron_miss = "Consistent"
    iron_feel = "No preference"

    if wants_irons:
        st.subheader("Irons")
        iron_shot_shape = st.selectbox("What is your iron shot shape?", SHOT_SHAPES, index=2)
        iron_goal = st.selectbox("Primary iron goal", GOALS, index=2)
        iron_trajectory = st.selectbox("Do you hit irons too high, too low, or about right?", TRAJECTORIES, index=1)
        iron_feel = st.selectbox("Preferred iron feel or look", IRON_FEELS, index=2)
        iron_miss = st.selectbox("Typical iron miss", IRON_MISSES, index=3)

    submitted = st.button("Recommend Clubs", type="primary")

if submitted:
    models = load_models()
    catalog = load_catalog()
    golfer = GolferInput(
        handicap=handicap,
        swing_speed=swing_speed,
        driver_carry=carry,
        shot_shape=driver_shot_shape,
        goal=driver_goal,
        iron_miss=iron_miss,
        iron_feel=iron_feel,
        shopping_for=shopping_for,
        driver_trajectory=driver_trajectory,
        iron_trajectory=iron_trajectory,
        iron_shot_shape=iron_shot_shape,
        iron_goal=iron_goal,
    )
    driver_row = build_model_row(
        handicap=handicap,
        swing_speed=swing_speed,
        driver_carry=carry,
        shot_shape=driver_shot_shape,
        goal=driver_goal,
        iron_miss=iron_miss,
        iron_feel=iron_feel,
    )
    iron_row = build_model_row(
        handicap=handicap,
        swing_speed=swing_speed,
        driver_carry=carry,
        shot_shape=iron_shot_shape,
        goal=iron_goal,
        iron_miss=iron_miss,
        iron_feel=iron_feel,
    )
    driver_specs = predict_specs(models, driver_row)
    iron_specs = predict_specs(models, iron_row)
    specs = {**driver_specs, "iron_category": iron_specs["iron_category"]}
    recommendations = []
    iron_recs = []
    if wants_driver:
        recommendations = recommend_clubs(
            catalog=catalog,
            golfer=golfer,
            predicted_loft=specs["driver_loft"],
            predicted_iron_category=specs["iron_category"],
            top_n=50,
        )
    if wants_irons:
        iron_recs = recommend_irons(
            catalog=catalog,
            golfer=golfer,
            predicted_iron_category=specs["iron_category"],
            top_n=50,
        )

    st.session_state.recommendation_result = {
        "specs": specs,
        "wants_driver": wants_driver,
        "wants_irons": wants_irons,
        "recommendations": recommendations,
        "iron_recs": iron_recs,
    }

result = st.session_state.recommendation_result

if result:
    specs = result["specs"]
    wants_driver = result["wants_driver"]
    wants_irons = result["wants_irons"]
    recommendations = result["recommendations"]
    iron_recs = result["iron_recs"]

    st.subheader("AI Predicted Fitting Specs")
    spec_cols = st.columns(3 if wants_driver and wants_irons else 2)
    if wants_driver:
        spec_cols[0].metric("Driver loft", f"{specs['driver_loft']} deg")
        spec_cols[1].metric("Shaft flex", specs["shaft_flex"])
    if wants_irons:
        spec_cols[-1].metric("Iron type", specs["iron_category"].replace("-", " ").title())

    if wants_driver and wants_irons:
        tab1, tab2 = st.tabs(["Driver Recommendations", "Iron Set Recommendations"])
    elif wants_driver:
        tab1 = st.container()
        tab2 = None
    else:
        tab1 = None
        tab2 = st.container()

    if wants_driver and tab1 is not None:
        with tab1:
            st.subheader("Driver Recommendations")
            driver_budget = st.slider("Driver budget", 300, 2500, 800, 25)
            filtered_recommendations = filter_recommendations_by_budget(
                recommendations, driver_budget, limit=5
            )
            if not filtered_recommendations:
                st.info("No driver recommendations are within this budget.")
            for rec in filtered_recommendations:
                render_recommendation_card(rec)

    if wants_irons and tab2 is not None:
        with tab2:
            st.subheader("Iron Set Recommendations")
            iron_budget = st.slider("Iron set budget", 500, 2500, 1400, 25)
            filtered_iron_recs = filter_recommendations_by_budget(
                iron_recs, iron_budget, limit=5
            )
            if not filtered_iron_recs:
                st.info("No iron set recommendations are within this budget.")
            for rec in filtered_iron_recs:
                render_recommendation_card(rec)
else:
    st.info("Enter a golfer profile in the sidebar and click Recommend Clubs.")
