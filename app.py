"""Custom web interface for the AI Golf Club Recommendation System."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

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
    ClubRecommendation,
    GolferInput,
    recommend_clubs,
    recommend_irons,
)
from train import LABEL_COLUMNS, train_from_csv


PROJECT_ROOT = Path(__file__).resolve().parent
EQUIPMENT_DIR = PROJECT_ROOT / "data" / "equipment" / "data"
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_PATH = PROJECT_ROOT / "data" / "golfers.csv"

app = Flask(__name__)


@lru_cache(maxsize=1)
def load_models() -> dict[str, object]:
    """Load trained Random Forest models, training them if artifacts are absent."""
    missing = [label for label in LABEL_COLUMNS if not (MODEL_DIR / f"{label}_model.joblib").exists()]
    if missing:
        train_from_csv(DATASET_PATH, MODEL_DIR)
    return {label: joblib.load(MODEL_DIR / f"{label}_model.joblib") for label in LABEL_COLUMNS}


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, list[dict[str, Any]]]:
    """Load equipment JSON records once for the web process."""
    return load_equipment_catalog(EQUIPMENT_DIR)


def predict_specs(models: dict[str, object], golfer_row: pd.DataFrame) -> dict[str, str]:
    """Predict each fitting label from a one-row golfer feature frame."""
    return {label: str(models[label].predict(golfer_row[INPUT_COLUMNS])[0]) for label in LABEL_COLUMNS}


def estimate_swing_speed_from_handicap(handicap: float) -> float:
    """Fallback driver speed estimate when the user only enters scoring information."""
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
    """Build the model input frame expected by the fitted scikit-learn pipelines."""
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


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _club_to_dict(recommendation: ClubRecommendation) -> dict[str, Any]:
    return {
        "name": recommendation.name,
        "score": recommendation.score,
        "reasons": recommendation.reasons[:4],
        "brand": recommendation.brand,
        "model": recommendation.model,
        "msrp": recommendation.msrp,
        "year": recommendation.year,
        "category": recommendation.category,
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        shopping_targets=SHOPPING_TARGETS,
        shot_shapes=SHOT_SHAPES,
        goals=GOALS,
        trajectories=TRAJECTORIES,
        iron_feels=IRON_FEELS,
        iron_misses=IRON_MISSES,
    )


@app.post("/api/recommend")
def api_recommend():
    payload = request.get_json(silent=True) or {}

    shopping_for = str(payload.get("shopping_for", "Both"))
    score_mode = str(payload.get("score_mode", "Handicap"))
    if score_mode == "Average score":
        handicap = float(estimate_handicap_from_average_score(_coerce_float(payload.get("average_score"), 90)))
    else:
        handicap = _coerce_float(payload.get("handicap"), 16)

    swing_speed = estimate_swing_speed_from_handicap(handicap)
    driver_carry = round(swing_speed * 2.35)
    speed_mode = str(payload.get("speed_mode", "Swing speed"))
    if speed_mode == "Carry distance":
        driver_carry = _coerce_float(payload.get("driver_carry"), driver_carry)
        swing_speed = round(driver_carry / 2.35, 1)
    else:
        swing_speed = _coerce_float(payload.get("swing_speed"), swing_speed)
        driver_carry = round(swing_speed * 2.35)

    driver_shot_shape = str(payload.get("driver_shot_shape", "Straight"))
    driver_goal = str(payload.get("driver_goal", "Accuracy"))
    driver_trajectory = str(payload.get("driver_trajectory", "About right"))
    iron_shot_shape = str(payload.get("iron_shot_shape", driver_shot_shape))
    iron_goal = str(payload.get("iron_goal", driver_goal))
    iron_trajectory = str(payload.get("iron_trajectory", "About right"))
    iron_feel = str(payload.get("iron_feel", "No preference"))
    iron_miss = str(payload.get("iron_miss", "Consistent"))

    wants_driver = shopping_for in {"Driver", "Both"}
    wants_irons = shopping_for in {"Irons", "Both"}

    models = load_models()
    catalog = load_catalog()
    golfer = GolferInput(
        handicap=handicap,
        swing_speed=swing_speed,
        driver_carry=driver_carry,
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
        driver_carry=driver_carry,
        shot_shape=driver_shot_shape,
        goal=driver_goal,
        iron_miss=iron_miss,
        iron_feel=iron_feel,
    )
    iron_row = build_model_row(
        handicap=handicap,
        swing_speed=swing_speed,
        driver_carry=driver_carry,
        shot_shape=iron_shot_shape,
        goal=iron_goal,
        iron_miss=iron_miss,
        iron_feel=iron_feel,
    )

    driver_specs = predict_specs(models, driver_row)
    iron_specs = predict_specs(models, iron_row)
    specs = {**driver_specs, "iron_category": iron_specs["iron_category"]}

    driver_recommendations: list[ClubRecommendation] = []
    iron_recommendations: list[ClubRecommendation] = []
    if wants_driver:
        driver_recommendations = recommend_clubs(
            catalog=catalog,
            golfer=golfer,
            predicted_loft=specs["driver_loft"],
            predicted_iron_category=specs["iron_category"],
            top_n=50,
        )
    if wants_irons:
        iron_recommendations = recommend_irons(
            catalog=catalog,
            golfer=golfer,
            predicted_iron_category=specs["iron_category"],
            top_n=50,
        )

    return jsonify(
        {
            "specs": specs,
            "profile": {
                "handicap": handicap,
                "swing_speed": swing_speed,
                "driver_carry": driver_carry,
                "shopping_for": shopping_for,
            },
            "wants_driver": wants_driver,
            "wants_irons": wants_irons,
            "recommendations": {
                "drivers": [_club_to_dict(rec) for rec in driver_recommendations],
                "irons": [_club_to_dict(rec) for rec in iron_recommendations],
            },
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8502, debug=True)
