"""Custom web interface for the AI Golf Club Recommendation System."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

from preprocess import (
    CLUB_CATEGORIES,
    DIVOT_DEPTHS,
    GOALS,
    INPUT_COLUMNS,
    IRON_FEELS,
    IRON_MISSES,
    SHOT_SHAPES,
    SWING_SPEED_BASE_MPH,
    SWING_SPEED_MAX_MPH,
    SWING_SPEED_MIN_MPH,
    SWING_SPEED_PER_HANDICAP_MPH,
    SWING_TEMPOS,
    TRAJECTORIES,
    TURF_FIRMNESS,
    WEDGE_LOFT_OPTIONS,
    WEDGE_SHOT_STYLES,
    estimate_handicap_from_average_score,
    load_equipment_catalog,
)
from recommend import (
    ClubRecommendation,
    GolferInput,
    recommend_clubs,
    recommend_fairway_woods,
    recommend_irons,
    recommend_wedges,
)
from train import LABEL_COLUMNS, train_from_csv


PROJECT_ROOT = Path(__file__).resolve().parent
EQUIPMENT_DIR = PROJECT_ROOT / "data" / "equipment" / "data"
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_PATH = PROJECT_ROOT / "data" / "golfers.csv"

app = Flask(__name__)
# Render/Cloudflare terminate TLS in front of this app and forward plain HTTP
# to gunicorn, so request.scheme (and anything built from it, like
# url_for(..., _external=True)) would read "http" without this - producing
# a canonical/OG URL for the insecure scheme on a page that's only ever
# actually served over HTTPS.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self'; "
    "img-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


@app.after_request
def _add_security_headers(response):
    # HSTS only matters over a connection ProxyFix confirms was actually
    # HTTPS - setting it on a plain-HTTP response (e.g. local dev) would be
    # a lie the browser can't act on anyway.
    if request.is_secure:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    # style-src needs 'unsafe-inline' for app.js's element.style.x = ...
    # positioning of the typeahead dropdown; script-src does not, since
    # that's the axis that actually matters against injection.
    response.headers["Content-Security-Policy"] = _CONTENT_SECURITY_POLICY
    if request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response


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


def predict_specs(
    models: dict[str, object],
    golfer_row: pd.DataFrame,
    labels: list[str] = LABEL_COLUMNS,
) -> dict[str, str]:
    """Predict the given fitting labels (default: all of them) from a one-row golfer feature frame."""
    return {label: str(models[label].predict(golfer_row[INPUT_COLUMNS])[0]) for label in labels}


# Matches synthetic_data.py's SHAFT_FLEX_LABELS ordering (softest to stiffest).
_FLEX_ORDER = ["L", "A", "R", "S", "X"]


def _adjust_flex_for_tempo(flex: str, tempo: str) -> str:
    """Nudge a speed-predicted flex letter one step for tempo, real-fitting style.

    Swing speed alone predicts a flex band; a fast transition through the ball
    (tempo) is the standard adjustment fitters make on top of that band, one
    step firmer or softer, not a full re-fit. Unrecognized flex letters (only
    possible if a model's label set ever drifts from _FLEX_ORDER) pass through
    unchanged rather than raising, since this is a display nicety, not a value
    the recommendation logic depends on.
    """
    if flex not in _FLEX_ORDER:
        return flex
    index = _FLEX_ORDER.index(flex)
    if tempo == "Aggressive":
        index = min(index + 1, len(_FLEX_ORDER) - 1)
    elif tempo == "Smooth":
        index = max(index - 1, 0)
    return _FLEX_ORDER[index]


def _loft_ladder(model: object) -> list[str]:
    """The loft labels this model can emit, lowest loft first.

    Read off the trained model rather than copied from synthetic_data's
    DRIVER_LOFT_LABELS, so the ladder is always the one the prediction being
    adjusted actually came from. _FLEX_ORDER above is a hand-written copy
    because flex order is semantic (L < A < R < S < X) and cannot be recovered
    by sorting; lofts are numbers, so they can.
    """
    try:
        return sorted((str(label) for label in getattr(model, "classes_", ())), key=float)
    except (TypeError, ValueError):
        # A label set that is not numeric means this is no longer a loft
        # ladder; adjusting it by position would be guesswork.
        return []


def _adjust_loft_for_trajectory(loft: str, trajectory: str, ladder: list[str]) -> str:
    """Nudge a predicted loft one rung for the golfer's ball flight.

    Ball flight is not one of the model's inputs (see INPUT_COLUMNS), so the
    predicted loft is byte-identical whether a golfer says their driver flies
    too low, too high, or about right - which is exactly what a golfer notices
    when they change that answer and the number does not move. Same situation
    as swing tempo and shaft flex above, and the same remedy: the model gives
    the band, the answer the model never saw nudges it one step, the way a
    fitter would rather than re-fitting from scratch.

    Loft is the primary launch lever on a wood, so the direction is the obvious
    one: flying too low earns more loft, too high earns less. recommend.py
    separately prefers higher-launch and higher-spin *heads* for the same
    answer; those are head construction rather than loft, so the two stack the
    way a real fitting does instead of double-counting one lever.

    Off-ladder lofts and unrecognised answers pass through unchanged rather
    than raising, matching _adjust_flex_for_tempo.
    """
    if trajectory not in {"Too low", "Too high"} or loft not in ladder:
        return loft
    index = ladder.index(loft)
    if trajectory == "Too low":
        index = min(index + 1, len(ladder) - 1)
    else:
        index = max(index - 1, 0)
    return ladder[index]


def estimate_swing_speed_from_handicap(handicap: float) -> float:
    """Fallback driver speed estimate when the user only enters scoring information."""
    return max(
        SWING_SPEED_MIN_MPH,
        min(SWING_SPEED_MAX_MPH, SWING_SPEED_BASE_MPH + SWING_SPEED_PER_HANDICAP_MPH * handicap),
    )


def build_model_row(
    *,
    handicap: float,
    swing_speed: float,
    driver_carry: float,
    shot_shape: str,
    goal: str,
    iron_miss: str,
    iron_feel: str,
    wedge_turf: str,
    divot_depth: str,
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
                "wedge_turf": wedge_turf,
                "divot_depth": divot_depth,
            }
        ]
    )


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _current_club_options(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build a sorted (id, label) list for the "current driver/irons" dropdowns."""
    options = [
        {
            "id": str(record["id"]),
            "label": f"{record.get('brand', 'Unknown')} {record.get('model', 'Unknown')} ({record.get('year', '?')})",
        }
        for record in records
        if record.get("id")
    ]
    return sorted(options, key=lambda option: option["label"])


def _resolve_current_club(
    catalog_key: str, club_id: str, catalog: dict[str, list[dict[str, Any]]]
) -> dict[str, Any] | None:
    """Look up a "current driver/irons" selection by id, or None if unset/unknown."""
    if not club_id:
        return None
    for record in catalog.get(catalog_key, []):
        if str(record.get("id")) == club_id:
            return record
    return None


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
        "years": recommendation.years,
    }


@app.get("/favicon.ico")
def favicon():
    """Serve the icon at the root path browsers probe regardless of <link> tags."""
    return send_from_directory(
        app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon"
    )


@app.get("/robots.txt")
def robots():
    """A plain static-file response, not a templated/catalog-driven one.

    Distinct from a 404 (which crawlers correctly treat as "no restrictions")
    - the actual risk this avoids is a 5xx here, which Google's crawler
    treats as "robots.txt unreachable" and can suppress crawling of the
    entire site while it persists. Routing through send_from_directory
    keeps this response independent of catalog loading, model inference, or
    template rendering, so an unrelated app-logic failure elsewhere can't
    take robots.txt down with it. It doesn't protect against the host
    process itself being asleep (Render free-tier hibernation) - only a
    CDN-level static response or an always-on tier can close that gap.
    """
    return send_from_directory(app.static_folder, "robots.txt", mimetype="text/plain")


@app.get("/")
def index():
    catalog = load_catalog()
    return render_template(
        "index.html",
        club_categories=CLUB_CATEGORIES,
        shot_shapes=SHOT_SHAPES,
        goals=GOALS,
        trajectories=TRAJECTORIES,
        iron_feels=IRON_FEELS,
        iron_misses=IRON_MISSES,
        turf_firmness=TURF_FIRMNESS,
        divot_depths=DIVOT_DEPTHS,
        swing_tempos=SWING_TEMPOS,
        wedge_shot_styles=WEDGE_SHOT_STYLES,
        wedge_lofts=WEDGE_LOFT_OPTIONS,
        current_drivers=_current_club_options(catalog.get("drivers", [])),
        current_iron_sets=_current_club_options(catalog.get("iron-sets", [])),
    )


@app.post("/api/recommend")
def api_recommend():
    payload = request.get_json(silent=True) or {}

    raw_shopping_for = payload.get("shopping_for")
    if not isinstance(raw_shopping_for, list):
        raw_shopping_for = None
    shopping_for = [str(item) for item in (raw_shopping_for or []) if str(item) in CLUB_CATEGORIES]
    if not shopping_for:
        shopping_for = ["Driver", "Irons"]

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

    swing_tempo = str(payload.get("swing_tempo", "Moderate"))
    if swing_tempo not in SWING_TEMPOS:
        swing_tempo = "Moderate"

    driver_shot_shape = str(payload.get("driver_shot_shape", "Straight"))
    if driver_shot_shape not in SHOT_SHAPES:
        driver_shot_shape = "Straight"
    driver_goal = str(payload.get("driver_goal", "Accuracy"))
    if driver_goal not in GOALS:
        driver_goal = "Accuracy"
    driver_trajectory = str(payload.get("driver_trajectory", "About right"))
    if driver_trajectory not in TRAJECTORIES:
        driver_trajectory = "About right"
    # None (not defaulted to the driver's value here) so GolferInput's own
    # effective_iron_shot_shape/effective_iron_goal fallback-to-driver logic
    # is the single place that decision happens - see the iron_row build
    # below, which must read the same effective_* values this feeds the
    # catalog scorer, or the ML model and the scorer can silently diverge.
    raw_iron_shot_shape = payload.get("iron_shot_shape")
    iron_shot_shape = str(raw_iron_shot_shape) if raw_iron_shot_shape in SHOT_SHAPES else None
    raw_iron_goal = payload.get("iron_goal")
    iron_goal = str(raw_iron_goal) if raw_iron_goal in GOALS else None
    iron_trajectory = str(payload.get("iron_trajectory", "About right"))
    if iron_trajectory not in TRAJECTORIES:
        iron_trajectory = "About right"
    iron_feel = str(payload.get("iron_feel", "No preference"))
    if iron_feel not in IRON_FEELS:
        iron_feel = "No preference"
    iron_miss = str(payload.get("iron_miss", "Consistent"))
    if iron_miss not in IRON_MISSES:
        iron_miss = "Consistent"
    wedge_turf = str(payload.get("wedge_turf", "Normal"))
    if wedge_turf not in TURF_FIRMNESS:
        wedge_turf = "Normal"
    divot_depth = str(payload.get("divot_depth", "Medium"))
    if divot_depth not in DIVOT_DEPTHS:
        divot_depth = "Medium"
    wedge_shot_style = str(payload.get("wedge_shot_style", "No preference"))
    if wedge_shot_style not in WEDGE_SHOT_STYLES:
        wedge_shot_style = "No preference"
    # 56 deg (matches GolferInput.wedge_loft's own default) is the single
    # most common "do-it-all" wedge loft - a reasonable fallback, not derived
    # from WEDGE_LOFT_OPTIONS's position, which could silently drift if that
    # list's contents ever change.
    wedge_loft = _coerce_float(payload.get("wedge_loft"), 56.0)
    wedge_loft = max(float(WEDGE_LOFT_OPTIONS[0]), min(float(WEDGE_LOFT_OPTIONS[-1]), wedge_loft))
    current_driver_id = str(payload.get("current_driver_id", "")).strip()
    current_iron_set_id = str(payload.get("current_iron_set_id", "")).strip()

    wants_driver = "Driver" in shopping_for
    wants_fairway_woods = "Fairway Wood" in shopping_for
    wants_irons = "Irons" in shopping_for
    wants_wedges = "Wedges" in shopping_for

    models = load_models()
    catalog = load_catalog()

    current_driver = _resolve_current_club("drivers", current_driver_id, catalog)
    current_iron_set = _resolve_current_club("iron-sets", current_iron_set_id, catalog)

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
        wedge_turf=wedge_turf,
        divot_depth=divot_depth,
        wedge_shot_style=wedge_shot_style,
        wedge_loft=wedge_loft,
        current_driver_launch=current_driver.get("launchChar") if current_driver else None,
        current_driver_spin=current_driver.get("spinChar") if current_driver else None,
        current_iron_launch=current_iron_set.get("launchChar") if current_iron_set else None,
        current_iron_spin=current_iron_set.get("spinChar") if current_iron_set else None,
    )

    driver_row = build_model_row(
        handicap=handicap,
        swing_speed=swing_speed,
        driver_carry=driver_carry,
        shot_shape=driver_shot_shape,
        goal=driver_goal,
        iron_miss=iron_miss,
        iron_feel=iron_feel,
        wedge_turf=wedge_turf,
        divot_depth=divot_depth,
    )
    iron_row = build_model_row(
        handicap=handicap,
        swing_speed=swing_speed,
        driver_carry=driver_carry,
        # Effective, not raw: must match what recommend.py's catalog scorer
        # sees for this same golfer, or the ML prediction and the scoring
        # reasons silently disagree about the golfer's iron shot shape/goal.
        shot_shape=golfer.effective_iron_shot_shape,
        goal=golfer.effective_iron_goal,
        iron_miss=iron_miss,
        iron_feel=iron_feel,
        wedge_turf=wedge_turf,
        divot_depth=divot_depth,
    )

    # fairway_wood_loft and wedge_bounce ride along in driver_specs: both are
    # predicted from driver_row since neither has its own shot_shape/goal
    # question, the same way iron_category gets its own prediction from
    # iron_row below because it does. Only iron_category is actually needed
    # from iron_row, so that's the only label predicted from it - predicting
    # (and discarding) the other 4 labels from iron_row was pure waste.
    driver_specs = predict_specs(models, driver_row)
    driver_specs["shaft_flex"] = _adjust_flex_for_tempo(driver_specs["shaft_flex"], swing_tempo)
    # Both woods take the driver's Ball flight answer: the question sits under
    # "Driver & Fairway Wood" in the form and recommend.py's _score_wood
    # already reads golfer.driver_trajectory for both categories, so leaving
    # the fairway wood's loft unmoved would have it fitted to a flight the
    # golfer just said was wrong.
    #
    # Applied here, before recommend_clubs() is handed specs["driver_loft"],
    # so the catalog scoring and every reason it writes ("Sold in your 12 deg
    # target loft") describe the adjusted target rather than the raw
    # prediction.
    for label in ("driver_loft", "fairway_wood_loft"):
        driver_specs[label] = _adjust_loft_for_trajectory(
            driver_specs[label], driver_trajectory, _loft_ladder(models[label])
        )
    iron_category = predict_specs(models, iron_row, labels=["iron_category"])["iron_category"]
    specs = {**driver_specs, "iron_category": iron_category}

    driver_recommendations: list[ClubRecommendation] = []
    fairway_wood_recommendations: list[ClubRecommendation] = []
    iron_recommendations: list[ClubRecommendation] = []
    wedge_recommendations: list[ClubRecommendation] = []
    if wants_driver:
        driver_recommendations = recommend_clubs(
            catalog=catalog,
            golfer=golfer,
            predicted_loft=specs["driver_loft"],
            predicted_iron_category=specs["iron_category"],
            top_n=len(catalog.get("drivers", [])),
        )
    if wants_fairway_woods:
        fairway_wood_recommendations = recommend_fairway_woods(
            catalog=catalog,
            golfer=golfer,
            predicted_loft=specs["fairway_wood_loft"],
            predicted_iron_category=specs["iron_category"],
            top_n=len(catalog.get("fairway-woods", [])),
        )
    if wants_irons:
        iron_recommendations = recommend_irons(
            catalog=catalog,
            golfer=golfer,
            predicted_iron_category=specs["iron_category"],
            top_n=len(catalog.get("iron-sets", [])),
        )
    if wants_wedges:
        wedge_recommendations = recommend_wedges(
            catalog=catalog,
            golfer=golfer,
            predicted_bounce=specs["wedge_bounce"],
            top_n=len(catalog.get("wedges", [])),
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
            "wants_fairway_woods": wants_fairway_woods,
            "wants_irons": wants_irons,
            "wants_wedges": wants_wedges,
            "recommendations": {
                "drivers": [_club_to_dict(rec) for rec in driver_recommendations],
                "fairway_woods": [_club_to_dict(rec) for rec in fairway_wood_recommendations],
                "irons": [_club_to_dict(rec) for rec in iron_recommendations],
                "wedges": [_club_to_dict(rec) for rec in wedge_recommendations],
            },
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8502, debug=True)
