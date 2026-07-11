"""Preprocessing helpers for golfer profiles and equipment JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SHOT_SHAPES = ["Slice", "Fade", "Straight", "Draw", "Hook"]
GOALS = ["Distance", "Accuracy", "Forgiveness"]
IRON_MISSES = ["Fat/Thin", "Left/Right", "Inconsistent", "Consistent"]
IRON_FEELS = ["Forged/Blade-like", "Confidence-inspiring", "No preference"]
SHOPPING_TARGETS = ["Driver", "Irons", "Both"]
TRAJECTORIES = ["Too low", "About right", "Too high"]
INPUT_COLUMNS = [
    "handicap",
    "swing_speed",
    "driver_carry",
    "shot_shape",
    "goal",
    "iron_miss",
    "iron_feel",
]
FEATURE_COLUMNS = [
    "handicap",
    "swing_speed",
    "driver_carry",
    "shot_shape_Slice",
    "shot_shape_Fade",
    "shot_shape_Straight",
    "shot_shape_Draw",
    "shot_shape_Hook",
    "goal_Distance",
    "goal_Accuracy",
    "goal_Forgiveness",
    "iron_miss_Fat/Thin",
    "iron_miss_Left/Right",
    "iron_miss_Inconsistent",
    "iron_miss_Consistent",
    "iron_feel_Forged/Blade-like",
    "iron_feel_Confidence-inspiring",
    "iron_feel_No preference",
]


def read_category_records(path: Path) -> list[dict[str, Any]]:
    """Read either a plain list or the database's {'records': [...]} shape."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    return []


def load_equipment_catalog(equipment_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load every category JSON file into a dict keyed by category name."""
    root = Path(equipment_dir)
    catalog: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            category = payload.get("_meta", {}).get("category", path.stem)
            records = payload.get("records", [])
        else:
            category = path.stem
            records = payload
        if isinstance(records, list):
            catalog[category] = records
    return catalog


def estimate_handicap_from_average_score(average_score: float, par: int = 72) -> int:
    """Estimate handicap when the golfer knows score but not handicap."""
    return int(max(0, min(36, round(float(average_score) - par))))


def make_feature_frame(golfers: pd.DataFrame) -> pd.DataFrame:
    """Convert raw golfer input columns into the fixed feature matrix."""
    frame = pd.DataFrame(index=golfers.index)
    for column in ["handicap", "swing_speed", "driver_carry"]:
        frame[column] = pd.to_numeric(golfers[column], errors="coerce").fillna(0)

    for value in SHOT_SHAPES:
        frame[f"shot_shape_{value}"] = (golfers["shot_shape"] == value).astype(int)
    for value in GOALS:
        frame[f"goal_{value}"] = (golfers["goal"] == value).astype(int)
    for value in IRON_MISSES:
        frame[f"iron_miss_{value}"] = (golfers["iron_miss"] == value).astype(int)
    for value in IRON_FEELS:
        frame[f"iron_feel_{value}"] = (golfers["iron_feel"] == value).astype(int)

    return frame[FEATURE_COLUMNS]


def golfer_dict_to_frame(golfer: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row DataFrame from web or CLI input."""
    return pd.DataFrame([{column: golfer[column] for column in INPUT_COLUMNS}])
