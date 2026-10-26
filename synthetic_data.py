"""Synthetic golfer dataset generation based on public fitting guidelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import (
    DIVOT_DEPTHS,
    GOALS,
    IRON_FEELS,
    IRON_MISSES,
    SHOT_SHAPES,
    SWING_SPEED_BASE_MPH,
    SWING_SPEED_MAX_MPH,
    SWING_SPEED_PER_HANDICAP_MPH,
    TURF_FIRMNESS,
)


DRIVER_LOFT_LABELS = ["8", "9", "10.5", "12"]
SHAFT_FLEX_LABELS = ["L", "A", "R", "S", "X"]
FAIRWAY_WOOD_LOFT_LABELS = ["15", "16.5", "18", "21"]
WEDGE_BOUNCE_LABELS = ["Low", "Mid", "High"]
IRON_CATEGORY_LABELS = [
    "blade",
    "players-cb",
    "players-distance",
    "game-improvement",
    "super-game-improvement",
]


def _with_adjacent_noise(label: str, labels: list[str], rng: np.random.Generator, rate: float) -> str:
    """Replace a label with a neighboring ordinal category at the requested rate."""
    if rng.random() >= rate:
        return label
    index = labels.index(label)
    alternatives = [labels[i] for i in (index - 1, index + 1) if 0 <= i < len(labels)]
    return str(rng.choice(alternatives))


def _driver_loft(speed: float, handicap: float, goal: str) -> str:
    if speed < 82:
        loft = "12"
    elif speed < 96:
        loft = "10.5"
    elif speed < 108:
        loft = "9"
    else:
        loft = "8"

    if handicap > 24 and loft in {"8", "9"}:
        loft = "10.5"
    if goal == "Forgiveness":
        if loft == "8":
            loft = "9"
        elif loft == "9" and speed < 102:
            loft = "10.5"
    return loft


def _shaft_flex(speed: float) -> str:
    if speed < 72:
        return "L"
    if speed < 84:
        return "A"
    if speed < 97:
        return "R"
    if speed < 110:
        return "S"
    return "X"


def _fairway_wood_loft(speed: float, handicap: float, goal: str) -> str:
    """Choose a fairway loft using speed, handicap, and goal thresholds."""
    if speed < 80:
        loft = "21"
    elif speed < 92:
        loft = "18"
    elif speed < 104:
        loft = "16.5"
    else:
        loft = "15"

    if handicap > 24 and loft in {"15", "16.5"}:
        loft = "18"
    if goal == "Forgiveness" and speed < 96:
        loft = "21"
    return loft


def _wedge_bounce(turf: str, divot_depth: str) -> str:
    """Combine turf firmness and divot depth into a bounce tier."""
    score = 0.0
    if turf == "Soft":
        score += 1.0
    elif turf == "Firm":
        score -= 1.0

    if divot_depth == "Deep":
        score += 1.2
    elif divot_depth == "Shallow":
        score -= 1.2

    if score <= -0.75:
        return "Low"
    if score <= 0.75:
        return "Mid"
    return "High"


def _iron_category(
    handicap: float,
    swing_speed: float,
    driver_carry: float,
    shot_shape: str,
    goal: str,
    iron_miss: str,
    iron_feel: str,
) -> str:
    """Map handicap, contact, speed, shot shape, goal, and feel to an iron category."""
    fit_score = 0.0

    if handicap >= 28:
        fit_score += 3.3
    elif handicap >= 20:
        fit_score += 2.6
    elif handicap >= 12:
        fit_score += 1.7
    elif handicap >= 6:
        fit_score += 0.9

    if swing_speed < 78 or driver_carry < 185:
        fit_score += 0.8
    elif swing_speed > 102 or driver_carry > 245:
        fit_score -= 0.6

    miss_adjustments = {
        "Fat/Thin": 1.2,
        "Inconsistent": 1.0,
        "Left/Right": 0.55,
        "Consistent": -0.7,
    }
    fit_score += miss_adjustments.get(iron_miss, 0.0)

    if goal == "Forgiveness":
        fit_score += 1.0
    elif goal == "Accuracy":
        fit_score -= 0.25
    elif goal == "Distance":
        fit_score -= 0.35

    if iron_feel == "Forged/Blade-like":
        fit_score -= 1.5
    elif iron_feel == "Confidence-inspiring":
        fit_score += 0.8

    if shot_shape == "Slice":
        fit_score += 0.45
    elif shot_shape in {"Draw", "Hook"}:
        fit_score -= 0.25

    if fit_score <= -0.8 and handicap <= 7:
        return "blade"
    if fit_score <= 1.0:
        return "players-cb"
    if fit_score <= 2.3:
        return "players-distance"
    if fit_score <= 3.6:
        return "game-improvement"
    return "super-game-improvement"


def predict_rule_labels(golfers: pd.DataFrame) -> pd.DataFrame:
    """Apply the generator's rules without label noise for a baseline comparison."""
    rows = [
        {
            "driver_loft": _driver_loft(r.swing_speed, r.handicap, r.goal),
            "shaft_flex": _shaft_flex(r.swing_speed),
            "fairway_wood_loft": _fairway_wood_loft(r.swing_speed, r.handicap, r.goal),
            "iron_category": _iron_category(
                r.handicap, r.swing_speed, r.driver_carry, r.shot_shape,
                r.goal, r.iron_miss, r.iron_feel,
            ),
            "wedge_bounce": _wedge_bounce(r.wedge_turf, r.divot_depth),
        }
        for r in golfers.itertuples()
    ]
    return pd.DataFrame(rows, index=golfers.index)


def generate_golfer_profiles(
    n: int = 12_000,
    seed: int = 42,
    noise_rate: float = 0.10,
) -> pd.DataFrame:
    """Generate synthetic golfers and labels for supervised learning."""
    if not 500 <= n <= 100_000:
        raise ValueError("n must be between 500 and 100,000")

    rng = np.random.default_rng(seed)
    handicaps = np.clip(rng.normal(17, 9, n), 0, 36).round(1)
    swing_speeds = np.clip(
        SWING_SPEED_BASE_MPH + SWING_SPEED_PER_HANDICAP_MPH * handicaps + rng.normal(0, 10, n),
        65,
        SWING_SPEED_MAX_MPH,
    ).round(1)
    carry_ratio = 2.45 - 0.009 * handicaps
    driver_carry = np.clip(swing_speeds * carry_ratio + rng.normal(0, 9, n), 120, 330).round(0)
    # A home course's turf firmness is a course/climate trait, not a skill
    # trait, so it is drawn independently of handicap rather than banded with
    # the fields below.
    wedge_turfs = [str(value) for value in rng.choice(TURF_FIRMNESS, size=n)]

    shot_shapes = []
    goals = []
    iron_misses = []
    iron_feels = []
    divot_depths = []
    for handicap in handicaps:
        if handicap >= 20:
            shot_probs = [0.38, 0.18, 0.24, 0.12, 0.08]
            goal_probs = [0.20, 0.25, 0.55]
            miss_probs = [0.4, 0.3, 0.25, 0.05]
            feel_probs = [0.1, 0.7, 0.2]
        elif handicap >= 10:
            shot_probs = [0.22, 0.22, 0.30, 0.18, 0.08]
            goal_probs = [0.30, 0.35, 0.35]
            miss_probs = [0.25, 0.35, 0.25, 0.15]
            feel_probs = [0.3, 0.4, 0.3]
        else:
            shot_probs = [0.10, 0.22, 0.36, 0.24, 0.08]
            goal_probs = [0.34, 0.48, 0.18]
            miss_probs = [0.1, 0.25, 0.15, 0.5]
            feel_probs = [0.42, 0.18, 0.40]
        shot_shapes.append(str(rng.choice(SHOT_SHAPES, p=shot_probs)))
        goals.append(str(rng.choice(GOALS, p=goal_probs)))
        iron_miss = str(rng.choice(IRON_MISSES, p=miss_probs))
        iron_misses.append(iron_miss)
        iron_feels.append(str(rng.choice(IRON_FEELS, p=feel_probs)))

        # Sample divot depth from the miss-pattern assumptions used by this generator.
        if iron_miss == "Fat/Thin":
            divot_probs = [0.10, 0.25, 0.65]
        elif iron_miss == "Inconsistent":
            divot_probs = [0.20, 0.40, 0.40]
        elif iron_miss == "Left/Right":
            divot_probs = [0.30, 0.45, 0.25]
        else:
            divot_probs = [0.35, 0.45, 0.20]
        divot_depths.append(str(rng.choice(DIVOT_DEPTHS, p=divot_probs)))

    rows = []
    for handicap, speed, carry, shape, goal, i_miss, i_feel, turf, divot in zip(
        handicaps,
        swing_speeds,
        driver_carry,
        shot_shapes,
        goals,
        iron_misses,
        iron_feels,
        wedge_turfs,
        divot_depths,
    ):
        loft = _with_adjacent_noise(_driver_loft(speed, handicap, goal), DRIVER_LOFT_LABELS, rng, noise_rate)
        flex = _with_adjacent_noise(_shaft_flex(speed), SHAFT_FLEX_LABELS, rng, noise_rate)
        fairway_loft = _with_adjacent_noise(
            _fairway_wood_loft(speed, handicap, goal), FAIRWAY_WOOD_LOFT_LABELS, rng, noise_rate
        )
        bounce = _with_adjacent_noise(_wedge_bounce(turf, divot), WEDGE_BOUNCE_LABELS, rng, noise_rate)
        iron = _with_adjacent_noise(
            _iron_category(handicap, speed, carry, shape, goal, i_miss, i_feel),
            IRON_CATEGORY_LABELS,
            rng,
            noise_rate,
        )
        rows.append(
            {
                "handicap": handicap,
                "swing_speed": speed,
                "driver_carry": int(carry),
                "shot_shape": shape,
                "goal": goal,
                "iron_miss": i_miss,
                "iron_feel": i_feel,
                "wedge_turf": turf,
                "divot_depth": divot,
                "driver_loft": loft,
                "shaft_flex": flex,
                "fairway_wood_loft": fairway_loft,
                "iron_category": iron,
                "wedge_bounce": bounce,
            }
        )

    return pd.DataFrame(rows)


def save_dataset(path: str | Path, n: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """Generate and save the golfer dataset to CSV."""
    frame = generate_golfer_profiles(n=n, seed=seed)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


if __name__ == "__main__":
    save_dataset(Path("data") / "golfers.csv")
