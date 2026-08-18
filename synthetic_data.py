"""Synthetic golfer dataset generation based on public fitting guidelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import GOALS, SHOT_SHAPES, IRON_MISSES, IRON_FEELS, TURF_FIRMNESS, DIVOT_DEPTHS


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


def _with_noise(label: str, labels: list[str], rng: np.random.Generator, rate: float) -> str:
    if rng.random() >= rate:
        return label
    alternatives = [candidate for candidate in labels if candidate != label]
    return str(rng.choice(alternatives))


def _driver_loft(speed: float, handicap: float, goal: str) -> str:
    if speed < 80:
        loft = "12"
    elif speed < 95:
        loft = "10.5"
    elif speed < 106:
        loft = "9"
    else:
        loft = "8"

    if handicap > 24 and loft in {"8", "9"}:
        loft = "10.5"
    if goal == "Forgiveness" and speed < 92:
        loft = "12"
    return loft


def _shaft_flex(speed: float) -> str:
    if speed < 72:
        return "L"
    if speed < 84:
        return "A"
    if speed < 97:
        return "R"
    if speed < 105:
        return "S"
    return "X"


def _fairway_wood_loft(speed: float, handicap: float, goal: str) -> str:
    """Same shape of rule as _driver_loft: slower speed needs more loft to launch.

    Fairway wood lofts run in finer steps than driver lofts (15/16.5/18/21 vs.
    8/9/10.5/12), so the thresholds are recalibrated, not reused.
    """
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
    """Bounce need combines turf/lie firmness with the golfer's attack angle.

    Soft turf and sand need sole material under the leading edge (high bounce)
    to keep the club from digging; firm turf needs the opposite so the leading
    edge can reach the ball before the sole does. Divot depth is a direct
    attack-angle proxy and carries at least as much weight as turf: a deep,
    steep divot needs bounce to survive even on firm ground, while a shallow,
    sweeping contact can get away with less bounce even on soft ground. The
    two signals are combined additively rather than one overriding the other,
    since neither alone is a reliable predictor.
    """
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
    """Choose iron category from multiple fitting signals, not handicap alone.

    Lower fit scores map to player-focused irons. Higher scores map to more
    forgiving game-improvement irons. Handicap is intentionally only one input;
    contact quality, speed, shot pattern, desired outcome, and preferred look
    can move a golfer into a different category.
    """
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

    if fit_score <= 0.3:
        return "blade"
    if fit_score <= 1.25:
        return "players-cb"
    if fit_score <= 2.35:
        return "players-distance"
    if fit_score <= 3.65:
        return "game-improvement"
    return "super-game-improvement"


def generate_golfer_profiles(
    n: int = 1500,
    seed: int = 42,
    noise_rate: float = 0.10,
) -> pd.DataFrame:
    """Generate synthetic golfers and labels for supervised learning."""
    if not 1000 <= n <= 2000:
        raise ValueError("n must be between 1,000 and 2,000 for this project")

    rng = np.random.default_rng(seed)
    handicaps = np.clip(rng.normal(17, 9, n), 0, 36).round(1)
    swing_speeds = np.clip(108 - 1.15 * handicaps + rng.normal(0, 7, n), 60, 120).round(1)
    driver_carry = np.clip(swing_speeds * 2.35 + rng.normal(0, 13, n), 130, 315).round(0)
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
            feel_probs = [0.6, 0.1, 0.3]
        shot_shapes.append(str(rng.choice(SHOT_SHAPES, p=shot_probs)))
        goals.append(str(rng.choice(GOALS, p=goal_probs)))
        iron_miss = str(rng.choice(IRON_MISSES, p=miss_probs))
        iron_misses.append(iron_miss)
        iron_feels.append(str(rng.choice(IRON_FEELS, p=feel_probs)))

        # Divot depth (attack angle) correlates with iron miss pattern in real
        # swings - a digging miss (Fat/Thin) comes from a steep, deep-divot
        # attack angle, while a consistently clean strike tends to sweep the
        # turf shallower. Sampled per-golfer, not per-handicap-band, since the
        # correlation is with the miss itself rather than skill level.
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
        loft = _with_noise(_driver_loft(speed, handicap, goal), DRIVER_LOFT_LABELS, rng, noise_rate)
        flex = _with_noise(_shaft_flex(speed), SHAFT_FLEX_LABELS, rng, noise_rate)
        fairway_loft = _with_noise(
            _fairway_wood_loft(speed, handicap, goal), FAIRWAY_WOOD_LOFT_LABELS, rng, noise_rate
        )
        bounce = _with_noise(_wedge_bounce(turf, divot), WEDGE_BOUNCE_LABELS, rng, noise_rate)
        iron = _with_noise(
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


def save_dataset(path: str | Path, n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate and save the golfer dataset to CSV."""
    frame = generate_golfer_profiles(n=n, seed=seed)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


if __name__ == "__main__":
    save_dataset(Path("data") / "golfers.csv")
