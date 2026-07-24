"""Synthetic golfer dataset generation based on public fitting guidelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import GOALS, SHOT_SHAPES, IRON_MISSES, IRON_FEELS


DRIVER_LOFT_LABELS = ["8", "9", "10.5", "12"]
SHAFT_FLEX_LABELS = ["L", "A", "R", "S", "X"]
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

    shot_shapes = []
    goals = []
    iron_misses = []
    iron_feels = []
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
        iron_misses.append(str(rng.choice(IRON_MISSES, p=miss_probs)))
        iron_feels.append(str(rng.choice(IRON_FEELS, p=feel_probs)))

    rows = []
    for handicap, speed, carry, shape, goal, i_miss, i_feel in zip(
        handicaps, swing_speeds, driver_carry, shot_shapes, goals, iron_misses, iron_feels
    ):
        loft = _with_noise(_driver_loft(speed, handicap, goal), DRIVER_LOFT_LABELS, rng, noise_rate)
        flex = _with_noise(_shaft_flex(speed), SHAFT_FLEX_LABELS, rng, noise_rate)
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
                "driver_loft": loft,
                "shaft_flex": flex,
                "iron_category": iron,
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
