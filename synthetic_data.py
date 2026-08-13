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


def _with_adjacent_noise(
    label: str, ordered_labels: list[str], rng: np.random.Generator, rate: float
) -> str:
    """Flip a label to a neighbouring category at `rate`.

    All three targets are ordinal (8 < 9 < 10.5 < 12 deg, L < A < R < S < X,
    blade < ... < super-game-improvement). Real fitting disagreement is between
    neighbouring categories, so flipping uniformly across the whole list would
    manufacture impossible rows - a 12 deg golfer relabelled 8 deg, or a
    super-game-improvement golfer relabelled into blades.

    The flip rate is unchanged, so the Bayes-optimal ceiling stays at
    1 - rate; only the *direction* of the disagreement becomes plausible.
    """
    if rng.random() >= rate:
        return label
    index = ordered_labels.index(label)
    neighbours = []
    if index > 0:
        neighbours.append(ordered_labels[index - 1])
    if index < len(ordered_labels) - 1:
        neighbours.append(ordered_labels[index + 1])
    return str(rng.choice(neighbours))


def _driver_loft(speed: float, handicap: float, goal: str) -> str:
    """Pick a head loft from clubhead speed, then adjust for skill and intent.

    Loft falls as speed rises. The retail market is dominated by 9 and 10.5 deg
    heads, so the bands are set so those two cover the bulk of golfers rather
    than pushing every slower swing to 12 deg.
    """
    if speed < 82:
        loft = "12"
    elif speed < 96:
        loft = "10.5"
    elif speed < 108:
        loft = "9"
    else:
        loft = "8"

    # High handicaps need launch help, so they are not sold the lowest lofts.
    if handicap > 24 and loft in {"8", "9"}:
        loft = "10.5"
    # Forgiveness seekers step up one band rather than jumping straight to 12.
    if goal == "Forgiveness":
        if loft == "8":
            loft = "9"
        elif loft == "9" and speed < 102:
            loft = "10.5"
    return loft


def _shaft_flex(speed: float) -> str:
    """Map clubhead speed to shaft flex using standard fitting bands.

    X-flex starts at 110 mph; below that even quick swings are better served by
    stiff, which keeps X-flex the small minority it is in the real market.
    """
    if speed < 72:
        return "L"
    if speed < 84:
        return "A"
    if speed < 97:
        return "R"
    if speed < 110:
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

    # True blades are a niche: a low fit score alone is not enough, the golfer
    # also has to have the handicap to justify them. Without the gate, wanting a
    # forged look was enough to be sold blades, which put 1 in 5 golfers in them.
    if fit_score <= -0.8 and handicap <= 7:
        return "blade"
    if fit_score <= 1.0:
        return "players-cb"
    if fit_score <= 2.3:
        return "players-distance"
    if fit_score <= 3.6:
        return "game-improvement"
    return "super-game-improvement"


def generate_golfer_profiles(
    n: int = 12_000,
    seed: int = 42,
    noise_rate: float = 0.10,
) -> pd.DataFrame:
    """Generate synthetic golfers and labels for supervised learning.

    Sample size matters more than model choice here. The iron_category rule
    sums six weighted terms and cuts the result at four thresholds, and a
    forest needs a lot of examples to place those cuts: measured on noise-free
    labels it scores 87% at 1 500 rows, 93% at 6 000 and 96% at 12 000. The
    default is set accordingly. Ordinal models, gradient boosting and extra
    derived features were all tried at 1 500 rows and none beat a plain forest.

    noise_rate flips that share of labels to a neighbouring category, which
    caps attainable accuracy at roughly 1 - noise_rate no matter how much data
    is supplied.
    """
    if not 500 <= n <= 100_000:
        raise ValueError("n must be between 500 and 100,000")

    rng = np.random.default_rng(seed)
    handicaps = np.clip(rng.normal(17, 9, n), 0, 36).round(1)

    # Speed falls with handicap, but far more loosely than a tight line: plenty
    # of high handicaps swing hard and wild, and plenty of low handicaps score
    # by being straight rather than long. A shallow slope plus a wide spread
    # keeps the correlation moderate and puts the bands near published figures
    # (scratch ~106 mph, male amateur average ~93, 30+ handicap ~82).
    swing_speeds = np.clip(
        106 - 0.68 * handicaps + rng.normal(0, 10, n), 65, 120
    ).round(1)

    # Carry is not a fixed multiple of speed: better players compress the ball
    # harder, so yards-per-mph rises as handicap falls. This stops carry being a
    # near-duplicate of speed and lets it carry strike-quality information.
    carry_ratio = 2.45 - 0.009 * handicaps
    driver_carry = np.clip(
        swing_speeds * carry_ratio + rng.normal(0, 9, n), 120, 330
    ).round(0)

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
            # Even single-figure golfers mostly game cavity backs; a blade-like
            # preference is a minority taste rather than the default.
            feel_probs = [0.42, 0.18, 0.40]
        shot_shapes.append(str(rng.choice(SHOT_SHAPES, p=shot_probs)))
        goals.append(str(rng.choice(GOALS, p=goal_probs)))
        iron_misses.append(str(rng.choice(IRON_MISSES, p=miss_probs)))
        iron_feels.append(str(rng.choice(IRON_FEELS, p=feel_probs)))

    rows = []
    for handicap, speed, carry, shape, goal, i_miss, i_feel in zip(
        handicaps, swing_speeds, driver_carry, shot_shapes, goals, iron_misses, iron_feels
    ):
        loft = _with_adjacent_noise(
            _driver_loft(speed, handicap, goal), DRIVER_LOFT_LABELS, rng, noise_rate
        )
        flex = _with_adjacent_noise(_shaft_flex(speed), SHAFT_FLEX_LABELS, rng, noise_rate)
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
                "driver_loft": loft,
                "shaft_flex": flex,
                "iron_category": iron,
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
