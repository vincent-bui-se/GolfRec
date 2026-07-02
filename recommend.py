"""Recommendation scoring for matching AI predictions to equipment records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GolferInput:
    handicap: float
    swing_speed: float
    driver_carry: float
    shot_shape: str
    goal: str
    iron_miss: str
    iron_feel: str


@dataclass(frozen=True)
class ClubRecommendation:
    name: str
    score: int
    reasons: list[str]
    brand: str
    model: str
    msrp: float | None
    category: str = "Driver"


def _closest_loft(
    lofts: list[Any],
    predicted_loft: str,
    adjust_range_deg: float = 0.0,
) -> tuple[float | None, float]:
    """Find the closest available head loft accounting for adjustable hosels.

    If the club has an adjustable hosel (adjustRangeDeg), the effective loft
    range for each head extends by ±adjust_range_deg.  We report the gap after
    this adjustment, so a club that can be dialled in to the target loft shows
    a gap of 0 and earns full loft points.
    """
    if not lofts:
        return None, 10.0
    target = float(predicted_loft)
    numeric_lofts = [float(loft) for loft in lofts]
    best_gap = min(max(0.0, abs(loft - target) - adjust_range_deg) for loft in numeric_lofts)
    closest = min(numeric_lofts, key=lambda loft: abs(loft - target))
    return closest, best_gap


def _speed_score(club: dict[str, Any], speed: float) -> tuple[float, str]:
    low = float(club.get("speedMinMph", 65))
    high = float(club.get("speedMaxMph", 120))
    if low <= speed <= high:
        return 25, f"Fits your {speed:.0f} mph swing speed."
    distance = min(abs(speed - low), abs(speed - high))
    return max(0, 25 - distance * 2), f"Close to your {speed:.0f} mph swing speed range."


def score_driver(
    club: dict[str, Any],
    golfer: GolferInput,
    predicted_loft: str,
) -> ClubRecommendation:
    """Score one driver against the golfer and predicted ideal loft.

    Scoring budget (max 100 pts):
      - Swing speed range match : 25 pts
      - Loft match (+ adjustable): 25 pts
      - Forgiveness / goal       : 20 pts
      - Launch characteristic    : 15 pts
      - Draw-bias / distance fit :  8 pts
      - Base score (always)      :  7 pts
    """
    score = 0.0
    reasons: list[str] = []

    # --- Swing speed (25 pts) ---
    speed_points, speed_reason = _speed_score(club, golfer.swing_speed)
    score += speed_points
    reasons.append(speed_reason)

    # --- Loft match with adjustable hosel awareness (25 pts) ---
    adjust_range = float(club.get("adjustRangeDeg", 0)) / 2  # total ÷ 2 → each direction
    closest_loft, loft_gap = _closest_loft(club.get("lofts", []), predicted_loft, adjust_range)
    loft_points = max(0.0, 25.0 - loft_gap * 10)
    score += loft_points
    if closest_loft is not None:
        if loft_gap == 0 and adjust_range > 0:
            reasons.append(
                f"Adjustable hosel can be set to the ideal {predicted_loft}° target."
            )
        elif loft_gap <= 0.5:
            reasons.append(f"Available loft {closest_loft:g}° closely matches the AI loft target.")

    # --- Forgiveness / goal fit (20 pts) ---
    forgiveness = float(club.get("forgivenessTier", 3))
    if golfer.goal == "Forgiveness":
        score += forgiveness * 3.5
        if club.get("family") in {"game-improvement", "max-forgiveness"}:
            score += 2.5
        if forgiveness >= 4:
            reasons.append("High forgiveness maximises your margin for off-centre hits.")
    elif golfer.goal == "Accuracy":
        score += 8 + min(forgiveness, 4) * 3
        if forgiveness >= 4:
            reasons.append("High forgiveness.")
    else:  # Distance
        if club.get("spinChar") in {"low", "low-mid"}:
            score += 18
            reasons.append("Low-spin design promotes extra distance.")
        elif club.get("spinChar") == "mid":
            score += 14
        else:
            score += 9

    # --- Launch characteristic (15 pts) ---
    launch = str(club.get("launchChar", "mid"))
    if golfer.swing_speed < 85 and "high" in launch:
        score += 15
        reasons.append("High launch helps slower swing speeds maximise carry.")
    elif golfer.swing_speed >= 100 and launch in {"low", "mid-low", "mid"}:
        score += 15
        reasons.append("Controlled launch keeps the ball penetrating into wind.")
    elif 85 <= golfer.swing_speed < 100 and launch in {"mid", "mid-high"}:
        score += 13
        reasons.append("Ideal mid-launch for your swing speed.")
    else:
        score += 8
        reasons.append(f"{launch.replace('-', ' ').title()} launch profile.")

    # --- Shot shape / family bonus (8 pts) ---
    if golfer.shot_shape == "Slice" and (
        club.get("drawBiasBuiltIn") or club.get("drawBiasAvailable") or club.get("family") == "draw-bias"
    ):
        score += 8
        reasons.append("Draw-bias option can help reduce a slice.")
    elif golfer.goal == "Distance" and club.get("family") in {"low-spin", "players"}:
        score += 6
        reasons.append("Low-spin player's head supports maximum distance.")
    elif golfer.shot_shape in {"Draw", "Hook"} and club.get("spinChar") in {"low", "low-mid"}:
        score += 4
        reasons.append("Low-spin design helps moderate a strong draw.")

    # --- Base score (7 pts) ---
    score += 7

    capped = int(round(max(0, min(score, 100))))
    name = f"{club.get('brand', 'Unknown')} {club.get('model', 'Unknown')}"
    return ClubRecommendation(
        name=name,
        score=capped,
        reasons=reasons,
        brand=str(club.get("brand", "Unknown")),
        model=str(club.get("model", "Unknown")),
        msrp=club.get("msrp") if isinstance(club.get("msrp"), (int, float)) else None,
    )


def score_iron_set(
    club: dict[str, Any],
    golfer: GolferInput,
    predicted_iron_category: str,
) -> ClubRecommendation:
    """Score one iron set against the golfer and predicted category.

    Scoring budget (max 100 pts):
      - Category match           : 30 pts
      - Forgiveness / miss type  : 25 pts
      - Construction / feel pref : 20 pts
      - Launch (speed vs. iron)  : 15 pts
      - Shot-shape bonus         :  5 pts
      - Base score               :  5 pts
    """
    score = 0.0
    reasons: list[str] = []

    # --- Category match (30 pts) ---
    category = str(club.get("ironCategory", ""))
    if category == predicted_iron_category:
        score += 30
        reasons.append(f"Matches the predicted {category.replace('-', ' ').title()} category.")
    else:
        # Adjacent categories still earn partial credit
        adjacent = {
            "blade": {"players-cb"},
            "players-cb": {"blade", "players-distance"},
            "players-distance": {"players-cb", "game-improvement"},
            "game-improvement": {"players-distance", "super-game-improvement"},
            "super-game-improvement": {"game-improvement"},
        }
        if category in adjacent.get(predicted_iron_category, set()):
            score += 18
            reasons.append(f"Near-match category: {category.replace('-', ' ').title()}.")
        else:
            score += 5
            reasons.append(f"Category: {category.replace('-', ' ').title()}.")

    # --- Forgiveness / miss type (25 pts) ---
    forgiveness = float(club.get("forgivenessTier", 3))
    if golfer.iron_miss in {"Fat/Thin", "Inconsistent"}:
        # Weight forgiveness heavily for inconsistent ball-strikers
        score += forgiveness * 5
        if forgiveness >= 4:
            reasons.append("High forgiveness helps with inconsistent contact.")
        elif forgiveness >= 3:
            reasons.append("Moderate forgiveness suits your miss tendency.")
    else:
        # Consistent ball-strikers earn flat points regardless of forgiveness
        score += 20
        if forgiveness >= 4:
            reasons.append("High forgiveness — extra margin even for consistent players.")

    # --- Construction / feel preference (20 pts) ---
    construction = str(club.get("construction", "")).lower()
    workability = str(club.get("workability", "")).lower()
    if golfer.iron_feel == "Forged/Blade-like":
        if "forged" in construction:
            score += 20
            reasons.append("Forged construction delivers the preferred feel.")
        elif workability in {"high", "medium-high"}:
            score += 14
            reasons.append("High workability approximates a forged feel.")
        else:
            score += 8
    elif golfer.iron_feel == "Confidence-inspiring":
        if "hollow-body" in construction or forgiveness >= 4:
            score += 20
            reasons.append("Wide sole and cavity back inspire confidence at address.")
        elif forgiveness >= 3:
            score += 14
            reasons.append("Cavity back design is confidence-inspiring.")
        else:
            score += 8
    else:  # No preference
        score += 14

    # --- Launch characteristic vs. swing speed (15 pts) ---
    iron_launch = str(club.get("launchChar", "mid")).lower()
    if golfer.swing_speed < 85 and "high" in iron_launch:
        score += 15
        reasons.append("High-launching irons help maximise carry for slower swing speeds.")
    elif golfer.swing_speed >= 100 and iron_launch in {"low", "mid-low"}:
        score += 15
        reasons.append("Lower-launching irons help control trajectory at faster speeds.")
    elif 85 <= golfer.swing_speed < 100 and "mid" in iron_launch:
        score += 12
        reasons.append("Mid-launch irons suit your swing speed well.")
    else:
        score += 8

    # --- Shot shape bonus (5 pts) ---
    if golfer.shot_shape == "Slice" and category in {"game-improvement", "super-game-improvement"}:
        score += 5
        reasons.append("Upright lie and offset help mitigate a slice.")
    elif golfer.shot_shape in {"Draw", "Hook"} and category in {"blade", "players-cb"}:
        score += 4
        reasons.append("Players irons offer the workability to shape shots.")

    # --- Base score (5 pts) ---
    score += 5

    capped = int(round(max(0, min(score, 100))))
    name = f"{club.get('brand', 'Unknown')} {club.get('model', 'Unknown')}"
    return ClubRecommendation(
        name=name,
        score=capped,
        reasons=reasons,
        brand=str(club.get("brand", "Unknown")),
        model=str(club.get("model", "Unknown")),
        msrp=club.get("msrp") if isinstance(club.get("msrp"), (int, float)) else None,
        category="Iron Set",
    )


def recommend_clubs(
    catalog: dict[str, list[dict[str, Any]]],
    golfer: GolferInput,
    predicted_loft: str,
    predicted_iron_category: str,
    top_n: int = 3,
) -> list[ClubRecommendation]:
    """Return ranked driver recommendations from the equipment database."""
    drivers = catalog.get("drivers", [])
    scored = [score_driver(club, golfer, predicted_loft) for club in drivers]

    if golfer.goal == "Forgiveness" or predicted_iron_category in {
        "game-improvement",
        "super-game-improvement",
    }:
        scored = [
            rec
            for rec in scored
            if rec.score >= 60 or "max" in rec.name.lower() or "draw" in rec.name.lower()
        ] or scored

    return sorted(scored, key=lambda rec: rec.score, reverse=True)[:top_n]


def recommend_irons(
    catalog: dict[str, list[dict[str, Any]]],
    golfer: GolferInput,
    predicted_iron_category: str,
    top_n: int = 3,
) -> list[ClubRecommendation]:
    """Return ranked iron set recommendations from the equipment database."""
    irons = catalog.get("iron-sets", [])
    scored = [score_iron_set(club, golfer, predicted_iron_category) for club in irons]

    if golfer.iron_miss in {"Fat/Thin", "Inconsistent"}:
        scored = [rec for rec in scored if rec.score >= 50] or scored

    return sorted(scored, key=lambda rec: rec.score, reverse=True)[:top_n]
