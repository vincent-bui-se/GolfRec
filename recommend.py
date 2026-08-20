"""Recommendation scoring for matching AI predictions to equipment records."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

_TRAILING_YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$")


def _display_name(brand: str, model: str) -> str:
    """Join brand and model without repeating a brand the model already carries.

    Some records bake the brand into the model ("Mizuno" + "Mizuno Pro 245"),
    which read as "Mizuno Mizuno Pro 245" when joined blindly.
    """
    if model.lower().startswith(brand.lower()):
        return model
    return f"{brand} {model}"


# Head-character blurbs. These carry no points; they exist so two clubs with
# identical fit scores still read differently in the results list.
_DRIVER_FAMILY_NOTES = {
    "low-spin": "Low-spin head trades help for speed.",
    "max-forgiveness": "Max-forgiveness build for the widest usable face.",
    "draw-bias": "Draw-bias build fights a left-to-right miss.",
    "players": "Compact players head rewards centre contact.",
    "versatile": "Versatile head balances help and workability.",
}


@dataclass(frozen=True)
class GolferInput:
    handicap: float
    swing_speed: float
    driver_carry: float
    shot_shape: str
    goal: str
    iron_miss: str
    iron_feel: str
    shopping_for: list[str] = field(default_factory=lambda: ["Driver", "Irons"])
    driver_trajectory: str = "About right"
    iron_trajectory: str = "About right"
    iron_shot_shape: str | None = None
    iron_goal: str | None = None
    wedge_turf: str = "Normal"
    divot_depth: str = "Medium"
    # Scoring-time-only preferences (not ML features - see preprocess.py).
    wedge_shot_style: str = "No preference"
    # What loft the golfer is shopping for - see _select_wedge_configuration_candidates.
    # Default (56) is the single most common "do-it-all" wedge loft.
    wedge_loft: float = 56.0
    # The golfer's *current* driver/iron launchChar+spinChar, resolved by the
    # caller from a catalog lookup (app.py) - None when they don't know or
    # skipped the question. "Too high"/"Too low" trajectory only means
    # something relative to what they already play: a golfer already on a
    # low-launch head who still balloons it needs a small nudge further down,
    # not the single lowest-launch head in the catalog (which is what an
    # absolute target would otherwise recommend, risking overcorrecting them
    # into "too low"). None falls back to the old absolute-target behavior.
    current_driver_launch: str | None = None
    current_driver_spin: str | None = None
    current_iron_launch: str | None = None
    current_iron_spin: str | None = None

    @property
    def driver_shot_shape(self) -> str:
        return self.shot_shape

    @property
    def driver_goal(self) -> str:
        return self.goal

    @property
    def effective_iron_shot_shape(self) -> str:
        return self.iron_shot_shape or self.shot_shape

    @property
    def effective_iron_goal(self) -> str:
        return self.iron_goal or self.goal


@dataclass(frozen=True)
class ClubRecommendation:
    name: str
    score: float
    reasons: list[str]
    brand: str
    model: str
    msrp: float | None
    year: int | None = None
    category: str = "Driver"
    years: list[int] | None = None


def _build_recommendation(
    club: dict[str, Any],
    score: float,
    reasons: list[str],
    category: str,
) -> ClubRecommendation:
    """Cap/round a raw score and assemble the final ClubRecommendation.

    Shared by _score_wood, score_iron_set, and score_wedge, which all ended
    with the same capping + name-building + construction logic - factored
    out so a future change to that finalization (e.g. rounding precision)
    only needs to happen once.
    """
    capped = round(max(0.0, min(score, 100.0)))
    name = _display_name(str(club.get("brand", "Unknown")), str(club.get("model", "Unknown")))
    return ClubRecommendation(
        name=name,
        score=capped,
        reasons=reasons,
        brand=str(club.get("brand", "Unknown")),
        model=str(club.get("model", "Unknown")),
        msrp=club.get("msrp") if isinstance(club.get("msrp"), (int, float)) else None,
        year=club.get("year") if isinstance(club.get("year"), int) else None,
        category=category,
    )


def _closest_loft(
    lofts: list[Any],
    predicted_loft: str,
    adjust_range_deg: float = 0.0,
) -> tuple[float | None, float]:
    """Find the closest available head loft accounting for adjustable hosels.

    If the club has an adjustable hosel (adjustRangeDeg), the effective loft
    range for each head extends by plus/minus adjust_range_deg. We report the gap after
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


# Launch/spin character on a 0-4 scale. "mid-low" and "low-mid" both appear
# in the catalog for the same underlying bucket (different curators, same
# meaning) - both must map to the same ordinal, which a literal string-set
# membership check doesn't guarantee: the old code checked `{"low",
# "mid-low"}` in some branches and `"mid" in launch` (a substring check that
# happens to also catch "mid-low"/"low-mid") in others, so a driver spelled
# "low-mid" could pass one branch's check and fail an equivalent one a few
# lines later purely on wording.
_LAUNCH_SPIN_ORDER = {
    "low": 0,
    "low-mid": 1,
    "mid-low": 1,
    "mid": 2,
    "mid-high": 3,
    "high": 4,
}


def _ordinal(value: object) -> int:
    return _LAUNCH_SPIN_ORDER.get(str(value).lower(), 2)


def _desired_ordinal(trajectory: str, baseline_ordinal: int) -> int:
    """One step toward correcting the complaint from the golfer's *current*
    launch/spin, not the extreme end of the scale - see GolferInput's
    current_driver_launch docstring for why."""
    if trajectory == "Too high":
        return max(0, baseline_ordinal - 1)
    if trajectory == "Too low":
        return min(4, baseline_ordinal + 1)
    return baseline_ordinal


def _relative_trajectory_points(
    candidate_value: object,
    trajectory: str,
    baseline_value: str,
    max_points: float,
) -> tuple[float, int]:
    """Score a candidate's launch or spin bucket against a one-step target
    from the golfer's known current bucket. Returns (points, gap) so the
    caller can phrase a reason only on a strong (small-gap) match."""
    baseline_ordinal = _ordinal(baseline_value)
    desired_ordinal = _desired_ordinal(trajectory, baseline_ordinal)
    gap = abs(_ordinal(candidate_value) - desired_ordinal)
    points = max(0.0, max_points - gap * (max_points / 3))
    return points, gap


def _speed_score(club: dict[str, Any], speed: float) -> tuple[float, str]:
    low = float(club.get("speedMinMph", 65))
    high = float(club.get("speedMaxMph", 120))
    if low <= speed <= high:
        # Say where in the head's window the golfer lands. Heads differ widely
        # here, and sitting at an edge is real fitting information.
        span = high - low
        position = (speed - low) / span if span > 0 else 0.5
        if position < 0.25:
            note = f"{speed:.0f} mph is at the low end of this head's {low:.0f}-{high:.0f} mph range."
        elif position > 0.75:
            note = f"{speed:.0f} mph is at the top of this head's {low:.0f}-{high:.0f} mph range."
        else:
            note = f"{speed:.0f} mph sits mid-range for this head ({low:.0f}-{high:.0f} mph)."
        return 10.0, note

    distance = min(abs(speed - low), abs(speed - high))
    # Provide a 2 mph buffer before penalizing
    effective_distance = max(0.0, distance - 2.0)
    score = max(0.0, 10.0 - effective_distance * 0.75)
    return score, f"Close to your {speed:.0f} mph swing speed range."


def filter_recommendations_by_budget(
    recommendations: list[ClubRecommendation],
    max_budget: float,
    limit: int | None = None,
) -> list[ClubRecommendation]:
    """Keep recommendations with a known MSRP at or below the selected budget.

    NOT called by app.py / the live app. The budget slider needs instant,
    server-round-trip-free filtering as the user drags it, so
    static/app.js's `filterByBudget` reimplements this exact logic
    client-side, and that JS copy is what actually runs in production. This
    Python version exists as a tested, readable reference and for any future
    server-side/API consumer - editing it has zero effect on the deployed
    app unless the JS copy is also updated to match, or app.py is wired to
    call this directly.
    """
    affordable = [
        recommendation
        for recommendation in recommendations
        if recommendation.msrp is not None and recommendation.msrp <= max_budget
    ]
    ordered = sorted(affordable, key=lambda rec: rec.score, reverse=True)
    return ordered if limit is None else ordered[:limit]


def select_recommendations_for_display(
    recommendations: list[ClubRecommendation],
    default_limit: int = 5,
) -> list[ClubRecommendation]:
    """Return default top recommendations, expanding flat top-score groups.

    NOT called by app.py / the live app - same situation as
    `filter_recommendations_by_budget` above: static/app.js's
    `selectForDisplay` is the client-side duplicate that actually runs.
    """
    if len(recommendations) <= default_limit:
        return recommendations

    default_slice = recommendations[:default_limit]
    first_score = default_slice[0].score
    if any(recommendation.score != first_score for recommendation in default_slice):
        return default_slice

    selected = list(default_slice)
    for recommendation in recommendations[default_limit:]:
        selected.append(recommendation)
        if recommendation.score < first_score:
            break
    return selected


def prioritise_distinctive_reasons(
    recommendations: list[ClubRecommendation],
) -> list[ClubRecommendation]:
    """Reorder each club's reasons so the least common ones come first.

    Reasons are appended in scoring order, which front-loads the statements
    nearly every club shares ("fits your swing speed", "adjustable hosel").
    Callers only show the first few, so without this the list reads identically
    down the page. Ranking by how many clubs in the set share a reason surfaces
    what actually separates this club, and ties keep their original order.
    """
    frequency = Counter(
        reason for recommendation in recommendations for reason in recommendation.reasons
    )
    ranked: list[ClubRecommendation] = []
    for recommendation in recommendations:
        order = {reason: index for index, reason in enumerate(recommendation.reasons)}
        ranked.append(
            replace(
                recommendation,
                reasons=sorted(
                    recommendation.reasons,
                    key=lambda reason: (frequency[reason], order[reason]),
                ),
            )
        )
    return ranked


def _score_wood(
    club: dict[str, Any],
    golfer: GolferInput,
    predicted_loft: str,
    category: str,
) -> ClubRecommendation:
    """Score one driver or fairway wood against the golfer and predicted loft.

    Drivers and fairway woods share the same catalog schema (lofts,
    adjustable hosel, launch/spin character, forgiveness score, family) and the
    same fitting logic - only the target loft range differs, which the caller
    already accounts for via `predicted_loft`. `category` only affects the
    label on the returned recommendation, never the scoring itself.

    Scoring weights (max 100 pts):
      - Launch characteristic    : 20 pts
      - Spin characteristic      : 18 pts
      - Forgiveness / goal       : 20 pts
      - Loft match (+ adjustable): 15 pts
      - Swing speed range match  : 10 pts
      - Shot shape / family bonus: 12 pts
      - Base score (always)      :  5 pts

    Launch+spin are weighted highest because optimizing them together for the
    golfer's swing speed is the single biggest lever in real club fitting -
    it drives carry distance and dispersion more than any other head trait.
    Swing-speed range is mostly a fit/legality gate (most modern heads clear
    a wide range) rather than a quality signal, so it carries less weight
    than the traits that actually differentiate one head from another for a
    golfer who already clears that gate.
    """
    score = 0.0
    reasons: list[str] = []

    # --- Swing speed (10 pts) ---
    speed_points, speed_reason = _speed_score(club, golfer.swing_speed)
    score += speed_points
    reasons.append(speed_reason)

    # --- Loft match with adjustable hosel awareness (15 pts) ---
    adjust_range = float(club.get("adjustRangeDeg", 0)) / 2  # total / 2 = each direction
    closest_loft, loft_gap = _closest_loft(club.get("lofts", []), predicted_loft, adjust_range)
    loft_points = max(0.0, 15.0 - loft_gap * 7.5)
    score += loft_points
    if closest_loft is not None:
        # Distinguish heads sold in the target loft from those that only reach it
        # via the hosel; "adjustable hosel" alone is true of nearly every driver.
        native_gap = abs(closest_loft - float(predicted_loft))
        if native_gap < 0.01:
            reasons.append(f"Sold in your {predicted_loft} deg target loft.")
        elif loft_gap == 0 and adjust_range > 0:
            reasons.append(
                f"{closest_loft:g} deg head dials to your {predicted_loft} deg "
                f"target (+/-{round(adjust_range, 1):g} deg hosel)."
            )
        elif loft_gap <= 0.5:
            reasons.append(f"Available loft {closest_loft:g} deg closely matches the AI loft target.")

    # --- Forgiveness / goal fit (20 pts) ---
    # forgivenessScore is 0-10 (was a 1-5 tier); default 6.0 mirrors the old
    # neutral default of 3/5.
    forgiveness = float(club.get("forgivenessScore", 6.0))
    if golfer.goal == "Forgiveness":
        score += forgiveness * 1.75
        # "game-improvement" is never a real driver/fairway-wood family
        # value in the catalog (see _DRIVER_FAMILY_NOTES's 5 keys) - that's
        # an ironCategory value, not a family one. Only "max-forgiveness"
        # can ever match here; kept as a single check rather than a set to
        # not imply a second live branch that doesn't exist.
        if club.get("family") == "max-forgiveness":
            score += 2.5
        if forgiveness >= 8:
            reasons.append("High forgiveness maximises your margin for off-centre hits.")
    elif golfer.goal == "Accuracy":
        score += 8 + min(forgiveness, 8) * 1.5
        if forgiveness >= 10:
            reasons.append("Top-tier stability holds your line on off-centre strikes.")
        elif forgiveness >= 8:
            reasons.append("Stable head keeps accuracy misses playable.")
    else:  # Distance
        # Not a spin re-check - the Spin bucket below already scores that.
        # Distance golfers still lose carry on a mis-hit, so forgiveness gets
        # a moderate (not full Forgiveness-goal-strength) credit here instead.
        score += 6 + forgiveness * 1.4
        if forgiveness >= 8:
            reasons.append("Forgiving head preserves distance on mis-hits.")

    # --- Launch characteristic (20 pts) ---
    # There's no separate current-fairway-wood-trajectory question, so this
    # baseline is always the golfer's current *driver* even when scoring a
    # fairway wood - that's an intentional proxy for their general ball-flight
    # tendency, but the reason text must say so explicitly for a fairway wood
    # card, or "than your current driver" reads like a category mismatch.
    baseline_suffix = "" if category == "Driver" else ", carrying that same fix into your fairway woods"
    launch = str(club.get("launchChar", "mid"))
    if golfer.current_driver_launch is not None:
        launch_points, launch_gap = _relative_trajectory_points(
            launch, golfer.driver_trajectory, golfer.current_driver_launch, 20
        )
        score += launch_points
        if launch_gap == 0 and golfer.driver_trajectory == "Too high":
            reasons.append(f"Lower launch than your current driver should bring your ball flight down without overcorrecting{baseline_suffix}.")
        elif launch_gap == 0 and golfer.driver_trajectory == "Too low":
            reasons.append(f"Higher launch than your current driver should bring your ball flight up without overcorrecting{baseline_suffix}.")
        elif launch_gap == 0:
            reasons.append(f"Launch matches your current driver, keeping your trajectory stable{baseline_suffix}.")
    elif golfer.driver_trajectory == "Too low" and _ordinal(launch) >= 3:
        score += 20
        reasons.append("Higher launch helps correct a low ball flight.")
    elif golfer.driver_trajectory == "Too high" and _ordinal(launch) <= 1:
        score += 20
        reasons.append("Lower launch helps bring down a high ball flight.")
    elif golfer.driver_trajectory == "About right" and _ordinal(launch) == 2:
        score += 19
        reasons.append("Mid launch keeps your current trajectory stable.")
    elif golfer.swing_speed < 85 and _ordinal(launch) >= 3:
        score += 17
        reasons.append("High launch helps slower swing speeds maximise carry.")
    elif golfer.swing_speed >= 100 and _ordinal(launch) <= 2:
        score += 20
        reasons.append("Controlled launch keeps the ball penetrating into wind.")
    elif 85 <= golfer.swing_speed < 100 and _ordinal(launch) in {2, 3}:
        score += 17
        reasons.append("Ideal mid-launch for your swing speed.")
    else:
        score += 11
        reasons.append(f"{launch.replace('-', ' ').title()} launch profile.")

    # --- Spin characteristic (18 pts) ---
    spin = str(club.get("spinChar", "mid")).lower()
    if golfer.current_driver_spin is not None:
        spin_points, spin_gap = _relative_trajectory_points(
            spin, golfer.driver_trajectory, golfer.current_driver_spin, 18
        )
        score += spin_points
        if spin_gap == 0 and golfer.driver_trajectory == "Too high":
            reasons.append(f"Lower spin than your current driver should help stop it ballooning{baseline_suffix}.")
        elif spin_gap == 0 and golfer.driver_trajectory == "Too low":
            reasons.append(f"Higher spin than your current driver should help keep it airborne longer{baseline_suffix}.")
        elif spin_gap == 0:
            reasons.append(f"Spin matches your current driver, maintaining your trajectory{baseline_suffix}.")
    elif golfer.driver_trajectory == "Too high" and _ordinal(spin) <= 1:
        score += 18
        reasons.append("Low spin helps prevent ballooning for your high trajectory.")
    elif golfer.driver_trajectory == "Too high" and _ordinal(spin) == 2:
        score += 10
    elif golfer.driver_trajectory == "Too low" and _ordinal(spin) >= 3:
        score += 18
        reasons.append("Higher spin helps keep the ball airborne longer for your low trajectory.")
    elif golfer.driver_trajectory == "Too low" and _ordinal(spin) == 2:
        score += 12
    elif golfer.driver_trajectory == "About right" and _ordinal(spin) == 2:
        score += 18
        reasons.append("Mid spin maintains your optimal trajectory.")
    elif golfer.driver_trajectory == "About right" and _ordinal(spin) in {1, 3}:
        score += 12
    else:
        score += 6

    # --- Shot shape / family bonus (12 pts) ---
    # Weighted up from the original 5: draw-bias fitting is a real,
    # well-documented corrective lever for the majority-slicing amateur
    # population, not a minor tiebreaker.
    if golfer.shot_shape == "Slice" and (
        club.get("drawBiasBuiltIn") or club.get("drawBiasAvailable") or club.get("family") == "draw-bias"
    ):
        score += 12
        reasons.append("Draw-bias option can help reduce a slice.")
    elif golfer.goal == "Distance" and club.get("family") in {"low-spin", "players"}:
        score += 10
        reasons.append("Low-spin player's head supports maximum distance.")
    elif golfer.shot_shape in {"Draw", "Hook"} and _ordinal(club.get("spinChar")) <= 1:
        # _ordinal(), not a raw string-set check - "low-mid" and "mid-low"
        # both appear in the catalog for the same bucket (see
        # _LAUNCH_SPIN_ORDER's comment), and a literal set membership check
        # only catches one spelling.
        score += 7
        reasons.append("Low-spin design helps moderate a strong draw.")

    # --- Head character (no points; describes what sets this head apart) ---
    family_note = _DRIVER_FAMILY_NOTES.get(str(club.get("family", "")))
    if family_note:
        reasons.append(family_note)

    # --- Base score (5 pts) ---
    score += 5

    return _build_recommendation(club, score, reasons, category)


def score_driver(
    club: dict[str, Any],
    golfer: GolferInput,
    predicted_loft: str,
) -> ClubRecommendation:
    """Score one driver against the golfer and predicted ideal loft. See `_score_wood`."""
    return _score_wood(club, golfer, predicted_loft, category="Driver")


def score_fairway_wood(
    club: dict[str, Any],
    golfer: GolferInput,
    predicted_loft: str,
) -> ClubRecommendation:
    """Score one fairway wood against the golfer and predicted ideal loft. See `_score_wood`."""
    return _score_wood(club, golfer, predicted_loft, category="Fairway Wood")


def score_iron_set(
    club: dict[str, Any],
    golfer: GolferInput,
    predicted_iron_category: str,
) -> ClubRecommendation:
    """Score one iron set against the golfer and predicted category.

    Scoring weights (max 100 pts):
      - Category match           : 32 pts
      - Forgiveness / miss type  : 20 pts
      - Launch (speed vs. iron)  : 18 pts
      - Spin characteristic      : 10 pts
      - Construction / feel pref :  8 pts (kept light; see the branch below)
      - Shot-shape bonus         :  7 pts
      - Base score               :  5 pts

    Category carries the most weight because matching construction to skill
    level (blade through super-game-improvement) is the single biggest
    practical iron-fitting decision - a badly mismatched category is a worse
    real-world outcome than any other factor scored here.
    """
    score = 0.0
    reasons: list[str] = []

    # --- Category match (32 pts) ---
    category = str(club.get("ironCategory", ""))
    if category == predicted_iron_category:
        score += 32
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
            score += 19
            reasons.append(f"Near-match category: {category.replace('-', ' ').title()}.")
        else:
            score += 5
            reasons.append(f"Category: {category.replace('-', ' ').title()}.")

    # --- Forgiveness / miss type (20 pts) ---
    forgiveness = float(club.get("forgivenessScore", 6.0))
    iron_goal = golfer.effective_iron_goal
    iron_shape = golfer.effective_iron_shot_shape

    if golfer.iron_miss in {"Fat/Thin", "Inconsistent"} or iron_goal == "Forgiveness":
        # Weight forgiveness heavily for inconsistent ball-strikers
        score += forgiveness * 2
        if forgiveness >= 8:
            reasons.append("High forgiveness helps with inconsistent contact.")
        elif forgiveness >= 6:
            reasons.append("Moderate forgiveness suits your miss tendency.")
    else:
        # Consistent ball-strikers mostly don't need forgiveness insurance, so
        # the baseline is high regardless - but the tiers must still differ in
        # points, not just in reason text, or a truly unforgiving iron ranks
        # identically to a max-forgiveness one for every consistent striker.
        score += 15
        if forgiveness >= 8:
            score += 5
            reasons.append("High forgiveness gives extra margin even for consistent players.")
        elif forgiveness >= 6:
            score += 3
        else:
            score += 1

    # --- Construction / feel preference (8 pts) ---
    # Kept deliberately light: this axis previously ran to 15 pts and swung
    # which irons got recommended more than a golfer could reliably
    # self-assess from a dropdown. It still counts, just as a lighter tiebreaker.
    construction = str(club.get("construction", "")).lower()
    workability = str(club.get("workability", "")).lower()
    if golfer.iron_feel == "Forged/Blade-like":
        if "forged" in construction:
            score += 8
            reasons.append("Forged construction delivers the preferred feel.")
        elif workability in {"high", "mid"}:
            # Was {"high", "medium-high"} - "medium-high" never exists in
            # the catalog (workability is only high/mid/low), so this
            # branch was silently only ever matching "high", leaving every
            # real mid-workability, non-forged iron scored identically to a
            # low-workability one on this axis.
            score += 5
            reasons.append("High workability approximates a forged feel.")
        else:
            score += 2
    elif golfer.iron_feel == "Confidence-inspiring":
        if "hollow-body" in construction:
            score += 8
            reasons.append("Hollow-body construction inspires confidence at address.")
        elif forgiveness >= 6:
            score += 5
            reasons.append("Forgiving head design is confidence-inspiring.")
        else:
            score += 2
    else:  # No preference
        score += 4

    # --- Launch characteristic vs. swing speed (18 pts) ---
    iron_launch = str(club.get("launchChar", "mid")).lower()
    if golfer.current_iron_launch is not None:
        iron_launch_points, iron_launch_gap = _relative_trajectory_points(
            iron_launch, golfer.iron_trajectory, golfer.current_iron_launch, 18
        )
        score += iron_launch_points
        if iron_launch_gap == 0 and golfer.iron_trajectory == "Too high":
            reasons.append("Lower launch than your current irons should bring your ball flight down without overcorrecting.")
        elif iron_launch_gap == 0 and golfer.iron_trajectory == "Too low":
            reasons.append("Higher launch than your current irons should bring your ball flight up without overcorrecting.")
        elif iron_launch_gap == 0:
            reasons.append("Launch matches your current irons, preserving your trajectory.")
    elif golfer.iron_trajectory == "Too low" and _ordinal(iron_launch) >= 3:
        score += 18
        reasons.append("High-launching irons help correct your low iron flight.")
    elif golfer.iron_trajectory == "Too high" and _ordinal(iron_launch) <= 1:
        score += 18
        reasons.append("Lower-launching irons help bring down your iron flight.")
    elif golfer.iron_trajectory == "About right" and _ordinal(iron_launch) == 2:
        score += 16
        reasons.append("Mid-launch irons preserve your current trajectory.")
    elif golfer.swing_speed < 85 and _ordinal(iron_launch) >= 3:
        score += 14
        reasons.append("High-launching irons help maximise carry for slower swing speeds.")
    elif golfer.swing_speed >= 100 and _ordinal(iron_launch) <= 1:
        score += 14
        reasons.append("Lower-launching irons help control trajectory at faster speeds.")
    elif 85 <= golfer.swing_speed < 100 and _ordinal(iron_launch) == 2:
        score += 14
        reasons.append("Mid-launch irons suit your swing speed well.")
    else:
        score += 10

    # --- Spin characteristic (10 pts) ---
    iron_spin = str(club.get("spinChar", "mid")).lower()
    if golfer.current_iron_spin is not None:
        iron_spin_points, iron_spin_gap = _relative_trajectory_points(
            iron_spin, golfer.iron_trajectory, golfer.current_iron_spin, 10
        )
        score += iron_spin_points
        if iron_spin_gap == 0 and golfer.iron_trajectory == "Too high":
            reasons.append("Lower spin than your current irons should help stop it ballooning.")
        elif iron_spin_gap == 0 and golfer.iron_trajectory == "Too low":
            reasons.append("Higher spin than your current irons should help it hold the green.")
        elif iron_spin_gap == 0:
            reasons.append("Spin matches your current irons for holding greens.")
    elif golfer.iron_trajectory == "Too high" and _ordinal(iron_spin) <= 1:
        score += 10
        reasons.append("Lower spin helps prevent ballooning for your high trajectory.")
    elif golfer.iron_trajectory == "Too high" and _ordinal(iron_spin) == 2:
        score += 6
    elif golfer.iron_trajectory == "Too low" and _ordinal(iron_spin) >= 3:
        score += 10
        reasons.append("Higher spin helps keep the ball airborne longer for your low trajectory.")
    elif golfer.iron_trajectory == "Too low" and _ordinal(iron_spin) == 2:
        score += 6
    elif golfer.iron_trajectory == "About right" and _ordinal(iron_spin) in {2, 3}:
        score += 10
        reasons.append("Good spin profile for holding greens.")
    else:
        score += 4

    # --- Shot shape bonus (7 pts) ---
    if iron_shape == "Slice" and category in {"game-improvement", "super-game-improvement"}:
        score += 7
        reasons.append("Upright lie and offset help mitigate a slice.")
    elif iron_shape in {"Draw", "Hook"} and category in {"blade", "players-cb"}:
        score += 7
        reasons.append("Players irons offer the workability to shape shots.")

    # --- Base score (5 pts) ---
    score += 5

    return _build_recommendation(club, score, reasons, "Iron Set")


def _recommend_woods(
    clubs: list[dict[str, Any]],
    golfer: GolferInput,
    predicted_loft: str,
    predicted_iron_category: str,
    top_n: int,
    category: str,
) -> list[ClubRecommendation]:
    """Shared ranking pass for drivers and fairway woods. See `_score_wood`."""
    scored = [(club, _score_wood(club, golfer, predicted_loft, category)) for club in clubs]

    if golfer.goal == "Forgiveness" or predicted_iron_category in {
        "game-improvement",
        "super-game-improvement",
    }:
        # A genuinely forgiving or draw-biased club should survive the floor
        # even if an unrelated axis (e.g. a speed or trajectory mismatch)
        # dragged its total score down - forgiveness/draw-bias is exactly the
        # trait that matters most for this golfer. Checked against the same
        # catalog fields _score_wood's own bonus reads, not a name-string
        # proxy for them: a club merely named "Max" for marketing reasons
        # isn't necessarily forgiving, and a genuinely draw-biased head like
        # the Ping G440 SFT ("Straight Flight Technology") has neither "max"
        # nor "draw" in its name.
        filtered = [
            (club, rec)
            for club, rec in scored
            if rec.score >= 60
            or float(club.get("forgivenessScore", 6.0)) >= 8
            or club.get("drawBiasBuiltIn")
            or club.get("drawBiasAvailable")
            or club.get("family") in {"max-forgiveness", "draw-bias"}
        ]
        scored = filtered or scored

    recommendations = [rec for _, rec in scored]
    ranked = sorted(recommendations, key=lambda rec: rec.score, reverse=True)[:top_n]
    return prioritise_distinctive_reasons(ranked)


def recommend_clubs(
    catalog: dict[str, list[dict[str, Any]]],
    golfer: GolferInput,
    predicted_loft: str,
    predicted_iron_category: str,
    top_n: int = 3,
) -> list[ClubRecommendation]:
    """Return ranked driver recommendations from the equipment database."""
    return _recommend_woods(
        catalog.get("drivers", []), golfer, predicted_loft, predicted_iron_category, top_n, "Driver"
    )


def recommend_fairway_woods(
    catalog: dict[str, list[dict[str, Any]]],
    golfer: GolferInput,
    predicted_loft: str,
    predicted_iron_category: str,
    top_n: int = 3,
) -> list[ClubRecommendation]:
    """Return ranked fairway wood recommendations from the equipment database."""
    return _recommend_woods(
        catalog.get("fairway-woods", []),
        golfer,
        predicted_loft,
        predicted_iron_category,
        top_n,
        "Fairway Wood",
    )


# ---------- Wedges ----------
# Wedges still don't get an ML-predicted loft target the way drivers/fairway
# woods/irons do - real wedge fitting is a multi-club gapping decision (a
# golfer typically carries 2-3 wedges spanning a loft range, informed by
# where their iron set stops), and recommending an assembled, gapped wedge
# set is a materially different feature from what's built here. What IS
# built here: the golfer states which single loft they're shopping for
# (golfer.wedge_loft), and that loft picks which configuration on each wedge
# gets evaluated. This matters because a wedge model commonly offers several
# bounce/grind options at different lofts, and comparing wedges on whichever
# configuration happens to have the closest bounce degree - regardless of
# loft - let a generic "works everywhere" grind win a Low-bounce request just
# by having a low-bounce loft option somewhere in its lineup, while a
# genuinely purpose-built low-bounce specialist grind on another wedge lost
# purely because its low-bounce loft wasn't the golfer's actual target loft.
# Comparing wedges at the same loft fixes that.

# Bounce tier boundaries, chosen from the catalog's own bounce values (4-14
# degrees): most low-bounce grinds sit at 4-7, most mid at 8-10, most high at
# 11+.
_LOW_BOUNCE_MAX = 7
_MID_BOUNCE_MAX = 10
_BOUNCE_TIER_CENTER = {"Low": 5.5, "Mid": 9.0, "High": 12.5}
_BOUNCE_TIER_ORDER = ["Low", "Mid", "High"]

# What each turf/lie condition needs from a grind, in the grind metadata's own
# vocabulary (each grind lists which conditions it's `bestFor`).
_TURF_BESTFOR_TAGS = {
    "Firm": {"firm-turf", "tight-lies"},
    "Normal": {"all-conditions"},
    "Soft": {"soft-turf", "sand-shots"},
}

# Divot depth is a direct attack-angle proxy for how much sole a golfer needs
# under the leading edge, independent of turf: a deep digger wants a wide sole
# regardless of ground firmness, a shallow sweeper wants a narrow sole so the
# leading edge (not the sole) meets the turf first.
_SOLE_WIDTH_FOR_DIVOT = {"Deep": "wide", "Medium": "medium", "Shallow": "narrow"}

# Greenside shot-shaping preference maps straight to the two most common
# `bestFor` tags in the wedge catalog that nothing previously scored against.
_SHOT_STYLE_TAGS = {
    "One repeatable shot": "square-face-work",
    "I like to shape shots around the green": "open-face-work",
}
# Reason-text phrasing for each shot style, kept separate from the dropdown
# option text above so the sentence reads naturally either way.
_SHOT_STYLE_REASON_TEXT = {
    "One repeatable shot": "your repeatable, square-face shot style",
    "I like to shape shots around the green": "shaping shots around the green",
}

_WEDGE_FAMILY_NOTES = {
    "max-forgiveness": "Wide, forgiving sole helps off-centre and heavy contact.",
    "players": "Compact players shape rewards precise strikes.",
    "versatile": "Versatile grind works across a range of lies.",
}


def _bounce_tier(bounce: float) -> str:
    if bounce <= _LOW_BOUNCE_MAX:
        return "Low"
    if bounce <= _MID_BOUNCE_MAX:
        return "Mid"
    return "High"


def _select_wedge_configuration_candidates(
    club: dict[str, Any], target_loft: float
) -> list[dict[str, Any]]:
    """All configurations at the loft closest to what the golfer is shopping for.

    Usually 1-4 (a wedge often offers several bounce/grind options at the
    same loft) - score_wedge scores each candidate fully and keeps whichever
    fits best, rather than this function guessing which one to pick.
    """
    configurations = club.get("configurations", [])
    if not configurations:
        return []
    closest_loft = min(
        configurations, key=lambda config: abs(float(config.get("loft", target_loft)) - target_loft)
    )["loft"]
    return [config for config in configurations if config.get("loft") == closest_loft]


def _score_wedge_configuration(
    club: dict[str, Any],
    golfer: GolferInput,
    predicted_bounce: str,
    config: dict[str, Any],
) -> tuple[float, list[str], set[str]]:
    """Score one specific loft/bounce/grind configuration's fit.

    Covers only the axes that depend on *which* configuration was picked
    (bounce tier, grind/turf, sole width, 60 pts combined) - score_wedge
    scores this once per same-loft candidate and keeps the best result.
    Everything config-independent (shot-style, forgiveness, workability,
    base) is scored once in score_wedge itself, not repeated per candidate.
    """
    points = 0.0
    reasons: list[str] = []
    best_for = _grind_best_for(club, config.get("grindCode"))

    # --- Bounce tier match (30 pts) ---
    chosen_bounce = float(config.get("bounce", 9))
    chosen_tier = _bounce_tier(chosen_bounce)
    tier_gap = abs(_BOUNCE_TIER_ORDER.index(chosen_tier) - _BOUNCE_TIER_ORDER.index(predicted_bounce))
    if tier_gap == 0:
        points += 30
        reasons.append(f"{chosen_bounce:g} deg bounce fits your {predicted_bounce.lower()}-bounce need.")
    elif tier_gap == 1:
        points += 17
        reasons.append(f"{chosen_bounce:g} deg bounce is close to your bounce need.")
    else:
        points += 7

    # --- Grind fit for stated turf/lie (18 pts) ---
    wanted_tags = _TURF_BESTFOR_TAGS.get(golfer.wedge_turf, set())
    if best_for & wanted_tags:
        points += 18
        reasons.append(f"Grind is built for your {golfer.wedge_turf.lower()} turf conditions.")
    elif "all-conditions" in best_for:
        points += 11
        reasons.append("All-conditions grind adapts to most turf.")
    else:
        points += 4

    # --- Sole-width fit for divot depth (12 pts) ---
    sole_width = _grind_sole_width(club, config.get("grindCode"))
    wanted_width = _SOLE_WIDTH_FOR_DIVOT.get(golfer.divot_depth, "medium")
    if sole_width == wanted_width:
        points += 12
        reasons.append(f"{sole_width.title()} sole matches your {golfer.divot_depth.lower()} divot.")
    elif sole_width == "medium" or wanted_width == "medium":
        points += 7
    else:
        points += 2

    return points, reasons, best_for


def _grind_best_for(club: dict[str, Any], grind_code: str | None) -> set[str]:
    for grind in club.get("grinds", []):
        if grind.get("grindCode") == grind_code:
            return set(grind.get("bestFor", []))
    return set()


def _grind_sole_width(club: dict[str, Any], grind_code: str | None) -> str | None:
    for grind in club.get("grinds", []):
        if grind.get("grindCode") == grind_code:
            width = grind.get("soleWidth")
            return str(width) if width else None
    return None


def score_wedge(
    club: dict[str, Any],
    golfer: GolferInput,
    predicted_bounce: str,
) -> ClubRecommendation:
    """Score one wedge against the golfer's stated loft and predicted bounce need.

    Scoring weights (max 100 pts):
      - Bounce tier match (from turf + divot depth) : 30 pts
      - Grind fit for stated turf/lie                : 18 pts
      - Sole-width fit for divot depth                : 12 pts
      - Greenside shot-style fit                      : 12 pts
      - Forgiveness / contact quality                 : 14 pts
      - Workability / shot-shaping                    :  9 pts
      - Base score (always)                           :  5 pts

    Bounce/grind/sole (turf interaction, 60 pts combined) keeps the largest
    share because wedge fitting is genuinely turf- and lie-centric - every
    major wedge maker's own fitting guide centers on it, more so than for
    full-swing clubs. Those three are scored per-candidate across every
    configuration the wedge offers at the golfer's stated loft
    (see _score_wedge_configuration) and the best-scoring one wins.
    Forgiveness/MOI is trimmed relative to the driver/iron scoring: it's a
    real but smaller factor for short-game clubs, where touch, spin, and
    turf interaction matter more than off-centre-hit margin.
    """
    candidates = _select_wedge_configuration_candidates(club, golfer.wedge_loft)
    if not candidates:
        score, reasons, best_for = 0.0, [], set()
    else:
        scored_candidates = [
            _score_wedge_configuration(club, golfer, predicted_bounce, config) for config in candidates
        ]
        score, reasons, best_for = max(scored_candidates, key=lambda candidate: candidate[0])
    reasons = list(reasons)

    # --- Greenside shot-style fit (12 pts) ---
    wanted_tag = _SHOT_STYLE_TAGS.get(golfer.wedge_shot_style)
    if wanted_tag is None:
        score += 7
    elif wanted_tag in best_for:
        score += 12
        style_text = _SHOT_STYLE_REASON_TEXT.get(golfer.wedge_shot_style, golfer.wedge_shot_style.lower())
        reasons.append(f"Grind suits {style_text}.")
    elif "all-conditions" in best_for:
        score += 7
    else:
        score += 2

    # --- Forgiveness / contact quality (14 pts) ---
    # Same combined trigger as score_iron_set: a digging/inconsistent miss or
    # an explicit Forgiveness goal both call for a forgiving wedge fit.
    forgiveness = float(club.get("forgivenessScore", 6.0))
    triggered = golfer.iron_miss in {"Fat/Thin", "Inconsistent"} or golfer.goal == "Forgiveness"
    if triggered:
        # Base caps at 12, not the bucket's full 14, reserving real headroom
        # for the versatility bonus below - at forgiveness=10 (4 real wedges
        # in the catalog sit there) the old 1.4x/14 cap left zero room for
        # the +1.4 bonus to add anything, so the reason text printed without
        # actually changing the score.
        forgiveness_points = min(forgiveness * 1.2, 12)
        if forgiveness >= 8:
            reasons.append("High forgiveness helps with off-centre wedge contact.")
        if "high-handicap-versatility" in best_for:
            forgiveness_points = min(forgiveness_points + 2, 14)
            reasons.append("Grind is built for high-handicap versatility.")
    else:
        forgiveness_points = 10
        if forgiveness >= 8:
            forgiveness_points += 3
            reasons.append("Forgiving sole gives extra margin even for consistent contact.")
    score += forgiveness_points

    # --- Workability / shot-shaping (9 pts) ---
    workability = str(club.get("workability", "mid")).lower()
    if golfer.goal in {"Accuracy", "Distance"}:
        if workability == "high":
            score += 9
            reasons.append("High workability supports precise, shaped wedge shots.")
        elif workability == "mid":
            score += 5
        else:
            score += 2
    else:  # Forgiveness
        if workability == "low":
            score += 9
            reasons.append("Low-workability shape favours consistency over shot-shaping.")
        elif workability == "mid":
            score += 5
        else:
            score += 2

    # --- Family character (no points; flavour text) ---
    family_note = _WEDGE_FAMILY_NOTES.get(str(club.get("family", "")))
    if family_note:
        reasons.append(family_note)

    # --- Base score (5 pts) ---
    score += 5

    return _build_recommendation(club, score, reasons, "Wedge")


def recommend_wedges(
    catalog: dict[str, list[dict[str, Any]]],
    golfer: GolferInput,
    predicted_bounce: str,
    top_n: int = 3,
) -> list[ClubRecommendation]:
    """Return ranked wedge recommendations from the equipment database."""
    wedges = catalog.get("wedges", [])
    scored = [score_wedge(club, golfer, predicted_bounce) for club in wedges]

    if golfer.iron_miss in {"Fat/Thin", "Inconsistent"}:
        scored = [rec for rec in scored if rec.score >= 50] or scored

    ranked = sorted(scored, key=lambda rec: rec.score, reverse=True)[:top_n]
    return prioritise_distinctive_reasons(ranked)


def _normalize_iron_model(model: str) -> str:
    """Strip a trailing " (YYYY)" year suffix some catalog entries bake into the model name."""
    return _TRAILING_YEAR_RE.sub("", model).strip()


def merge_same_name_iron_sets(
    recommendations: list[ClubRecommendation],
) -> list[ClubRecommendation]:
    """Collapse same-brand/same-model iron sets that only differ by year into one entry.

    Sets are merged only when their score and reasons are identical across years
    (i.e. the years don't actually change the fit). The merged entry's `year` is
    set to the oldest year in the group so the existing used-condition filter
    (`year <= USED_MAX_YEAR`) still includes it correctly whenever any year in
    the group qualifies.
    """
    groups: dict[tuple[str, str], list[ClubRecommendation]] = defaultdict(list)
    for rec in recommendations:
        key = (rec.brand, _normalize_iron_model(rec.model))
        groups[key].append(rec)

    merged: list[ClubRecommendation] = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue

        by_signature: dict[tuple[int, tuple[str, ...]], list[ClubRecommendation]] = defaultdict(list)
        for rec in group:
            by_signature[(rec.score, tuple(rec.reasons))].append(rec)

        for signature_group in by_signature.values():
            if len(signature_group) == 1:
                merged.append(signature_group[0])
                continue

            years = sorted({rec.year for rec in signature_group if rec.year is not None})
            msrps = [rec.msrp for rec in signature_group if rec.msrp is not None]
            representative = signature_group[0]
            normalized_model = _normalize_iron_model(representative.model)
            merged.append(
                replace(
                    representative,
                    model=normalized_model,
                    name=_display_name(representative.brand, normalized_model),
                    year=min(years) if years else representative.year,
                    years=years if len(years) > 1 else None,
                    msrp=min(msrps) if msrps else representative.msrp,
                )
            )

    return merged


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

    merged = merge_same_name_iron_sets(scored)
    ranked = sorted(merged, key=lambda rec: rec.score, reverse=True)[:top_n]
    return prioritise_distinctive_reasons(ranked)
