from pathlib import Path

from preprocess import load_equipment_catalog
from recommend import GolferInput, recommend_clubs, score_driver


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EQUIPMENT_DIR = PROJECT_ROOT / "data" / "equipment" / "data"


def test_score_driver_prefers_speed_loft_budget_and_goal_matches():
    golfer = GolferInput(
        handicap=16,
        swing_speed=92,
        driver_carry=225,
        shot_shape="Slice",
        goal="Forgiveness",
        iron_miss="Fat/Thin",
        iron_feel="Forged/Blade-like",
    )
    club = {
        "brand": "Test",
        "model": "Max",
        "lofts": [10.5, 12],
        "forgivenessTier": 5,
        "launchChar": "mid-high",
        "spinChar": "mid",
        "speedMinMph": 80,
        "speedMaxMph": 100,
        "msrp": 599,
        "family": "game-improvement",
    }

    scored = score_driver(club, golfer, predicted_loft="10.5")

    assert scored.score >= 90
    assert "92 mph swing speed" in " ".join(scored.reasons)


def test_recommend_clubs_returns_ranked_top_five_drivers():
    catalog = load_equipment_catalog(EQUIPMENT_DIR)
    golfer = GolferInput(
        handicap=22,
        swing_speed=84,
        driver_carry=205,
        shot_shape="Slice",
        goal="Forgiveness",
        iron_miss="Fat/Thin",
        iron_feel="Forged/Blade-like",
    )

    recommendations = recommend_clubs(
        catalog=catalog,
        golfer=golfer,
        predicted_loft="12",
        predicted_iron_category="game-improvement",
        top_n=5,
    )

    assert len(recommendations) == 5
    assert all(0 <= rec.score <= 100 for rec in recommendations)
    assert recommendations == sorted(recommendations, key=lambda rec: rec.score, reverse=True)
    assert all(rec.name for rec in recommendations)
