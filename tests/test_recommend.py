from pathlib import Path

from preprocess import load_equipment_catalog
from recommend import (
    ClubRecommendation,
    GolferInput,
    filter_recommendations_by_budget,
    recommend_clubs,
    score_driver,
    score_iron_set,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EQUIPMENT_DIR = PROJECT_ROOT / "data" / "equipment" / "data"


def test_score_driver_prefers_speed_loft_and_goal_matches():
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


def test_driver_trajectory_changes_launch_scoring():
    low_launch_driver = {
        "brand": "Test",
        "model": "Low",
        "lofts": [9, 10.5],
        "forgivenessTier": 4,
        "launchChar": "low",
        "spinChar": "low",
        "speedMinMph": 85,
        "speedMaxMph": 110,
        "msrp": 599,
        "family": "low-spin",
    }
    high_launch_driver = {**low_launch_driver, "model": "High", "launchChar": "high"}
    golfer = GolferInput(
        handicap=12,
        swing_speed=95,
        driver_carry=235,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        driver_trajectory="Too low",
    )

    high_score = score_driver(high_launch_driver, golfer, predicted_loft="10.5").score
    low_score = score_driver(low_launch_driver, golfer, predicted_loft="10.5").score

    assert high_score > low_score


def test_iron_specific_shot_shape_goal_and_trajectory_affect_scoring():
    iron = {
        "brand": "Test",
        "model": "GI",
        "ironCategory": "game-improvement",
        "construction": "cavity-back",
        "forgivenessTier": 5,
        "workability": "mid",
        "launchChar": "high",
        "msrp": 999,
    }
    golfer = GolferInput(
        handicap=20,
        swing_speed=82,
        driver_carry=195,
        shot_shape="Draw",
        goal="Distance",
        iron_miss="Inconsistent",
        iron_feel="Confidence-inspiring",
        iron_shot_shape="Slice",
        iron_goal="Forgiveness",
        iron_trajectory="Too low",
    )

    scored = score_iron_set(iron, golfer, "game-improvement")

    assert scored.score >= 90
    assert "slice" in " ".join(scored.reasons).lower()


def test_filter_recommendations_by_budget_keeps_affordable_clubs():
    recommendations = [
        ClubRecommendation(
            name="Affordable",
            score=90,
            reasons=[],
            brand="A",
            model="One",
            msrp=599,
            year=2024,
        ),
        ClubRecommendation(
            name="Premium",
            score=95,
            reasons=[],
            brand="P",
            model="Two",
            msrp=899,
            year=2024,
        ),
        ClubRecommendation(
            name="Unknown",
            score=80,
            reasons=[],
            brand="U",
            model="Three",
            msrp=None,
            year=2024,
        ),
    ]

    filtered = filter_recommendations_by_budget(recommendations, max_budget=600)

    assert [rec.name for rec in filtered] == ["Affordable"]


def test_filter_recommendations_by_budget_preserves_score_order():
    recommendations = [
        ClubRecommendation("Second", 80, [], "B", "Two", 700, 2024),
        ClubRecommendation("First", 90, [], "A", "One", 650, 2024),
        ClubRecommendation("Too Much", 99, [], "C", "Three", 1200, 2024),
    ]

    filtered = filter_recommendations_by_budget(recommendations, max_budget=800)

    assert [rec.name for rec in filtered] == ["First", "Second"]


def test_driver_recommendations_include_all_years_when_budget_filter_is_separate():
    golfer = GolferInput(
        handicap=12,
        swing_speed=94,
        driver_carry=230,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
    )
    catalog = {
        "drivers": [
            {
                "brand": "Older",
                "model": "Best",
                "year": 2022,
                "lofts": [10.5],
                "forgivenessTier": 5,
                "launchChar": "high",
                "spinChar": "mid",
                "speedMinMph": 80,
                "speedMaxMph": 105,
                "msrp": 799,
                "family": "versatile",
            },
            {
                "brand": "Newer",
                "model": "Best",
                "year": 2025,
                "lofts": [10.5],
                "forgivenessTier": 4,
                "launchChar": "mid",
                "spinChar": "mid",
                "speedMinMph": 80,
                "speedMaxMph": 105,
                "msrp": 699,
                "family": "versatile",
            },
        ]
    }

    recommendations = recommend_clubs(catalog, golfer, "10.5", "players-distance", top_n=5)

    assert {rec.year for rec in recommendations} == {2022, 2025}
