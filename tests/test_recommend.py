from pathlib import Path

from preprocess import load_equipment_catalog
from recommend import (
    ClubRecommendation,
    GolferInput,
    filter_recommendations_by_budget,
    merge_same_name_iron_sets,
    prioritise_distinctive_reasons,
    recommend_clubs,
    recommend_fairway_woods,
    recommend_irons,
    recommend_wedges,
    select_recommendations_for_display,
    score_driver,
    score_fairway_wood,
    score_iron_set,
    score_wedge,
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
        "forgivenessScore": 10.0,
        "launchChar": "mid-high",
        "spinChar": "mid",
        "speedMinMph": 80,
        "speedMaxMph": 100,
        "msrp": 599,
        "family": "game-improvement",
        "drawBiasBuiltIn": True,
    }

    scored = score_driver(club, golfer, predicted_loft="10.5")

    assert scored.score >= 90
    # The speed reason names the golfer's speed and the head's fitted window,
    # rather than a fixed phrase, so assert on the facts it must carry.
    speed_reason = next(reason for reason in scored.reasons if "92 mph" in reason)
    assert "80-100 mph" in speed_reason


def test_reasons_lead_with_what_distinguishes_each_club():
    """The displayed reasons must not read identically down the results list.

    Reasons are appended in scoring order, which front-loads statements nearly
    every club shares. Callers show only the first few, so those must be
    reordered by how rare they are within the result set.
    """
    catalog = load_equipment_catalog(EQUIPMENT_DIR)
    golfer = GolferInput(
        handicap=16,
        swing_speed=92,
        driver_carry=216,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
    )

    ranked = recommend_clubs(
        catalog=catalog,
        golfer=golfer,
        predicted_loft="10.5",
        predicted_iron_category="players-distance",
        top_n=16,
    )

    shown = [tuple(rec.reasons[:3]) for rec in ranked]
    assert len(set(shown)) >= 8, "top results still share the same leading reasons"

    # No single reason may lead every club; that is the flattening this guards.
    leads = {rec.reasons[0] for rec in ranked if rec.reasons}
    assert len(leads) > 1


def test_prioritise_distinctive_reasons_preserves_scores_and_content():
    """Reordering is presentational: it must not add, drop, or rescore anything."""
    before = [
        ClubRecommendation(
            name="A", score=90, reasons=["shared", "rare-a"], brand="A", model="A", msrp=1.0
        ),
        ClubRecommendation(
            name="B", score=80, reasons=["shared", "rare-b"], brand="B", model="B", msrp=2.0
        ),
    ]

    after = prioritise_distinctive_reasons(before)

    assert [rec.score for rec in after] == [90, 80]
    assert [rec.name for rec in after] == ["A", "B"]
    assert [sorted(rec.reasons) for rec in after] == [sorted(rec.reasons) for rec in before]
    # The reason unique to each club now leads.
    assert after[0].reasons[0] == "rare-a"
    assert after[1].reasons[0] == "rare-b"


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
        "forgivenessScore": 8.0,
        "launchChar": "low",
        "spinChar": "low",
        "speedMinMph": 85,
        "speedMaxMph": 110,
        "msrp": 599,
        "family": "low-spin",
    }
    high_launch_driver = {**low_launch_driver, "model": "High", "launchChar": "high", "spinChar": "high"}
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


def test_known_current_driver_prefers_a_modest_step_over_the_extreme():
    """The core scenario a real current-setup baseline is meant to fix: a
    golfer already on a "mid" launch driver who says their flight is too high
    needs a mild reduction (mid-low), not the single most extreme low-launch
    head in the catalog - jumping straight to the extreme risks overcorrecting
    them into "too low", which is exactly the failure mode a baseline avoids.
    """
    base_spec = {
        "brand": "Test",
        "lofts": [10.5],
        "forgivenessScore": 8.0,
        "spinChar": "mid",
        "speedMinMph": 85,
        "speedMaxMph": 110,
        "msrp": 599,
        "family": "versatile",
    }
    modest_step_driver = {**base_spec, "model": "Modest", "launchChar": "mid-low"}
    extreme_driver = {**base_spec, "model": "Extreme", "launchChar": "low"}

    golfer_with_baseline = GolferInput(
        handicap=14,
        swing_speed=95,
        driver_carry=225,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        driver_trajectory="Too high",
        current_driver_launch="mid",
        current_driver_spin="mid",
    )

    modest_score = score_driver(modest_step_driver, golfer_with_baseline, predicted_loft="10.5").score
    extreme_score = score_driver(extreme_driver, golfer_with_baseline, predicted_loft="10.5").score

    assert modest_score > extreme_score
    assert any("current driver" in reason.lower() for reason in
               score_driver(modest_step_driver, golfer_with_baseline, predicted_loft="10.5").reasons)

    # Without a known baseline, the old undifferentiated behavior holds: both
    # count as "low enough" for a "Too high" complaint and score identically
    # on the launch axis specifically - confirming the difference above comes
    # from the baseline, not some other change to these two fixtures.
    golfer_without_baseline = GolferInput(
        handicap=14,
        swing_speed=95,
        driver_carry=225,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        driver_trajectory="Too high",
    )
    modest_score_no_baseline = score_driver(modest_step_driver, golfer_without_baseline, predicted_loft="10.5").score
    extreme_score_no_baseline = score_driver(extreme_driver, golfer_without_baseline, predicted_loft="10.5").score
    assert modest_score_no_baseline == extreme_score_no_baseline


def test_launch_char_spelling_variants_score_identically_without_a_baseline():
    """Regression test: the catalog has both "mid-low" and "low-mid" for the
    same real bucket (different curators, same meaning). The old code checked
    literal string sets like {"low", "mid-low"} in some branches, so a club
    spelled "low-mid" could silently fall through to a worse-scoring branch
    than an otherwise-identical club spelled "mid-low"."""
    base_spec = {
        "brand": "Test",
        "lofts": [10.5],
        "forgivenessScore": 8.0,
        "spinChar": "mid",
        "speedMinMph": 85,
        "speedMaxMph": 110,
        "msrp": 599,
        "family": "versatile",
    }
    spelled_mid_low = {**base_spec, "model": "A", "launchChar": "mid-low"}
    spelled_low_mid = {**base_spec, "model": "B", "launchChar": "low-mid"}
    golfer = GolferInput(
        handicap=14,
        swing_speed=95,
        driver_carry=225,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        driver_trajectory="Too high",
    )

    assert (
        score_driver(spelled_mid_low, golfer, predicted_loft="10.5").score
        == score_driver(spelled_low_mid, golfer, predicted_loft="10.5").score
    )


def test_iron_trajectory_changes_spin_scoring():
    low_spin_iron = {
        "brand": "Test",
        "model": "Low",
        "ironCategory": "players-distance",
        "construction": "hollow-body",
        "forgivenessScore": 6.0,
        "workability": "mid",
        "launchChar": "mid",
        "spinChar": "low",
        "msrp": 999,
    }
    high_spin_iron = {**low_spin_iron, "model": "High", "spinChar": "high"}
    golfer = GolferInput(
        handicap=10,
        swing_speed=95,
        driver_carry=240,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        iron_trajectory="Too low",
    )

    high_score = score_iron_set(high_spin_iron, golfer, "players-distance").score
    low_score = score_iron_set(low_spin_iron, golfer, "players-distance").score

    assert high_score > low_score


def test_known_current_irons_prefers_a_modest_step_over_the_extreme():
    """Mirrors the driver case for irons: a golfer already on weak/low-launch
    irons who still says their flight is too high needs a small further
    reduction, not the single lowest-launching iron set in the catalog."""
    base_spec = {
        "brand": "Test",
        "ironCategory": "players-distance",
        "construction": "cavity-back",
        "forgivenessScore": 6.0,
        "workability": "mid",
        "spinChar": "mid",
        "msrp": 999,
    }
    modest_step_irons = {**base_spec, "model": "Modest", "launchChar": "mid-low"}
    extreme_irons = {**base_spec, "model": "Extreme", "launchChar": "low"}

    golfer_with_baseline = GolferInput(
        handicap=14,
        swing_speed=95,
        driver_carry=225,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        iron_trajectory="Too high",
        current_iron_launch="mid",
        current_iron_spin="mid",
    )

    modest_score = score_iron_set(modest_step_irons, golfer_with_baseline, "players-distance").score
    extreme_score = score_iron_set(extreme_irons, golfer_with_baseline, "players-distance").score

    assert modest_score > extreme_score

    golfer_without_baseline = GolferInput(
        handicap=14,
        swing_speed=95,
        driver_carry=225,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        iron_trajectory="Too high",
    )
    modest_score_no_baseline = score_iron_set(modest_step_irons, golfer_without_baseline, "players-distance").score
    extreme_score_no_baseline = score_iron_set(extreme_irons, golfer_without_baseline, "players-distance").score
    assert modest_score_no_baseline == extreme_score_no_baseline


def test_iron_specific_shot_shape_goal_and_trajectory_affect_scoring():
    iron = {
        "brand": "Test",
        "model": "GI",
        "ironCategory": "game-improvement",
        "construction": "cavity-back",
        "forgivenessScore": 10.0,
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

    assert scored.score >= 85
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


def test_select_recommendations_for_display_uses_default_when_top_score_is_clear():
    recommendations = [
        ClubRecommendation(f"Club {score}", score, [], "B", "M", 500, 2024)
        for score in [98, 96, 95, 94, 93, 92, 91]
    ]

    selected = select_recommendations_for_display(recommendations, default_limit=5)

    assert [rec.score for rec in selected] == [98, 96, 95, 94, 93]


def test_select_recommendations_for_display_expands_flat_top_five_until_score_drops():
    recommendations = [
        ClubRecommendation(f"Club {index}", score, [], "B", "M", 500, 2024)
        for index, score in enumerate([91, 91, 91, 91, 91, 91, 89, 88], start=1)
    ]

    selected = select_recommendations_for_display(recommendations, default_limit=5)

    assert [rec.score for rec in selected] == [91, 91, 91, 91, 91, 91, 89]


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
                "forgivenessScore": 10.0,
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
                "forgivenessScore": 8.0,
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


def test_forgiveness_floor_bypass_checks_real_fields_not_the_product_name():
    """Regression test: the floor-bypass for Forgiveness/game-improvement
    golfers used to check whether "max" or "draw" appeared in the club's
    *name*, not whether it was actually forgiving or draw-biased. A real
    catalog club - the Ping G440 SFT ("Straight Flight Technology") - is
    genuinely draw-bias (drawBiasBuiltIn=True, family="draw-bias") but its
    name contains neither substring, so it used to get dropped by the score
    floor even for the exact golfer profile that filter exists to protect.
    """
    low_score_draw_bias = {
        "brand": "Ping",
        "model": "G440 SFT",
        "lofts": [8],
        "forgivenessScore": 4.0,
        "launchChar": "low",
        "spinChar": "low",
        "speedMinMph": 95,
        "speedMaxMph": 125,
        "msrp": 599,
        "family": "draw-bias",
        "drawBiasBuiltIn": True,
    }
    control_driver = {
        "brand": "Test",
        "model": "Control",
        "lofts": [10.5],
        "forgivenessScore": 10.0,
        "launchChar": "high",
        "spinChar": "mid",
        "speedMinMph": 60,
        "speedMaxMph": 80,
        "msrp": 599,
        "family": "max-forgiveness",
    }
    golfer = GolferInput(
        handicap=24,
        swing_speed=70,
        driver_carry=165,
        shot_shape="Slice",
        goal="Forgiveness",
        iron_miss="Fat/Thin",
        iron_feel="Confidence-inspiring",
        driver_trajectory="About right",
    )
    catalog = {"drivers": [low_score_draw_bias, control_driver]}

    # Confirm the setup actually exercises the floor (score genuinely < 60,
    # not accidentally rescued by the "if nothing survives, keep everything"
    # fallback because the control club alone would pass).
    raw_score = score_driver(low_score_draw_bias, golfer, predicted_loft="10.5").score
    assert raw_score < 60

    recommendations = recommend_clubs(catalog, golfer, "10.5", "game-improvement", top_n=5)

    assert "G440 SFT" in " ".join(rec.model for rec in recommendations)


def test_merge_same_name_iron_sets_combines_identical_ratings_across_years():
    same_signature = dict(brand="Titleist", model="T100", msrp=1499, reasons=["Great feel"], score=80)
    recommendations = [
        ClubRecommendation(name="Titleist T100", year=2023, **same_signature),
        ClubRecommendation(name="Titleist T100", year=2021, msrp=1399, reasons=same_signature["reasons"], score=80, brand="Titleist", model="T100 (2021)"),
        ClubRecommendation(name="Titleist T100", year=2019, msrp=1399, reasons=same_signature["reasons"], score=80, brand="Titleist", model="T100"),
    ]

    merged = merge_same_name_iron_sets(recommendations)

    assert len(merged) == 1
    combined = merged[0]
    assert combined.years == [2019, 2021, 2023]
    assert combined.year == 2019, "year is set to the oldest so the used-condition filter still includes it"
    assert combined.msrp == 1399, "merged msrp is the minimum across the merged years"
    assert combined.model == "T100", "the (YYYY) suffix is stripped from the merged display model"


def test_merge_same_name_iron_sets_keeps_different_ratings_separate():
    recommendations = [
        ClubRecommendation(
            name="Titleist T100", score=80, reasons=["Great feel"], brand="Titleist", model="T100", msrp=1499, year=2023
        ),
        ClubRecommendation(
            name="Titleist T100", score=55, reasons=["Harder to hit"], brand="Titleist", model="T100 (2019)", msrp=1399, year=2019
        ),
    ]

    merged = merge_same_name_iron_sets(recommendations)

    assert len(merged) == 2
    assert {rec.years for rec in merged} == {None}, "clubs whose score/reasons differ across years stay separate, unmerged"
    assert {rec.year for rec in merged} == {2019, 2023}


def test_merge_same_name_iron_sets_passes_through_unpaired_year_suffixed_model():
    solo = ClubRecommendation(
        name="Callaway Apex CB (2024)",
        score=55,
        reasons=["Forged feel"],
        brand="Callaway",
        model="Apex CB (2024)",
        msrp=1399,
        year=2024,
    )

    merged = merge_same_name_iron_sets([solo])

    assert merged == [solo], "a lone year-suffixed model with no same-name sibling is left untouched"


def test_recommend_irons_merges_same_name_different_years_end_to_end():
    golfer = GolferInput(
        handicap=14,
        swing_speed=90,
        driver_carry=215,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
    )
    base = {
        "brand": "Titleist",
        "ironCategory": "players-cb",
        "construction": "forged-carbon",
        "forgivenessScore": 6.0,
        "workability": "high",
        "launchChar": "mid",
        "spinChar": "mid",
    }
    catalog = {
        "iron-sets": [
            {**base, "model": "T100", "year": 2019, "msrp": 1399},
            {**base, "model": "T100 (2021)", "year": 2021, "msrp": 1399},
            {**base, "model": "T100 (2023)", "year": 2023, "msrp": 1499},
        ]
    }

    recommendations = recommend_irons(catalog, golfer, "players-cb", top_n=3)

    assert len(recommendations) == 1
    merged = recommendations[0]
    assert merged.years == [2019, 2021, 2023]
    assert merged.year == 2019
    USED_MAX_YEAR_EQUIVALENT = 2026 - 4
    assert merged.year <= USED_MAX_YEAR_EQUIVALENT, (
        "merged year must satisfy the client's used-condition filter whenever any merged year qualifies"
    )


def test_score_fairway_wood_shares_driver_scoring_but_uses_its_own_category():
    club = {
        "brand": "Test",
        "model": "Max",
        "lofts": [16.5, 18],
        "forgivenessScore": 8.0,
        "launchChar": "mid-high",
        "spinChar": "mid",
        "speedMinMph": 80,
        "speedMaxMph": 105,
        "msrp": 429,
        "family": "versatile",
    }
    golfer = GolferInput(
        handicap=18,
        swing_speed=88,
        driver_carry=210,
        shot_shape="Slice",
        goal="Forgiveness",
        iron_miss="Fat/Thin",
        iron_feel="Forged/Blade-like",
    )

    driver_score = score_driver(club, golfer, predicted_loft="18")
    fairway_wood_score = score_fairway_wood(club, golfer, predicted_loft="18")

    assert driver_score.score == fairway_wood_score.score
    assert driver_score.reasons == fairway_wood_score.reasons
    assert driver_score.category == "Driver"
    assert fairway_wood_score.category == "Fairway Wood"


def test_wood_launch_reasons_are_category_agnostic():
    # score_driver and score_fairway_wood share one implementation, so a
    # launch-correction reason written with "driver" in it would misdescribe
    # a fairway wood recommendation.
    club = {
        "brand": "Test",
        "model": "Low",
        "lofts": [9, 10.5],
        "forgivenessScore": 8.0,
        "launchChar": "high",
        "spinChar": "mid",
        "speedMinMph": 85,
        "speedMaxMph": 110,
        "msrp": 599,
        "family": "versatile",
    }
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

    scored = score_fairway_wood(club, golfer, predicted_loft="10.5")

    assert not any("driver" in reason.lower() for reason in scored.reasons)


def test_recommend_fairway_woods_returns_ranked_recommendations_end_to_end():
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

    recommendations = recommend_fairway_woods(
        catalog=catalog,
        golfer=golfer,
        predicted_loft="18",
        predicted_iron_category="game-improvement",
        top_n=5,
    )

    assert len(recommendations) == 5
    assert all(0 <= rec.score <= 100 for rec in recommendations)
    assert recommendations == sorted(recommendations, key=lambda rec: rec.score, reverse=True)
    assert all(rec.category == "Fairway Wood" for rec in recommendations)


def test_score_wedge_bounce_tier_match_scores_higher_than_mismatch():
    matching_bounce_club = {
        "brand": "Test",
        "model": "Match",
        "forgivenessScore": 6.0,
        "workability": "high",
        "family": "players",
        "msrp": 179,
        "configurations": [{"loft": 56, "bounce": 12, "grindCode": "K"}],
        "grinds": [{"grindCode": "K", "bestFor": ["all-conditions"]}],
    }
    mismatched_bounce_club = {
        **matching_bounce_club,
        "model": "Mismatch",
        "configurations": [{"loft": 56, "bounce": 6, "grindCode": "L"}],
        "grinds": [{"grindCode": "L", "bestFor": ["all-conditions"]}],
    }
    golfer = GolferInput(
        handicap=14,
        swing_speed=90,
        driver_carry=215,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        wedge_turf="Normal",
    )

    matched = score_wedge(matching_bounce_club, golfer, predicted_bounce="High")
    mismatched = score_wedge(mismatched_bounce_club, golfer, predicted_bounce="High")

    assert matched.score > mismatched.score
    assert any("bounce" in reason.lower() for reason in matched.reasons)


def test_score_wedge_grind_fit_rewards_matching_turf_condition():
    club_firm_grind = {
        "brand": "Test",
        "model": "FirmGrind",
        "forgivenessScore": 6.0,
        "workability": "high",
        "msrp": 179,
        "configurations": [{"loft": 56, "bounce": 9, "grindCode": "L"}],
        "grinds": [{"grindCode": "L", "bestFor": ["firm-turf", "tight-lies"]}],
    }
    club_soft_grind = {
        **club_firm_grind,
        "model": "SoftGrind",
        "grinds": [{"grindCode": "L", "bestFor": ["soft-turf", "sand-shots"]}],
    }
    firm_turf_golfer = GolferInput(
        handicap=14,
        swing_speed=90,
        driver_carry=215,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        wedge_turf="Firm",
    )

    matches_turf = score_wedge(club_firm_grind, firm_turf_golfer, predicted_bounce="Mid")
    mismatches_turf = score_wedge(club_soft_grind, firm_turf_golfer, predicted_bounce="Mid")

    assert matches_turf.score > mismatches_turf.score
    assert any("turf" in reason.lower() for reason in matches_turf.reasons)


def test_score_wedge_forgiveness_axis_favors_high_forgiveness_for_digging_miss():
    high_forgiveness = {
        "brand": "Test",
        "model": "Forgiving",
        "forgivenessScore": 10.0,
        "workability": "low",
        "family": "max-forgiveness",
        "msrp": 149,
        "configurations": [{"loft": 56, "bounce": 9, "grindCode": "F"}],
        "grinds": [{"grindCode": "F", "bestFor": ["all-conditions"]}],
    }
    low_forgiveness = {
        **high_forgiveness,
        "model": "Players",
        "forgivenessScore": 4.0,
        "workability": "high",
        "family": "players",
    }
    golfer = GolferInput(
        handicap=22,
        swing_speed=82,
        driver_carry=195,
        shot_shape="Slice",
        goal="Accuracy",
        iron_miss="Fat/Thin",
        iron_feel="Confidence-inspiring",
        wedge_turf="Normal",
    )

    forgiving_score = score_wedge(high_forgiveness, golfer, predicted_bounce="Mid").score
    players_score = score_wedge(low_forgiveness, golfer, predicted_bounce="Mid").score

    assert forgiving_score > players_score


def test_score_wedge_workability_axis_favors_low_workability_for_forgiveness_goal():
    low_workability = {
        "brand": "Test",
        "model": "Low",
        "forgivenessScore": 6.0,
        "workability": "low",
        "msrp": 149,
        "configurations": [{"loft": 56, "bounce": 9, "grindCode": "F"}],
        "grinds": [{"grindCode": "F", "bestFor": ["all-conditions"]}],
    }
    high_workability = {**low_workability, "model": "High", "workability": "high"}
    golfer = GolferInput(
        handicap=22,
        swing_speed=82,
        driver_carry=195,
        shot_shape="Slice",
        goal="Forgiveness",
        iron_miss="Consistent",
        iron_feel="Confidence-inspiring",
        wedge_turf="Normal",
    )

    low_score = score_wedge(low_workability, golfer, predicted_bounce="Mid").score
    high_score = score_wedge(high_workability, golfer, predicted_bounce="Mid").score

    assert low_score > high_score


def test_score_wedge_sole_width_fit_rewards_matching_divot_depth():
    wide_sole_club = {
        "brand": "Test",
        "model": "Wide",
        "forgivenessScore": 6.0,
        "workability": "mid",
        "msrp": 179,
        "configurations": [{"loft": 56, "bounce": 9, "grindCode": "W"}],
        "grinds": [{"grindCode": "W", "bestFor": ["all-conditions"], "soleWidth": "wide"}],
    }
    narrow_sole_club = {
        **wide_sole_club,
        "model": "Narrow",
        "grinds": [{"grindCode": "W", "bestFor": ["all-conditions"], "soleWidth": "narrow"}],
    }
    deep_divot_golfer = GolferInput(
        handicap=14,
        swing_speed=90,
        driver_carry=215,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        wedge_turf="Normal",
        divot_depth="Deep",
    )

    matches_divot = score_wedge(wide_sole_club, deep_divot_golfer, predicted_bounce="Mid")
    mismatches_divot = score_wedge(narrow_sole_club, deep_divot_golfer, predicted_bounce="Mid")

    assert matches_divot.score > mismatches_divot.score
    assert any("divot" in reason.lower() for reason in matches_divot.reasons)


def test_score_wedge_shot_style_fit_rewards_matching_bestfor_tag():
    square_face_club = {
        "brand": "Test",
        "model": "Square",
        "forgivenessScore": 6.0,
        "workability": "mid",
        "msrp": 179,
        "configurations": [{"loft": 56, "bounce": 9, "grindCode": "S"}],
        "grinds": [{"grindCode": "S", "bestFor": ["square-face-work"]}],
    }
    open_face_club = {
        **square_face_club,
        "model": "Open",
        "grinds": [{"grindCode": "S", "bestFor": ["open-face-work"]}],
    }
    shaper_golfer = GolferInput(
        handicap=14,
        swing_speed=90,
        driver_carry=215,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        wedge_turf="Normal",
        wedge_shot_style="I like to shape shots around the green",
    )

    matches_style = score_wedge(open_face_club, shaper_golfer, predicted_bounce="Mid")
    mismatches_style = score_wedge(square_face_club, shaper_golfer, predicted_bounce="Mid")

    assert matches_style.score > mismatches_style.score
    assert any("shots around the green" in reason.lower() for reason in matches_style.reasons)


def test_score_wedge_high_handicap_versatility_bonus_applies_when_forgiveness_triggered():
    versatile_grind = {
        "brand": "Test",
        "model": "Versatile",
        "forgivenessScore": 8.0,
        "workability": "mid",
        "msrp": 179,
        "configurations": [{"loft": 56, "bounce": 9, "grindCode": "V"}],
        "grinds": [{"grindCode": "V", "bestFor": ["all-conditions", "high-handicap-versatility"]}],
    }
    plain_grind = {
        **versatile_grind,
        "model": "Plain",
        "grinds": [{"grindCode": "V", "bestFor": ["all-conditions"]}],
    }
    digging_golfer = GolferInput(
        handicap=24,
        swing_speed=82,
        driver_carry=195,
        shot_shape="Slice",
        goal="Accuracy",
        iron_miss="Fat/Thin",
        iron_feel="Confidence-inspiring",
        wedge_turf="Normal",
    )

    versatile_score = score_wedge(versatile_grind, digging_golfer, predicted_bounce="Mid").score
    plain_score = score_wedge(plain_grind, digging_golfer, predicted_bounce="Mid").score

    assert versatile_score > plain_score


def test_score_wedge_uses_the_configuration_at_the_golfers_stated_loft():
    """A club offering different grinds at different lofts must be judged on
    the one at golfer.wedge_loft, not on whichever configuration elsewhere
    in its lineup happens to have the closest bounce degree."""
    club = {
        "brand": "Test",
        "model": "TwoLoft",
        "forgivenessScore": 6.0,
        "workability": "mid",
        "msrp": 179,
        "configurations": [
            {"loft": 52, "bounce": 6, "grindCode": "L"},
            {"loft": 58, "bounce": 12, "grindCode": "S"},
        ],
        "grinds": [
            {"grindCode": "L", "bestFor": ["firm-turf", "tight-lies"]},
            {"grindCode": "S", "bestFor": ["soft-turf", "sand-shots"]},
        ],
    }
    golfer = GolferInput(
        handicap=14,
        swing_speed=90,
        driver_carry=215,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        wedge_turf="Soft",
        wedge_loft=58,
    )

    result = score_wedge(club, golfer, predicted_bounce="High")

    assert any("12" in reason for reason in result.reasons)
    assert not any("6 deg bounce" in reason for reason in result.reasons)


def test_score_wedge_picks_the_best_overall_fit_among_same_loft_candidates():
    """Regression test: a genuinely purpose-built specialist grind must beat
    a generic 'works everywhere' grind at the *same* loft when the
    specialist is the real fit - a generic grind should not win just by
    scoring adequately on every axis while the specialist correctly scores
    low on axes it was never designed for."""
    specialist_club = {
        "brand": "Test",
        "model": "Specialist",
        "forgivenessScore": 4.0,
        "workability": "high",
        "family": "players",
        "msrp": 160,
        "configurations": [{"loft": 58, "bounce": 6, "grindCode": "X"}],
        "grinds": [
            {
                "grindCode": "X",
                "bestFor": ["firm-turf", "tight-lies"],
                "soleWidth": "narrow",
            }
        ],
    }
    generic_club = {
        "brand": "Test",
        "model": "Generic",
        "forgivenessScore": 4.0,
        "workability": "high",
        "family": "players",
        "msrp": 160,
        "configurations": [{"loft": 58, "bounce": 6, "grindCode": "S"}],
        "grinds": [
            {
                "grindCode": "S",
                "bestFor": ["all-conditions", "square-face-work"],
                "soleWidth": "medium",
            }
        ],
    }
    # Turf/divot/shot-style all point at the specialist's exact specialty.
    golfer = GolferInput(
        handicap=12,
        swing_speed=98,
        driver_carry=230,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        wedge_turf="Firm",
        divot_depth="Shallow",
        wedge_loft=58,
    )

    specialist_result = score_wedge(specialist_club, golfer, predicted_bounce="Low")
    generic_result = score_wedge(generic_club, golfer, predicted_bounce="Low")

    assert specialist_result.score > generic_result.score


def test_recommend_wedges_returns_ranked_recommendations_end_to_end():
    catalog = load_equipment_catalog(EQUIPMENT_DIR)
    golfer = GolferInput(
        handicap=14,
        swing_speed=90,
        driver_carry=215,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
        wedge_turf="Soft",
    )

    recommendations = recommend_wedges(
        catalog=catalog,
        golfer=golfer,
        predicted_bounce="High",
        top_n=5,
    )

    assert len(recommendations) == 5
    assert all(0 <= rec.score <= 100 for rec in recommendations)
    assert recommendations == sorted(recommendations, key=lambda rec: rec.score, reverse=True)
    assert all(rec.category == "Wedge" for rec in recommendations)
