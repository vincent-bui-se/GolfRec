"""Tests for shot-shape, spin, offset, and wedge-loft fitting rules."""

from recommend import GolferInput, score_driver, score_iron_set, score_wedge


def _golfer(**overrides) -> GolferInput:
    base = dict(
        handicap=12,
        swing_speed=92,
        driver_carry=215,
        shot_shape="Straight",
        goal="Accuracy",
        iron_miss="Consistent",
        iron_feel="No preference",
    )
    base.update(overrides)
    return GolferInput(**base)


def _driver(**overrides) -> dict:
    club = {
        "brand": "Test",
        "model": "Neutral",
        "lofts": [9, 10.5, 12],
        "adjustRangeDeg": 4,
        "forgivenessScore": 9.0,
        "launchChar": "mid",
        "spinChar": "mid",
        "speedMinMph": 75,
        "speedMaxMph": 110,
        "msrp": 599,
        "family": "versatile",
        "drawBiasBuiltIn": False,
        "drawBiasAvailable": True,
    }
    club.update(overrides)
    return club


def test_a_driver_that_fits_every_axis_reaches_100_for_a_straight_hitter():
    scored = score_driver(_driver(forgivenessScore=10.0), _golfer(), predicted_loft="10.5")
    assert scored.score == 100


def test_built_in_draw_bias_is_penalized_for_a_hook_and_rewarded_for_a_slice():
    neutral = _driver()
    draw_biased = _driver(model="Draw", drawBiasBuiltIn=True, family="draw-bias")

    hooker = _golfer(shot_shape="Hook")
    slicer = _golfer(shot_shape="Slice")

    assert score_driver(draw_biased, hooker, "10.5").score < score_driver(neutral, hooker, "10.5").score
    assert score_driver(draw_biased, slicer, "10.5").score > score_driver(neutral, slicer, "10.5").score
    hook_reasons = score_driver(draw_biased, hooker, "10.5").reasons
    assert any("make a hook worse" in reason for reason in hook_reasons)


def test_low_spin_head_costs_a_slow_swinger_but_helps_a_fast_distance_seeker():
    low_spin = _driver(model="LS", spinChar="low", launchChar="low-mid", family="low-spin")
    mid_spin = _driver()

    slow = _golfer(swing_speed=78, goal="Distance")
    fast = _golfer(swing_speed=112, goal="Distance")

    assert score_driver(low_spin, slow, "12").score < score_driver(mid_spin, slow, "12").score
    assert score_driver(low_spin, fast, "9").score > score_driver(mid_spin, fast, "9").score
    slow_reasons = score_driver(low_spin, slow, "12").reasons
    assert any("fall out of the air" in reason for reason in slow_reasons)


def test_speed_at_the_edge_of_a_heads_window_scores_below_the_middle():
    golfer = _golfer(swing_speed=92)
    centred = _driver(speedMinMph=75, speedMaxMph=110)
    at_edge = _driver(speedMinMph=91, speedMaxMph=125)
    assert score_driver(at_edge, golfer, "10.5").score < score_driver(centred, golfer, "10.5").score


def _iron_set(category: str, offset_mm: float | None) -> dict:
    positions = [{"position": "7", "loft": 30}]
    if offset_mm is not None:
        positions[0]["offsetMm"] = offset_mm
    return {
        "brand": "Test",
        "model": category,
        "ironCategory": category,
        "construction": "cast-stainless",
        "forgivenessScore": 7.0,
        "launchChar": "mid",
        "spinChar": "mid",
        "workability": "mid",
        "msrp": 999,
        "ironPositions": positions,
    }


def test_iron_offset_helps_a_slice_and_hurts_a_hook():
    low_offset = _iron_set("players-distance", 2.0)
    high_offset = _iron_set("players-distance", 4.5)

    slicer = _golfer(iron_shot_shape="Slice")
    hooker = _golfer(iron_shot_shape="Hook")

    assert score_iron_set(high_offset, slicer, "players-distance").score > score_iron_set(
        low_offset, slicer, "players-distance"
    ).score
    assert score_iron_set(low_offset, hooker, "players-distance").score > score_iron_set(
        high_offset, hooker, "players-distance"
    ).score
    assert any("4.5 mm" in reason for reason in score_iron_set(high_offset, hooker, "players-distance").reasons)


def test_iron_offset_falls_back_to_the_category_median_without_naming_a_number():
    unmeasured = _iron_set("super-game-improvement", None)
    reasons = score_iron_set(unmeasured, _golfer(iron_shot_shape="Hook"), "super-game-improvement").reasons
    offset_reason = next(reason for reason in reasons if "offset" in reason.lower())
    assert "mm" not in offset_reason


def _wedge(lofts: list[int], **overrides) -> dict:
    club = {
        "brand": "Test",
        "model": "Wedge",
        "forgivenessScore": 6.0,
        "workability": "high",
        "msrp": 179,
        "configurations": [{"loft": loft, "bounce": 10, "grindCode": "S"} for loft in lofts],
        "grinds": [{"grindCode": "S", "soleWidth": "medium", "bestFor": ["all-conditions"]}],
    }
    club.update(overrides)
    return club


def test_wedge_not_sold_in_the_requested_loft_is_penalized_and_says_so():
    golfer = _golfer(wedge_loft=62)
    has_62 = score_wedge(_wedge([58, 60, 62]), golfer, "Mid")
    tops_out_at_60 = score_wedge(_wedge([56, 58, 60]), golfer, "Mid")

    assert tops_out_at_60.score < has_62.score
    assert any("closest loft is 60°" in reason for reason in tops_out_at_60.reasons)


def test_wedge_forgiveness_follows_the_iron_goal_not_the_driver_goal():
    forgiving = _wedge([56], forgivenessScore=9.5)
    unforgiving = _wedge([56], model="Blade", forgivenessScore=3.0)

    # Driver goal is Distance, but the golfer wants forgiveness from irons.
    golfer = _golfer(goal="Distance", iron_goal="Forgiveness")
    gap_with_iron_goal = (
        score_wedge(forgiving, golfer, "Mid").score - score_wedge(unforgiving, golfer, "Mid").score
    )
    gap_without = (
        score_wedge(forgiving, _golfer(goal="Distance"), "Mid").score
        - score_wedge(unforgiving, _golfer(goal="Distance"), "Mid").score
    )
    assert gap_with_iron_goal > gap_without


def test_consistent_strikers_still_see_wedge_forgiveness_differences():
    golfer = _golfer()
    assert score_wedge(_wedge([56], forgivenessScore=7.5), golfer, "Mid").score > score_wedge(
        _wedge([56], forgivenessScore=4.0), golfer, "Mid"
    ).score
