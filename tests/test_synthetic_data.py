from synthetic_data import (
    _fairway_wood_loft,
    _iron_category,
    _wedge_bounce,
    generate_golfer_profiles,
)


def test_generate_golfer_profiles_creates_course_dataset_shape():
    golfers = generate_golfer_profiles(n=1200, seed=7)

    assert len(golfers) == 1200
    assert {
        "handicap",
        "swing_speed",
        "driver_carry",
        "shot_shape",
        "goal",
        "iron_miss",
        "iron_feel",
        "wedge_turf",
        "divot_depth",
        "driver_loft",
        "shaft_flex",
        "fairway_wood_loft",
        "iron_category",
        "wedge_bounce",
    }.issubset(golfers.columns)
    assert golfers["swing_speed"].between(60, 120).all()
    assert set(golfers["wedge_turf"].unique()).issubset({"Firm", "Normal", "Soft"})
    assert set(golfers["divot_depth"].unique()).issubset({"Shallow", "Medium", "Deep"})


def test_generate_golfer_profiles_has_non_trivial_label_variety():
    golfers = generate_golfer_profiles(n=1500, seed=11)

    assert golfers["driver_loft"].nunique() >= 4
    assert golfers["shaft_flex"].nunique() >= 4
    assert golfers["fairway_wood_loft"].nunique() >= 3
    assert golfers["iron_category"].nunique() >= 3
    assert golfers["wedge_bounce"].nunique() >= 2


def test_fairway_wood_loft_uses_speed_and_handicap():
    low_loft = _fairway_wood_loft(speed=110, handicap=6, goal="Distance")
    high_loft = _fairway_wood_loft(speed=74, handicap=28, goal="Forgiveness")

    assert float(low_loft) < float(high_loft)


def test_wedge_bounce_reflects_turf_and_divot_depth():
    assert _wedge_bounce(turf="Firm", divot_depth="Medium") == "Low"
    assert _wedge_bounce(turf="Normal", divot_depth="Medium") == "Mid"
    assert _wedge_bounce(turf="Soft", divot_depth="Medium") == "High"
    # A deep divot (steep attack angle) pushes the need up even on firm turf,
    # since a thin-soled wedge punishes a digging strike regardless of surface.
    assert _wedge_bounce(turf="Firm", divot_depth="Deep") == "Mid"
    # A shallow, sweeping strike moderates the need down even on soft turf.
    assert _wedge_bounce(turf="Soft", divot_depth="Shallow") == "Mid"
    # The two signals combine rather than either alone deciding the tier.
    assert _wedge_bounce(turf="Normal", divot_depth="Deep") == "High"
    assert _wedge_bounce(turf="Normal", divot_depth="Shallow") == "Low"


def test_iron_category_uses_more_than_handicap():
    player_style = _iron_category(
        handicap=14,
        swing_speed=104,
        driver_carry=252,
        shot_shape="Draw",
        goal="Distance",
        iron_miss="Consistent",
        iron_feel="Forged/Blade-like",
    )
    forgiveness_style = _iron_category(
        handicap=14,
        swing_speed=78,
        driver_carry=178,
        shot_shape="Slice",
        goal="Forgiveness",
        iron_miss="Fat/Thin",
        iron_feel="Confidence-inspiring",
    )

    assert player_style in {"blade", "players-cb", "players-distance"}
    assert forgiveness_style in {"game-improvement", "super-game-improvement"}
    assert player_style != forgiveness_style
