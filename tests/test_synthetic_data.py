from synthetic_data import _iron_category, generate_golfer_profiles


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
        "driver_loft",
        "shaft_flex",
        "iron_category",
    }.issubset(golfers.columns)
    assert golfers["swing_speed"].between(60, 120).all()


def test_generate_golfer_profiles_has_non_trivial_label_variety():
    golfers = generate_golfer_profiles(n=1500, seed=11)

    assert golfers["driver_loft"].nunique() >= 4
    assert golfers["shaft_flex"].nunique() >= 4
    assert golfers["iron_category"].nunique() >= 3


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
