from synthetic_data import generate_golfer_profiles


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
