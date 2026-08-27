import pytest

import app as app_module


@pytest.fixture()
def client():
    app_module.app.testing = True
    return app_module.app.test_client()


def test_current_club_options_sorts_by_label_and_skips_recordless_ids():
    records = [
        {"id": "b-driver", "brand": "Zeta", "model": "One", "year": 2024},
        {"id": "a-driver", "brand": "Alpha", "model": "Two", "year": 2023},
        {"brand": "Missing Id", "model": "Three", "year": 2022},
    ]

    options = app_module._current_club_options(records)

    assert options == [
        {"id": "a-driver", "label": "Alpha Two (2023)"},
        {"id": "b-driver", "label": "Zeta One (2024)"},
    ]


def test_resolve_current_club_finds_by_id():
    catalog = {"drivers": [{"id": "known-driver", "launchChar": "low", "spinChar": "mid"}]}

    record = app_module._resolve_current_club("drivers", "known-driver", catalog)

    assert record == {"id": "known-driver", "launchChar": "low", "spinChar": "mid"}


def test_resolve_current_club_returns_none_for_empty_id():
    catalog = {"drivers": [{"id": "known-driver"}]}

    assert app_module._resolve_current_club("drivers", "", catalog) is None


def test_resolve_current_club_returns_none_for_unknown_id():
    catalog = {"drivers": [{"id": "known-driver"}]}

    assert app_module._resolve_current_club("drivers", "not-in-catalog", catalog) is None


def test_index_page_renders_current_club_typeahead_fields(client):
    """"Current driver"/"current irons" are typeahead text inputs (a visible
    label field plus a hidden id field resolved client-side from a
    <datalist>), not plain <select> dropdowns - see static/app.js."""
    catalog = app_module.load_catalog()
    a_real_driver = catalog["drivers"][0]
    expected_label = f"{a_real_driver['brand']} {a_real_driver['model']} ({a_real_driver['year']})"

    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'name="current_driver_label"' in html
    assert 'name="current_iron_set_label"' in html
    assert 'type="hidden" name="current_driver_id"' in html
    assert 'type="hidden" name="current_iron_set_id"' in html
    assert expected_label in html


def test_index_page_renders_swing_tempo_field_and_disclaimer(client):
    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'name="swing_tempo"' in html
    for tempo in app_module.SWING_TEMPOS:
        assert f'value="{tempo}"' in html
    assert "a starting point" in html


def test_index_page_separates_speed_fields_from_driver_specific_fields(client):
    """Swing speed/tempo feed shaft_flex and iron_category both, so they need
    their own section (shown for Driver, Fairway Wood, or Irons) separate
    from the driver/fairway-wood-only fields (current driver, shot shape,
    goal) that stay in .wood-fields - see updateConditionalFields() in
    app.js."""
    response = client.get("/")
    html = response.get_data(as_text=True)

    speed_section = html.index('class="form-section speed-fields"')
    tempo_field = html.index('name="swing_tempo"')
    wood_section = html.index('class="form-section wood-fields"')
    current_driver_field = html.index('name="current_driver_label"')

    assert speed_section < tempo_field < wood_section < current_driver_field


def test_recommend_endpoint_accepts_current_club_selections(client):
    catalog = app_module.load_catalog()
    real_driver_id = catalog["drivers"][0]["id"]
    real_iron_set_id = catalog["iron-sets"][0]["id"]

    response = client.post(
        "/api/recommend",
        json={
            "shopping_for": ["Driver", "Irons"],
            "handicap": 14,
            "speed_mode": "Swing speed",
            "swing_speed": 95,
            "current_driver_id": real_driver_id,
            "current_iron_set_id": real_iron_set_id,
            "driver_trajectory": "Too high",
            "iron_trajectory": "Too high",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["wants_driver"] is True
    assert payload["wants_irons"] is True
    assert len(payload["recommendations"]["drivers"]) > 0
    assert len(payload["recommendations"]["irons"]) > 0


def test_recommend_endpoint_ignores_unknown_current_club_id(client):
    """An id that doesn't match anything (stale option, tampered request)
    should fall back to the no-baseline behavior rather than error."""
    response = client.post(
        "/api/recommend",
        json={
            "shopping_for": ["Driver"],
            "handicap": 14,
            "speed_mode": "Swing speed",
            "swing_speed": 95,
            "current_driver_id": "not-a-real-id",
        },
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("flex", "tempo", "expected"),
    [
        ("R", "Aggressive", "S"),
        ("R", "Smooth", "A"),
        ("R", "Moderate", "R"),
        ("R", "Unknown value", "R"),  # invalid tempo behaves like Moderate
        ("X", "Aggressive", "X"),  # already stiffest: clamps, doesn't overflow
        ("L", "Smooth", "L"),  # already softest: clamps, doesn't underflow
        ("Z", "Aggressive", "Z"),  # unrecognized flex letter passes through
    ],
)
def test_adjust_flex_for_tempo(flex, tempo, expected):
    assert app_module._adjust_flex_for_tempo(flex, tempo) == expected


def test_recommend_endpoint_aggressive_tempo_stiffens_flex_by_one_step(client):
    payload = {
        "shopping_for": ["Driver"],
        "handicap": 14,
        "speed_mode": "Swing speed",
        "swing_speed": 95,
    }

    moderate = client.post("/api/recommend", json={**payload, "swing_tempo": "Moderate"}).get_json()
    aggressive = client.post("/api/recommend", json={**payload, "swing_tempo": "Aggressive"}).get_json()

    assert app_module._adjust_flex_for_tempo(moderate["specs"]["shaft_flex"], "Aggressive") == (
        aggressive["specs"]["shaft_flex"]
    )


def test_recommend_endpoint_defaults_swing_tempo_to_moderate(client):
    """No swing_tempo in the request should score identically to an explicit "Moderate"."""
    payload = {
        "shopping_for": ["Driver"],
        "handicap": 14,
        "speed_mode": "Swing speed",
        "swing_speed": 95,
    }

    omitted = client.post("/api/recommend", json=payload).get_json()
    explicit = client.post("/api/recommend", json={**payload, "swing_tempo": "Moderate"}).get_json()

    assert omitted["specs"]["shaft_flex"] == explicit["specs"]["shaft_flex"]


_LOFTS = ["8", "9", "10.5", "12"]


@pytest.mark.parametrize(
    ("loft", "trajectory", "expected"),
    [
        ("9", "Too low", "10.5"),  # flying low earns more loft
        ("10.5", "Too high", "9"),  # flying high earns less
        ("9", "About right", "9"),
        ("9", "Unknown value", "9"),  # invalid answer behaves like About right
        ("12", "Too low", "12"),  # already the most loft: clamps, doesn't overflow
        ("8", "Too high", "8"),  # already the least: clamps, doesn't underflow
        ("7.5", "Too low", "7.5"),  # off-ladder loft passes through
    ],
)
def test_adjust_loft_for_trajectory(loft, trajectory, expected):
    assert app_module._adjust_loft_for_trajectory(loft, trajectory, _LOFTS) == expected


def test_loft_ladder_is_sorted_numerically_not_alphabetically():
    """The rungs come off the trained model's own label set, and sklearn orders
    classes_ as strings - which puts "10.5" between "1" and "8" rather than
    where a golfer would put it. Sorting by float is what makes index +/- 1 mean
    one step of loft."""

    class FakeModel:
        classes_ = ["10.5", "12", "8", "9"]

    assert app_module._loft_ladder(FakeModel()) == ["8", "9", "10.5", "12"]


def test_loft_ladder_is_empty_for_a_non_numeric_label_set():
    """Adjusting a non-numeric label set by position would be guesswork, so the
    ladder comes back empty and _adjust_loft_for_trajectory passes values
    through untouched."""

    class FakeModel:
        classes_ = ["Blade", "Cavity back"]

    assert app_module._loft_ladder(FakeModel()) == []


def test_ball_flight_changes_the_recommended_driver_loft(client):
    """The regression this was filed as: "when I say my driver goes too low, it
    still recommends the same loft".

    Ball flight is not one of the model's nine INPUT_COLUMNS, so the raw
    prediction is identical for all three answers - the number a golfer is
    watching never moved no matter what they told it. Asserts the three answers
    do not all agree, rather than naming lofts, so the label set can change
    without making this a test of the label set."""
    payload = {
        "shopping_for": ["Driver"],
        "handicap": 14,
        "speed_mode": "Swing speed",
        "swing_speed": 95,
    }

    lofts = {
        answer: client.post(
            "/api/recommend", json={**payload, "driver_trajectory": answer}
        ).get_json()["specs"]["driver_loft"]
        for answer in ("Too low", "About right", "Too high")
    }

    assert len(set(lofts.values())) > 1, f"ball flight moved nothing: {lofts}"
    assert float(lofts["Too low"]) > float(lofts["About right"]) >= float(lofts["Too high"])


@pytest.mark.parametrize("spec", ["driver_loft", "fairway_wood_loft"])
def test_ball_flight_adjusts_both_woods_by_one_rung(client, spec):
    """The fairway wood rides the driver's Ball flight answer: the question sits
    under "Driver & Fairway Wood" in the form, and recommend.py's _score_wood
    already reads golfer.driver_trajectory for both categories.

    Compares against the helper rather than a literal, the way the swing-tempo
    test above does, so the assertion stays true if the ladders change."""
    payload = {
        "shopping_for": ["Driver", "Fairway Wood"],
        "handicap": 14,
        "speed_mode": "Swing speed",
        "swing_speed": 95,
    }
    ladder = app_module._loft_ladder(app_module.load_models()[spec])

    about = client.post(
        "/api/recommend", json={**payload, "driver_trajectory": "About right"}
    ).get_json()
    too_low = client.post(
        "/api/recommend", json={**payload, "driver_trajectory": "Too low"}
    ).get_json()

    assert app_module._adjust_loft_for_trajectory(
        about["specs"][spec], "Too low", ladder
    ) == too_low["specs"][spec]


def test_ball_flight_defaults_to_about_right(client):
    """No driver_trajectory in the request should behave as "About right"."""
    payload = {
        "shopping_for": ["Driver"],
        "handicap": 14,
        "speed_mode": "Swing speed",
        "swing_speed": 95,
    }

    omitted = client.post("/api/recommend", json=payload).get_json()
    explicit = client.post(
        "/api/recommend", json={**payload, "driver_trajectory": "About right"}
    ).get_json()

    assert omitted["specs"]["driver_loft"] == explicit["specs"]["driver_loft"]


def test_a_golfer_already_on_the_top_rung_still_gets_different_clubs(client):
    """A slow swinger is predicted at the most lofted driver made, so "too low"
    has no rung left to climb - and that golfer is exactly the one most likely
    to give that answer. The loft is pinned by the ladder, deliberately: a
    fitter facing "already the most loft and still too low" changes the head,
    not the number.

    What has to keep working is that the answer still does something visible,
    which it does through recommend.py's launch/spin preferences."""
    payload = {
        "shopping_for": ["Driver"],
        "handicap": 16,
        "speed_mode": "Swing speed",
        "swing_speed": 75,
    }

    about = client.post(
        "/api/recommend", json={**payload, "driver_trajectory": "About right"}
    ).get_json()
    too_low = client.post(
        "/api/recommend", json={**payload, "driver_trajectory": "Too low"}
    ).get_json()

    ladder = app_module._loft_ladder(app_module.load_models()["driver_loft"])
    assert about["specs"]["driver_loft"] == ladder[-1], "profile no longer pins the top rung"
    assert too_low["specs"]["driver_loft"] == about["specs"]["driver_loft"]

    names = [
        [club["name"] for club in result["recommendations"]["drivers"][:3]]
        for result in (about, too_low)
    ]
    assert names[0] != names[1], "ball flight changed neither the loft nor the clubs"
