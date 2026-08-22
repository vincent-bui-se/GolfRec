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
    assert "not a full fitting" in html


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
