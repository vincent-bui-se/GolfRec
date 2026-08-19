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


def test_index_page_renders_current_club_dropdowns(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"current_driver_id" in response.data
    assert b"current_iron_set_id" in response.data
    assert b"Not sure / skip" in response.data


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
