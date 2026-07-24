from pathlib import Path

import pandas as pd

from preprocess import (
    FEATURE_COLUMNS,
    estimate_handicap_from_average_score,
    load_equipment_catalog,
    make_feature_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EQUIPMENT_DIR = PROJECT_ROOT / "data" / "equipment" / "data"


def test_load_equipment_catalog_reads_category_records():
    catalog = load_equipment_catalog(EQUIPMENT_DIR)

    assert "drivers" in catalog
    assert "iron-sets" in catalog
    assert len(catalog["drivers"]) >= 10
    assert {"brand", "model", "lofts", "forgivenessTier", "msrp"}.issubset(
        catalog["drivers"][0].keys()
    )


def test_driver_catalog_includes_callaway_rogue_st_family():
    catalog = load_equipment_catalog(EQUIPMENT_DIR)
    driver_ids = {driver["id"] for driver in catalog["drivers"]}

    assert {
        "callaway-rogue-st-max",
        "callaway-rogue-st-max-d",
        "callaway-rogue-st-max-ls",
        "callaway-rogue-st-triple-diamond-ls",
    }.issubset(driver_ids)


def test_make_feature_frame_encodes_expected_inputs():
    golfers = pd.DataFrame(
        [
            {
                "handicap": 18,
                "swing_speed": 91,
                "driver_carry": 225,
                "shot_shape": "Slice",
                "goal": "Forgiveness",
                "iron_miss": "Fat/Thin",
                "iron_feel": "Forged/Blade-like",
            }
        ]
    )

    frame = make_feature_frame(golfers)

    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame.loc[0, "shot_shape_Slice"] == 1
    assert frame.loc[0, "goal_Forgiveness"] == 1
    assert frame.loc[0, "swing_speed"] == 91


def test_estimate_handicap_from_average_score_uses_par_72_baseline():
    assert estimate_handicap_from_average_score(90) == 18
    assert estimate_handicap_from_average_score(68) == 0
    assert estimate_handicap_from_average_score(112) == 36
