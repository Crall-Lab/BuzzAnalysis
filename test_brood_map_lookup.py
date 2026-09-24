import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).with_name("analyze_clean_data_0.2.py")
SPEC = importlib.util.spec_from_file_location("analyze_clean_data_0_2", MODULE_PATH)
analyze_clean_data_0_2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze_clean_data_0_2)


def test_find_brood_map_matches_colony_and_date_with_zero_padding(tmp_path):
    brood_dir = tmp_path / "brood"
    brood_dir.mkdir()

    target = brood_dir / "col_01-2021-06-12-nest_image.csv"
    target.write_text("x,y\n", encoding="utf-8")
    (brood_dir / "col_01-2021-06-13-nest_image.csv").write_text("x,y\n", encoding="utf-8")
    (brood_dir / "col_02-2021-06-12-nest_image.csv").write_text("x,y\n", encoding="utf-8")

    result = analyze_clean_data_0_2._find_brood_map("col_1-2021-06-12", str(brood_dir), "_nest_image.csv")

    assert result == str(target)


def test_find_brood_map_finds_nested_match(tmp_path):
    brood_dir = tmp_path / "brood"
    nested_dir = brood_dir / "2021-06-12"
    nested_dir.mkdir(parents=True)

    target = nested_dir / "col_15-2021-06-12-nest_image.csv"
    target.write_text("x,y\n", encoding="utf-8")

    result = analyze_clean_data_0_2._find_brood_map("col_15-2021-06-12", str(brood_dir), "_nest_image.csv")

    assert result == str(target)


def test_find_brood_map_raises_for_ambiguous_matches(tmp_path):
    brood_dir = tmp_path / "brood"
    brood_dir.mkdir()

    (brood_dir / "col_1-2021-06-12-nest_image.csv").write_text("x,y\n", encoding="utf-8")
    (brood_dir / "col_01-2021-06-12-nest_image.csv").write_text("x,y\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Multiple brood maps matched"):
        analyze_clean_data_0_2._find_brood_map("col_1-2021-06-12", str(brood_dir), "_nest_image.csv")


def test_analysis_brood_objects_keeps_location_perimeters_and_excludes_calibration():
    full = pd.DataFrame({
        "label": [
            "Arena perimeter (polygon)",
            "Foraging perimeter (polygon)",
            "Nest perimeter (polygon)",
            "Calibration A->B (line)",
        ],
        "x": [1, 2, 3, 4],
    })

    brood = analyze_clean_data_0_2._analysis_brood_objects(full)

    assert brood["label"].tolist() == [
        "Arena perimeter (polygon)",
        "Foraging perimeter (polygon)",
        "Nest perimeter (polygon)",
    ]


def test_frame_level_table_exports_location_distances_flags_and_contacts():
    columns = pd.MultiIndex.from_tuples([
        ("centroidX", 1),
        ("centroidY", 1),
        ("centroidX", 2),
        ("centroidY", 2),
        ("distM_Arena perimeter (polygon)_0", 1),
        ("distM_Foraging perimeter (polygon)_1", 1),
        ("distM_Nest perimeter (polygon)_2", 1),
        ("distM_pollen balls (circles)_3", 1),
        ("distM_nectar source (circle)_4", 1),
        ("distM_Larvae (circles)_5", 1),
        ("distM_Wax pots (circles)_6", 1),
        ("distM_Eggs perimeter (polygons)_7", 1),
        ("distM_Pupae (circles)_8", 1),
        ("distM_Queen larva (circles)_9", 1),
        ("distM_Queen pupae (circles)_10", 1),
    ])
    pivot = pd.DataFrame(
        [
            [0, 0, 0, 4, 0, 0, 8, 2, 7, 4, 9, 12, 6, 11, 13],
            [3, 4, 30, 40, 0, 12, 0, 9, 3, 6, 2, 1, 8, 2, 12],
        ],
        index=[0, 1],
        columns=columns,
    )
    opt = {
        "behavior_fps": 1,
        "max_behavior_gap_sec": 5,
        "speed_cutoff": 1,
        "max_speed_cutoff": 0,
        "inactive_gap_max_frames": 0,
        "interaction_distance_cutoff": 5,
        "nest_on_distance_px": 5,
        "frame_level_include_distc": False,
    }

    frame_level = analyze_clean_data_0_2.frame_level_table(
        pivot,
        opt,
        source_file="col_01-2021-06-12_00-00-00_Whole_clean.csv",
        worker="01",
        date="2021-06-12",
        time="00-00-00",
        lr="Whole",
    )

    bee1 = frame_level[frame_level["bee_ID"] == 1].reset_index(drop=True)
    assert bee1.loc[0, "contact_count"] == 1
    assert bee1.loc[1, "contact_count"] == 0
    assert bee1.loc[1, "activity"] == 1
    assert np.isnan(bee1.loc[0, "activity"])
    assert bee1.loc[0, "distance_to_arena_perimeter_px"] == 0
    assert bool(bee1.loc[0, "inside_arena_perimeter"])
    assert bool(bee1.loc[0, "inside_foraging_perimeter"])
    assert not bool(bee1.loc[0, "inside_nest_perimeter"])
    assert bool(bee1.loc[0, "on_pollen"])
    assert bool(bee1.loc[0, "on_brood"])
    assert bee1.loc[0, "distance_to_larvae_px"] == 4
    assert bool(bee1.loc[0, "on_larvae"])
    assert bee1.loc[0, "distance_to_eggs_px"] == 12
    assert not bool(bee1.loc[0, "on_eggs"])
    assert bee1.loc[0, "nearest_small_object_label"] == "pollen balls (circles)"
    assert bee1.loc[0, "location_zone"] == "foraging"
    assert bool(bee1.loc[1, "inside_nest_perimeter"])
    assert bee1.loc[1, "location_zone"] == "nest"
    assert bool(bee1.loc[1, "on_eggs"])
    assert not bool(bee1.loc[1, "on_pupae"])
    assert bool(bee1.loc[1, "on_queen_larva"])
    assert not bool(bee1.loc[1, "on_queen_pupae"])
    assert bool(bee1.loc[1, "on_nectar"])
    assert bool(bee1.loc[1, "on_wax"])
    assert "distm_arena_perimeter_polygon_0" in frame_level.columns
    assert "distm_foraging_perimeter_polygon_1" in frame_level.columns
