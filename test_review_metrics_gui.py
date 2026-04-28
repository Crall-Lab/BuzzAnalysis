from pathlib import Path

import numpy as np
import pandas as pd

from review_metrics_gui import (
    ReviewSession,
    assess_nest_map_status,
    classify_activity_value,
    compute_activity_tables,
    compute_interaction_count_table,
    compute_nearest_neighbor_distance_table,
    compute_social_center_distance_table,
    contiguous_true_ranges,
    convert_speed_cutoff_units,
    convert_speed_table_units,
    discover_sessions,
    find_tracking_csv_by_pattern,
    find_related_csvs,
    find_tracking_csvs,
    tracking_csv_pattern,
    tracking_csv_pattern_label,
)


def write_tracking(path: Path):
    pd.DataFrame(
        {
            "frame": [0, 1, 10],
            "ID": [1, 1, 1],
            "centroidX": [0.0, 4.0, 20.0],
            "centroidY": [0.0, 0.0, 0.0],
        }
    ).to_csv(path, index=False)


def test_find_tracking_csvs_matches_nested_clean_file(tmp_path):
    video = tmp_path / "worker23_2022-06-20_18-25-30.mjpeg"
    video.write_bytes(b"")
    nested = tmp_path / video.stem
    nested.mkdir()
    csv_path = nested / f"{video.stem}_Whole_clean.csv"
    write_tracking(csv_path)

    assert find_tracking_csvs(video, tmp_path) == (csv_path,)


def test_find_tracking_csvs_matches_video_stem_variants(tmp_path):
    video = tmp_path / "bumblebox-17_2024-08-11_00_30_02.mp4"
    video.write_bytes(b"")
    cleaned = tmp_path / "bumblebox-17_2024-08-11_00_30_02_newtracks_cleaned.csv"
    raw = tmp_path / "bumblebox-17_2024-08-11_00_30_02_raw.csv"
    no_id = tmp_path / "bumblebox-17_2024-08-11_00_30_02_noID.csv"
    unrelated = tmp_path / "bumblebox-17_2024-08-11_00_30_03_newtracks_cleaned.csv"
    write_tracking(cleaned)
    write_tracking(raw)
    pd.DataFrame(
        {
            "frame": [0],
            "centroidX": [0.0],
            "centroidY": [0.0],
        }
    ).to_csv(no_id, index=False)
    write_tracking(unrelated)

    assert find_related_csvs(video, tmp_path) == (cleaned, raw, no_id)
    assert find_tracking_csvs(video, tmp_path) == (cleaned, raw)


def test_discover_sessions_for_single_video(tmp_path):
    video = tmp_path / "example.mp4"
    video.write_bytes(b"")
    csv_path = tmp_path / "example.csv"
    write_tracking(csv_path)

    sessions = discover_sessions(video)

    assert len(sessions) == 1
    assert sessions[0].video_path == video.resolve()
    assert sessions[0].tracking_paths == (csv_path,)


def test_tracking_csv_pattern_reuses_same_suffix_for_later_videos(tmp_path):
    first_video = tmp_path / "video_a.mp4"
    first_csv = tmp_path / "video_a_Whole_clean.csv"
    next_video = tmp_path / "video_b.mp4"
    next_csv = tmp_path / "video_b_Whole_clean.csv"
    first_video.write_bytes(b"")
    next_video.write_bytes(b"")
    write_tracking(first_csv)
    write_tracking(next_csv)

    pattern = tracking_csv_pattern(first_video, first_csv)
    session = ReviewSession(next_video, (next_csv,), tmp_path)

    assert pattern == "_Whole_clean.csv"
    assert tracking_csv_pattern_label(pattern) == "<video name>_Whole_clean.csv"
    assert find_tracking_csv_by_pattern(session, pattern) == next_csv


def test_tracking_csv_pattern_handles_exact_stem_csv(tmp_path):
    video = tmp_path / "example.mp4"
    csv_path = tmp_path / "example.csv"
    video.write_bytes(b"")
    write_tracking(csv_path)

    pattern = tracking_csv_pattern(video, csv_path)

    assert pattern == ".csv"
    assert tracking_csv_pattern_label(pattern) == "<video name>.csv"


def test_compute_activity_tables_respects_gui_cutoff_and_gap():
    tracking = pd.DataFrame(
        {
            "frame": [0, 1, 20],
            "ID": [1, 1, 1],
            "centroidX": [0.0, 4.0, 104.0],
            "centroidY": [0.0, 0.0, 0.0],
        }
    )

    act, speed = compute_activity_tables(
        tracking,
        frame_rate=5,
        max_gap_seconds=3,
        speed_cutoff=3.5,
    )

    assert np.isclose(speed.loc[1, 1], 4.0)
    assert classify_activity_value(act.loc[1, 1]) == "active"
    assert classify_activity_value(act.loc[20, 1]) == "unknown"


def test_metric_tables_compute_distances_and_interactions():
    tracking = pd.DataFrame(
        {
            "frame": [0, 0, 0, 1],
            "ID": [1, 2, 3, 1],
            "centroidX": [0.0, 3.0, 30.0, 0.0],
            "centroidY": [0.0, 4.0, 40.0, 10.0],
        }
    )

    nearest = compute_nearest_neighbor_distance_table(tracking)
    interactions = compute_interaction_count_table(tracking, cutoff=6)
    social = compute_social_center_distance_table(tracking)

    assert np.isclose(nearest.loc[0, 1], 5.0)
    assert np.isclose(nearest.loc[0, 2], 5.0)
    assert interactions.loc[0, 1] == 1
    assert interactions.loc[0, 3] == 0
    assert interactions.loc[1, 1] == 0
    assert set(social.columns) == {1, 2, 3}


def test_convert_speed_units_and_cutoff():
    speed = pd.DataFrame({1: [1.0, 2.0]}, index=[0, 1])

    px_sec = convert_speed_table_units(speed, "px/sec", frame_rate=5, px_per_cm=10)
    cm_sec = convert_speed_table_units(speed, "cm/sec", frame_rate=5, px_per_cm=10)

    assert px_sec.loc[1, 1] == 10.0
    assert cm_sec.loc[1, 1] == 1.0
    assert convert_speed_cutoff_units(2, "px/sec", frame_rate=5, px_per_cm=10) == 10
    assert convert_speed_cutoff_units(2, "cm/sec", frame_rate=5, px_per_cm=10) == 1


def test_contiguous_true_ranges_splits_false_values_and_frame_gaps():
    mask = pd.Series(
        [True, True, False, True, True, True],
        index=[1, 2, 3, 7, 8, 10],
    )

    assert contiguous_true_ranges(mask) == [(1, 2), (7, 8), (10, 10)]


def test_assess_nest_map_status_finds_matching_brood_csv(tmp_path):
    video = tmp_path / "col_15-2021-06-12.mjpeg"
    video.write_bytes(b"")
    tracking = tmp_path / "col_15-2021-06-12_Whole_clean.csv"
    write_tracking(tracking)
    brood = tmp_path / "maps" / "col_15-2021-06-12-nest_image.csv"
    brood.parent.mkdir()
    brood.write_text("object index,label,label ID,vertex ID,shape,x,y,radius\n", encoding="utf-8")

    status = assess_nest_map_status(ReviewSession(video, (tracking,), tmp_path))

    assert status.state == "ready"
    assert status.csv_path == brood.resolve()
    assert status.target == (15, "2021-06-12")


def test_assess_nest_map_status_detects_json_without_csv(tmp_path):
    video = tmp_path / "example.mjpeg"
    video.write_bytes(b"")
    tracking = tmp_path / "col_01-2021-06-13_Whole_clean.csv"
    write_tracking(tracking)
    image = tmp_path / "col_1-2021-06-13-nest_image.png"
    image.write_bytes(b"")
    image.with_suffix(".json").write_text('{"shapes": []}', encoding="utf-8")

    status = assess_nest_map_status(ReviewSession(video, (tracking,), tmp_path))

    assert status.state == "json-only"
    assert status.image_path == image.resolve()
    assert status.json_path == image.with_suffix(".json").resolve()
