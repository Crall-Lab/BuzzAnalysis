from pathlib import Path

import numpy as np
import pandas as pd

from review_metrics_gui import (
    ReviewSession,
    assess_nest_map_status,
    classify_activity_value,
    compute_activity_tables,
    discover_sessions,
    find_related_csvs,
    find_tracking_csvs,
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
