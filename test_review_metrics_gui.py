from pathlib import Path

import numpy as np
import pandas as pd

from review_metrics_gui import (
    NestComponent,
    ReviewHub,
    ReviewSession,
    assess_nest_map_status,
    classify_activity_value,
    compute_activity_tables,
    compute_interaction_count_table,
    compute_nest_component_tables,
    compute_nearest_neighbor_distance_table,
    compute_raw_speed_table,
    compute_social_center_distance_table,
    distance_to_nest_component,
    ethogram_state_table,
    contiguous_true_ranges,
    contiguous_value_ranges,
    convert_speed_cutoff_units,
    convert_speed_table_units,
    discover_sessions,
    find_noid_tag_csvs,
    find_tracking_csv_by_pattern,
    find_related_csvs,
    find_tracking_csvs,
    is_over_max_speed,
    load_noid_tags,
    overlay_speed_value,
    social_center_from_tracking_paths,
    social_center_tracking_paths_for_sessions,
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


def test_find_noid_tag_csvs_uses_related_noid_file_without_id(tmp_path):
    video = tmp_path / "bumblebox-17_2024-08-11_00_30_02.mp4"
    video.write_bytes(b"")
    cleaned = tmp_path / "bumblebox-17_2024-08-11_00_30_02_newtracks_cleaned.csv"
    no_id = tmp_path / "bumblebox-17_2024-08-11_00_30_02_noID.csv"
    write_tracking(cleaned)
    pd.DataFrame(
        {
            "frame": [0, 1],
            "centroidX": [12.5, 14.0],
            "centroidY": [22.5, 24.0],
        }
    ).to_csv(no_id, index=False)
    session = discover_sessions(video)[0]

    assert session.tracking_paths == (cleaned,)
    assert find_noid_tag_csvs(session) == (no_id,)


def test_load_noid_tags_accepts_csv_without_id_column(tmp_path):
    no_id = tmp_path / "example_noID.csv"
    pd.DataFrame(
        {
            "frame": ["2", "bad", "3"],
            "centroidX": ["10.0", "11.0", None],
            "centroidY": ["20.0", "21.0", "22.0"],
        }
    ).to_csv(no_id, index=False)

    tags = load_noid_tags((no_id,))

    assert list(tags["frame"]) == [2]
    assert tags.loc[0, "centroidX"] == 10.0
    assert tags.loc[0, "centroidY"] == 20.0
    assert tags.loc[0, "csv_path"] == str(no_id)


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


def test_raw_speed_table_preserves_values_that_max_speed_filter_marks_unknown():
    tracking = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "ID": [1, 1, 1],
            "centroidX": [0.0, 10.0, 210.0],
            "centroidY": [0.0, 0.0, 0.0],
        }
    )

    act, filtered_speed = compute_activity_tables(
        tracking,
        frame_rate=5,
        max_gap_seconds=3,
        speed_cutoff=3.5,
        max_speed_cutoff=50,
    )
    raw_speed = compute_raw_speed_table(
        tracking,
        frame_rate=5,
        max_gap_seconds=3,
    )

    assert np.isclose(raw_speed.loc[1, 1], 10.0)
    assert np.isclose(raw_speed.loc[2, 1], 200.0)
    assert np.isnan(filtered_speed.loc[2, 1])
    assert classify_activity_value(act.loc[2, 1]) == "unknown"
    assert is_over_max_speed(raw_speed.loc[2, 1], 50)
    assert not is_over_max_speed(raw_speed.loc[1, 1], 50)
    assert overlay_speed_value(filtered_speed.loc[2, 1], raw_speed.loc[2, 1], True) == raw_speed.loc[2, 1]


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


def test_social_center_distance_uses_supplied_whole_day_center():
    tracking = pd.DataFrame(
        {
            "frame": [0],
            "ID": [1],
            "centroidX": [13.0],
            "centroidY": [24.0],
        }
    )

    social = compute_social_center_distance_table(tracking, social_center=(10.0, 20.0))

    assert np.isclose(social.loc[0, 1], 5.0)


def test_social_center_from_tracking_paths_averages_across_files(tmp_path):
    first = tmp_path / "col_1-2024-01-01_00_00_00_Whole_clean.csv"
    second = tmp_path / "col_1-2024-01-01_01_00_00_Whole_clean.csv"
    pd.DataFrame({"centroidX": [0.0, 10.0], "centroidY": [10.0, 20.0]}).to_csv(first, index=False)
    pd.DataFrame({"centroidX": [20.0], "centroidY": [30.0]}).to_csv(second, index=False)

    center = social_center_from_tracking_paths((first, second))

    assert center == (10.0, 20.0)


def test_social_center_tracking_paths_match_day_and_csv_pattern(tmp_path):
    video_a = tmp_path / "col_1-2024-01-01_00_00_00.mp4"
    video_b = tmp_path / "col_1-2024-01-01_01_00_00.mp4"
    video_other_day = tmp_path / "col_1-2024-01-02_00_00_00.mp4"
    for path in (video_a, video_b, video_other_day):
        path.write_bytes(b"")
    csv_a = tmp_path / "col_1-2024-01-01_00_00_00_Whole_clean.csv"
    csv_b = tmp_path / "col_1-2024-01-01_01_00_00_Whole_clean.csv"
    csv_b_left = tmp_path / "col_1-2024-01-01_01_00_00_Left_clean.csv"
    csv_other_day = tmp_path / "col_1-2024-01-02_00_00_00_Whole_clean.csv"
    for path in (csv_a, csv_b, csv_b_left, csv_other_day):
        write_tracking(path)
    sessions = (
        ReviewSession(video_a, (csv_a,), tmp_path),
        ReviewSession(video_b, (csv_b, csv_b_left), tmp_path),
        ReviewSession(video_other_day, (csv_other_day,), tmp_path),
    )

    paths = social_center_tracking_paths_for_sessions(
        sessions,
        target=(1, "2024-01-01"),
        pattern="_Whole_clean.csv",
    )

    assert paths == (csv_a.resolve(), csv_b.resolve())


def test_nest_component_tables_use_shape_distances_and_ignore_broad_perimeters():
    tracking = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "ID": [1, 2, 1],
            "centroidX": [10.0, 15.0, 30.0],
            "centroidY": [10.0, 10.0, 30.0],
        }
    )
    brood_map = pd.DataFrame(
        {
            "object index": [0, 1, 2, 3, 3, 3, 3],
            "label": [
                "Arena perimeter (polygon)",
                "Larvae (circles)",
                "Eggs (points)",
                "Nest perimeter (polygon)",
                "Nest perimeter (polygon)",
                "Nest perimeter (polygon)",
                "Nest perimeter (polygon)",
            ],
            "shape": ["polygon", "circle", "point", "polygon", "polygon", "polygon", "polygon"],
            "x": [0.0, 10.0, 30.0, 0.0, 40.0, 40.0, 0.0],
            "y": [0.0, 10.0, 30.0, 0.0, 0.0, 40.0, 40.0],
            "radius": [np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )

    distance, interactions = compute_nest_component_tables(tracking, brood_map, cutoff=2.5)

    assert distance.loc[0, 1] == 0.0
    assert np.isclose(distance.loc[0, 2], 3.0)
    assert distance.loc[1, 1] == 0.0
    assert interactions.loc[0, 1] == 1
    assert interactions.loc[0, 2] == 0
    assert interactions.loc[1, 1] == 1


def test_distance_to_nest_component_handles_polygon_interior():
    component = NestComponent(
        label="Nest perimeter (polygon)",
        shape="polygon",
        points=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
    )

    inside, _ = distance_to_nest_component((5.0, 5.0), component)
    outside, _ = distance_to_nest_component((15.0, 5.0), component)

    assert inside == 0.0
    assert np.isclose(outside, 5.0)


def test_ethogram_state_table_binarizes_event_metrics():
    table = pd.DataFrame({1: [0.0, 2.0, np.nan], 2: [1.0, 0.0, 0.0]}, index=[0, 1, 2])

    states = ethogram_state_table(table, "nest_interactions", cutoff=5)

    assert states.loc[0, 1] == 0.0
    assert states.loc[1, 1] == 1.0
    assert np.isnan(states.loc[2, 1])
    assert states.loc[0, 2] == 1.0


def test_ethogram_state_table_thresholds_distance_metrics():
    table = pd.DataFrame({1: [2.0, 8.0, np.nan]}, index=[0, 1, 2])

    states = ethogram_state_table(table, "nest_distance", cutoff=5)

    assert states.loc[0, 1] == 1.0
    assert states.loc[1, 1] == 0.0
    assert np.isnan(states.loc[2, 1])


def test_contiguous_value_ranges_keeps_unknown_runs():
    series = pd.Series([0.0, 0.0, 1.0, np.nan, np.nan, 1.0], index=[0, 1, 2, 3, 4, 7])

    assert contiguous_value_ranges(series) == [
        (0, 1, 0),
        (2, 2, 1),
        (3, 4, None),
        (7, 7, 1),
    ]


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


def test_frame_ticks_include_first_and_last_values():
    window = ReviewHub.__new__(ReviewHub)
    ticks = window.frame_ticks(3, 73)

    assert ticks[0] == (3, "3")
    assert ticks[-1] == (73, "73")
    assert len(ticks) <= 6


class StubTimeline:
    def __init__(self):
        self.range = None

    def setRange(self, minimum, maximum):
        self.range = (minimum, maximum)


def test_unknown_video_frame_count_uses_tracking_frame_range():
    window = ReviewHub.__new__(ReviewHub)
    window.tracking = pd.DataFrame({"frame": [0, 12]})
    window.noid_tags = pd.DataFrame()
    window.video_frame_count_known = False
    window.frame_count = 0
    window.current_frame = 0
    window.timeline = StubTimeline()
    window.sessions = [ReviewSession(Path("example.mjpeg"), (), None)]
    window.session_index = 0

    window.update_effective_frame_range()

    assert window.frame_count == 13
    assert window.timeline.range == (0, 12)


def test_mjpeg_frame_count_can_be_extended_from_tracking_range():
    window = ReviewHub.__new__(ReviewHub)
    window.tracking = pd.DataFrame({"frame": [0, 42]})
    window.noid_tags = pd.DataFrame()
    window.video_frame_count_known = True
    window.frame_count = 2
    window.current_frame = 0
    window.timeline = StubTimeline()
    window.sessions = [ReviewSession(Path("example.mjpeg"), (), None)]
    window.session_index = 0

    window.update_effective_frame_range()

    assert window.frame_count == 43
    assert window.timeline.range == (0, 42)


class SequentialCapture:
    def __init__(self):
        self.index = 0
        self.set_calls = []

    def read(self):
        value = self.index
        self.index += 1
        return True, value

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        self.index = int(value)
        return True


class StubTimer:
    def __init__(self, active=False):
        self.active = active
        self.starts = []
        self.stops = 0

    def isActive(self):
        return self.active

    def start(self, interval):
        self.active = True
        self.starts.append(interval)

    def stop(self):
        self.active = False
        self.stops += 1


class StubButton:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


def test_sequential_frame_read_does_not_seek_between_adjacent_frames():
    window = ReviewHub.__new__(ReviewHub)
    capture = SequentialCapture()
    window.capture = capture
    window._capture_next_frame = 0
    window._last_frame_index = None
    window._last_frame_bgr = None

    assert window.read_video_frame(0) == 0
    assert window.read_video_frame(1) == 1
    assert capture.set_calls == []


def test_playback_started_from_last_frame_rewinds_before_starting():
    window = ReviewHub.__new__(ReviewHub)
    window.timer = StubTimer(active=False)
    window.play_button = StubButton()
    window.frame_count = 10
    window.current_frame = 9
    window.video_fps = 5
    window._capture_needs_reset = False
    targets = []

    def seek_frame(frame):
        targets.append(frame)
        window.current_frame = frame
        window._capture_needs_reset = False
        return True

    window.seek_frame = seek_frame

    window.toggle_playback()

    assert targets == [0]
    assert window.current_frame == 0
    assert window.timer.starts == [200]
    assert window.play_button.text == "Pause"


def test_stopping_on_last_frame_marks_capture_for_reset():
    window = ReviewHub.__new__(ReviewHub)
    window.timer = StubTimer(active=True)
    window.play_button = StubButton()
    window.frame_count = 3
    window.current_frame = 2
    window._capture_needs_reset = False

    window.stop_playback()

    assert not window.timer.isActive()
    assert window.play_button.text == "Play"
    assert window._capture_needs_reset is True


def test_frame_read_after_end_reopens_instead_of_using_stale_cache():
    window = ReviewHub.__new__(ReviewHub)
    window.capture = object()
    window._capture_next_frame = 3
    window._capture_needs_reset = True
    window._last_frame_index = 2
    window._last_frame_bgr = "stale-frame"
    reopen_calls = []

    def read_frame_by_reopening(frame_index):
        reopen_calls.append(frame_index)
        return f"fresh-frame-{frame_index}"

    window.read_frame_by_reopening = read_frame_by_reopening

    assert window.read_video_frame(2) == "fresh-frame-2"
    assert reopen_calls == [2]
    assert window._last_frame_bgr == "fresh-frame-2"
    assert window._capture_next_frame == 3
    assert window._capture_needs_reset is False


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


def test_assess_nest_map_status_uses_bumblebox_video_colony_date(tmp_path):
    video = tmp_path / "bumblebox-17_2024-08-11_00_30_02.mp4"
    video.write_bytes(b"")
    tracking = tmp_path / "bumblebox-17_2024-08-11_00_30_02_newtracks_cleaned.csv"
    write_tracking(tracking)
    brood = tmp_path / "maps" / "col_17-2024-08-11-nest_image.csv"
    brood.parent.mkdir()
    brood.write_text("object index,label,label ID,vertex ID,shape,x,y,radius\n", encoding="utf-8")

    status = assess_nest_map_status(ReviewSession(video, (tracking,), tmp_path))

    assert status.state == "ready"
    assert status.csv_path == brood.resolve()
    assert status.target == (17, "2024-08-11")


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
