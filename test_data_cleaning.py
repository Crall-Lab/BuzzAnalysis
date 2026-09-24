from pathlib import Path
import runpy
from types import SimpleNamespace

import pandas as pd

from data_cleaning import remove_jumps


def load_clean_script_function(name):
    namespace = runpy.run_path(Path(__file__).with_name("02_clean_data.py"), run_name="clean_script_test")
    return namespace[name]


def make_args(tmp_path: Path, threshold: int = 50):
    source = tmp_path / "dataset"
    return SimpleNamespace(source=str(source), remove_jumps=threshold)


def test_remove_jumps_drops_flagged_rows_and_writes_log(tmp_path):
    args = make_args(tmp_path, threshold=50)
    df = pd.DataFrame(
        {
            "ID": [7, 7, 7],
            "frame": [0, 1, 2],
            "centroidX": [0.0, 100.0, 0.0],
            "centroidY": [0.0, 0.0, 0.0],
        }
    )

    cleaned = remove_jumps(df, args, filename="video_a")

    assert cleaned["frame"].tolist() == [0, 2]
    assert "flagged_as_jump" not in cleaned.columns

    log_path = tmp_path / "dataset_jump_log.csv"
    assert log_path.exists()

    log_df = pd.read_csv(log_path)
    assert log_df["label"].tolist() == ["neighbor", "jump", "neighbor"]
    assert log_df.loc[log_df["label"] == "jump", "frame"].tolist() == [1]


def test_remove_jumps_keeps_non_jump_rows(tmp_path):
    args = make_args(tmp_path, threshold=50)
    df = pd.DataFrame(
        {
            "ID": [9, 9, 9],
            "frame": [0, 1, 2],
            "centroidX": [0.0, 10.0, 20.0],
            "centroidY": [0.0, 0.0, 0.0],
        }
    )

    cleaned = remove_jumps(df, args, filename="video_b")

    assert cleaned["frame"].tolist() == [0, 1, 2]
    assert "flagged_as_jump" not in cleaned.columns
    assert not (tmp_path / "dataset_jump_log.csv").exists()


def test_remove_jumps_keeps_and_logs_ambiguous_one_way_high_speed_transition(tmp_path):
    args = make_args(tmp_path, threshold=50)
    df = pd.DataFrame(
        {
            "ID": [7, 7, 7],
            "frame": [0, 1, 2],
            "centroidX": [0.0, 100.0, 200.0],
            "centroidY": [0.0, 0.0, 0.0],
        }
    )

    cleaned = remove_jumps(df, args, filename="video_c")

    assert cleaned["frame"].tolist() == [0, 1, 2]

    log_path = tmp_path / "dataset_jump_log.csv"
    assert log_path.exists()

    log_df = pd.read_csv(log_path)
    assert set(log_df["action"]) == {"kept_ambiguous_high_speed_transition"}
    assert log_df["label"].tolist() == [
        "transition_start",
        "transition_end",
        "transition_start",
        "transition_end",
    ]
    assert log_df["transition_speed_px_per_frame"].tolist() == [100.0, 100.0, 100.0, 100.0]


def test_remove_jumps_requires_plausible_bridge_before_dropping_spike(tmp_path):
    args = make_args(tmp_path, threshold=50)
    df = pd.DataFrame(
        {
            "ID": [7, 7, 7],
            "frame": [0, 1, 2],
            "centroidX": [0.0, 100.0, 120.0],
            "centroidY": [0.0, 0.0, 0.0],
        }
    )

    cleaned = remove_jumps(df, args, filename="video_d")

    assert cleaned["frame"].tolist() == [0, 1, 2]

    log_path = tmp_path / "dataset_jump_log.csv"
    assert log_path.exists()

    log_df = pd.read_csv(log_path)
    assert "removed_isolated_spike" not in set(log_df["action"])
    assert set(log_df["action"]) == {"kept_ambiguous_high_speed_transition"}


def test_clean_settings_file_uses_max_speed_for_spike_removal(tmp_path):
    settings = tmp_path / "buzzanalysis_settings.json"
    settings.write_text('{"max_speed_cutoff": 450}', encoding="utf-8")
    args = SimpleNamespace(settings=str(settings), remove_jumps=None)

    apply_settings_file = load_clean_script_function("apply_settings_file")
    updated = apply_settings_file(args)

    assert updated.remove_jumps == 450


def test_clean_settings_file_does_not_override_explicit_spike_threshold(tmp_path):
    settings = tmp_path / "buzzanalysis_settings.json"
    settings.write_text('{"max_speed_cutoff": 450}', encoding="utf-8")
    args = SimpleNamespace(settings=str(settings), remove_jumps=300)

    apply_settings_file = load_clean_script_function("apply_settings_file")
    updated = apply_settings_file(args)

    assert updated.remove_jumps == 300
