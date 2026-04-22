from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from data_cleaning import remove_jumps


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
