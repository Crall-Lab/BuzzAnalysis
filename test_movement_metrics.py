import numpy as np
import pandas as pd

from aux import close_inactive_activity_gaps, movement_metrics


def make_pivot(frames, xs):
    columns = pd.MultiIndex.from_tuples([("centroidX", 1), ("centroidY", 1)])
    return pd.DataFrame(
        {
            ("centroidX", 1): xs,
            ("centroidY", 1): [0.0] * len(xs),
        },
        index=frames,
        columns=columns,
    )


def test_movement_metrics_ignores_gaps_longer_than_threshold():
    one_lr = make_pivot([0, 1, 20], [0.0, 4.0, 104.0])

    act, speed = movement_metrics(one_lr, frame_rate=5, max_gap_seconds=3)

    assert np.isclose(speed.loc[1, 1], 4.0)
    assert act.loc[1, 1] == 1
    assert np.isnan(speed.loc[20, 1])
    assert np.isnan(act.loc[20, 1])


def test_movement_metrics_normalizes_valid_multi_frame_gaps():
    one_lr = make_pivot([0, 10], [0.0, 20.0])

    act, speed = movement_metrics(one_lr, frame_rate=5, max_gap_seconds=3)

    assert np.isclose(speed.loc[10, 1], 2.0)
    assert act.loc[10, 1] == 0


def test_movement_metrics_marks_speeds_above_max_as_unknown():
    one_lr = make_pivot([0, 1, 2], [0.0, 4.0, 104.0])

    act, speed = movement_metrics(
        one_lr,
        frame_rate=5,
        max_gap_seconds=3,
        speed_cutoff=3.5,
        max_speed_cutoff=50,
    )

    assert np.isclose(speed.loc[1, 1], 4.0)
    assert act.loc[1, 1] == 1
    assert np.isnan(speed.loc[2, 1])
    assert np.isnan(act.loc[2, 1])


def test_movement_metrics_max_speed_zero_disables_filter():
    one_lr = make_pivot([0, 1], [0.0, 100.0])

    act, speed = movement_metrics(
        one_lr,
        frame_rate=5,
        max_gap_seconds=3,
        speed_cutoff=3.5,
        max_speed_cutoff=0,
    )

    assert np.isclose(speed.loc[1, 1], 100.0)
    assert act.loc[1, 1] == 1


def test_movement_metrics_closes_short_inactive_blips_inside_activity_bout():
    one_lr = make_pivot([0, 1, 2, 3, 4, 5], [0.0, 4.0, 8.0, 8.5, 12.5, 16.5])

    act, speed = movement_metrics(
        one_lr,
        frame_rate=5,
        max_gap_seconds=3,
        speed_cutoff=3.5,
        inactive_gap_max_frames=1,
    )

    assert np.isclose(speed.loc[3, 1], 0.5)
    assert act.loc[3, 1] == 1


def test_movement_metrics_does_not_close_long_inactive_bouts():
    one_lr = make_pivot([0, 1, 2, 3, 4, 5], [0.0, 4.0, 8.0, 8.5, 9.0, 13.0])

    act, _speed = movement_metrics(
        one_lr,
        frame_rate=5,
        max_gap_seconds=3,
        speed_cutoff=3.5,
        inactive_gap_max_frames=1,
    )

    assert act.loc[3, 1] == 0
    assert act.loc[4, 1] == 0


def test_movement_metrics_does_not_fill_unknown_tracking_gaps():
    one_lr = make_pivot([0, 1, 20, 21], [0.0, 4.0, 4.5, 8.5])

    act, speed = movement_metrics(
        one_lr,
        frame_rate=5,
        max_gap_seconds=1,
        speed_cutoff=3.5,
        inactive_gap_max_frames=100,
    )

    assert np.isnan(speed.loc[20, 1])
    assert np.isnan(act.loc[20, 1])


def test_activity_smoothing_does_not_bridge_unknown_gap():
    activity = pd.DataFrame({1: [1.0, np.nan, 0.0, 1.0]}, index=[1, 2, 3, 4])

    smoothed = close_inactive_activity_gaps(activity, max_gap_frames=1)

    assert np.isnan(smoothed.loc[2, 1])
    assert smoothed.loc[3, 1] == 0
