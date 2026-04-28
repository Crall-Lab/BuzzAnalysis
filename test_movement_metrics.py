import numpy as np
import pandas as pd

from aux import movement_metrics


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
