"""Dated-folder grouping must affect actual distSC outputs, including worker jobs."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("analysis_social_centers", ROOT / "analyze_clean_data_0.2.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def write_tracking(path, xs, ys=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": range(len(xs)), "ID": [1] * len(xs),
                  "centroidX": xs, "centroidY": ys if ys is not None else [0.] * len(xs)}).to_csv(path, index=False)
    return str(path)


def run_analysis(source, output, *options):
    return subprocess.run([sys.executable, str(ROOT / "analyze_clean_data_0.2.py"),
                           "-s", str(source), "-o", str(output), *options],
                          text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=45)


@pytest.mark.parametrize("folder", ["2026-05-01", "2026_05_01", "col_1-2021-06-10", "col_01-2021-06-10", "bumblebox-05_2026-05-01"])
def test_date_folder_with_nested_recording_folder(tmp_path, folder):
    day = tmp_path / folder
    recording = day / 'bumblebox-05_2026-05-01_12_00_00' / 'tracks_clean.csv'
    assert analysis.find_social_center_date_folder(recording, tmp_path) == day


def test_source_itself_can_be_the_date_folder(tmp_path):
    root = tmp_path / '2026-05-01'
    assert analysis.find_social_center_date_folder(root / 'tracks.csv', root) == root


def test_nearest_date_folder_wins(tmp_path):
    root = tmp_path / '2026-05-01'
    day = root / '2026-05-02'
    assert analysis.find_social_center_date_folder(day / 'nested' / 'tracks.csv', root) == day


def test_date_search_does_not_escape_source(tmp_path):
    root = tmp_path / '2026-05-01' / 'undated-source'
    with pytest.raises(ValueError, match='dated folder'):
        analysis.find_social_center_date_folder(root / 'recording' / 'tracks.csv', root)


@pytest.mark.parametrize('folder', ['2026-02-30', '2026-13-01', '20260501', '2026-05-01_12_00_00'])
def test_invalid_or_unsupported_date_folder_is_rejected(tmp_path, folder):
    with pytest.raises(ValueError, match='date|dated'):
        analysis.find_social_center_date_folder(tmp_path / folder / 'tracks.csv', tmp_path)


def test_different_colonies_same_day_remain_separate_and_centers_are_row_weighted(tmp_path):
    first = write_tracking(tmp_path / 'colony-a' / '2026-05-01' / 'first.csv', [0.], [2.])
    second = write_tracking(tmp_path / 'colony-a' / '2026-05-01' / 'nested' / 'second.csv', [4., 4., 4.], [6., 6., 6.])
    other = write_tracking(tmp_path / 'colony-b' / '2026-05-01' / 'third.csv', [100.], [200.])
    centers = analysis.social_centers_for_files([first, second, other], tmp_path, by_date_folder=True)
    assert np.allclose(centers[first], [3., 5.])
    assert np.allclose(centers[second], [3., 5.])
    assert np.allclose(centers[other], [100., 200.])


@pytest.mark.parametrize('cores', ['1', '2'])
def test_flag_computes_one_center_per_day_in_full_cli(tmp_path, cores):
    source = tmp_path / 'source'
    first = 'bumblebox-05_2026-05-01_12_00_00_Whole_clean.csv'
    second = 'bumblebox-05_2026-05-01_13_00_00_Whole_clean.csv'
    third = 'bumblebox-05_2026-05-02_12_00_00_Whole_clean.csv'
    write_tracking(source / '2026-05-01' / first.removesuffix('_Whole_clean.csv') / first, [0., 2.])
    write_tracking(source / '2026-05-01' / second.removesuffix('_Whole_clean.csv') / second, [8., 10.])
    write_tracking(source / '2026-05-02' / third.removesuffix('_Whole_clean.csv') / third, [100., 102.])
    output = tmp_path / 'summary.csv'
    run = run_analysis(source, output, '--social-center-by-date-folder', '-c', cores)
    assert run.returncode == 0, run.stdout + run.stderr
    result = pd.read_csv(output).sort_values(['Date', 'Time'])
    assert result.distSC.tolist() == [4., 4., 1.]
    assert 'Social center' in run.stdout and '2026-05-01' in run.stdout


def test_default_keeps_one_batch_center(tmp_path):
    source = tmp_path / 'source'
    for date, time, xs in [('2026-05-01', '12_00_00', [0., 2.]),
                            ('2026-05-01', '13_00_00', [8., 10.]),
                            ('2026-05-02', '12_00_00', [100., 102.])]:
        write_tracking(source / date / f'bumblebox-05_{date}_{time}_Whole_clean.csv', xs)
    output = tmp_path / 'summary.csv'
    run = run_analysis(source, output)
    assert run.returncode == 0, run.stdout + run.stderr
    result = pd.read_csv(output).sort_values(['Date', 'Time'])
    assert result.distSC.tolist() == [36., 28., 64.]


def test_missing_day_folder_fails_before_any_output_is_written(tmp_path):
    source = tmp_path / 'undated'
    path = source / 'bumblebox-05_2026-05-01_12_00_00_Whole_clean.csv'
    write_tracking(path, [0., 2.])
    output = tmp_path / 'summary.csv'
    run = run_analysis(source, output, '--social-center-by-date-folder', '--save-frame-level')
    assert run.returncode != 0
    assert 'dated folder' in run.stderr
    assert str(path) in run.stderr
    assert not output.exists()
    assert not list(source.rglob('*_frame_level.csv'))


def test_limit_applies_before_group_center_calculation(tmp_path):
    source = tmp_path / '2026-05-01'
    write_tracking(source / 'bumblebox-05_2026-05-01_12_00_00_Whole_clean.csv', [0., 2.])
    write_tracking(source / 'bumblebox-05_2026-05-01_13_00_00_Whole_clean.csv', [100., 102.])
    output = tmp_path / 'summary.csv'
    run = run_analysis(source, output, '--social-center-by-date-folder', '-l', '1')
    assert run.returncode == 0, run.stdout + run.stderr
    assert pd.read_csv(output).distSC.tolist() == [1.]
