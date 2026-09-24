"""Regression coverage for repository review findings and real CLI workflows."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import aux
import baseFunctions
import broodFunctions
import data_cleaning
import preliminary_newtracks_analysis as preliminary
import summarize_inactive_to_active as inactive
import summarize_nest_transition_rates as nest
from utils_io import save_df

ROOT = Path(__file__).parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pivot(xs, ys=None, frames=None):
    xs = np.asarray(xs, dtype=float)
    if xs.ndim == 1:
        xs = xs[:, None]
    ys = np.zeros_like(xs) if ys is None else np.asarray(ys, dtype=float)
    columns = pd.MultiIndex.from_product([['centroidX', 'centroidY'], range(1, xs.shape[1]+1)])
    return pd.DataFrame(np.concatenate([xs, ys], axis=1), columns=columns,
                        index=pd.Index(frames if frames is not None else range(len(xs)), name='frame'))


def run_cli(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                          cwd=cwd or ROOT, text=True, capture_output=True, stdin=subprocess.DEVNULL,
                          env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}, timeout=45)


def test_save_df_accepts_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_df(pd.DataFrame({'value': [1]}), 'out.csv')
    assert pd.read_csv('out.csv').value.tolist() == [1]


def test_jump_log_relative_source_is_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Path(data_cleaning._jump_log_path(SimpleNamespace(source='data'))) == tmp_path / 'data_jump_log.csv'


def test_cleaning_resolves_minimal_tracking_schema():
    df = pd.DataFrame({'frame': [0, 1, 1, 2], 'ID': [1]*4,
                       'centroidX': [0., 1., 100., 2.], 'centroidY': [0.]*4})
    work, flag = data_cleaning.return_duplicate_bees(df)
    cleaned = data_cleaning.drop_duplicates_clean(work, flag)
    assert cleaned.centroidX.tolist() == [0., 1., 2.]


def test_interpolation_accepts_centroid_only_numeric_data():
    df = pd.DataFrame({'ID': [1, 1], 'frame': [0, 2], 'centroidX': [0., 4.], 'centroidY': [0., 2.]})
    result = data_cleaning.interpolate(df, 1, 5)
    assert result.frame.tolist() == [0, 1, 2]
    assert result.centroidX.tolist() == [0., 2., 4.]
    assert result.interpolated.tolist() == [0, 1, 0]


def test_activity_smoothing_does_not_cross_absent_frames():
    activity = pd.DataFrame({1: [1., 0., 1.]}, index=[0, 10, 11])
    assert aux.close_inactive_activity_gaps(activity, 2).loc[10, 1] == 0


def test_colocated_bees_count_as_contacts(monkeypatch):
    monkeypatch.setattr(baseFunctions, 'pixels_per_cm', 10)
    table = pivot([[0, 0], [1, 1]])
    assert np.allclose(baseFunctions.totalInt(table), [2, 2])
    assert np.allclose(baseFunctions.totalIntFrames(table), [2, 2])
    assert np.allclose(baseFunctions.meanIBD(table), [0, 0])


def test_enriched_single_bee_has_no_pair_opportunities():
    table = pivot([0, 1])
    table[('distM_Larvae (circles)_0', 1)] = [2, 2]
    assert np.allclose(baseFunctions.totalIntFrames(table), [0])


def test_inactivity_ignores_arena_and_foraging_geometry():
    table = pivot([0, 0, 0])
    table[('distM_Arena perimeter (polygon)_0', 1)] = 0.
    table[('distM_Foraging perimeter (polygon)_1', 1)] = 0.
    table[('distM_Larvae (circles)_2', 1)] = 100.
    assert broodFunctions.PropInactiveTime(table).loc[1] == 1.


def test_merge_empty_transition_parts():
    assert preliminary.merge_transition_parts([pd.DataFrame()], fps=5).empty


def test_blank_legacy_contact_index(tmp_path):
    (tmp_path / 'contact_pairs_file_index.csv').write_text('\n')
    assert inactive.load_contact_index(tmp_path) == {}


def contact_frame():
    return pd.DataFrame({'bee_ID': [1, 2], 'frame': [0, 0], 'source_file': ['v', 'v'],
                         'centroidX': [0., 0.], 'centroidY': [0., 0.],
                         'contact_count': [1., 1.], 'social_contact': [True, True],
                         'excluded_from_analysis': [False, True]})


def test_missing_contact_file_does_not_fabricate_no_contact(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError), match='contact|Contact'):
        inactive.recompute_contacts(contact_frame(), tmp_path / 'missing.parquet', set())


def test_upstream_excluded_bees_stay_excluded(tmp_path):
    pairs = tmp_path / 'pairs.parquet'
    pd.DataFrame({'source_file': ['v'], 'frame': [0], 'bee_ID_1': [1], 'bee_ID_2': [2]}).to_parquet(pairs)
    result, _ = inactive.recompute_contacts(contact_frame(), pairs, set())
    assert result.bee_ID.tolist() == [1]
    assert result.contact_count.tolist() == [0.]


def test_missing_pair_index_rejects_positive_contacts():
    with pytest.raises(ValueError, match='contact|Contact'):
        inactive.recompute_contacts(contact_frame(), None, set())


def test_empty_nest_collapse_preserves_schema():
    result = nest.collapse_transition_rates(pd.DataFrame(), [nest.NEST_STATUS_COL, 'social_contact'], 5)
    assert nest.NEST_STATUS_COL in result.columns


def test_clean_cli_runs_without_terminal_input(tmp_path):
    source = tmp_path / 'raw'
    source.mkdir()
    pd.DataFrame({'frame': [0, 1], 'ID': [1, 1], 'centroidX': [0., 1.], 'centroidY': [0., 0.]}).to_csv(
        source / 'bumblebox-05_2026-05-01_12_00_00_Whole.csv', index=False)
    result = run_cli('02_clean_data.py', '-s', source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(source.glob('*_clean.csv'))) == 1


@pytest.mark.parametrize('side', ['Left', 'Right', 'Whole'])
def test_analysis_cli_processes_sides_and_creates_output(tmp_path, side):
    source = tmp_path / 'source'
    source.mkdir()
    path = source / f'bumblebox-05_2026-05-01_12_00_00_{side}_clean.csv'
    pd.DataFrame({'frame': [0, 1, 2], 'ID': [1]*3, 'centroidX': [0., 4., 8.], 'centroidY': [0.]*3}).to_csv(path, index=False)
    output = tmp_path / 'results' / 'Analysis.csv'
    result = run_cli('analyze_clean_data_0.2.py', '-s', source, '-o', output, '--save-frame-level')
    assert result.returncode == 0, result.stdout + result.stderr
    table = pd.read_csv(output, dtype={'pi_ID': str})
    assert table.LR.tolist() == [side]
    assert table.Date.tolist() == ['2026-05-01']
    assert table.Time.tolist() == ['12-00-00']
    assert int(table.pi_ID.iloc[0]) == 5


def test_analysis_custom_extension_does_not_overwrite_input(tmp_path):
    path = tmp_path / 'col_01-2021-06-12_12-00-00_custom.csv'
    pd.DataFrame({'frame': [0, 1], 'ID': [1, 1], 'centroidX': [0., 1.], 'centroidY': [0., 0.]}).to_csv(path, index=False)
    original = path.read_bytes()
    result = run_cli('analyze_clean_data_0.2.py', '-s', tmp_path, '-e', '_custom.csv',
                     '-o', tmp_path / 'analysis.csv', '--save-pivots')
    assert result.returncode == 0, result.stdout + result.stderr
    assert path.read_bytes() == original
    assert len(list(tmp_path.glob('*_pivot_enriched.feather'))) == 1


def test_intermediate_and_aggregate_cli(tmp_path):
    source = tmp_path / 'nested'
    source.mkdir()
    pd.DataFrame({'frame': [0, 1, 2], 'ID': [1]*3, 'centroidX': [0., 4., 8.], 'centroidY': [0.]*3}).to_csv(
        source / 'bumblebox-05_2026-05-01_12_00_00_Whole_clean.csv', index=False)
    result = run_cli('03_compute_intermediates.py', '-s', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(source.glob('*_pivot.feather'))) == 1
    result = run_cli('04_aggregate_video_means.py', '-s', tmp_path, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = pd.read_csv(tmp_path / 'Analysis.csv')
    assert summary.distSC.notna().all()
    assert summary.video.iloc[0] == 'bumblebox-05_2026-05-01_12_00_00_Whole'


def test_centroid_distances_keep_bee_coordinates_together():
    import processBroodFunctions as geometry
    table = pivot([[1, 10]], [[2, 20]])
    points = pd.DataFrame({'x': [0.], 'y': [0.], 'label': ['Eggs (points)'], 'object index': [0], 'vertex ID': [1]}, index=[7])
    distances = geometry.distanceFromCentroid_new(table, points)
    assert np.isclose(distances.iloc[0, 0], np.hypot(1, 2))
    assert np.isclose(distances.iloc[0, 1], np.hypot(10, 20))


def test_segment_distance_endpoints_and_degenerate_segment():
    from processBroodFunctions import minDistance
    assert minDistance([0, 0], [1, 0], [-3, 4]) == 5
    assert minDistance([0, 0], [0, 0], [3, 4]) == 5


def test_settings_preserve_explicit_cli_choices(tmp_path, monkeypatch):
    settings = tmp_path / 'settings.json'
    settings.write_text('{"speed_cutoff": 9, "pixels_per_cm": 200}')
    script = load_script('analyze_clean_data_0.2.py')
    monkeypatch.setattr(sys, 'argv', ['analysis', '--settings', str(settings), '--speed-cutoff=2'])
    options = script.apply_settings_file(script.cli())
    assert options['speed_cutoff'] == 2
    assert options['pixels_per_cm'] == 200


@pytest.mark.parametrize('contact,short,excluded', [(False, False, False), (True, False, False), (False, True, False), (True, False, True)])
def test_preliminary_and_summary_cli(tmp_path, contact, short, excluded):
    source = tmp_path / 'source'
    labels = tmp_path / 'labels'
    results = tmp_path / 'results'
    source.mkdir(); labels.mkdir()
    frames = [0] if short else list(range(8))
    positions = [0, 0, 0, 10, 20, 20, 20, 30]
    rows = [{'frame': f, 'ID': bee, 'centroidX': positions[f] + (0 if bee == 1 else (2 if contact else 200)),
             'centroidY': 0.} for bee in (1, 2) for f in frames]
    pd.DataFrame(rows).to_csv(source / 'bumblebox-05_2026-05-01_12_00_00_newtracks.csv', index=False)
    extras = ['--exclude-bee-ids', '2'] if excluded else []
    run = run_cli('preliminary_newtracks_analysis.py', '--source', source, '--labels', labels,
                  '--output', results, '--colony-number', '65', '--box-id', 'bumblebox-05',
                  '--inactive-gap-max-frames', '0', *extras)
    assert run.returncode == 0, run.stdout + run.stderr
    for script in ['summarize_inactive_to_active.py', 'summarize_nest_transition_rates.py']:
        run = run_cli(script, results)
        assert run.returncode == 0, run.stdout + run.stderr
    if not short:
        suffix = '2' if excluded else 'none'
        summary = results / f'inactive_to_active_excluding_{suffix}' / 'inactive_to_active_rates_by_date_location_contact.csv'
        run = run_cli('plot_inactive_to_active_over_time.py', summary, '--min-opportunities', '1',
                      '--no-separate', '--dpi', '50', '--output-prefix', results / 'plots' / 'activation')
        assert run.returncode == 0, run.stdout + run.stderr
        run = run_cli('plot_activity_transition_rates.py', results / 'activity_transition_rates_by_location_contact.csv',
                      '--no-separate', '--dpi', '50')
        assert run.returncode == 0, run.stdout + run.stderr


def test_brood_analysis_parallel_and_enriched_export(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    labels = tmp_path / 'labels'; labels.mkdir()
    rows = [{'frame': f, 'ID': b, 'centroidX': 5. + f + b, 'centroidY': 5.} for f in range(3) for b in (1, 2)]
    for time in ['12_00_00', '12_01_00']:
        pd.DataFrame(rows).to_csv(source / f'bumblebox-05_2026-05-01_{time}_Whole_clean.csv', index=False)
    pd.DataFrame([
        {'object index': 0, 'label': 'Nest perimeter (polygon)', 'shape': 'polygon', 'vertex ID': i+1,
         'x': x, 'y': y, 'radius': np.nan} for i, (x, y) in enumerate([(0, 0), (20, 0), (20, 20), (0, 20)])
    ] + [{'object index': 1, 'label': 'Larvae (circles)', 'shape': 'circle', 'vertex ID': 1, 'x': 10, 'y': 10, 'radius': 1}]
    ).to_csv(labels / 'bumblebox-05-2026-05-01-nest_image.csv', index=False)
    run = run_cli('analyze_clean_data_0.2.py', '-s', source, '-b', labels, '-c', 2,
                  '--save-pivots', '--save-frame-level', '-o', tmp_path / 'summary.csv')
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'failed on' not in run.stdout
    summary = pd.read_csv(tmp_path / 'summary.csv')
    assert len(summary) == 4
    assert summary.PropBroodTime.notna().all()
    enriched = pd.read_feather(next(source.glob('*_pivot_enriched.feather')))
    assert 'activity' in enriched.columns.get_level_values(0)
    assert 'speed' in enriched.columns.get_level_values(0)
    frames = pd.read_csv(next(source.glob('*_frame_level.csv')))
    assert (frames.location_zone == 'nest').all()


def test_split_rerun_does_not_split_generated_files(tmp_path):
    path = tmp_path / 'worker23_2022-06-20_18-25-30.csv'
    pd.DataFrame({'frame': [0], 'ID': [1], 'centroidX': [1], 'centroidY': [2]}).to_csv(path, index=False)
    for _ in range(2):
        run = run_cli('01_split_lr.py', '-s', tmp_path, '-w')
        assert run.returncode == 0, run.stdout + run.stderr
    assert len(list(tmp_path.rglob('*_Whole.csv'))) == 1


def test_upstream_colony_determines_size_file(tmp_path):
    (tmp_path / 'run_metadata.json').write_text('{"colony_number": "12"}')
    (tmp_path / 'size-over-time-colony-65.csv').write_text('Date,size\n')
    assert inactive.default_colony_size_file(tmp_path) is None


def test_legacy_jump_log_remains_readable(tmp_path):
    args = SimpleNamespace(source=str(tmp_path / 'raw'), remove_jumps=50)
    log = Path(data_cleaning._jump_log_path(args))
    pd.DataFrame({'video': ['old'], 'ID': [7], 'frame': [1], 'centroidX': [100], 'centroidY': [0], 'label': ['jump']}).to_csv(log, index=False)
    data_cleaning.remove_jumps(pd.DataFrame({'ID': [7]*3, 'frame': [0, 1, 2],
                                'centroidX': [0., 100., 0.], 'centroidY': [0.]*3}), args, 'new')
    result = pd.read_csv(log)
    assert len(result) == 4
    assert result.action.eq('removed_isolated_spike').all()


def test_repeated_spikes_preserve_the_good_middle_observation(tmp_path):
    args = SimpleNamespace(source=str(tmp_path / 'raw'), remove_jumps=50)
    rows = pd.DataFrame({'ID': [1]*5, 'frame': range(5), 'centroidX': [0., 100., 0., 100., 0.], 'centroidY': [0.]*5})
    assert data_cleaning.remove_jumps(rows, args, 'v').frame.tolist() == [0, 2, 4]
