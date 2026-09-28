# BuzzAnalysis

Analyze BumbleBox bee tracking data: clean detections, review tracks over video, label nest features, and calculate activity, proximity contacts, nest use, and activity-transition rates.

**Start here:** [Three-page introductory manual (PDF)](docs/intro_manual.pdf) · [Editable manual (HTML)](docs/intro_manual.html) · [Review findings and validation](docs/review_report.md)

## Setup

The current workflow is on `testing_May25`. From a fresh checkout:

```bash
git clone --branch testing_May25 https://github.com/Crall-Lab/BuzzAnalysis.git
cd BuzzAnalysis
conda env create -f environment.yml
conda activate BuzzAnalysis
```

The environment includes Python 3.13, NumPy, pandas, SciPy, Shapely 2, PyArrow, tqdm, Matplotlib, PySide6, pyqtgraph, Pillow, OpenCV, LabelMe, and pytest. If you already have the environment, activate it; `install_gui_dependencies.sh` installs missing analysis/GUI dependencies into an existing named Conda environment.

## First analysis

Run from the repository root. This example copies the included sample CSV before producing derived files:

```bash
mkdir -p demo/raw
cp worker23_2022-06-20_18-25-30.csv demo/raw/
python 01_split_lr.py -s demo/raw -e .csv --whole
python 02_clean_data.py -s demo/raw
python analyze_clean_data_0.2.py -s demo/raw -o demo/Analysis.csv --save-frame-level
```

Tracking CSVs require `frame`, `ID`, `centroidX`, and `centroidY`. Frame numbers and IDs are integers; positions are pixels. The main analysis recognizes worker/col/colony/bumblebox filenames containing date and time. For a custom name layout, use `--filename-positions` with the inclusive positions in `params.py`.

`Analysis.csv` contains one row per bee/video/side. Optional frame-level CSVs are saved beside the cleaned recordings. `--save-pivots` additionally exports enriched Feather tables. Use `-e` to select a different input suffix, `-b /path/to/labels` for nest maps, `-l 1` for a small first run, and `-c 4` for parallel processing.

## Social center per dated folder

Use `--social-center-by-date-folder` to calculate a separate social center for each dated folder while still producing one combined analysis CSV:

```bash
python analyze_clean_data_0.2.py \
  -s "/path/to/parent-folder" \
  --social-center-by-date-folder \
  -o "/path/to/results/Analysis.csv"
```

For example, every matching tracking file beneath `parent/2021-06-10/` shares one center, and files beneath `parent/2021-06-11/` share another. Tracking files can be nested further inside recording folders.

The nearest parent folder whose name ends in a valid `YYYY-MM-DD` or `YYYY_MM_DD` date defines the group. Names such as `col_1-2021-06-10` and `col_01-2021-06-10` are supported. Timestamped recording folders such as `bumblebox-05_2021-06-10_12_00_00` are skipped when finding the day folder. The search includes `--source` itself and stops there.

Groups use the full folder path: `colony-a/2021-06-10/` and `colony-b/2021-06-10/` have separate centers. Files without a dated parent cause an error before output is written. Use a consistent colony/camera coordinate system within each dated folder.

The center is the mean of the selected detections' X/Y coordinates, so recordings contribute in proportion to their valid detection rows. File filters (`-e`, `-l`) also limit which data contribute to a center. Each group's center and file count are printed, and the center is used for that group's `distSC` metric in both serial and parallel runs. Without the flag, all selected files share one batch-wide center. This flag applies to the main batch analysis, not the GUI's separate social-center display.

## Review and settings

```bash
python review_metrics_gui.py /path/to/working-data
python LabelNests_GUI.1.16.py /path/to/labels
```

In the review GUI, select a matching tracking CSV under **Metric Controls**, inspect videos, and **Export settings**. Both cleanup and analysis accept `--settings settings.json`. Explicit CLI options override JSON values, which override defaults.

Check FPS, pixel calibration, thresholds, and smoothing against your recordings. Use a compatible colony/camera coordinate system for each social-center group (the entire batch by default, or each dated folder with the flag above). Missing activity is unknown, not inactive. Contacts describe proximity, not verified biological interactions. The manual explains units and output meanings.

## Preliminary activity-transition workflow

`preliminary_newtracks_analysis.py` creates daily Parquet tables and indexes from modern `*_newtracks.csv` files. It requires `--source`, `--labels`, `--output`, `--box-id`, and `--colony-number`. Use a fresh output directory per run. Its raw-track duplicate averaging is separate from the cleaning workflow.

Run `summarize_inactive_to_active.py RESULTS` or `summarize_nest_transition_rates.py RESULTS` on that output. Excluded tags stay excluded, and contacts are recomputed without them. Plot summary CSVs with `plot_activity_transition_rates.py` and `plot_inactive_to_active_over_time.py`; use the latter's `--fps` option for your recording frame rate. See page 3 of the manual for a complete starting example.

## Validation and documentation

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
QT_QPA_PLATFORM=offscreen python docs/build_manual.py
```

The test suite covers complete CLI workflows, plot generation, and social-center grouping. The manual builder uses installed PySide6 and rejects content that would overflow its three PDF pages. Review scope, corrected result-affecting issues, and tested dependency versions are recorded in [the review report](docs/review_report.md).

Historical `runMe*.py`, `analyze_data_0.1.py`, and `analyze_clean_data_0.1.py` scripts remain available. Start new analyses with `analyze_clean_data_0.2.py`. The coordinate-only `03_compute_intermediates.py` / `04_aggregate_video_means.py` route is optional; brood-aware analysis uses the main runner.
