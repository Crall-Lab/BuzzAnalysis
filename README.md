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

## Review and settings

```bash
python review_metrics_gui.py /path/to/working-data
python LabelNests_GUI.1.16.py /path/to/labels
```

In the review GUI, select a matching tracking CSV under **Metric Controls**, inspect videos, and **Export settings**. Both cleanup and analysis accept `--settings settings.json`. Explicit CLI options override JSON values, which override defaults.

Check FPS, pixel calibration, thresholds, and smoothing against your recordings. Run a compatible colony/camera coordinate system per batch: the main runner pools selected files for its social center. Missing activity is unknown, not inactive. Contacts describe proximity, not verified biological interactions. The manual explains units and output meanings.

## Preliminary activity-transition workflow

`preliminary_newtracks_analysis.py` creates daily Parquet tables and indexes from modern `*_newtracks.csv` files. It requires `--source`, `--labels`, `--output`, `--box-id`, and `--colony-number`. Use a fresh output directory per run. Its raw-track duplicate averaging is separate from the cleaning workflow.

Run `summarize_inactive_to_active.py RESULTS` or `summarize_nest_transition_rates.py RESULTS` on that output. Excluded tags stay excluded, and contacts are recomputed without them. Plot summary CSVs with `plot_activity_transition_rates.py` and `plot_inactive_to_active_over_time.py`; use the latter's `--fps` option for your recording frame rate. See page 3 of the manual for a complete starting example.

## Validation and documentation

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
QT_QPA_PLATFORM=offscreen python docs/build_manual.py
```

The reviewed code passes 86 tests, including complete CLI workflows and plot generation. The manual builder uses installed PySide6 and rejects content that would overflow its three PDF pages. Review scope, corrected result-affecting issues, and tested dependency versions are recorded in [the review report](docs/review_report.md).

Historical `runMe*.py`, `analyze_data_0.1.py`, and `analyze_clean_data_0.1.py` scripts remain available. Start new analyses with `analyze_clean_data_0.2.py`. The coordinate-only `03_compute_intermediates.py` / `04_aggregate_video_means.py` route is optional; brood-aware analysis uses the main runner.
