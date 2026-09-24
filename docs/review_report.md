# BuzzAnalysis repository review

Reviewed checkout: `testing_May25`, starting from commit `72a0eb5` plus the existing 14 modified files and five new scripts. Those local changes are included in this delivery.

## Findings and fixes

| Finding | Effect before the fix | Resolution |
| --- | --- | --- |
| Coordinate layout in `distanceFromCentroid_new` | With multiple bees, reshaping mixed one bee's X coordinate with another bee's Y coordinate, corrupting exported `distC` distances. Nonconsecutive map indexes also failed. | Select X/Y explicitly by bee ID and normalize map row indexes. |
| Inactivity included every non-coordinate column | Arena/foraging distances of zero made inactive bees appear to occupy a nest/food object; unrelated columns could also affect scoring. | Restrict `PropInactiveTime` to minimum distances for nest/brood/food objects. Align on-object threshold boundaries. |
| Zero-distance bee pairs were treated as self pairs | Distinct colocated bees disappeared from contact and distance metrics. | Mask the diagonal only, retain real zero-distance pairs, and align inclusive contact thresholds. |
| Smoothing crossed missing frame numbers | Short inactive observations could become active across an unobserved gap. | Require contiguous frame indexes across both bounding active observations. |
| Duplicate cleanup required undocumented metadata | Minimal tracking CSVs silently skipped duplicate resolution. | Accept the four required tracking columns, preserve optional video/colony grouping, and use unambiguous nearby observations. |
| Interpolation assumed front coordinates and integer-valued row types | Centroid-only numeric tables could raise missing-column or float-to-range errors. | Interpolate available coordinate columns, explicitly convert frame counts, and validate timing. |
| Cleanup prompts and jump logs | Unused prompt blocked unattended runs; relative source paths could put logs at filesystem root; old log schemas became corrupt when appended. | Remove the prompt, normalize log paths, migrate legacy columns, and preserve good observations between repeated spikes. |
| Batch file selection and filenames | The analysis ignored `-e`, skipped Left/Right files, required interactive confirmation, and could silently slice metadata incorrectly. Saving a custom-suffix pivot could overwrite input. | Honor sorted suffix selection; parse known recording names; provide explicit positional fallback; use safe sidecar paths and create output directories. |
| Settings-file precedence | JSON silently replaced explicit CLI thresholds. | Explicit CLI choices override validated JSON objects; defaults fill unspecified values. |
| Intermediate workflow | Relative subdirectory traversal failed, bare output filenames failed, recording IDs were truncated, and `distSC` lacked its required center. | Repair recursive discovery, preserve frame indexes and full video names, calculate per-video centers, and reject unsupported brood aggregation with an actionable message. |
| Empty preliminary outputs and audits | No transitions, no contacts, or no excluded-tag detections could crash otherwise valid runs. | Handle empty concatenations and preserve index/summary schemas. |
| Contact reconstruction and exclusions | Summaries could reintroduce upstream-excluded tags or turn missing pair files into false no-contact observations. | Carry exclusions forward, recompute contacts without those tags, and reject missing/inconsistent pair data. |
| Experiment metadata and memory | Defaults silently labeled inputs as colony 65/box 05; colony-size lookup could select colony 65 for other experiments; every daily frame table was retained in memory. | Require device/colony IDs, use matching run metadata for colony-size lookup, store portable indexes, and retain compact daily summaries between dates. |
| Geometry helpers and exports | Endpoint distance used the wrong exponent; zero-length segments divided by zero; polygon IDs could combine labels; enriched pivots lacked promised activity/speed. | Correct distance arithmetic, constrain object grouping, and include activity/speed in enriched exports. |
| Setup, splitting, and plot output | Installer paths only worked on one computer, dependencies were missing, split reruns consumed generated files, and a new plot output directory failed before plotting. | Add `environment.yml`, resolve environments through Conda, skip generated split products, and create plot output directories first. |

## Validation

- Baseline: **54 tests passed**.
- Reproduced **19 failures** with the first regression set before fixes.
- Final code suite: **86 tests passed** using `QT_QPA_PLATFORM=offscreen python -m pytest -q`.
- Regression coverage includes minimal-schema cleaning/interpolation, actual CLI runs for Left/Right/Whole/custom suffixes, preservation of input files, nested intermediate processing, parallel brood analysis, distance numerics, exclusions, short recordings, no-contact recordings, summary scripts, and PNG/PDF plot generation.
- Documentation checks: bundled sample commands completed successfully and produced five bee/video summary rows; all Python files compiled; installer shell syntax passed. The manual PDF has exactly three A4 pages, checked for fit and visual readability.
- Tested environment: Python 3.13.13; NumPy 2.4.4; pandas 3.0.2; SciPy 1.17.1; Shapely 2.1.2; PyArrow 23.0.1; PySide6 6.11.0; OpenCV 4.13.0; Matplotlib 3.10.9; pytest 9.0.3.

## Scope and interpretation

This review validates software behavior on regression fixtures and small complete workflows; it does not establish biological validity for a particular experiment or certify all historical scripts. Existing `runMe*` and older analysis variants remain available, but the current main entry point is `analyze_clean_data_0.2.py`. The new environment specification was checked against installed dependencies; a fresh Conda environment was not downloaded/rebuilt.

Thresholds, FPS, calibration, and optional activity smoothing remain experimental choices. Main batch social-center calculation pools selected files, so keep a compatible colony/camera coordinate system per run. The preliminary raw-track workflow averages duplicate coordinates and can infer foraging from the rightmost arena unless `--no-infer-foraging` is supplied. Missing detections remain unknown; observed tag counts are not biological colony size.

The corrected geometry, inactivity, contact, and exclusion behavior can change earlier results. Re-run affected analyses from preserved input files and compare representative recordings before combining old and new outputs. Headless tests cover GUI logic; a complete interactive LabelMe editing session was not performed.
