#!/usr/bin/env python3
"""
analyze_clean_data_0.2.py
────────────────────────────────────────────────────────────────────────────
Successor of runMe13.py that starts from *_clean.csv files. Produces
per-bee summaries, optional frame-level CSVs, and optional coordinate/distance pivots.

Typical call
------------
python analyze_clean_data_0.2.py \
       -s  <project_root> \
       -b  <brood_map_dir> \
       --save-pivots \
       -o  Analysis.csv \
       -c  4
"""

# ── std-lib ───────────────────────────────────────────────────────────────
import argparse, json, os, sys, re, warnings
from settings_io import read_settings, parse_with_overrides
from inspect import getmembers, isfunction
from multiprocessing import Pool
from pathlib import Path
# ── third-party ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils_io import save_df
# ── project modules ───────────────────────────────────────────────────────
import baseFunctions
import broodFunctions
import processBroodFunctions
from aux import mean_centroids_across_files
from aux import configure_behavior_timing, movement_metrics
from nest_labeling import extract_colony_and_date
import pdb
from params import colony_number_position, Date_position, H_position, M_position, S_position
from params import (
    digital_noise_speed_cutoff,
    frame_per_sec,
    inactive_gap_max_frames,
    interaction_distance_cutoff,
    max_behavior_gap_seconds,
    max_speed_cutoff,
    onDist,
    pixels_per_cm,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s','--source', default='.',
                    help='Root searched recursively for *_clean.csv')
    ap.add_argument('-e','--extension', default='_clean.csv')
    ap.add_argument('-b','--brood', type=str, default=None,
                    help='Folder that holds brood maps')
    ap.add_argument('-x','--broodExtension', default='_nest_image.csv')
    ap.add_argument('-o','--outFile', default='Analysis.csv')
    ap.add_argument('-c','--cores', type=int, default=1,
                    help='Parallel workers (default 1 = serial)')
    ap.add_argument('-l','--limit', type=int, default=None,
                    help='Process only N files (debug)')
    ap.add_argument('--save-pivots', action='store_true',
                    help='Write *_pivot_enriched.feather per video')
    ap.add_argument('--save-frame-level', action='store_true',
                    help='Write *_frame_level.csv per video with coordinates, activity, contacts, object distances, and location/on-object classifications')
    ap.add_argument('--frame-level-include-distc', action='store_true',
                    help='Also include raw distC vertex/centroid distance columns in *_frame_level.csv')
    ap.add_argument('--settings', type=str, default=None,
                    help='JSON settings file exported from review_metrics_gui.py')
    ap.add_argument('--behavior-fps', type=float, default=frame_per_sec,
                    help='Frame rate used to decide whether behavior gaps are short enough to score')
    ap.add_argument('--max-behavior-gap-sec', type=float, default=max_behavior_gap_seconds,
                    help='Maximum elapsed seconds between detections for behavior calculations')
    ap.add_argument('--speed-cutoff', type=float, default=digital_noise_speed_cutoff,
                    help='Speed threshold in px/frame for active vs inactive')
    ap.add_argument('--max-speed-cutoff', type=float, default=max_speed_cutoff,
                    help='Speeds above this px/frame value are treated as unknown; 0 disables this filter')
    ap.add_argument('--inactive-gap-max-frames', type=int, default=inactive_gap_max_frames,
                    help='Inactive runs this many frames or shorter are closed when active frames surround them')
    ap.add_argument('--interaction-distance-cutoff', type=float, default=interaction_distance_cutoff,
                    help='Bee-to-bee contact threshold in pixels')
    ap.add_argument('--nest-on-distance-px', type=float, default=onDist,
                    help='Distance threshold in pixels for a bee to be on a nest object')
    ap.add_argument('--pixels-per-cm', type=float, default=pixels_per_cm,
                    help='Pixel-to-cm scale')
    ap.add_argument('--filename-positions', action='store_true',
                    help='Use the inclusive character positions in params.py instead of automatic filename parsing')
    args = parse_with_overrides(ap)
    if args.cores < 1 or (args.limit is not None and args.limit < 1):
        ap.error("--cores and --limit must be positive")
    if not args.extension:
        ap.error("--extension must not be empty")
    return vars(args)

def apply_settings_file(opt: dict) -> dict:
    settings_path = opt.get('settings')
    if not settings_path:
        return opt

    path = Path(settings_path).expanduser()
    if not path.exists():
        sys.exit(f"❌  Settings file not found: {path}")
    try:
        data = read_settings(path)
    except Exception as exc:
        sys.exit(f"❌  Could not read settings file {path}: {exc}")

    key_map = {
        'frame_per_sec': 'behavior_fps',
        'behavior_fps': 'behavior_fps',
        'max_behavior_gap_seconds': 'max_behavior_gap_sec',
        'max_behavior_gap_sec': 'max_behavior_gap_sec',
        'digital_noise_speed_cutoff': 'speed_cutoff',
        'speed_cutoff': 'speed_cutoff',
        'max_speed_cutoff': 'max_speed_cutoff',
        'inactive_gap_max_frames': 'inactive_gap_max_frames',
        'interaction_distance_cutoff': 'interaction_distance_cutoff',
        'onDist': 'nest_on_distance_px',
        'nest_on_distance_px': 'nest_on_distance_px',
        'pixels_per_cm': 'pixels_per_cm',
        'save_frame_level': 'save_frame_level',
        'frame_level_include_distc': 'frame_level_include_distc',
    }
    for settings_key, opt_key in key_map.items():
        if settings_key in data and opt_key not in opt.get('_explicit_options', set()):
            opt[opt_key] = data[settings_key]

    print(f"Using settings file: {path}")
    return opt


def apply_runtime_settings(opt: dict):
    configure_behavior_timing(
        frame_rate=opt['behavior_fps'],
        max_gap_seconds=opt['max_behavior_gap_sec'],
        speed_cutoff=opt['speed_cutoff'],
        max_speed_cutoff=opt['max_speed_cutoff'],
        inactive_gap_max_frames=opt['inactive_gap_max_frames'],
    )

    updates = {
        'digital_noise_speed_cutoff': float(opt['speed_cutoff']),
        'max_speed_cutoff': float(opt['max_speed_cutoff']),
        'interaction_distance_cutoff': float(opt['interaction_distance_cutoff']),
        'onDist': float(opt['nest_on_distance_px']),
        'pixels_per_cm': float(opt['pixels_per_cm']),
        'inactive_gap_max_frames': int(opt['inactive_gap_max_frames']),
    }
    for module in (baseFunctions, broodFunctions):
        for name, value in updates.items():
            if hasattr(module, name):
                setattr(module, name, value)

# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════
def parse_recording_name(name: str, use_positions: bool = False):
    """Return device/colony ID, date and time; reject silently misparsed metadata."""
    from datetime import datetime
    if use_positions:
        def take(position):
            return name[position[0]:position[-1] + 1]
        device = take(colony_number_position)
        date = take(Date_position)
        time = '-'.join(take(pos) for pos in (H_position, M_position, S_position))
    else:
        match = re.search(
            r"(?:colony|col|bumblebox|worker)[-_ ]*(\d+)[-_](\d{4}[-_]\d{2}[-_]\d{2})"
            r"[-_](\d{2})[-_](\d{2})[-_](\d{2})", name, re.IGNORECASE)
        if match is None:
            raise ValueError(f"Cannot parse device/date/time from {name}; use a supported name or --filename-positions with params.py")
        device, date, *clock = match.groups()
        date = date.replace('_', '-')
        time = '-'.join(clock)
    int(device)
    datetime.strptime(f"{date} {time}", "%Y-%m-%d %H-%M-%S")
    return device, date, time


def pivot_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: MultiIndex (coord, ID) × frame."""
    return (df.pivot_table(index='frame', columns='ID',
                           values=['centroidX','centroidY'])
              .sort_index(axis=1)
              .apply(pd.to_numeric, errors='coerce'))

def _extract_colony_and_date(name: str):
    """Parse colony number and YYYY-MM-DD date from a tracking or brood filename."""
    return extract_colony_and_date(name)


def _find_brood_map(stem: str, root: str, ext: str):
    """Find a unique brood map matching the same colony and date as *stem*."""
    target = _extract_colony_and_date(stem)
    if target is None:
        raise ValueError(f"Could not parse colony/date from tracking stem: {stem}")

    root_path = Path(root)
    if not root_path.exists():
        return None

    ext_clean = ext.lstrip("_-")
    valid_suffixes = {
        ext_clean,
        f"_{ext_clean}",
        f"-{ext_clean}",
    }

    matches = []
    for path in root_path.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if not any(path.name.endswith(suffix) for suffix in valid_suffixes):
            continue

        parsed = _extract_colony_and_date(path.name)
        if parsed == target:
            matches.append(path)

    if not matches:
        return None

    matches = sorted(matches)
    if len(matches) > 1:
        match_list = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Multiple brood maps matched colony/date {target[0]} on {target[1]}: {match_list}"
        )

    return str(matches[0])


def _analysis_brood_objects(full: pd.DataFrame) -> pd.DataFrame:
    """Return labeled objects that should contribute to analysis distances."""
    if "label" not in full.columns:
        return full.copy()

    labels = full["label"].astype(str)
    exclude = labels.str.contains("calibration", case=False, na=False)
    return full.loc[~exclude].copy()


def _safe_column_name(text) -> str:
    clean = re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_").lower()
    return clean or "value"


def _object_group(label: str) -> str:
    label_lower = str(label).lower()
    if "arena perimeter" in label_lower:
        return "arena_perimeter"
    if "foraging perimeter" in label_lower:
        return "foraging_perimeter"
    if "nest perimeter" in label_lower:
        return "nest_perimeter"
    if "pollen" in label_lower:
        return "pollen"
    if "nectar" in label_lower:
        return "nectar"
    if "queen" in label_lower and any(term in label_lower for term in ("larvae", "larva")):
        return "queen_larva"
    if "queen" in label_lower and any(term in label_lower for term in ("pupae", "pupa")):
        return "queen_pupae"
    if "egg" in label_lower:
        return "eggs"
    if any(term in label_lower for term in ("larvae", "larva")):
        return "larvae"
    if any(term in label_lower for term in ("pupae", "pupa")):
        return "pupae"
    if "wax" in label_lower:
        return "wax"
    if "temp probe" in label_lower:
        return "temp_probe"
    if "calibration" in label_lower:
        return "calibration"
    return "other"


def _distance_feature_info(feature):
    match = re.match(r"^(dist[CM])_(.+)_(\d+)(?:_[A-Za-z]\d+)?$", str(feature))
    if match is None:
        return None

    label = match.group(2)
    return {
        "kind": match.group(1),
        "label": label,
        "object_index": int(match.group(3)),
        "group": _object_group(label),
    }


def _distance_features(pivot: pd.DataFrame, kind: str) -> tuple[list[str], dict[str, dict]]:
    features = []
    meta = {}
    if not isinstance(pivot.columns, pd.MultiIndex):
        return features, meta

    seen = set()
    for feature in pivot.columns.get_level_values(0):
        if feature in seen:
            continue
        info = _distance_feature_info(feature)
        if info is None or info["kind"] != kind:
            continue
        seen.add(feature)
        features.append(feature)
        meta[feature] = info
    return features, meta


def _bee_ids(pivot: pd.DataFrame) -> list:
    if not isinstance(pivot.columns, pd.MultiIndex):
        return []
    return list(dict.fromkeys(bee for feature, bee in pivot.columns if feature == "centroidX"))


def _series_for_bee(pivot: pd.DataFrame, feature: str, bee_id):
    if (feature, bee_id) in pivot.columns:
        return pd.to_numeric(pivot[(feature, bee_id)], errors="coerce")
    return pd.Series(np.nan, index=pivot.index, dtype="float64")


def _distance_table_for_bee(pivot: pd.DataFrame, features: list[str], bee_id) -> pd.DataFrame:
    cols = [(feature, bee_id) for feature in features if (feature, bee_id) in pivot.columns]
    if not cols:
        return pd.DataFrame(index=pivot.index)
    table = pivot.loc[:, cols].copy()
    table.columns = [feature for feature, _bee in cols]
    return table.apply(pd.to_numeric, errors="coerce")


def _min_distance(distances: pd.DataFrame, meta: dict[str, dict], groups: set[str]) -> pd.Series:
    cols = [feature for feature in distances.columns if meta[feature]["group"] in groups]
    if not cols:
        return pd.Series(np.nan, index=distances.index, dtype="float64")
    return distances.loc[:, cols].min(axis=1, skipna=True)


def _nearest_object(distances: pd.DataFrame, meta: dict[str, dict], groups: set[str] | None = None):
    cols = [
        feature for feature in distances.columns
        if groups is None or meta[feature]["group"] in groups
    ]
    empty_label = pd.Series(pd.NA, index=distances.index, dtype="object")
    empty_feature = pd.Series(pd.NA, index=distances.index, dtype="object")
    empty_object = pd.Series(pd.NA, index=distances.index, dtype="object")
    empty_distance = pd.Series(np.nan, index=distances.index, dtype="float64")
    if not cols:
        return empty_feature, empty_label, empty_object, empty_distance

    subset = distances.loc[:, cols]
    nearest_distance = subset.min(axis=1, skipna=True)
    nearest_feature = subset.idxmin(axis=1, skipna=True).where(nearest_distance.notna(), pd.NA)
    nearest_label = nearest_feature.map(
        lambda feature: meta[feature]["label"] if pd.notna(feature) else pd.NA
    )
    nearest_object_index = nearest_feature.map(
        lambda feature: meta[feature]["object_index"] if pd.notna(feature) else pd.NA
    )
    return nearest_feature, nearest_label, nearest_object_index, nearest_distance


def _inside_from_distance(distance: pd.Series) -> pd.Series:
    inside = pd.Series(pd.NA, index=distance.index, dtype="boolean")
    valid = distance.notna()
    inside.loc[valid] = np.isclose(distance.loc[valid], 0)
    return inside


def _on_from_distance(distance: pd.Series, threshold: float) -> pd.Series:
    on_object = pd.Series(pd.NA, index=distance.index, dtype="boolean")
    valid = distance.notna()
    on_object.loc[valid] = distance.loc[valid] <= threshold
    return on_object


def _location_zone(frame_level: pd.DataFrame) -> pd.Series:
    zone = pd.Series("outside_annotated_perimeters", index=frame_level.index, dtype="object")
    known = pd.Series(False, index=frame_level.index)
    for column in ("inside_arena_perimeter", "inside_foraging_perimeter", "inside_nest_perimeter"):
        if column in frame_level:
            known = known | frame_level[column].notna()

    zone = zone.where(known, "unknown")
    if "inside_arena_perimeter" in frame_level:
        zone = zone.mask(frame_level["inside_arena_perimeter"].fillna(False).astype(bool), "arena")
    if "inside_foraging_perimeter" in frame_level:
        zone = zone.mask(frame_level["inside_foraging_perimeter"].fillna(False).astype(bool), "foraging")
    if "inside_nest_perimeter" in frame_level:
        zone = zone.mask(frame_level["inside_nest_perimeter"].fillna(False).astype(bool), "nest")
    return zone


def _contact_count_table(pivot: pd.DataFrame, bee_ids: list, cutoff: float) -> pd.DataFrame:
    counts = pd.DataFrame(np.nan, index=pivot.index, columns=bee_ids, dtype="float64")
    if cutoff <= 0 or len(bee_ids) == 0:
        return counts
    if len(bee_ids) == 1:
        valid = (_series_for_bee(pivot, "centroidX", bee_ids[0]).notna()
                 & _series_for_bee(pivot, "centroidY", bee_ids[0]).notna())
        counts.loc[valid, bee_ids[0]] = 0
        return counts

    xs = pivot.loc[:, [("centroidX", bee) for bee in bee_ids]].to_numpy(dtype=float, copy=False)
    ys = pivot.loc[:, [("centroidY", bee) for bee in bee_ids]].to_numpy(dtype=float, copy=False)
    valid = np.isfinite(xs) & np.isfinite(ys)
    cutoff_sq = float(cutoff) ** 2
    chunk_size = max(1, min(20000, 2_000_000 // max(1, len(bee_ids) ** 2)))

    for start in range(0, len(pivot.index), chunk_size):
        end = min(start + chunk_size, len(pivot.index))
        x_chunk = xs[start:end]
        y_chunk = ys[start:end]
        valid_chunk = valid[start:end]
        dx = x_chunk[:, :, None] - x_chunk[:, None, :]
        dy = y_chunk[:, :, None] - y_chunk[:, None, :]
        pair_valid = valid_chunk[:, :, None] & valid_chunk[:, None, :]
        contact = ((dx * dx + dy * dy) <= cutoff_sq) & pair_valid
        diag = np.arange(len(bee_ids))
        contact[:, diag, diag] = False
        chunk_counts = contact.sum(axis=2).astype("float64")
        chunk_counts[~valid_chunk] = np.nan
        counts.iloc[start:end, :] = chunk_counts

    return counts


def _sidecar_path(fpath: str, extension: str, suffix: str) -> Path:
    path = Path(fpath)
    name = path.name
    if extension and name.endswith(extension):
        return path.with_name(f"{name[:-len(extension)]}{suffix}")
    return path.with_name(f"{path.stem}{suffix}")


def frame_level_table(
    pivot: pd.DataFrame,
    opt: dict,
    source_file: str,
    worker: str,
    date: str,
    time: str,
    lr: str,
) -> pd.DataFrame:
    bee_ids = _bee_ids(pivot)
    if not bee_ids:
        return pd.DataFrame()

    act, speed = movement_metrics(
        pivot,
        frame_rate=opt["behavior_fps"],
        max_gap_seconds=opt["max_behavior_gap_sec"],
        speed_cutoff=opt["speed_cutoff"],
        max_speed_cutoff=opt["max_speed_cutoff"],
        inactive_gap_max_frames=opt["inactive_gap_max_frames"],
    )
    contact_count = _contact_count_table(pivot, bee_ids, opt["interaction_distance_cutoff"])

    distm_features, distm_meta = _distance_features(pivot, "distM")
    distc_features, _distc_meta = _distance_features(pivot, "distC")
    brood_groups = {"eggs", "larvae", "pupae", "queen_larva", "queen_pupae"}
    small_groups = {*brood_groups, "pollen", "nectar", "wax"}
    all_groups = {
        "arena_perimeter",
        "foraging_perimeter",
        "nest_perimeter",
        *brood_groups,
        "pollen",
        "nectar",
        "wax",
        "temp_probe",
        "other",
    }

    rows = []
    for bee_id in bee_ids:
        frame_level = pd.DataFrame({
            "source_file": source_file,
            "pi_ID": worker,
            "Date": date,
            "Time": time,
            "LR": lr,
            "frame": pivot.index,
            "bee_ID": bee_id,
            "centroidX": _series_for_bee(pivot, "centroidX", bee_id).to_numpy(),
            "centroidY": _series_for_bee(pivot, "centroidY", bee_id).to_numpy(),
            "speed_px_per_frame": speed[bee_id].to_numpy() if bee_id in speed else np.nan,
            "activity": act[bee_id].to_numpy() if bee_id in act else np.nan,
            "contact_count": contact_count[bee_id].to_numpy() if bee_id in contact_count else np.nan,
        })
        social_contact = pd.Series(pd.NA, index=frame_level.index, dtype="boolean")
        valid_contact = frame_level["contact_count"].notna()
        social_contact.loc[valid_contact] = frame_level.loc[valid_contact, "contact_count"] > 0
        frame_level["social_contact"] = social_contact
        frame_level["activity_speed_cutoff_px_per_frame"] = opt["speed_cutoff"]
        frame_level["max_speed_cutoff_px_per_frame"] = opt["max_speed_cutoff"]
        frame_level["contact_distance_threshold_px"] = opt["interaction_distance_cutoff"]
        frame_level["on_distance_threshold_px"] = opt["nest_on_distance_px"]

        distm = _distance_table_for_bee(pivot, distm_features, bee_id)
        frame_level["distance_to_arena_perimeter_px"] = _min_distance(distm, distm_meta, {"arena_perimeter"}).to_numpy()
        frame_level["distance_to_foraging_perimeter_px"] = _min_distance(distm, distm_meta, {"foraging_perimeter"}).to_numpy()
        frame_level["distance_to_nest_perimeter_px"] = _min_distance(distm, distm_meta, {"nest_perimeter"}).to_numpy()
        frame_level["distance_to_brood_px"] = _min_distance(distm, distm_meta, brood_groups).to_numpy()
        for name in ("eggs", "larvae", "pupae", "queen_larva", "queen_pupae"):
            frame_level[f"distance_to_{name}_px"] = _min_distance(distm, distm_meta, {name}).to_numpy()
        frame_level["distance_to_pollen_px"] = _min_distance(distm, distm_meta, {"pollen"}).to_numpy()
        frame_level["distance_to_nectar_px"] = _min_distance(distm, distm_meta, {"nectar"}).to_numpy()
        frame_level["distance_to_wax_px"] = _min_distance(distm, distm_meta, {"wax"}).to_numpy()
        frame_level["distance_to_any_small_object_px"] = _min_distance(distm, distm_meta, small_groups).to_numpy()
        frame_level["distance_to_any_labeled_object_px"] = _min_distance(distm, distm_meta, all_groups).to_numpy()

        for name in ("arena_perimeter", "foraging_perimeter", "nest_perimeter"):
            dist_col = f"distance_to_{name}_px"
            frame_level[f"inside_{name}"] = _inside_from_distance(frame_level[dist_col])
            frame_level[f"within_{name}_threshold"] = _on_from_distance(
                frame_level[dist_col],
                opt["nest_on_distance_px"],
            )

        for name in (
            "brood",
            "eggs",
            "larvae",
            "pupae",
            "queen_larva",
            "queen_pupae",
            "pollen",
            "nectar",
            "wax",
            "any_small_object",
            "any_labeled_object",
        ):
            frame_level[f"on_{name}"] = _on_from_distance(
                frame_level[f"distance_to_{name}_px"],
                opt["nest_on_distance_px"],
            )

        frame_level["location_zone"] = _location_zone(frame_level)

        nearest_feature, nearest_label, nearest_object, nearest_distance = _nearest_object(
            distm,
            distm_meta,
            all_groups,
        )
        frame_level["nearest_labeled_object_feature"] = nearest_feature.to_numpy()
        frame_level["nearest_labeled_object_label"] = nearest_label.to_numpy()
        frame_level["nearest_labeled_object_index"] = nearest_object.to_numpy()
        frame_level["nearest_labeled_object_distance_px"] = nearest_distance.to_numpy()

        nearest_feature, nearest_label, nearest_object, nearest_distance = _nearest_object(
            distm,
            distm_meta,
            small_groups,
        )
        frame_level["nearest_small_object_feature"] = nearest_feature.to_numpy()
        frame_level["nearest_small_object_label"] = nearest_label.to_numpy()
        frame_level["nearest_small_object_index"] = nearest_object.to_numpy()
        frame_level["nearest_small_object_distance_px"] = nearest_distance.to_numpy()

        used_names = set(frame_level.columns)
        for feature in distm.columns:
            raw_name = _safe_column_name(feature)
            if raw_name in used_names:
                raw_name = f"raw_{raw_name}"
            used_names.add(raw_name)
            frame_level[raw_name] = distm[feature].to_numpy()

        if opt.get("frame_level_include_distc"):
            distc = _distance_table_for_bee(pivot, distc_features, bee_id)
            for feature in distc.columns:
                raw_name = _safe_column_name(feature)
                if raw_name in used_names:
                    raw_name = f"raw_{raw_name}"
                used_names.add(raw_name)
                frame_level[raw_name] = distc[feature].to_numpy()

        rows.append(frame_level)

    return pd.concat(rows, ignore_index=True)



#    print(len(hits))
#    if not hits:
#        print("Error: Stem string for the brood map path couldnt be found")
#        return None
#    else:
#        for hit in hits:
#            if hit.endswith(ext):
#                return hit

def processBrood_test(basename, oneLR, LR, brood_dir, brood_ext):
    """Attach brood-distance matrices (if map present)."""

    # --- locate brood map file ---
    p = Path(brood_dir)
    if p.is_file() and p.suffix.lower() == ".csv":
        mp = brood_dir
    else:
        stem = basename
        mp = _find_brood_map(stem, brood_dir, brood_ext)
        if mp is None:
            raise FileNotFoundError(f"No brood map matched {basename} in {brood_dir}")
        print(type(mp))
        print(mp)

    full = pd.read_csv(mp)

    # --- Apply left/right filtering ---
    THR = 2000
    if LR == 'Left':
        full = full[full['x'] < THR]
    elif LR == 'Right':
        full = full[full['x'] > THR]

    # Keep Arena/Foraging/Nest perimeters for location analysis; only
    # calibration helper labels are excluded from behavior distances.
    brood = _analysis_brood_objects(full)

    # --- Partition brood objects cleanly ---
    circles = brood[
        brood["shape"].str.contains("circle", case=False) &
        brood["radius"].notna()
    ].copy()

    polygons = brood[
        brood["shape"].str.contains("polygon", case=False)
    ].copy()

    # Everything else (e.g. rectangles) appears only in centroid-based distances
    # They do *not* go into circle or polygon distance calculations.
    other = brood[
        ~brood.index.isin(circles.index) &
        ~brood.index.isin(polygons.index)
    ].copy()

    # --- Combine for centroid-based distances ---
    # Centroid-func can safely handle all brood objects
    all_for_centroid = pd.concat([circles, polygons, other], ignore_index=True)

    #print(f"[processBrood] circles:  {circles.shape}")
    #print(f"[processBrood] polygons: {polygons.shape}")
    #print(f"[processBrood] other:    {other.shape}")
    #print(f"[processBrood] centroid total: {all_for_centroid.shape}")

    # --- Call each distance method with correct subset ---
    d1 = processBroodFunctions.distanceFromCentroid_new(oneLR, all_for_centroid)
    #print(f"[processBrood] d1 (centroid) shape: {d1.shape}")

    d2 = processBroodFunctions.minimumDistanceCircle_new(circles, oneLR)
    #print(f"[processBrood] d2 (circle) shape:   {d2.shape}")

    d3 = processBroodFunctions.minimumDistancePolygon_new(oneLR, polygons)
    #print(f"[processBrood] d3 (polygon) shape:  {d3.shape}")

    # --- Combine all ---
    return pd.concat([oneLR, d1, d2, d3], axis=1)


def processBrood(basename, oneLR, LR, brood_dir, brood_ext):
    """Attach brood-distance matrices (if map present)."""
    stem = basename



    mp = _find_brood_map(stem, brood_dir, brood_ext)
    if mp is None:
        print("ERROR: Brood map could not be found")
        #raise TypeError
        return oneLR

    full = pd.read_csv(mp)
    THR = 2000
    if LR == 'Left':  full = full[full['x'] < THR]
    elif LR == 'Right': full = full[full['x'] > THR]

    brood = _analysis_brood_objects(full).reset_index(drop=True)
    #eggs  = brood[brood['radius'].isna()]
    #allb  = brood.dropna(subset=['radius'])
    
    circles = brood[
        brood["shape"].str.contains("circle", case=False) 
        & brood["radius"].notna()
        ].reset_index(drop=True)
    
    polygons = brood[
    brood["shape"].str.contains("polygon", case=False)
    ].reset_index(drop=True)
    

    #print(f"allb shape: {allb.shape}")
    #for df in (brood, eggs, allb):
    #    df.reset_index(drop=True, inplace=True)

    d1 = processBroodFunctions.distanceFromCentroid(oneLR, brood)
    #print(f"d1 shape: {d1.shape}")
    d2 = processBroodFunctions.minimumDistanceCircle(circles, oneLR)
    #print(f"d2 shape: {d2.shape}")
    d3 = processBroodFunctions.minimumDistancePolygon(oneLR, polygons)
    #print(f"d3 shape: {d3.shape}")
    #pdb.set_trace()
    return pd.concat([oneLR, d1, d2, d3], axis=1)

# ══════════════════════════════════════════════════════════════════════════
#  Per-file job
# ══════════════════════════════════════════════════════════════════════════
def analyse_one(fpath, opt, funcs, social_center):
    base = os.path.basename(fpath)
    df = pd.read_csv(fpath, dtype={
        "frame":"int32","ID":"int32",
        "centroidX":"float32","centroidY":"float32",
        "frontX":"float32","frontY":"float32"})
    
    worker, Date, time = parse_recording_name(base, opt.get('filename_positions', False))
    H, M, S = time.split('-')

    LRm = re.search(r'_(Left|Right|Whole)_', base)
    LR  = LRm.group(1) if LRm else 'Whole'

    pivot = pivot_clean(df)              # build wide table
    #print(fpath)
    #print(pivot.head)

    if opt['brood']:
        pivot = processBrood_test(base, pivot, LR,
                             brood_dir=opt['brood'],
                             brood_ext=opt['broodExtension'])
        
        #dupes = pivot.columns[pivot.columns.duplicated()]
        #print("Duplicated column names:")
        #for d in dupes:
        #    print(d)

        #print("Before reset_index: ", pivot.columns.nlevels)
        #print("Are column names unique before reset?", pivot.columns.is_unique)
        #print("Are column names unique after reset?", pivot.reset_index().columns.is_unique)
        


    bee_ids = pivot.columns.levels[1]
    if not len(bee_ids):
        return None

    if opt['save_frame_level']:
        frame_level = frame_level_table(
            pivot=pivot,
            opt=opt,
            source_file=base,
            worker=worker,
            date=Date,
            time=f"{H}-{M}-{S}",
            lr=LR,
        )
        if not frame_level.empty:
            frame_level_path = _sidecar_path(fpath, opt['extension'], '_frame_level.csv')
            frame_level.to_csv(frame_level_path, index=False)
            print(f"Frame-level file: {frame_level_path}")

    out = pd.DataFrame(index=bee_ids)
    out['pi_ID'] = worker
    out['bee_ID'] = bee_ids
    out['Date']   = Date
    out['Time']   = f"{H}-{M}-{S}"
    out['LR']     = LR

    for name, fn in funcs:

        if name == "distSC":
            try:
                res = fn(pivot, social_center)
                if isinstance(res, (list, np.ndarray)):
                    res = pd.Series(res, index=bee_ids)
                if isinstance(res, pd.Series):
                    out[name] = res.reindex(bee_ids)
                else:
                    out[name] = res
            except Exception as e:
                print(f"⚠ {name} failed on {base}: {e}")
                out[name] = np.nan

        else:
            try:
                res = fn(pivot)
                if isinstance(res, (list, np.ndarray)):
                    res = pd.Series(res, index=bee_ids)
                if isinstance(res, pd.Series):
                    out[name] = res.reindex(bee_ids)
                else:
                    out[name] = res
            except Exception as e:
                print(f"⚠ {name} failed on {base}: {e}")
                out[name] = np.nan

    if opt['save_pivots']:
        act, speed = movement_metrics(pivot)
        enriched = pd.concat([pivot, pd.concat({'activity': act, 'speed': speed}, axis=1)], axis=1)
        feather = _sidecar_path(fpath, opt['extension'], '_pivot_enriched.feather')
        
        #raise Exception
        enriched.to_feather(feather)
        print(f"Feather file: {feather}")
    return out

def job(arg):
    _fpath, opt, _funcs, _social_center = arg
    apply_runtime_settings(opt)
    return analyse_one(*arg)

# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    opt = cli()
    opt = apply_settings_file(opt)
    apply_runtime_settings(opt)
    for key in ('pixels_per_cm', 'interaction_distance_cutoff', 'nest_on_distance_px'):
        if not np.isfinite(float(opt[key])) or float(opt[key]) <= 0:
            sys.exit(f"{key} must be finite and positive")
    print(
        "Behavior settings: "
        f"fps={opt['behavior_fps']}, "
        f"speed_cutoff={opt['speed_cutoff']} px/frame, "
        f"max_speed={opt['max_speed_cutoff']} px/frame, "
        f"max_gap={opt['max_behavior_gap_sec']} sec, "
        f"inactive_blip_max={opt['inactive_gap_max_frames']} frames"
    )
    print(
        "Distance settings: "
        f"bee_contact={opt['interaction_distance_cutoff']} px, "
        f"nest_on={opt['nest_on_distance_px']} px, "
        f"pixels_per_cm={opt['pixels_per_cm']}"
    )

    funcs = [(n, f) for n, f in getmembers(baseFunctions)
             if isfunction(f) and f.__module__ == 'baseFunctions']
    if opt['brood']:
        funcs += [(n, f) for n, f in getmembers(broodFunctions)
                  if isfunction(f) and f.__module__ == 'broodFunctions']

    #print(Path(opt['source']))

    files = sorted(str(p) for p in Path(opt['source']).rglob(f"*{opt['extension']}") if p.is_file())
    if opt['limit']:
        files = files[:opt['limit']]
    if not files:
        sys.exit(f"No files ending with {opt['extension']} found under {opt['source']}")
    # Validate metadata before writing any output.
    for path in files:
        parse_recording_name(Path(path).name, opt.get('filename_positions', False))

    print("Calculating the social center from your data now...")
    mean_sc_dict = mean_centroids_across_files(files, chunksize=None)
    mean_x = mean_sc_dict["mean_centroidX"]
    mean_y = mean_sc_dict["mean_centroidY"]
    social_center = [mean_x, mean_y]

    it = [(fp, opt, funcs, social_center) for fp in files]
    if opt['cores'] > 1:
        with Pool(opt['cores']) as pool:
            results = list(tqdm(pool.imap_unordered(job, it),
                                total=len(files), desc="Files"))
    else:
        results = [job(a) for a in tqdm(it, desc="Files")]

    results = [r for r in results if r is not None]
    if not results:
        sys.exit("Nothing processed!")

    combined = pd.concat(results, ignore_index=True)
    save_df(combined, opt['outFile'])
    print(f"✅  Saved {opt['outFile']}   ({len(combined)} bee-video rows)")

if __name__ == "__main__":
    main()
