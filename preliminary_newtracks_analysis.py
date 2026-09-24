#!/usr/bin/env python3
"""Preliminary frame-level analysis for copied BumbleBox *_newtracks.csv files.

Outputs are written outside the source tree:
- one compact frame-level parquet per date
- one sparse contact-pairs parquet per date
- activity transition summaries by location/contact context
- coverage and run metadata tables
"""

from __future__ import annotations

import argparse
import json
from settings_io import read_settings, parse_with_overrides
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from tqdm import tqdm

from aux import configure_behavior_timing, movement_metrics
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


TRACK_RE = re.compile(
    r"^(?P<box>bumblebox-\d+)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})_newtracks\.csv$"
)

BROOD_GROUPS = ("eggs", "larvae", "pupae", "queen_larva", "queen_pupae")
SMALL_GROUPS = (*BROOD_GROUPS, "pollen", "nectar", "wax")
PERIMETER_GROUPS = ("arena_perimeter", "foraging_perimeter", "nest_perimeter")
DISTANCE_GROUPS = (
    *PERIMETER_GROUPS,
    "brood",
    *BROOD_GROUPS,
    "pollen",
    "nectar",
    "wax",
    "any_small_object",
    "any_labeled_object",
)
ON_GROUPS = (
    "brood",
    *BROOD_GROUPS,
    "pollen",
    "nectar",
    "wax",
    "any_small_object",
    "any_labeled_object",
)


@dataclass(frozen=True)
class LabeledObject:
    object_index: int
    label: str
    shape: str
    group: str
    geometry: Any
    source: str = "label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build preliminary frame-level activity/location/contact outputs from *_newtracks.csv files."
    )
    parser.add_argument("--source", required=True, type=Path, help="Root containing copied *_newtracks.csv files.")
    parser.add_argument("--labels", required=True, type=Path, help="Folder containing LabelMe-derived nest CSVs.")
    parser.add_argument("--output", required=True, type=Path, help="Output folder for preliminary results.")
    parser.add_argument("--colony-number", required=True, help="Biological colony number to write in outputs.")
    parser.add_argument("--box-id", required=True, help="BumbleBox ID expected in filenames.")
    parser.add_argument("--limit", type=int, help="Only process the first N files after sorting.")
    parser.add_argument("--date", action="append", help="Process only this YYYY-MM-DD date; may be repeated.")
    parser.add_argument("--start-date", help="Process files on or after this YYYY-MM-DD date.")
    parser.add_argument("--end-date", help="Process files on or before this YYYY-MM-DD date.")
    parser.add_argument(
        "--exclude-bee-ids",
        nargs="*",
        type=int,
        default=[],
        help="Bee IDs to drop before activity/contact/location summaries are computed.",
    )
    parser.add_argument("--settings", type=Path, help="JSON settings file exported from review_metrics_gui.py.")
    parser.add_argument("--fps", type=float, default=frame_per_sec)
    parser.add_argument("--max-behavior-gap-sec", type=float, default=max_behavior_gap_seconds)
    parser.add_argument("--speed-cutoff", type=float, default=digital_noise_speed_cutoff)
    parser.add_argument("--max-speed-cutoff", type=float, default=max_speed_cutoff)
    parser.add_argument("--inactive-gap-max-frames", type=int, default=inactive_gap_max_frames)
    parser.add_argument("--contact-distance-px", type=float, default=interaction_distance_cutoff)
    parser.add_argument("--nest-on-distance-px", type=float, default=onDist)
    parser.add_argument("--pixels-per-cm", type=float, default=pixels_per_cm)
    parser.add_argument(
        "--no-infer-foraging",
        action="store_true",
        help="Do not infer foraging perimeter from the rightmost arena rectangle when no explicit foraging label exists.",
    )
    return parse_with_overrides(parser)


def apply_settings_file(args: argparse.Namespace) -> argparse.Namespace:
    if args.settings is None:
        return args

    path = args.settings.expanduser().resolve()
    if not path.exists():
        sys.exit(f"Settings file not found: {path}")
    try:
        data = read_settings(path)
    except Exception as exc:
        sys.exit(f"Could not read settings file {path}: {exc}")

    key_map = {
        "frame_per_sec": "fps",
        "behavior_fps": "fps",
        "fps": "fps",
        "max_behavior_gap_seconds": "max_behavior_gap_sec",
        "max_behavior_gap_sec": "max_behavior_gap_sec",
        "digital_noise_speed_cutoff": "speed_cutoff",
        "speed_cutoff": "speed_cutoff",
        "max_speed_cutoff": "max_speed_cutoff",
        "inactive_gap_max_frames": "inactive_gap_max_frames",
        "interaction_distance_cutoff": "contact_distance_px",
        "contact_distance_px": "contact_distance_px",
        "onDist": "nest_on_distance_px",
        "nest_on_distance_px": "nest_on_distance_px",
        "pixels_per_cm": "pixels_per_cm",
        "excluded_bee_ids": "exclude_bee_ids",
        "exclude_bee_ids": "exclude_bee_ids",
    }
    for settings_key, arg_name in key_map.items():
        if settings_key in data and arg_name not in getattr(args, "_explicit_options", set()):
            setattr(args, arg_name, data[settings_key])

    if isinstance(args.exclude_bee_ids, str):
        args.exclude_bee_ids = [int(bee_id) for bee_id in re.split(r"[\s,]+", args.exclude_bee_ids) if bee_id]
    else:
        args.exclude_bee_ids = [int(bee_id) for bee_id in (args.exclude_bee_ids or [])]

    args.settings = path
    print(f"Using settings file: {path}")
    return args


def track_info(path: Path) -> dict[str, str] | None:
    match = TRACK_RE.match(path.name)
    if match is None:
        return None
    info = match.groupdict()
    info["time"] = f"{info['hour']}-{info['minute']}-{info['second']}"
    info["datetime"] = f"{info['date']} {info['hour']}:{info['minute']}:{info['second']}"
    info["stem"] = path.name.removesuffix("_newtracks.csv")
    return info


def label_group(label: str) -> str:
    clean = str(label).strip().lower()
    if "foraging perimeter" in clean:
        return "foraging_perimeter"
    if "arena perimeter" in clean:
        return "arena_perimeter"
    if "nest perimeter" in clean:
        return "nest_perimeter"
    if "queen" in clean and "larv" in clean:
        return "queen_larva"
    if "queen" in clean and "pup" in clean:
        return "queen_pupae"
    if "egg" in clean:
        return "eggs"
    if "larv" in clean:
        return "larvae"
    if "pup" in clean:
        return "pupae"
    if "pollen" in clean:
        return "pollen"
    if "nectar" in clean:
        return "nectar"
    if "wax" in clean:
        return "wax"
    if "temp" in clean or "probe" in clean:
        return "temp_probe"
    return "other"


def repair_geometry(geom: Any) -> Any:
    if geom is None:
        return geom
    if getattr(geom, "is_empty", False):
        return geom
    if getattr(geom, "is_valid", True):
        return geom
    repaired = geom.buffer(0)
    return repaired if not repaired.is_empty else geom


def object_geometry(rows: pd.DataFrame) -> Any:
    shape = str(rows["shape"].iloc[0]).strip().lower()
    rows = rows.sort_values("vertex ID") if "vertex ID" in rows.columns else rows
    xy = rows[["x", "y"]].to_numpy(dtype=float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        return None

    if "circle" in shape and pd.notna(rows["radius"].iloc[0]):
        radius = float(rows["radius"].iloc[0])
        if radius <= 0 or not math.isfinite(radius):
            return Point(float(xy[0, 0]), float(xy[0, 1]))
        return Point(float(xy[0, 0]), float(xy[0, 1])).buffer(radius)

    if "rectangle" in shape and len(xy) >= 2:
        return box(float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max()))

    if "polygon" in shape and len(xy) >= 3:
        return repair_geometry(Polygon(xy))

    if len(xy) >= 3:
        return repair_geometry(Polygon(xy))
    if len(xy) == 2:
        return LineString(xy)
    return Point(float(xy[0, 0]), float(xy[0, 1]))


def load_label_objects(path: Path | None, infer_foraging: bool) -> tuple[list[LabeledObject], str]:
    if path is None or not path.exists():
        return [], "missing"

    raw = pd.read_csv(path)
    required = {"object index", "label", "shape", "x", "y"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if "radius" not in raw.columns:
        raw["radius"] = np.nan
    if "vertex ID" not in raw.columns:
        raw["vertex ID"] = np.arange(len(raw)) + 1

    for column in ("object index", "vertex ID", "x", "y", "radius"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    objects: list[LabeledObject] = []
    grouped = raw.dropna(subset=["object index", "x", "y"]).groupby(
        ["object index", "label", "shape"],
        dropna=False,
        sort=False,
    )
    for (object_index, label, shape), rows in grouped:
        if "calibration" in str(label).lower():
            continue
        geom = object_geometry(rows)
        if geom is None or getattr(geom, "is_empty", False):
            continue
        objects.append(
            LabeledObject(
                object_index=int(object_index),
                label=str(label),
                shape=str(shape),
                group=label_group(str(label)),
                geometry=geom,
            )
        )

    source_note = "explicit"
    has_foraging = any(obj.group == "foraging_perimeter" for obj in objects)
    arena_objects = [obj for obj in objects if obj.group == "arena_perimeter"]
    if infer_foraging and not has_foraging and len(arena_objects) >= 2:
        rightmost = max(arena_objects, key=lambda obj: obj.geometry.centroid.x)
        objects.append(
            LabeledObject(
                object_index=rightmost.object_index,
                label=f"{rightmost.label} [foraging inferred from rightmost arena]",
                shape=rightmost.shape,
                group="foraging_perimeter",
                geometry=rightmost.geometry,
                source="inferred_rightmost_arena",
            )
        )
        source_note = "foraging_inferred_from_rightmost_arena"
    elif not has_foraging:
        source_note = "no_foraging_perimeter"

    return objects, source_note


def group_unions(objects: list[LabeledObject]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for obj in objects:
        grouped[obj.group].append(obj.geometry)
        if obj.group in BROOD_GROUPS:
            grouped["brood"].append(obj.geometry)
        if obj.group in SMALL_GROUPS:
            grouped["any_small_object"].append(obj.geometry)
        grouped["any_labeled_object"].append(obj.geometry)

    unions = {}
    for group, geoms in grouped.items():
        geoms = [geom for geom in geoms if geom is not None and not geom.is_empty]
        if geoms:
            unions[group] = unary_union(geoms)
    return unions


def find_label_maps(labels_root: Path, box_id: str) -> dict[str, Path]:
    maps = {}
    pattern = f"{box_id}-*-nest_image.csv"
    for path in sorted(labels_root.rglob(pattern)):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
        if match:
            date = match.group(1)
            if date in maps:
                raise ValueError(f"Multiple nest maps matched {box_id} on {date}")
            maps[date] = path
    return maps


def pivot_tracking(df: pd.DataFrame) -> pd.DataFrame:
    for column in ("frame", "ID", "centroidX", "centroidY"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["frame", "ID"])
    df["frame"] = df["frame"].astype(int)
    df["ID"] = df["ID"].astype(int)
    pivot = (
        df.pivot_table(index="frame", columns="ID", values=["centroidX", "centroidY"], aggfunc="mean")
        .sort_index(axis=1)
        .sort_index()
    )
    if not pivot.empty:
        pivot = pivot.reindex(np.arange(int(pivot.index.min()), int(pivot.index.max()) + 1))
    return pivot


def bee_ids_from_pivot(pivot: pd.DataFrame) -> list[int]:
    if pivot.empty or not isinstance(pivot.columns, pd.MultiIndex):
        return []
    return sorted(int(bee) for feature, bee in pivot.columns if feature == "centroidX")


def contact_count_and_pairs(
    pivot: pd.DataFrame,
    bee_ids: list[int],
    cutoff: float,
    *,
    source_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.DataFrame(np.nan, index=pivot.index, columns=bee_ids, dtype="float64")
    pair_rows = []
    if cutoff <= 0 or len(bee_ids) == 0:
        return counts, pd.DataFrame()
    xs = pivot.loc[:, [("centroidX", bee) for bee in bee_ids]].to_numpy(dtype=float, copy=False)
    ys = pivot.loc[:, [("centroidY", bee) for bee in bee_ids]].to_numpy(dtype=float, copy=False)
    valid = np.isfinite(xs) & np.isfinite(ys)
    cutoff_sq = float(cutoff) ** 2

    for frame_idx, frame in enumerate(pivot.index):
        valid_idx = np.flatnonzero(valid[frame_idx])
        if len(valid_idx) == 0:
            continue
        if len(valid_idx) == 1:
            counts.iat[frame_idx, valid_idx[0]] = 0
            continue
        x = xs[frame_idx, valid_idx]
        y = ys[frame_idx, valid_idx]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx * dx + dy * dy
        contact = dist_sq <= cutoff_sq
        np.fill_diagonal(contact, False)
        counts.iloc[frame_idx, valid_idx] = contact.sum(axis=1)

        upper_i, upper_j = np.triu_indices(len(valid_idx), k=1)
        keep = contact[upper_i, upper_j]
        if not np.any(keep):
            continue
        for local_i, local_j in zip(upper_i[keep], upper_j[keep]):
            bee_a = int(bee_ids[valid_idx[local_i]])
            bee_b = int(bee_ids[valid_idx[local_j]])
            pair_rows.append(
                {
                    "source_file": source_file,
                    "frame": int(frame),
                    "bee_ID_1": min(bee_a, bee_b),
                    "bee_ID_2": max(bee_a, bee_b),
                    "distance_px": float(math.sqrt(dist_sq[local_i, local_j])),
                    "contact_distance_threshold_px": float(cutoff),
                }
            )
    return counts, pd.DataFrame(pair_rows)


def geometry_metrics(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    unions: dict[str, Any],
    bee_ids: list[int],
    frame_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    distance_mats = {
        group: np.full((frame_count, len(bee_ids)), np.nan, dtype="float32")
        for group in DISTANCE_GROUPS
    }
    inside_mats = {
        group: np.full((frame_count, len(bee_ids)), np.nan, dtype="float32")
        for group in PERIMETER_GROUPS
    }
    valid = np.isfinite(x_matrix) & np.isfinite(y_matrix)
    if not np.any(valid):
        return distance_mats, inside_mats

    flat_valid = valid.reshape(-1)
    flat_x = x_matrix.reshape(-1)[flat_valid]
    flat_y = y_matrix.reshape(-1)[flat_valid]
    points = shapely.points(flat_x, flat_y)

    for group in DISTANCE_GROUPS:
        geom = unions.get(group)
        if geom is None or geom.is_empty:
            continue
        values = np.full(flat_valid.shape, np.nan, dtype="float32")
        values[flat_valid] = np.asarray(shapely.distance(geom, points), dtype="float32")
        distance_mats[group] = values.reshape(frame_count, len(bee_ids))

    for group in PERIMETER_GROUPS:
        geom = unions.get(group)
        if geom is None or geom.is_empty:
            continue
        values = np.full(flat_valid.shape, np.nan, dtype="float32")
        values[flat_valid] = np.asarray(shapely.covers(geom, points), dtype="float32")
        inside_mats[group] = values.reshape(frame_count, len(bee_ids))

    return distance_mats, inside_mats


def nullable_bool(values: np.ndarray) -> pd.Series:
    series = pd.Series(values)
    valid = series.notna()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out.loc[valid] = series.loc[valid].astype(float) > 0
    return out


def location_zone(arena: pd.Series, foraging: pd.Series, nest: pd.Series, has_map: bool) -> pd.Series:
    zone = pd.Series(pd.NA, index=arena.index, dtype="object")
    if not has_map:
        return zone
    any_valid = arena.notna() | foraging.notna() | nest.notna()
    zone.loc[any_valid] = "neither"
    zone = zone.mask(arena.fillna(False).astype(bool), "arena")
    zone = zone.mask(foraging.fillna(False).astype(bool), "foraging")
    zone = zone.mask(nest.fillna(False).astype(bool), "nest")
    return zone


def frame_level_for_file(
    path: Path,
    info: dict[str, str],
    objects: list[LabeledObject],
    label_path: Path | None,
    label_source: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    pivot = pivot_tracking(df)
    bee_ids = bee_ids_from_pivot(pivot)
    excluded_bee_ids = set(getattr(args, "exclude_bee_ids", []) or [])
    if not bee_ids:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {"source_file": info["stem"], "date": info["date"], "status": "no_bees"},
        )
    analysis_bee_ids = [bee_id for bee_id in bee_ids if bee_id not in excluded_bee_ids]

    act, speed = movement_metrics(
        pivot,
        frame_rate=args.fps,
        max_gap_seconds=args.max_behavior_gap_sec,
        speed_cutoff=args.speed_cutoff,
        max_speed_cutoff=args.max_speed_cutoff,
        inactive_gap_max_frames=args.inactive_gap_max_frames,
    )
    contact_count, contact_pairs = contact_count_and_pairs(
        pivot,
        analysis_bee_ids,
        args.contact_distance_px,
        source_file=info["stem"],
    )

    x_matrix = pivot.loc[:, [("centroidX", bee) for bee in bee_ids]].to_numpy(dtype=float, copy=False)
    y_matrix = pivot.loc[:, [("centroidY", bee) for bee in bee_ids]].to_numpy(dtype=float, copy=False)
    unions = group_unions(objects)
    distances, inside = geometry_metrics(x_matrix, y_matrix, unions, bee_ids, len(pivot.index))

    rows = []
    for bee_col, bee_id in enumerate(bee_ids):
        out = pd.DataFrame(
            {
                "source_file": info["stem"],
                "box_id": info["box"],
                "colony_number": str(args.colony_number),
                "Date": info["date"],
                "Time": info["time"],
                "datetime": info["datetime"],
                "frame": pivot.index.to_numpy(dtype=int),
                "bee_ID": int(bee_id),
                "excluded_from_analysis": bool(bee_id in excluded_bee_ids),
                "centroidX": x_matrix[:, bee_col],
                "centroidY": y_matrix[:, bee_col],
                "speed_px_per_frame": speed[bee_id].to_numpy() if bee_id in speed else np.nan,
                "activity": act[bee_id].to_numpy() if bee_id in act else np.nan,
                "contact_count": contact_count[bee_id].to_numpy() if bee_id in contact_count else np.nan,
                "has_nest_map": bool(objects),
                "nest_map_path": str(label_path) if label_path else pd.NA,
                "foraging_perimeter_source": label_source,
                "activity_speed_cutoff_px_per_frame": float(args.speed_cutoff),
                "max_speed_cutoff_px_per_frame": float(args.max_speed_cutoff),
                "contact_distance_threshold_px": float(args.contact_distance_px),
                "on_distance_threshold_px": float(args.nest_on_distance_px),
                "pixels_per_cm": float(args.pixels_per_cm),
            }
        )
        social_contact = pd.Series(pd.NA, index=out.index, dtype="boolean")
        valid_contact = out["contact_count"].notna()
        social_contact.loc[valid_contact] = out.loc[valid_contact, "contact_count"] > 0
        out["social_contact"] = social_contact
        out["activity_state"] = pd.Series(np.where(out["activity"] == 1, "active", "inactive"), index=out.index)
        out.loc[out["activity"].isna(), "activity_state"] = pd.NA

        for group in DISTANCE_GROUPS:
            out[f"distance_to_{group}_px"] = distances[group][:, bee_col]
        for group in PERIMETER_GROUPS:
            out[f"inside_{group}"] = nullable_bool(inside[group][:, bee_col])
            out[f"within_{group}_threshold"] = pd.Series(
                out[f"distance_to_{group}_px"] <= float(args.nest_on_distance_px),
                dtype="boolean",
            ).where(out[f"distance_to_{group}_px"].notna(), pd.NA)
        for group in ON_GROUPS:
            out[f"on_{group}"] = pd.Series(
                out[f"distance_to_{group}_px"] <= float(args.nest_on_distance_px),
                dtype="boolean",
            ).where(out[f"distance_to_{group}_px"].notna(), pd.NA)

        out["location_zone"] = location_zone(
            out["inside_arena_perimeter"],
            out["inside_foraging_perimeter"],
            out["inside_nest_perimeter"],
            bool(objects),
        )

        small_distance_cols = [f"distance_to_{group}_px" for group in SMALL_GROUPS]
        small_distances = out[small_distance_cols]
        all_small_missing = small_distances.isna().all(axis=1)
        nearest_small = pd.Series(pd.NA, index=out.index, dtype="object")
        if (~all_small_missing).any():
            nearest_small.loc[~all_small_missing] = (
                small_distances.loc[~all_small_missing]
                .idxmin(axis=1)
                .str.removeprefix("distance_to_")
                .str.removesuffix("_px")
            )
        out["nearest_small_object_group"] = nearest_small
        out["nearest_small_object_distance_px"] = small_distances.min(axis=1, skipna=True)
        rows.append(out)

    frame_level = pd.concat(rows, ignore_index=True)
    summary = {
        "source_file": info["stem"],
        "date": info["date"],
        "frames": int(len(pivot.index)),
        "bee_ids": int(len(bee_ids)),
        "analysis_bee_ids": int(len(analysis_bee_ids)),
        "excluded_bee_ids": ",".join(str(bee_id) for bee_id in sorted(excluded_bee_ids)),
        "frame_level_rows": int(len(frame_level)),
        "detected_rows": int(frame_level["centroidX"].notna().sum()),
        "activity_rows": int(frame_level["activity"].notna().sum()),
        "contact_pair_rows": int(len(contact_pairs)),
        "has_nest_map": bool(objects),
        "nest_map_path": str(label_path) if label_path else "",
        "foraging_perimeter_source": label_source,
        "status": "ok",
    }
    return frame_level, contact_pairs, summary


def transition_counts(frame_level: pd.DataFrame) -> pd.DataFrame:
    if frame_level.empty:
        return pd.DataFrame()
    work = frame_level.sort_values(["source_file", "bee_ID", "frame"]).copy()
    if "excluded_from_analysis" in work.columns:
        work = work.loc[~work["excluded_from_analysis"].fillna(False)].copy()
        if work.empty:
            return pd.DataFrame()
    grouped = work.groupby(["source_file", "bee_ID"], sort=False)
    work["next_activity"] = grouped["activity"].shift(-1)
    work["next_frame"] = grouped["frame"].shift(-1)
    valid = (
        work["activity"].isin([0, 1])
        & work["next_activity"].isin([0, 1])
        & ((work["next_frame"] - work["frame"]) == 1)
    )
    work = work.loc[valid].copy()
    if work.empty:
        return pd.DataFrame()
    work["activity_from"] = work["activity"].astype(int)
    work["activity_to"] = work["next_activity"].astype(int)
    work["social_contact"] = work["social_contact"].astype("boolean").astype("string").fillna("unknown")
    for col in ("on_brood", "on_pollen", "on_nectar", "on_wax", "on_any_small_object"):
        work[col] = work[col].astype("boolean").astype("string").fillna("unknown")
    work["location_zone"] = work["location_zone"].fillna("unknown")
    group_cols = [
        "Date",
        "location_zone",
        "social_contact",
        "on_brood",
        "on_pollen",
        "on_nectar",
        "on_wax",
        "on_any_small_object",
        "activity_from",
        "activity_to",
    ]
    return work.groupby(group_cols, dropna=False).size().reset_index(name="n_transitions")


def summarize_transition_rates(transitions: pd.DataFrame, fps: float) -> pd.DataFrame:
    if transitions.empty:
        return transitions
    denom_cols = [c for c in transitions.columns if c not in {"activity_to", "n_transitions"}]
    out = transitions.copy()
    out["n_opportunities"] = out.groupby(denom_cols, dropna=False)["n_transitions"].transform("sum")
    out["transition_rate_per_frame"] = out["n_transitions"] / out["n_opportunities"]
    out["transition_rate_per_second"] = out["transition_rate_per_frame"] * float(fps)
    out["transition_label"] = out["activity_from"].astype(str) + "_to_" + out["activity_to"].astype(str)
    return out


def location_activity_summary(frame_level_parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not frame_level_parts:
        return pd.DataFrame()
    cols = ["Date", "location_zone", "social_contact", "activity", "contact_count"]
    kept_parts = []
    for part in frame_level_parts:
        if part.empty:
            continue
        if "excluded_from_analysis" in part.columns:
            part = part.loc[~part["excluded_from_analysis"].fillna(False)]
        if not part.empty:
            kept_parts.append(part[cols])
    if not kept_parts:
        return pd.DataFrame()
    work = pd.concat(kept_parts, ignore_index=True)
    if work.empty:
        return pd.DataFrame()
    work["social_contact"] = work["social_contact"].astype("boolean").astype("string").fillna("unknown")
    work["location_zone"] = work["location_zone"].fillna("unknown")
    grouped = work.groupby(["Date", "location_zone", "social_contact"], dropna=False)
    return grouped.agg(
        rows=("activity", "size"),
        activity_scored_rows=("activity", "count"),
        proportion_active=("activity", "mean"),
        mean_contact_count=("contact_count", "mean"),
    ).reset_index()


def merge_transition_parts(parts: list[pd.DataFrame], fps: float) -> pd.DataFrame:
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    if merged.empty:
        return pd.DataFrame()
    group_cols = [col for col in merged.columns if col != "n_transitions"]
    merged = merged.groupby(group_cols, dropna=False)["n_transitions"].sum().reset_index()
    return summarize_transition_rates(merged, fps)


def main() -> int:
    args = apply_settings_file(parse_args())
    for key in ("contact_distance_px", "nest_on_distance_px", "pixels_per_cm"):
        value = float(getattr(args, key))
        if not np.isfinite(value) or value <= 0:
            raise SystemExit(f"{key} must be finite and positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    source_root = args.source.expanduser().resolve()
    labels_root = args.labels.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame_dir = output_root / "frame_level_by_date"
    contact_dir = output_root / "contact_pairs_by_date"
    frame_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    configure_behavior_timing(
        frame_rate=args.fps,
        max_gap_seconds=args.max_behavior_gap_sec,
        speed_cutoff=args.speed_cutoff,
        max_speed_cutoff=args.max_speed_cutoff,
        inactive_gap_max_frames=args.inactive_gap_max_frames,
    )

    label_maps = find_label_maps(labels_root, args.box_id)
    requested_dates = set(args.date or [])
    files = []
    for path in sorted(source_root.rglob("*_newtracks.csv")):
        info = track_info(path)
        if info is None:
            continue
        if info["box"] != args.box_id:
            continue
        if requested_dates and info["date"] not in requested_dates:
            continue
        if args.start_date and info["date"] < args.start_date:
            continue
        if args.end_date and info["date"] > args.end_date:
            continue
        files.append(path)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("No *_newtracks.csv files found for requested filters.")
        return 1

    run_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source_root),
        "labels": str(labels_root),
        "output": str(output_root),
        "colony_number": str(args.colony_number),
        "box_id": args.box_id,
        "n_input_files": len(files),
        "settings_file": str(args.settings) if args.settings else None,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "settings": {
            "fps": args.fps,
            "max_behavior_gap_sec": args.max_behavior_gap_sec,
            "speed_cutoff": args.speed_cutoff,
            "max_speed_cutoff": args.max_speed_cutoff,
            "inactive_gap_max_frames": args.inactive_gap_max_frames,
            "contact_distance_px": args.contact_distance_px,
            "nest_on_distance_px": args.nest_on_distance_px,
            "pixels_per_cm": args.pixels_per_cm,
            "infer_foraging": not args.no_infer_foraging,
            "exclude_bee_ids": sorted(int(bee_id) for bee_id in args.exclude_bee_ids),
        },
    }
    (output_root / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    all_summaries = []
    transition_parts = []
    location_summary_parts = []
    frame_index_rows = []
    contact_index_rows = []
    label_cache: dict[str, tuple[list[LabeledObject], str]] = {}
    failures = []

    files_by_date: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        info = track_info(path)
        files_by_date[info["date"]].append(path)

    for date, date_files in tqdm(sorted(files_by_date.items()), desc="Dates"):
        label_path = label_maps.get(date)
        if date not in label_cache:
            try:
                label_cache[date] = load_label_objects(label_path, infer_foraging=not args.no_infer_foraging)
            except Exception as exc:
                failures.append({"date": date, "file": "", "error": f"label map error: {exc}"})
                label_cache[date] = ([], "label_error")
        objects, label_source = label_cache[date]

        frame_parts = []
        contact_parts = []
        for path in tqdm(date_files, desc=date, leave=False):
            info = track_info(path)
            try:
                frame_level, contact_pairs, summary = frame_level_for_file(
                    path,
                    info,
                    objects,
                    label_path,
                    label_source,
                    args,
                )
                all_summaries.append(summary)
                if not frame_level.empty:
                    frame_parts.append(frame_level)
                if not contact_pairs.empty:
                    contact_pairs["Date"] = date
                    contact_pairs["box_id"] = args.box_id
                    contact_pairs["colony_number"] = str(args.colony_number)
                    contact_parts.append(contact_pairs)
            except Exception as exc:
                failures.append({"date": date, "file": str(path), "error": str(exc)})
                all_summaries.append({"source_file": path.name, "date": date, "status": "error", "error": str(exc)})

        if frame_parts:
            date_frame = pd.concat(frame_parts, ignore_index=True)
            frame_path = frame_dir / f"{args.box_id}_{date}_frame_level.parquet"
            date_frame.to_parquet(frame_path, index=False)
            frame_index_rows.append({"Date": date, "path": str(frame_path.relative_to(output_root)), "rows": len(date_frame)})
            transition_parts.append(transition_counts(date_frame))
            location_summary_parts.append(location_activity_summary([date_frame]))

        if contact_parts:
            date_contacts = pd.concat(contact_parts, ignore_index=True)
            contact_path = contact_dir / f"{args.box_id}_{date}_contact_pairs.parquet"
            date_contacts.to_parquet(contact_path, index=False)
            contact_index_rows.append({"Date": date, "path": str(contact_path.relative_to(output_root)), "rows": len(date_contacts)})

    pd.DataFrame(all_summaries).to_csv(output_root / "video_processing_summary.csv", index=False)
    pd.DataFrame(frame_index_rows, columns=["Date", "path", "rows"]).to_csv(output_root / "frame_level_file_index.csv", index=False)
    pd.DataFrame(contact_index_rows, columns=["Date", "path", "rows"]).to_csv(output_root / "contact_pairs_file_index.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_root / "processing_failures.csv", index=False)

    transitions = merge_transition_parts(transition_parts, args.fps)
    if not transitions.empty:
        transitions.to_csv(output_root / "activity_transition_rates_by_context.csv", index=False)
        collapsed_cols = ["location_zone", "social_contact", "activity_from", "activity_to"]
        collapsed = transitions.groupby(collapsed_cols, dropna=False)["n_transitions"].sum().reset_index()
        collapsed = summarize_transition_rates(collapsed, args.fps)
        collapsed.to_csv(output_root / "activity_transition_rates_by_location_contact.csv", index=False)

    location_summary = pd.concat(location_summary_parts, ignore_index=True) if location_summary_parts else pd.DataFrame()
    if not location_summary.empty:
        location_summary.to_csv(output_root / "location_contact_activity_summary.csv", index=False)

    print(f"Processed files: {len(files)}")
    print(f"Frame-level date files: {len(frame_index_rows)}")
    print(f"Contact-pair date files: {len(contact_index_rows)}")
    print(f"Failures: {len(failures)}")
    print(f"Output: {output_root}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
