#!/usr/bin/env python3
"""Interactive metric review hub for BumbleBox tracking videos."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QProcess, QSettings, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGraphicsRectItem,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from aux import mean_centroids_across_files, movement_metrics
from nest_labeling import (
    LABEL_COLORS_BGR,
    NEST_IMAGE_EXTENSIONS,
    convert_labelme_json_to_csv,
    extract_colony_and_date,
    find_matching_brood_csvs,
    find_matching_nest_images,
)
from params import (
    digital_noise_speed_cutoff,
    frame_per_sec,
    interaction_distance_cutoff,
    max_behavior_gap_seconds,
    pixels_per_cm,
)


VIDEO_EXTENSIONS = {".mp4", ".mjpeg", ".avi", ".mov", ".mkv"}
REQUIRED_TRACKING_COLUMNS = {"frame", "ID", "centroidX", "centroidY"}
REQUIRED_NOID_COLUMNS = {"frame", "centroidX", "centroidY"}
SETTINGS_ORG = "BuzzAnalysis"
SETTINGS_APP = "MetricReviewHub"
LAST_SOURCE_PATH_KEY = "last_source/path"
LAST_SOURCE_RANDOMIZE_KEY = "last_source/randomize"
NOID_TAG_COLOR = QColor("#ff4fb8")
NEST_MAP_LINE_WIDTH = 6
TRAIL_LINE_WIDTH = 4
NEST_INTERACTION_LINE_WIDTH = 4
NEST_DISTANCE_LINE_WIDTH = 6
NEST_INTERACTION_HIGHLIGHT_ALPHA = 128
PLOT_LINE_WIDTH = 4
PLOT_SELECTED_LINE_WIDTH = 7


@dataclass(frozen=True)
class ReviewSession:
    video_path: Path
    tracking_paths: tuple[Path, ...]
    source_root: Path | None = None
    related_csv_paths: tuple[Path, ...] = ()

    @property
    def label(self) -> str:
        if self.tracking_paths:
            return f"{self.video_path.name} ({len(self.tracking_paths)} compatible CSV)"
        if self.related_csv_paths:
            return f"{self.video_path.name} ({len(self.related_csv_paths)} related CSV, none compatible)"
        return f"{self.video_path.name} (missing CSV)"


@dataclass(frozen=True)
class NestMapStatus:
    target: tuple[int, str] | None
    roots: tuple[Path, ...]
    csv_path: Path | None
    image_path: Path | None
    json_path: Path | None
    state: str
    message: str
    csv_matches: tuple[Path, ...] = ()
    image_matches: tuple[Path, ...] = ()


@dataclass(frozen=True)
class NestComponent:
    label: str
    shape: str
    points: tuple[tuple[float, float], ...]
    radius: float = 0.0


def stable_color(tag: int | str) -> QColor:
    rng = random.Random(int(tag))
    return QColor(rng.randint(55, 235), rng.randint(55, 235), rng.randint(55, 235))


def missing_tracking_columns(path: Path) -> tuple[str, ...]:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return tuple(sorted(REQUIRED_TRACKING_COLUMNS))
    return tuple(sorted(REQUIRED_TRACKING_COLUMNS - set(header.columns)))


def is_tracking_csv(path: Path) -> bool:
    return not missing_tracking_columns(path)


def missing_noid_columns(path: Path) -> tuple[str, ...]:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return tuple(sorted(REQUIRED_NOID_COLUMNS))
    return tuple(sorted(REQUIRED_NOID_COLUMNS - set(header.columns)))


def is_noid_tag_csv(path: Path) -> bool:
    return "noid" in path.stem.lower() and not missing_noid_columns(path)


def _csv_priority(path: Path, stem: str) -> tuple[int, str]:
    name = path.name
    priorities = [
        f"{stem}_Whole_clean.csv",
        f"{stem}_Left_clean.csv",
        f"{stem}_Right_clean.csv",
    ]
    try:
        return priorities.index(name), str(path)
    except ValueError:
        pass
    if name.startswith(f"{stem}_") and name.endswith(("_clean.csv", "_cleaned.csv")):
        return 10, str(path)
    if name == f"{stem}.csv":
        return 20, str(path)
    if name.startswith(f"{stem}_") and "interpolated" in name.lower():
        return 30, str(path)
    if name.startswith(f"{stem}_") and "newtracks" in name.lower():
        return 40, str(path)
    if name.startswith(f"{stem}_") and "raw" in name.lower():
        return 50, str(path)
    if name.startswith(f"{stem}_") and "noid" in name.lower():
        return 60, str(path)
    return 99, str(path)


def find_related_csvs(video_path: Path, root: Path | None = None) -> tuple[Path, ...]:
    stem = video_path.stem
    search_roots = [video_path.parent, video_path.parent / stem]
    if root is not None:
        search_roots.append(root)

    candidates: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*.csv"):
            name = path.name
            exact_match = name == f"{stem}.csv"
            stem_prefix_match = name.startswith(f"{stem}_")
            if exact_match or stem_prefix_match:
                candidates.add(path)

    return tuple(sorted(candidates, key=lambda path: _csv_priority(path, stem)))


def find_tracking_csvs(video_path: Path, root: Path | None = None) -> tuple[Path, ...]:
    return tuple(path for path in find_related_csvs(video_path, root) if is_tracking_csv(path))


def find_noid_tag_csvs(session: ReviewSession) -> tuple[Path, ...]:
    return tuple(path for path in session.related_csv_paths if is_noid_tag_csv(path))


def tracking_csv_pattern(video_path: Path, csv_path: Path) -> str:
    """Return the reusable CSV filename pattern relative to a video stem."""
    stem = video_path.stem
    name = csv_path.name
    if name == f"{stem}.csv":
        return ".csv"
    if name.startswith(stem):
        return name[len(stem):]
    return name


def tracking_csv_pattern_label(pattern: str) -> str:
    if pattern == ".csv":
        return "<video name>.csv"
    if pattern.startswith(("_", "-", ".")):
        return f"<video name>{pattern}"
    return pattern


def find_tracking_csv_by_pattern(session: ReviewSession, pattern: str) -> Path | None:
    for path in session.tracking_paths:
        if tracking_csv_pattern(session.video_path, path) == pattern:
            return path
    return None


def _unique_existing_roots(paths: Iterable[Path | None]) -> tuple[Path, ...]:
    roots = []
    for path in paths:
        if path is None:
            continue
        path = path.expanduser().resolve()
        if path.is_file():
            path = path.parent
        if path.exists() and path not in roots:
            roots.append(path)
    return tuple(roots)


def session_colony_date(session: ReviewSession) -> tuple[int, str] | None:
    for path in (*session.tracking_paths, session.video_path):
        parsed = extract_colony_and_date(path.name)
        if parsed is not None:
            return parsed
    return None


def session_search_roots(session: ReviewSession) -> tuple[Path, ...]:
    return _unique_existing_roots(
        [
            session.source_root,
            session.video_path.parent,
            *(path.parent for path in session.tracking_paths),
        ]
    )


def assess_nest_map_status(session: ReviewSession) -> NestMapStatus:
    target = session_colony_date(session)
    roots = session_search_roots(session)
    if target is None:
        return NestMapStatus(
            target=None,
            roots=roots,
            csv_path=None,
            image_path=None,
            json_path=None,
            state="unknown-id",
            message="Could not parse colony/date from this session.",
        )

    csv_matches = tuple(
        sorted({path for root in roots for path in find_matching_brood_csvs(root, target)})
    )
    image_matches = tuple(
        sorted({path for root in roots for path in find_matching_nest_images(root, target)})
    )

    if len(csv_matches) > 1:
        return NestMapStatus(
            target=target,
            roots=roots,
            csv_path=None,
            image_path=image_matches[0] if image_matches else None,
            json_path=None,
            state="ambiguous-csv",
            message=f"Multiple brood CSVs match colony {target[0]} on {target[1]}.",
            csv_matches=csv_matches,
            image_matches=image_matches,
        )

    csv_path = csv_matches[0] if csv_matches else None
    image_path = image_matches[0] if image_matches else None
    if csv_path and image_path is None:
        same_stem_image = [
            csv_path.with_suffix(ext)
            for ext in sorted(NEST_IMAGE_EXTENSIONS)
            if csv_path.with_suffix(ext).exists()
        ]
        image_path = same_stem_image[0] if same_stem_image else None

    json_path = None
    if image_path is not None:
        candidate_json = image_path.with_suffix(".json")
        json_path = candidate_json if candidate_json.exists() else candidate_json
    elif csv_path is not None:
        candidate_json = csv_path.with_suffix(".json")
        json_path = candidate_json if candidate_json.exists() else None

    if csv_path is not None:
        if json_path is not None and json_path.exists() and json_path.stat().st_mtime > csv_path.stat().st_mtime:
            state = "stale-csv"
            message = f"Brood CSV is older than the LabelMe JSON: {csv_path.name}"
        else:
            state = "ready"
            message = f"Brood CSV ready: {csv_path.name}"
        return NestMapStatus(
            target=target,
            roots=roots,
            csv_path=csv_path,
            image_path=image_path,
            json_path=json_path,
            state=state,
            message=message,
            csv_matches=csv_matches,
            image_matches=image_matches,
        )

    if image_path is not None and json_path is not None and json_path.exists():
        state = "json-only"
        message = f"LabelMe JSON found; brood CSV has not been generated: {json_path.name}"
    elif image_path is not None:
        state = "image-only"
        message = f"Nest image found but not labeled yet: {image_path.name}"
    else:
        state = "missing-image"
        message = f"No nest image or brood CSV found for colony {target[0]} on {target[1]}."

    return NestMapStatus(
        target=target,
        roots=roots,
        csv_path=None,
        image_path=image_path,
        json_path=json_path,
        state=state,
        message=message,
        csv_matches=csv_matches,
        image_matches=image_matches,
    )


def discover_sessions(source: Path, randomize: bool = False) -> list[ReviewSession]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() in VIDEO_EXTENSIONS:
            related = find_related_csvs(source, source.parent)
            compatible = tuple(path for path in related if is_tracking_csv(path))
            return [ReviewSession(source, compatible, source.parent, related)]
        raise ValueError(f"Expected a video file or folder, got: {source}")

    videos = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    sessions = []
    for video in videos:
        related = find_related_csvs(video, source)
        compatible = tuple(path for path in related if is_tracking_csv(path))
        sessions.append(ReviewSession(video, compatible, source, related))
    if randomize:
        random.shuffle(sessions)
    return sessions


def load_tracking(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        missing = REQUIRED_TRACKING_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        df = df.copy()
        df["csv_path"] = str(path)
        frames.append(df)

    if not frames:
        raise ValueError("No tracking CSVs were found for this video.")

    out = pd.concat(frames, ignore_index=True)
    for col in ("frame", "ID", "centroidX", "centroidY"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["frame", "ID", "centroidX", "centroidY"])
    out["frame"] = out["frame"].astype(int)
    out["ID"] = out["ID"].astype(int)
    if "interpolated" not in out.columns:
        out["interpolated"] = 0
    return out.sort_values(["frame", "ID"]).reset_index(drop=True)


def load_noid_tags(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        missing = REQUIRED_NOID_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        df = df.copy()
        df["csv_path"] = str(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["frame", "centroidX", "centroidY", "csv_path"])

    out = pd.concat(frames, ignore_index=True)
    for col in ("frame", "centroidX", "centroidY"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["frame", "centroidX", "centroidY"])
    out["frame"] = out["frame"].astype(int)
    return out.sort_values(["frame", "csv_path"]).reset_index(drop=True)


def pivot_tracking(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pivot_table(index="frame", columns="ID", values=["centroidX", "centroidY"])
        .sort_index(axis=1)
        .apply(pd.to_numeric, errors="coerce")
    )


def compute_activity_tables(
    tracking: pd.DataFrame,
    *,
    frame_rate: float,
    max_gap_seconds: float,
    speed_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = pivot_tracking(tracking)
    return movement_metrics(
        pivot,
        frame_rate=frame_rate,
        max_gap_seconds=max_gap_seconds,
        speed_cutoff=speed_cutoff,
    )


def social_center_from_tracking_paths(paths: Iterable[Path]) -> tuple[float, float] | None:
    result = mean_centroids_across_files(paths)
    center_x = result["mean_centroidX"]
    center_y = result["mean_centroidY"]
    if not np.isfinite(center_x) or not np.isfinite(center_y):
        return None
    return float(center_x), float(center_y)


def social_center_tracking_paths_for_sessions(
    sessions: Iterable[ReviewSession],
    target: tuple[int, str] | None,
    pattern: str | None,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for session in sessions:
        if target is not None and session_colony_date(session) != target:
            continue
        if pattern is not None:
            path = find_tracking_csv_by_pattern(session, pattern)
            if path is None:
                continue
            paths.append(path)
        elif session.tracking_paths:
            paths.append(session.tracking_paths[0])

    unique_paths: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in unique_paths:
            unique_paths.append(resolved)
    return tuple(unique_paths)


def compute_social_center_distance_table(
    tracking: pd.DataFrame,
    social_center: tuple[float, float] | None = None,
) -> pd.DataFrame:
    pivot = pivot_tracking(tracking)
    if social_center is None:
        center_x = float(tracking["centroidX"].mean())
        center_y = float(tracking["centroidY"].mean())
    else:
        center_x, center_y = social_center
    return np.sqrt((pivot["centroidX"] - center_x) ** 2 + (pivot["centroidY"] - center_y) ** 2)


def compute_nearest_neighbor_distance_table(tracking: pd.DataFrame) -> pd.DataFrame:
    frames = sorted(int(frame) for frame in tracking["frame"].unique())
    bee_ids = sorted(int(bee_id) for bee_id in tracking["ID"].unique())
    out = pd.DataFrame(np.nan, index=frames, columns=bee_ids, dtype=float)

    for frame, rows in tracking.groupby("frame"):
        rows = rows.sort_values("ID")
        if len(rows) < 2:
            continue
        coords = rows[["centroidX", "centroidY"]].to_numpy(dtype=float)
        ids = rows["ID"].to_numpy(dtype=int)
        delta = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt((delta ** 2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)
        out.loc[int(frame), ids] = np.min(distances, axis=1)

    return out


def compute_interaction_count_table(tracking: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    frames = sorted(int(frame) for frame in tracking["frame"].unique())
    bee_ids = sorted(int(bee_id) for bee_id in tracking["ID"].unique())
    out = pd.DataFrame(np.nan, index=frames, columns=bee_ids, dtype=float)

    for frame, rows in tracking.groupby("frame"):
        rows = rows.sort_values("ID")
        coords = rows[["centroidX", "centroidY"]].to_numpy(dtype=float)
        ids = rows["ID"].to_numpy(dtype=int)
        if len(rows) < 2:
            out.loc[int(frame), ids] = 0
            continue
        delta = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt((delta ** 2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)
        out.loc[int(frame), ids] = np.sum(distances <= cutoff, axis=1)

    return out


def is_nest_metric_label(label: str) -> bool:
    clean = label.strip().lower()
    return bool(clean) and not clean.startswith("arena perimeter") and "calibration" not in clean


def nest_components_from_brood_map(brood_map: pd.DataFrame, *, metrics_only: bool = True) -> list[NestComponent]:
    if brood_map.empty:
        return []

    components: list[NestComponent] = []
    grouped = brood_map.groupby(["object index", "label", "shape"], dropna=False)
    for (_, label, shape), rows in grouped:
        label = "" if pd.isna(label) else str(label)
        if metrics_only and not is_nest_metric_label(label):
            continue
        shape = "" if pd.isna(shape) else str(shape)
        points = tuple((float(row.x), float(row.y)) for _, row in rows.iterrows())
        if not points:
            continue
        radius = 0.0
        if "radius" in rows.columns and not pd.isna(rows.iloc[0].radius):
            radius = float(rows.iloc[0].radius)
        components.append(NestComponent(label=label, shape=shape, points=points, radius=radius))
    return components


def _distance_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0:
        return float(np.linalg.norm(point - a)), a
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    nearest = a + t * ab
    return float(np.linalg.norm(point - nearest)), nearest


def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    inside = False
    x, y = point
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_at_y = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def _polyline_distance(point: np.ndarray, vertices: np.ndarray, *, closed: bool) -> tuple[float, np.ndarray]:
    if len(vertices) == 1:
        return float(np.linalg.norm(point - vertices[0])), vertices[0]

    segments = list(zip(vertices[:-1], vertices[1:]))
    if closed and len(vertices) > 2:
        segments.append((vertices[-1], vertices[0]))

    best_distance = float("inf")
    best_point = vertices[0]
    for a, b in segments:
        distance, nearest = _distance_to_segment(point, a, b)
        if distance < best_distance:
            best_distance = distance
            best_point = nearest
    return best_distance, best_point


def _rectangle_vertices(points: np.ndarray) -> np.ndarray:
    p1, p2 = points[0], points[1]
    min_x, max_x = sorted((p1[0], p2[0]))
    min_y, max_y = sorted((p1[1], p2[1]))
    return np.array(
        [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ],
        dtype=float,
    )


def distance_to_nest_component(point_xy: tuple[float, float], component: NestComponent) -> tuple[float, tuple[float, float]]:
    point = np.array(point_xy, dtype=float)
    points = np.array(component.points, dtype=float)
    shape = component.shape.lower()

    if shape == "circle":
        center = points[0]
        radius = max(0.0, float(component.radius))
        delta = point - center
        center_distance = float(np.linalg.norm(delta))
        if center_distance <= radius:
            return 0.0, (float(point[0]), float(point[1]))
        if center_distance == 0:
            nearest = center + np.array([radius, 0.0])
        else:
            nearest = center + delta / center_distance * radius
        return float(center_distance - radius), (float(nearest[0]), float(nearest[1]))

    if shape == "point":
        distance = float(np.linalg.norm(point - points[0]))
        return distance, (float(points[0][0]), float(points[0][1]))

    if shape == "rectangle" and len(points) >= 2:
        points = _rectangle_vertices(points)
        shape = "polygon"

    if shape == "polygon" and len(points) >= 3:
        if _point_in_polygon(point, points):
            return 0.0, (float(point[0]), float(point[1]))
        distance, nearest = _polyline_distance(point, points, closed=True)
        return distance, (float(nearest[0]), float(nearest[1]))

    if len(points) >= 2:
        distance, nearest = _polyline_distance(point, points, closed=False)
        return distance, (float(nearest[0]), float(nearest[1]))

    distance = float(np.linalg.norm(point - points[0]))
    return distance, (float(points[0][0]), float(points[0][1]))


def nearest_nest_component(
    point_xy: tuple[float, float],
    components: list[NestComponent],
) -> tuple[float, tuple[float, float], NestComponent] | None:
    best: tuple[float, tuple[float, float], NestComponent] | None = None
    for component in components:
        distance, nearest = distance_to_nest_component(point_xy, component)
        if best is None or distance < best[0]:
            best = (distance, nearest, component)
    return best


def compute_nest_component_tables(
    tracking: pd.DataFrame,
    brood_map: pd.DataFrame,
    cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    components = nest_components_from_brood_map(brood_map)
    if tracking.empty or not components:
        return pd.DataFrame(), pd.DataFrame()

    frames = sorted(int(frame) for frame in tracking["frame"].unique())
    bee_ids = sorted(int(bee_id) for bee_id in tracking["ID"].unique())
    distance = pd.DataFrame(np.nan, index=frames, columns=bee_ids, dtype=float)
    interactions = pd.DataFrame(np.nan, index=frames, columns=bee_ids, dtype=float)

    for row in tracking.itertuples(index=False):
        point = (float(row.centroidX), float(row.centroidY))
        distances = [distance_to_nest_component(point, component)[0] for component in components]
        frame = int(row.frame)
        bee_id = int(row.ID)
        distance.loc[frame, bee_id] = min(distances)
        interactions.loc[frame, bee_id] = sum(value <= cutoff for value in distances)

    return distance, interactions


def ethogram_state_table(
    table: pd.DataFrame,
    metric: str,
    cutoff: float,
    activity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()

    if metric == "activity":
        return table.copy()
    if metric == "speed" and activity is not None and not activity.empty:
        return activity.reindex(index=table.index, columns=table.columns)

    if metric in {"interactions", "nest_interactions"}:
        state = table > 0
    elif metric in {"nearest_neighbor", "nest_distance"}:
        state = table <= cutoff
    elif metric == "speed":
        state = table > cutoff
    else:
        state = table.notna()

    return state.astype(float).where(~table.isna())


def contiguous_value_ranges(series: pd.Series) -> list[tuple[int, int, int | None]]:
    ranges: list[tuple[int, int, int | None]] = []
    start = None
    previous = None
    previous_state: int | None = None

    for frame, value in series.sort_index().items():
        frame = int(frame)
        state = None if pd.isna(value) else int(value)
        starts_new_range = (
            start is None
            or previous is None
            or frame > previous + 1
            or state != previous_state
        )
        if starts_new_range:
            if start is not None:
                ranges.append((start, previous, previous_state))
            start = frame
            previous_state = state
        previous = frame

    if start is not None and previous is not None:
        ranges.append((start, previous, previous_state))
    return ranges


def convert_speed_table_units(speed: pd.DataFrame, unit: str, frame_rate: float, px_per_cm: float) -> pd.DataFrame:
    if unit == "px/sec":
        return speed * frame_rate
    if unit == "cm/sec":
        return speed * frame_rate / px_per_cm
    return speed.copy()


def convert_speed_cutoff_units(cutoff: float, unit: str, frame_rate: float, px_per_cm: float) -> float:
    if unit == "px/sec":
        return cutoff * frame_rate
    if unit == "cm/sec":
        return cutoff * frame_rate / px_per_cm
    return cutoff


def contiguous_true_ranges(mask: pd.Series) -> list[tuple[int, int]]:
    """Return inclusive frame ranges where a boolean mask is true."""
    if mask.empty:
        return []

    ranges = []
    start = None
    previous = None
    for frame, value in mask.sort_index().items():
        frame = int(frame)
        if bool(value):
            if start is None or (previous is not None and frame > previous + 1):
                if start is not None:
                    ranges.append((start, previous))
                start = frame
            previous = frame
        elif start is not None:
            ranges.append((start, previous))
            start = None
            previous = None
    if start is not None:
        ranges.append((start, previous if previous is not None else start))
    return ranges


def classify_activity_value(value) -> str:
    if pd.isna(value):
        return "unknown"
    return "active" if int(value) == 1 else "inactive"


def review_settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def source_kind(path: Path) -> str:
    return "Folder" if path.is_dir() else "File"


def last_source_from_settings() -> Path | None:
    raw = review_settings().value(LAST_SOURCE_PATH_KEY, "")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    return path if path.exists() else None


def last_source_randomize() -> bool:
    raw = review_settings().value(LAST_SOURCE_RANDOMIZE_KEY, False)
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in {"1", "true", "yes"}


def save_last_source(source: Path, randomize: bool = False):
    settings = review_settings()
    settings.setValue(LAST_SOURCE_PATH_KEY, str(source.expanduser().resolve()))
    settings.setValue(LAST_SOURCE_RANDOMIZE_KEY, bool(randomize))
    settings.sync()


def default_open_dir() -> Path:
    last = last_source_from_settings()
    if last is None:
        return Path.cwd()
    return last if last.is_dir() else last.parent


def make_collapsible_group(title: str) -> QGroupBox:
    box = QGroupBox(title)
    box.setCheckable(True)
    box.setChecked(True)
    box.toggled.connect(lambda checked, group=box: set_group_collapsed(group, checked))
    return box


def set_group_collapsed(group: QGroupBox, expanded: bool):
    layout = group.layout()
    if layout is None:
        return
    for index in range(layout.count()):
        item = layout.itemAt(index)
        widget = item.widget()
        if widget is not None:
            widget.setVisible(expanded)
            continue
        child_layout = item.layout()
        if child_layout is not None:
            set_layout_visible(child_layout, expanded)


def set_layout_visible(layout: QLayout, visible: bool):
    for index in range(layout.count()):
        item = layout.itemAt(index)
        widget = item.widget()
        if widget is not None:
            widget.setVisible(visible)
            continue
        child_layout = item.layout()
        if child_layout is not None:
            set_layout_visible(child_layout, visible)


class FolderModeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Folder playback")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("How should videos from this folder be ordered?"))
        self.mode = QComboBox()
        self.mode.addItems(["Sequential", "Random"])
        layout.addWidget(self.mode)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def randomize(self) -> bool:
        return self.mode.currentText() == "Random"


class StartupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.choice = None
        self.last_source = last_source_from_settings()
        self.setWindowTitle("Open review source")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose a video or a folder of videos to review."))

        if self.last_source is not None:
            kind = source_kind(self.last_source)
            last_row = QHBoxLayout()
            last_label = QLabel(f"Last {kind}: {self.last_source.name}")
            last_label.setWordWrap(True)
            last_label.setToolTip(str(self.last_source))
            last_button = QPushButton("Open Last")
            last_button.setToolTip(str(self.last_source))
            last_button.clicked.connect(lambda: self._choose("last"))
            last_row.addWidget(last_label, stretch=1)
            last_row.addWidget(last_button)
            layout.addLayout(last_row)

        row = QHBoxLayout()
        video = QPushButton("Open Video")
        folder = QPushButton("Open Folder")
        cancel = QPushButton("Cancel")
        video.clicked.connect(lambda: self._choose("video"))
        folder.clicked.connect(lambda: self._choose("folder"))
        cancel.clicked.connect(self.reject)
        row.addWidget(video)
        row.addWidget(folder)
        row.addWidget(cancel)
        layout.addLayout(row)

    def _choose(self, choice: str):
        self.choice = choice
        self.accept()


class TrackingCsvChoiceDialog(QDialog):
    def __init__(self, session: ReviewSession, paths: tuple[Path, ...], message: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose tracking CSV")
        self.paths = paths

        layout = QVBoxLayout(self)
        intro = QLabel(f"{message}\n\nVideo: {session.video_path.name}")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.csv_choice = QComboBox()
        for path in paths:
            pattern = tracking_csv_pattern(session.video_path, path)
            label = f"{path.name}  ({tracking_csv_pattern_label(pattern)})"
            self.csv_choice.addItem(label, str(path))
            self.csv_choice.setItemData(self.csv_choice.count() - 1, str(path), Qt.ToolTipRole)
        layout.addWidget(self.csv_choice)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selected_path(self) -> Path | None:
        data = self.csv_choice.currentData()
        return Path(data) if data else None


class VideoLabel(QLabel):
    zoomChanged = Signal(float)
    legendVisibilityChanged = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(520, 292)
        self.setStyleSheet("background: #101418; color: #d8e2e7;")
        self.setText("Open a video or folder to begin review")
        self.setMouseTracking(True)
        self.setToolTip("Scroll to zoom. Drag to pan. Double-click the key to hide it, or double-click elsewhere to reset zoom.")

        self._pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._max_zoom = 16.0
        self._pan = QPointF(0.0, 0.0)
        self._drag_start: QPointF | None = None
        self._drag_start_pan = QPointF(0.0, 0.0)
        self._legend_items: list[tuple[str, str, QColor]] = []
        self._legend_visible = True
        self._legend_rect = QRectF()

    @property
    def zoom_factor(self) -> float:
        return self._zoom

    def setPixmap(self, pixmap: QPixmap):  # noqa: N802 - Qt override
        self.set_video_pixmap(pixmap)

    def set_video_pixmap(self, pixmap: QPixmap | None):
        if pixmap is None or pixmap.isNull():
            self._pixmap = None
            self.update()
            return

        previous_size = self._pixmap.size() if self._pixmap is not None else None
        self._pixmap = QPixmap(pixmap)
        if previous_size is not None and previous_size != self._pixmap.size():
            self.reset_zoom()
        else:
            self._clamp_pan()
            self.update()

    def set_legend_items(self, items: list[tuple[str, str, QColor]]):
        self._legend_items = [(kind, label, QColor(color)) for kind, label, color in items]
        self.update()

    def legend_visible(self) -> bool:
        return self._legend_visible

    def set_legend_visible(self, visible: bool):
        if self._legend_visible == visible:
            return
        self._legend_visible = visible
        self.legendVisibilityChanged.emit(visible)
        self.update()

    def toggle_legend(self):
        self.set_legend_visible(not self._legend_visible)

    def reset_zoom(self):
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._drag_start = None
        self._update_cursor()
        self.zoomChanged.emit(self._zoom)
        self.update()

    def zoom_in(self):
        self.zoom_at(QPointF(self.rect().center()), 1.25)

    def zoom_out(self):
        self.zoom_at(QPointF(self.rect().center()), 0.8)

    def zoom_at(self, widget_pos: QPointF, factor: float):
        if self._pixmap is None or self._pixmap.isNull():
            return

        old_zoom = self._zoom
        new_zoom = max(1.0, min(self._max_zoom, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 0.0001:
            return

        old_rect = self._target_rect()
        old_scale = self._scale_for_zoom(old_zoom)
        image_x = (widget_pos.x() - old_rect.x()) / old_scale
        image_y = (widget_pos.y() - old_rect.y()) / old_scale

        self._zoom = new_zoom
        new_scale = self._scale_for_zoom(self._zoom)
        target_w = self._pixmap.width() * new_scale
        target_h = self._pixmap.height() * new_scale
        base_x = (self.width() - target_w) / 2.0
        base_y = (self.height() - target_h) / 2.0
        desired_x = widget_pos.x() - image_x * new_scale
        desired_y = widget_pos.y() - image_y * new_scale
        self._pan = QPointF(desired_x - base_x, desired_y - base_y)
        self._clamp_pan()
        self._update_cursor()
        self.zoomChanged.emit(self._zoom)
        self.update()

    def _scale_for_zoom(self, zoom: float) -> float:
        if self._pixmap is None or self._pixmap.isNull():
            return 1.0
        fit = min(
            max(1, self.width()) / max(1, self._pixmap.width()),
            max(1, self.height()) / max(1, self._pixmap.height()),
        )
        return fit * zoom

    def _clamped_pan(self, pan: QPointF) -> QPointF:
        if self._pixmap is None or self._pixmap.isNull():
            return QPointF(0.0, 0.0)

        scale = self._scale_for_zoom(self._zoom)
        target_w = self._pixmap.width() * scale
        target_h = self._pixmap.height() * scale
        view_w = max(1, self.width())
        view_h = max(1, self.height())
        base_x = (view_w - target_w) / 2.0
        base_y = (view_h - target_h) / 2.0

        x = base_x + pan.x()
        y = base_y + pan.y()
        if target_w <= view_w:
            x = base_x
        else:
            x = min(0.0, max(view_w - target_w, x))
        if target_h <= view_h:
            y = base_y
        else:
            y = min(0.0, max(view_h - target_h, y))

        return QPointF(x - base_x, y - base_y)

    def _clamp_pan(self):
        self._pan = self._clamped_pan(self._pan)

    def _target_rect(self) -> QRectF:
        if self._pixmap is None or self._pixmap.isNull():
            return QRectF()
        self._clamp_pan()
        scale = self._scale_for_zoom(self._zoom)
        target_w = self._pixmap.width() * scale
        target_h = self._pixmap.height() * scale
        x = (self.width() - target_w) / 2.0 + self._pan.x()
        y = (self.height() - target_h) / 2.0 + self._pan.y()
        return QRectF(x, y, target_w, target_h)

    def _update_cursor(self):
        if self._zoom > 1.0001:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def resizeEvent(self, event):
        self._clamp_pan()
        super().resizeEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.2 if delta > 0 else 1 / 1.2
        self.zoom_at(event.position(), factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._zoom > 1.0001:
            self._drag_start = event.position()
            self._drag_start_pan = QPointF(self._pan)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            delta = event.position() - self._drag_start
            self._pan = self._drag_start_pan + delta
            self._clamp_pan()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            self._drag_start = None
            self._update_cursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._legend_visible and self._legend_rect.contains(event.position()):
                self.set_legend_visible(False)
                event.accept()
                return
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        if self._pixmap is None or self._pixmap.isNull():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#101418"))
        painter.drawPixmap(self._target_rect(), self._pixmap, QRectF(self._pixmap.rect()))
        self._draw_legend(painter)
        painter.end()

    def _draw_legend(self, painter: QPainter):
        self._legend_rect = QRectF()
        if not self._legend_visible or not self._legend_items:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        font_metrics = painter.fontMetrics()
        row_h = max(20, font_metrics.height() + 5)
        title = "Key"
        max_item_rows = max(1, int((self.height() - 38) / row_h) - 1)
        visible_items = self._legend_items
        if len(visible_items) > max_item_rows:
            n_hidden = len(visible_items) - max_item_rows + 1
            visible_items = [
                *visible_items[: max(0, max_item_rows - 1)],
                ("text", f"+ {n_hidden} more", QColor("#f6f8fb")),
            ]
        text_w = max(
            [font_metrics.horizontalAdvance(title)]
            + [font_metrics.horizontalAdvance(label) for _, label, _ in visible_items]
        )
        width = min(max(180, text_w + 54), max(180, self.width() - 24))
        height = 14 + row_h * (len(visible_items) + 1)
        x = self.width() - width - 12
        y = 12
        rect = QRectF(x, y, width, height)
        self._legend_rect = rect

        painter.setPen(QPen(QColor(236, 241, 245, 150), 1))
        painter.setBrush(QColor(16, 20, 24, 222))
        painter.drawRoundedRect(rect, 6, 6)

        text_x = x + 36
        row_y = y + 7
        painter.setPen(QColor("#f6f8fb"))
        painter.drawText(QRectF(x + 10, row_y, width - 20, row_h), Qt.AlignVCenter, title)

        for kind, label, color in visible_items:
            row_y += row_h
            center_y = row_y + row_h / 2.0
            swatch_x = x + 16
            painter.setPen(QPen(color, 3))
            painter.setBrush(color)

            if kind == "dot":
                painter.drawEllipse(QPointF(swatch_x + 6, center_y), 6, 6)
            elif kind == "ring":
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(swatch_x + 6, center_y), 7, 7)
            elif kind == "diamond":
                painter.drawPolygon(
                    QPolygonF(
                        [
                            QPointF(swatch_x + 7, center_y - 7),
                            QPointF(swatch_x + 14, center_y),
                            QPointF(swatch_x + 7, center_y + 7),
                            QPointF(swatch_x, center_y),
                        ]
                    )
                )
            elif kind == "cross":
                painter.drawLine(QPointF(swatch_x, center_y), QPointF(swatch_x + 14, center_y))
                painter.drawLine(QPointF(swatch_x + 7, center_y - 7), QPointF(swatch_x + 7, center_y + 7))
            elif kind == "text":
                painter.setPen(QPen(color, 1))
                painter.drawText(QRectF(swatch_x - 2, row_y, 26, row_h), Qt.AlignVCenter, "12")
            else:
                painter.drawLine(QPointF(swatch_x - 1, center_y), QPointF(swatch_x + 15, center_y))

            painter.setPen(QColor("#edf3f7"))
            painter.drawText(QRectF(text_x, row_y, width - 46, row_h), Qt.AlignVCenter, label)

        painter.restore()


class ReviewHub(QMainWindow):
    def __init__(self, source: Path | None = None):
        super().__init__()
        self.setWindowTitle("BuzzAnalysis Metric Review Hub")
        self.resize(1280, 760)
        self.setMinimumSize(1100, 680)

        self.sessions: list[ReviewSession] = []
        self.session_index = -1
        self.capture: cv2.VideoCapture | None = None
        self.current_frame = 0
        self.frame_count = 0
        self.video_frame_count_known = False
        self.video_fps = frame_per_sec
        self._capture_next_frame: int | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self.tracking = pd.DataFrame()
        self.rows_by_frame: dict[int, pd.DataFrame] = {}
        self.noid_tags = pd.DataFrame()
        self.noid_rows_by_frame: dict[int, pd.DataFrame] = {}
        self.noid_tag_paths: tuple[Path, ...] = ()
        self.noid_status_message = "No noID CSV loaded"
        self.activity = pd.DataFrame()
        self.speed = pd.DataFrame()
        self.social_center_distance = pd.DataFrame()
        self.nearest_neighbor_distance = pd.DataFrame()
        self.interaction_count = pd.DataFrame()
        self.nest_component_distance = pd.DataFrame()
        self.nest_component_interaction_count = pd.DataFrame()
        self.plot_series_by_bee: dict[int, pd.Series] = {}
        self.heatmap_bees: list[int] = []
        self.bee_ids: list[int] = []
        self.social_center: tuple[float, float] | None = None
        self.social_center_file_count = 0
        self.active_tracking_path: Path | None = None
        self.tracking_status_message = "No tracking loaded"
        self.tracking_csv_preference: str | None = None
        self.tracking_csv_prompted = False
        self.nest_status: NestMapStatus | None = None
        self.brood_map = pd.DataFrame()
        self.labeling_process: QProcess | None = None
        self.labeling_image_path: Path | None = None
        self.labeling_csv_target: Path | None = None
        self._last_frame_bgr = None
        self._last_frame_index: int | None = None

        self._build_ui()
        if source is None:
            QTimer.singleShot(0, self.open_startup_dialog)
        else:
            self.load_source(source)

    def _build_ui(self):
        toolbar = QToolBar("Session")
        self.addToolBar(toolbar)
        open_video = QAction("Open Video", self)
        open_folder = QAction("Open Folder", self)
        open_video.triggered.connect(self.open_video)
        open_folder.triggered.connect(self.open_folder)
        toolbar.addAction(open_video)
        toolbar.addAction(open_folder)

        splitter = QSplitter()
        self.setCentralWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.sessions_group = make_collapsible_group("Sessions")
        sessions_layout = QVBoxLayout(self.sessions_group)
        self.session_list = QListWidget()
        self.session_list.currentRowChanged.connect(self.load_session_at)
        sessions_layout.addWidget(self.session_list)
        nav_row = QHBoxLayout()
        prev_video = QPushButton("Previous")
        next_video = QPushButton("Next")
        prev_video.clicked.connect(self.previous_session)
        next_video.clicked.connect(self.next_session)
        nav_row.addWidget(prev_video)
        nav_row.addWidget(next_video)
        sessions_layout.addLayout(nav_row)
        left_layout.addWidget(self.sessions_group, stretch=1)
        splitter.addWidget(left)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.video_label = VideoLabel()
        center_layout.addWidget(self.video_label, stretch=1)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.sliderMoved.connect(self.seek_frame)
        center_layout.addWidget(self.timeline)

        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        prev_frame = QPushButton("-1 frame")
        next_frame = QPushButton("+1 frame")
        prev_frame.clicked.connect(self.previous_frame)
        next_frame.clicked.connect(self.next_frame)
        self.frame_label = QLabel("Frame 0 / 0")
        zoom_out = QPushButton("Zoom -")
        zoom_in = QPushButton("Zoom +")
        reset_zoom = QPushButton("Reset zoom")
        self.legend_button = QPushButton()
        self.zoom_label = QLabel("Zoom 100%")
        zoom_out.setToolTip("Zoom out")
        zoom_in.setToolTip("Zoom in")
        reset_zoom.setToolTip("Reset zoom and recenter")
        zoom_out.clicked.connect(self.video_label.zoom_out)
        zoom_in.clicked.connect(self.video_label.zoom_in)
        reset_zoom.clicked.connect(self.video_label.reset_zoom)
        self.legend_button.clicked.connect(self.video_label.toggle_legend)
        self.video_label.zoomChanged.connect(self.update_zoom_label)
        self.video_label.legendVisibilityChanged.connect(self.update_legend_button)
        self.update_legend_button(self.video_label.legend_visible())
        controls.addWidget(self.play_button)
        controls.addWidget(prev_frame)
        controls.addWidget(next_frame)
        controls.addWidget(self.frame_label)
        controls.addWidget(zoom_out)
        controls.addWidget(zoom_in)
        controls.addWidget(reset_zoom)
        controls.addWidget(self.zoom_label)
        controls.addWidget(self.legend_button)
        controls.addStretch()
        center_layout.addLayout(controls)
        splitter.addWidget(center)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)
        self.layer_group = self._make_layer_group()
        self.params_group = self._make_params_group()
        self.nest_group = self._make_nest_group()
        self.inspector_group = self._make_inspector_group()
        right_layout.addWidget(self.layer_group)
        right_layout.addWidget(self.params_group)
        right_layout.addWidget(self.nest_group)
        right_layout.addWidget(self.inspector_group, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([220, 760, 300])

    def _make_layer_group(self) -> QWidget:
        box = make_collapsible_group("Metric Layers")
        layout = QGridLayout(box)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(2)
        self.show_tracks = QCheckBox("Tracking points")
        self.show_activity = QCheckBox("Activity state")
        self.show_noid_tags = QCheckBox("noID tags")
        self.show_speed_labels = QCheckBox("Speed labels")
        self.show_interactions = QCheckBox("Interactions")
        self.show_nest_interactions = QCheckBox("Nest interactions")
        self.show_nest_distances = QCheckBox("Nest distances")
        self.show_social_center = QCheckBox("Social center")
        self.show_trails = QCheckBox("Trails")
        self.show_nest_map = QCheckBox("Nest map")
        layer_options = [
            (self.show_tracks, True),
            (self.show_activity, True),
            (self.show_noid_tags, False),
            (self.show_speed_labels, False),
            (self.show_interactions, False),
            (self.show_nest_interactions, False),
            (self.show_nest_distances, False),
            (self.show_social_center, False),
            (self.show_trails, False),
            (self.show_nest_map, True),
        ]
        for index, (checkbox, checked) in enumerate(layer_options):
            checkbox.setChecked(checked)
            checkbox.stateChanged.connect(self.render_current_frame)
            layout.addWidget(checkbox, index // 2, index % 2)
        self.update_noid_layer_availability()
        return box

    def _make_params_group(self) -> QWidget:
        box = make_collapsible_group("Metric Controls")
        layout = QFormLayout(box)

        self.tracking_source = QComboBox()
        self.tracking_source.currentIndexChanged.connect(self.change_tracking_source)

        self.speed_cutoff = QDoubleSpinBox()
        self.speed_cutoff.setRange(0, 1000)
        self.speed_cutoff.setDecimals(2)
        self.speed_cutoff.setSingleStep(0.25)
        self.speed_cutoff.setValue(float(digital_noise_speed_cutoff))
        self.speed_cutoff.valueChanged.connect(self.recompute_metrics)

        self.behavior_fps = QDoubleSpinBox()
        self.behavior_fps.setRange(0.01, 1000)
        self.behavior_fps.setDecimals(2)
        self.behavior_fps.setValue(float(frame_per_sec))
        self.behavior_fps.valueChanged.connect(self.recompute_metrics)

        self.max_gap = QDoubleSpinBox()
        self.max_gap.setRange(0.01, 120)
        self.max_gap.setDecimals(2)
        self.max_gap.setSingleStep(0.25)
        self.max_gap.setValue(float(max_behavior_gap_seconds))
        self.max_gap.valueChanged.connect(self.recompute_metrics)

        self.interaction_cutoff = QDoubleSpinBox()
        self.interaction_cutoff.setRange(0, 10000)
        self.interaction_cutoff.setDecimals(1)
        self.interaction_cutoff.setValue(float(interaction_distance_cutoff))
        self.interaction_cutoff.valueChanged.connect(self.recompute_metrics)

        self.trail_frames = QSpinBox()
        self.trail_frames.setRange(1, 500)
        self.trail_frames.setValue(20)
        self.trail_frames.valueChanged.connect(self.render_current_frame)

        self.focus_bee = QComboBox()
        self.focus_bee.currentIndexChanged.connect(self.update_focus_plot)
        self.focus_bee.currentIndexChanged.connect(self.render_current_frame)

        layout.addRow("Tracking CSV", self.tracking_source)
        layout.addRow("Speed cutoff", self.speed_cutoff)
        layout.addRow("Behavior FPS", self.behavior_fps)
        layout.addRow("Max gap sec", self.max_gap)
        layout.addRow("Interaction px", self.interaction_cutoff)
        layout.addRow("Trail frames", self.trail_frames)
        layout.addRow("Focus bee", self.focus_bee)
        box.setChecked(False)
        return box

    def _make_nest_group(self) -> QWidget:
        box = make_collapsible_group("Nest Map")
        layout = QVBoxLayout(box)
        self.nest_status_label = QLabel("No session loaded")
        self.nest_status_label.setWordWrap(True)
        layout.addWidget(self.nest_status_label)

        row = QHBoxLayout()
        self.open_labeler_btn = QPushButton("Label nest image")
        self.generate_brood_csv_btn = QPushButton("Generate CSV")
        self.refresh_nest_btn = QPushButton("Refresh")
        self.open_labeler_btn.clicked.connect(self.open_nest_labeler)
        self.generate_brood_csv_btn.clicked.connect(self.generate_current_brood_csv)
        self.refresh_nest_btn.clicked.connect(self.refresh_nest_status)
        row.addWidget(self.open_labeler_btn)
        row.addWidget(self.generate_brood_csv_btn)
        row.addWidget(self.refresh_nest_btn)
        layout.addLayout(row)
        self._set_nest_controls_enabled(False)
        box.setChecked(False)
        return box

    def _make_inspector_group(self) -> QWidget:
        box = make_collapsible_group("Frame Inspector")
        layout = QVBoxLayout(box)
        self.stats_label = QLabel("No tracking loaded")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)
        self.expand_plot_btn = QPushButton("Expand plot")
        self.expand_plot_btn.setCheckable(True)
        self.expand_plot_btn.setToolTip("Collapse the upper right-panel sections so the plot has more room")
        self.expand_plot_btn.toggled.connect(self.set_plot_expanded)
        self.restore_plot_btn = QPushButton("Restore plot defaults")
        self.restore_plot_btn.setToolTip("Reset plot metric, units, mode, focus bee, and frame range")
        self.restore_plot_btn.clicked.connect(self.restore_plot_defaults)
        plot_button_row = QHBoxLayout()
        plot_button_row.addWidget(self.expand_plot_btn)
        plot_button_row.addWidget(self.restore_plot_btn)
        layout.addLayout(plot_button_row)

        plot_controls = QGridLayout()
        plot_controls.setContentsMargins(0, 0, 0, 0)
        plot_controls.setHorizontalSpacing(6)
        plot_controls.setVerticalSpacing(2)
        self.plot_metric = QComboBox()
        self.plot_metric.addItem("Speed", "speed")
        self.plot_metric.addItem("Activity", "activity")
        self.plot_metric.addItem("Social center", "dist_sc")
        self.plot_metric.addItem("Nearest neighbor", "nearest_neighbor")
        self.plot_metric.addItem("Interactions", "interactions")
        self.plot_metric.addItem("Nest distance", "nest_distance")
        self.plot_metric.addItem("Nest interactions", "nest_interactions")
        self.plot_metric.currentIndexChanged.connect(self.update_focus_plot)
        self.plot_metric.setMinimumContentsLength(13)

        self.speed_units = QComboBox()
        self.speed_units.addItems(["px/frame", "px/sec", "cm/sec"])
        self.speed_units.currentIndexChanged.connect(self.update_focus_plot)
        self.speed_units.setMinimumContentsLength(8)

        self.plot_mode = QComboBox()
        self.plot_mode.addItem("Lines", "lines")
        self.plot_mode.addItem("Heatmap", "heatmap")
        self.plot_mode.addItem("Ethogram", "ethogram")
        self.plot_mode.currentIndexChanged.connect(self.update_focus_plot)
        self.plot_mode.setMinimumContentsLength(8)

        for combo in (self.plot_metric, self.speed_units, self.plot_mode):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)

        plot_controls.addWidget(QLabel("Metric"), 0, 0)
        plot_controls.addWidget(QLabel("Units"), 0, 1)
        plot_controls.addWidget(QLabel("Mode"), 0, 2)
        plot_controls.addWidget(self.plot_metric, 1, 0)
        plot_controls.addWidget(self.speed_units, 1, 1)
        plot_controls.addWidget(self.plot_mode, 1, 2)
        plot_controls.setColumnStretch(0, 2)
        plot_controls.setColumnStretch(1, 1)
        plot_controls.setColumnStretch(2, 1)
        layout.addLayout(plot_controls)

        self.speed_plot = pg.PlotWidget()
        self.speed_plot.setMinimumHeight(180)
        self.speed_plot.showAxis("bottom", True)
        bottom_axis = self.speed_plot.getAxis("bottom")
        bottom_axis.setHeight(38)
        bottom_axis.setPen(pg.mkPen("#d8e2e7"))
        bottom_axis.setTextPen(pg.mkPen("#d8e2e7"))
        tick_font = QFont()
        tick_font.setPixelSize(12)
        bottom_axis.setStyle(
            showValues=True,
            tickTextHeight=22,
            tickTextOffset=8,
            autoExpandTextSpace=True,
            tickFont=tick_font,
        )
        self.speed_plot.showGrid(x=True, y=True, alpha=0.2)
        self.speed_plot.setLabel("left", "Speed", units="px/frame")
        self.speed_plot.setLabel("bottom", "Frame (CSV)")
        self.frame_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#e6c229", width=1))
        self.speed_plot.addItem(self.frame_marker)
        self.speed_plot.scene().sigMouseClicked.connect(self.on_speed_plot_clicked)
        layout.addWidget(self.speed_plot)
        box.setChecked(False)
        return box

    def open_startup_dialog(self):
        dialog = StartupDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        if dialog.choice == "video":
            self.open_video()
        elif dialog.choice == "folder":
            self.open_folder()
        elif dialog.choice == "last" and dialog.last_source is not None:
            self.load_source(dialog.last_source, randomize=last_source_randomize())

    def set_plot_expanded(self, expanded: bool):
        groups = (self.layer_group, self.params_group, self.nest_group)
        if expanded:
            self._pre_expand_group_states = {group: group.isChecked() for group in groups}
            for group in groups:
                group.setChecked(False)
            self.expand_plot_btn.setText("Restore panels")
            self.speed_plot.setMinimumHeight(360)
        else:
            for group in groups:
                group.setChecked(getattr(self, "_pre_expand_group_states", {}).get(group, True))
            self.expand_plot_btn.setText("Expand plot")
            self.speed_plot.setMinimumHeight(180)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            str(default_open_dir()),
            "Videos (*.mp4 *.mjpeg *.avi *.mov *.mkv)",
        )
        if path:
            self.load_source(Path(path))

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open folder", str(default_open_dir()))
        if not path:
            return
        mode = FolderModeDialog(self)
        if mode.exec() != QDialog.Accepted:
            return
        self.load_source(Path(path), randomize=mode.randomize)

    def load_source(self, source: Path, randomize: bool = False):
        try:
            self.sessions = discover_sessions(source, randomize=randomize)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load source", str(exc))
            return

        self.session_list.clear()
        self.session_index = -1
        self.tracking_csv_preference = None
        self.tracking_csv_prompted = False
        for session in self.sessions:
            item = QListWidgetItem(session.label)
            if not session.tracking_paths:
                color = "#d8903f" if session.related_csv_paths else "#b64b4b"
                item.setForeground(QColor(color))
            self.session_list.addItem(item)

        if not self.sessions:
            QMessageBox.warning(self, "No videos found", "No supported video files were found.")
            return
        save_last_source(source, randomize=randomize)
        self.session_list.setCurrentRow(0)

    def load_session_at(self, row: int):
        if row < 0 or row >= len(self.sessions) or row == self.session_index:
            return
        self.stop_playback()
        self.session_index = row
        session = self.sessions[row]
        self.load_session(session)

    def load_session(self, session: ReviewSession):
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(str(session.video_path))
        if not self.capture.isOpened():
            QMessageBox.critical(self, "Video error", f"Could not open {session.video_path}")
            return

        reported_frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.video_frame_count_known = reported_frame_count > 1
        self.frame_count = reported_frame_count if self.video_frame_count_known else 0
        self.video_fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or frame_per_sec
        self.current_frame = 0
        self._capture_next_frame = 0
        self._last_frame_bgr = None
        self._last_frame_index = None
        self.timeline.setRange(0, max(0, self.frame_count - 1))
        self.video_label.reset_zoom()
        self.load_noid_tags_for_session(session)
        self.populate_tracking_sources(session)
        self.apply_tracking_csv_choice(session)
        self.load_selected_tracking(show_warning=True)

    def data_max_frame(self) -> int:
        max_frame = -1
        if not self.tracking.empty and "frame" in self.tracking.columns:
            max_frame = max(max_frame, int(self.tracking["frame"].max()))
        if not self.noid_tags.empty and "frame" in self.noid_tags.columns:
            max_frame = max(max_frame, int(self.noid_tags["frame"].max()))
        return max_frame

    def update_effective_frame_range(self):
        data_max = self.data_max_frame()
        session = self.current_session()
        is_mjpeg = session is not None and session.video_path.suffix.lower() == ".mjpeg"
        should_use_data_range = data_max >= 0 and (
            not self.video_frame_count_known
            or (is_mjpeg and data_max + 1 > self.frame_count)
        )
        if should_use_data_range:
            self.frame_count = data_max + 1
        elif not self.video_frame_count_known:
            self.frame_count = 0
        if self.frame_count:
            self.current_frame = min(self.current_frame, self.frame_count - 1)
        self.timeline.setRange(0, max(0, self.frame_count - 1))

    def populate_tracking_sources(self, session: ReviewSession):
        self.tracking_source.blockSignals(True)
        self.tracking_source.clear()
        choices = session.related_csv_paths or session.tracking_paths
        if not choices:
            self.tracking_source.addItem("No related CSV files found", None)
            self.tracking_source.blockSignals(False)
            return

        default_index = 0
        for index, path in enumerate(choices):
            missing = missing_tracking_columns(path)
            label = path.name
            if missing:
                label = f"{path.name} (missing {', '.join(missing)})"
            elif path in session.tracking_paths and default_index == 0:
                default_index = index
            self.tracking_source.addItem(label, str(path))
            self.tracking_source.setItemData(index, str(path), Qt.ToolTipRole)

        self.tracking_source.setCurrentIndex(default_index)
        self.tracking_source.blockSignals(False)

    def select_tracking_source_path(self, path: Path) -> bool:
        for index in range(self.tracking_source.count()):
            data = self.tracking_source.itemData(index)
            if data and Path(data) == path:
                self.tracking_source.blockSignals(True)
                self.tracking_source.setCurrentIndex(index)
                self.tracking_source.blockSignals(False)
                return True
        return False

    def prompt_tracking_csv_choice(self, session: ReviewSession, message: str) -> Path | None:
        if not session.tracking_paths:
            return None
        dialog = TrackingCsvChoiceDialog(session, session.tracking_paths, message, self)
        if dialog.exec() != QDialog.Accepted:
            return None
        return dialog.selected_path

    def apply_tracking_csv_choice(self, session: ReviewSession):
        if not session.tracking_paths:
            return

        if self.tracking_csv_preference is not None:
            preferred = find_tracking_csv_by_pattern(session, self.tracking_csv_preference)
            if preferred is not None:
                self.select_tracking_source_path(preferred)
                return

            QMessageBox.warning(
                self,
                "Tracking CSV unavailable",
                (
                    "The selected CSV pattern is not available for this video:\n"
                    f"{tracking_csv_pattern_label(self.tracking_csv_preference)}\n\n"
                    "Choose one of the available tracking CSVs for this video."
                ),
            )
            chosen = self.prompt_tracking_csv_choice(
                session,
                "Choose a tracking CSV for this video. This new choice will be reused when possible.",
            )
            if chosen is not None:
                self.tracking_csv_preference = tracking_csv_pattern(session.video_path, chosen)
                self.select_tracking_source_path(chosen)
            return

        if self.tracking_csv_prompted:
            return

        self.tracking_csv_prompted = True
        chosen = self.prompt_tracking_csv_choice(
            session,
            "Choose which associated tracking CSV to use. This choice will be reused for later videos when possible.",
        )
        if chosen is not None:
            self.tracking_csv_preference = tracking_csv_pattern(session.video_path, chosen)
            self.select_tracking_source_path(chosen)

    def change_tracking_source(self, _index: int):
        session = self.current_session()
        path = self.selected_tracking_path()
        if session is not None and path in session.tracking_paths:
            self.tracking_csv_preference = tracking_csv_pattern(session.video_path, path)
            self.tracking_csv_prompted = True
        self.load_selected_tracking(show_warning=True)

    def selected_tracking_path(self) -> Path | None:
        data = self.tracking_source.currentData()
        if not data:
            return None
        return Path(data)

    def clear_tracking_state(self, message: str = "No tracking loaded"):
        self.tracking = pd.DataFrame()
        self.rows_by_frame = {}
        self.activity = pd.DataFrame()
        self.speed = pd.DataFrame()
        self.social_center_distance = pd.DataFrame()
        self.nearest_neighbor_distance = pd.DataFrame()
        self.interaction_count = pd.DataFrame()
        self.nest_component_distance = pd.DataFrame()
        self.nest_component_interaction_count = pd.DataFrame()
        self.plot_series_by_bee = {}
        self.heatmap_bees = []
        self.bee_ids = []
        self.social_center = None
        self.social_center_file_count = 0
        self.active_tracking_path = None
        self.tracking_status_message = message
        self.focus_bee.blockSignals(True)
        self.focus_bee.clear()
        self.focus_bee.addItem("All bees", None)
        self.focus_bee.blockSignals(False)
        self.update_effective_frame_range()
        self.video_label.set_legend_items(self.current_legend_items())

    def update_noid_layer_availability(self):
        if not hasattr(self, "show_noid_tags"):
            return
        available = not self.noid_tags.empty
        self.show_noid_tags.setEnabled(available)
        if available:
            names = ", ".join(path.name for path in self.noid_tag_paths[:2])
            if len(self.noid_tag_paths) > 2:
                names += f", +{len(self.noid_tag_paths) - 2} more"
            self.show_noid_tags.setToolTip(f"Show noID detections from {names}")
        else:
            self.show_noid_tags.setToolTip(self.noid_status_message)

    def clear_noid_tags(self, message: str = "No noID CSV found"):
        self.noid_tags = pd.DataFrame(columns=["frame", "centroidX", "centroidY", "csv_path"])
        self.noid_rows_by_frame = {}
        self.noid_tag_paths = ()
        self.noid_status_message = message
        self.update_noid_layer_availability()

    def load_noid_tags_for_session(self, session: ReviewSession):
        paths = find_noid_tag_csvs(session)
        if not paths:
            self.clear_noid_tags("No noID CSV found for this video")
            return
        try:
            self.noid_tags = load_noid_tags(paths)
        except Exception as exc:
            self.clear_noid_tags(f"Could not load noID CSV:\n{exc}")
            return

        self.noid_tag_paths = paths
        if self.noid_tags.empty:
            self.noid_rows_by_frame = {}
            self.noid_status_message = "noID CSV loaded, but it has no usable rows"
        else:
            self.noid_rows_by_frame = {int(frame): rows for frame, rows in self.noid_tags.groupby("frame")}
            self.noid_status_message = f"noID tags: {', '.join(path.name for path in paths)}"
        self.update_noid_layer_availability()
        self.update_effective_frame_range()

    def load_selected_tracking(self, show_warning: bool = False):
        path = self.selected_tracking_path()
        if path is None:
            self.clear_tracking_state("No tracking CSV selected")
            self.refresh_nest_status()
            self.render_current_frame()
            return
        try:
            self.tracking = load_tracking((path,))
        except Exception as exc:
            self.clear_tracking_state(f"Could not load tracking CSV:\n{path.name}\n{exc}")
            self.refresh_nest_status()
            if show_warning:
                QMessageBox.warning(self, "Tracking CSV not compatible", str(exc))
            self.render_current_frame()
            return

        self.active_tracking_path = path
        self.tracking_status_message = f"Tracking: {path.name}"
        self.rows_by_frame = {int(frame): rows for frame, rows in self.tracking.groupby("frame")}
        self.bee_ids = sorted(int(x) for x in self.tracking["ID"].unique())
        self.update_effective_frame_range()
        self.focus_bee.blockSignals(True)
        self.focus_bee.clear()
        self.focus_bee.addItem("All bees", None)
        for bee_id in self.bee_ids:
            self.focus_bee.addItem(str(bee_id), bee_id)
        self.focus_bee.blockSignals(False)
        self.refresh_nest_status()
        self.recompute_metrics()
        self.render_current_frame()

    def current_session(self) -> ReviewSession | None:
        if 0 <= self.session_index < len(self.sessions):
            return self.sessions[self.session_index]
        return None

    def _set_nest_controls_enabled(self, enabled: bool):
        self.open_labeler_btn.setEnabled(enabled)
        self.generate_brood_csv_btn.setEnabled(False)
        self.refresh_nest_btn.setEnabled(enabled)

    def refresh_nest_status(self):
        session = self.current_session()
        if session is None:
            self.nest_status = None
            self.brood_map = pd.DataFrame()
            self.nest_status_label.setText("No session loaded")
            self._set_nest_controls_enabled(False)
            return

        self.nest_status = assess_nest_map_status(session)
        self.load_brood_map()
        self.update_nest_controls()
        if not self.tracking.empty:
            self.update_focus_plot()
        self.render_current_frame()

    def update_nest_controls(self):
        status = self.nest_status
        if status is None:
            self.nest_status_label.setText("No session loaded")
            self._set_nest_controls_enabled(False)
            return

        text = status.message
        if status.target is not None:
            text = f"Colony {status.target[0]}  Date {status.target[1]}\n{text}"
        if not self.brood_map.empty:
            text += f"\nOverlay objects loaded: {self.brood_map['object index'].nunique()}"
        self.nest_status_label.setText(text)

        can_label = status.target is not None and status.state != "ambiguous-csv"
        can_generate = status.json_path is not None and status.json_path.exists()
        can_generate = can_generate and status.state in {"json-only", "stale-csv"}
        self.open_labeler_btn.setEnabled(can_label)
        self.open_labeler_btn.setText("Open labeler" if status.image_path else "Choose nest image")
        self.generate_brood_csv_btn.setEnabled(can_generate)
        self.refresh_nest_btn.setEnabled(True)

    def load_brood_map(self):
        status = self.nest_status
        if status is None or status.csv_path is None or not status.csv_path.exists():
            self.brood_map = pd.DataFrame()
            self.nest_component_distance = pd.DataFrame()
            self.nest_component_interaction_count = pd.DataFrame()
            return
        try:
            brood = pd.read_csv(status.csv_path)
            required = {"object index", "label", "shape", "x", "y", "radius"}
            if not required.issubset(brood.columns):
                raise ValueError(f"missing columns: {sorted(required - set(brood.columns))}")
            for col in ("object index", "x", "y", "radius"):
                brood[col] = pd.to_numeric(brood[col], errors="coerce")
            self.brood_map = brood.dropna(subset=["object index", "x", "y"])
            self.recompute_nest_component_metrics()
        except Exception as exc:
            self.brood_map = pd.DataFrame()
            self.nest_component_distance = pd.DataFrame()
            self.nest_component_interaction_count = pd.DataFrame()
            QMessageBox.warning(self, "Brood map error", f"Could not read brood map:\n{exc}")

    def _choose_nest_image(self) -> Path | None:
        status = self.nest_status
        start_dir = Path.cwd()
        if status is not None and status.roots:
            start_dir = status.roots[0]
        extensions = " ".join(f"*{ext}" for ext in sorted(NEST_IMAGE_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose nest image",
            str(start_dir),
            f"Nest images ({extensions})",
        )
        if not path:
            return None
        image_path = Path(path)
        if status is not None and status.target is not None:
            image_target = extract_colony_and_date(image_path.name)
            if image_target != status.target:
                QMessageBox.warning(
                    self,
                    "Filename does not match session",
                    (
                        "The selected image does not encode the same colony/date as this session.\n\n"
                        f"Session: colony {status.target[0]} on {status.target[1]}\n"
                        f"Selected: {image_path.name}\n\n"
                        "Choose or rename a nest image like col_15-2021-06-12-nest_image.png."
                    ),
                )
                return None
        return image_path

    def open_nest_labeler(self):
        if self.labeling_process is not None and self.labeling_process.state() == QProcess.Running:
            QMessageBox.information(self, "Labeler already open", "The nest-labeling tool is already running.")
            return

        status = self.nest_status
        if status is None:
            return
        image_path = status.image_path or self._choose_nest_image()
        if image_path is None:
            return

        script_path = Path(__file__).with_name("LabelNests_GUI.1.16.py")
        if not script_path.exists():
            QMessageBox.critical(self, "Missing labeler", f"Could not find:\n{script_path}")
            return

        labelmerc_path = Path(__file__).with_name("labelmerc")
        args = [str(script_path), str(image_path.parent), "--start-image", image_path.name]
        if labelmerc_path.exists():
            args.extend(["--labelmerc", str(labelmerc_path)])

        self.labeling_image_path = image_path
        self.labeling_csv_target = status.csv_path or image_path.with_suffix(".csv")
        self.labeling_process = QProcess(self)
        self.labeling_process.setProgram(sys.executable)
        self.labeling_process.setArguments(args)
        self.labeling_process.setProcessChannelMode(QProcess.SeparateChannels)
        self.labeling_process.readyReadStandardOutput.connect(self._dump_labeler_output)
        self.labeling_process.readyReadStandardError.connect(self._dump_labeler_error)
        self.labeling_process.finished.connect(self.on_nest_labeler_finished)
        self.labeling_process.start()
        self.nest_status_label.setText(f"Nest labeler opened for:\n{image_path.name}")

    def _dump_labeler_output(self):
        if self.labeling_process is None:
            return
        out = bytes(self.labeling_process.readAllStandardOutput()).decode(errors="ignore")
        if out.strip():
            print("LabelNests:", out.strip())

    def _dump_labeler_error(self):
        if self.labeling_process is None:
            return
        err = bytes(self.labeling_process.readAllStandardError()).decode(errors="ignore")
        if err.strip():
            print("LabelNests stderr:", err.strip())

    def on_nest_labeler_finished(self, exit_code: int, _exit_status):
        image_path = self.labeling_image_path
        csv_target = self.labeling_csv_target
        self.labeling_process = None
        self.labeling_image_path = None
        self.labeling_csv_target = None

        if exit_code != 0:
            QMessageBox.warning(self, "Labeler closed with error", f"Labeler exited with code {exit_code}.")

        if image_path is not None:
            json_path = image_path.with_suffix(".json")
            csv_path = csv_target or image_path.with_suffix(".csv")
            if json_path.exists() and (
                not csv_path.exists() or json_path.stat().st_mtime > csv_path.stat().st_mtime
            ):
                try:
                    convert_labelme_json_to_csv(json_path, csv_path)
                except Exception as exc:
                    QMessageBox.warning(self, "CSV generation failed", f"Could not generate brood CSV:\n{exc}")
        self.refresh_nest_status()

    def generate_current_brood_csv(self):
        status = self.nest_status
        if status is None or status.json_path is None or not status.json_path.exists():
            QMessageBox.warning(self, "No LabelMe JSON", "No LabelMe JSON is available to convert yet.")
            return
        csv_path = status.csv_path
        if csv_path is None and status.image_path is not None:
            csv_path = status.image_path.with_suffix(".csv")
        try:
            written = convert_labelme_json_to_csv(status.json_path, csv_path)
            self.nest_status_label.setText(f"Generated brood CSV:\n{written}")
        except Exception as exc:
            QMessageBox.warning(self, "CSV generation failed", f"Could not generate brood CSV:\n{exc}")
        self.refresh_nest_status()

    def recompute_nest_component_metrics(self):
        self.nest_component_distance, self.nest_component_interaction_count = compute_nest_component_tables(
            self.tracking,
            self.brood_map,
            cutoff=self.interaction_cutoff.value(),
        )

    def current_tracking_pattern(self) -> str | None:
        if self.tracking_csv_preference is not None:
            return self.tracking_csv_preference
        session = self.current_session()
        path = self.selected_tracking_path()
        if session is not None and path in session.tracking_paths:
            return tracking_csv_pattern(session.video_path, path)
        return None

    def social_center_tracking_paths(self) -> tuple[Path, ...]:
        session = self.current_session()
        path = self.selected_tracking_path()
        if session is None:
            return (path.expanduser().resolve(),) if path is not None else ()

        target = session_colony_date(session)
        if target is None:
            return (path.expanduser().resolve(),) if path is not None else ()

        paths = social_center_tracking_paths_for_sessions(
            self.sessions,
            target,
            self.current_tracking_pattern(),
        )
        if paths:
            return paths
        return (path.expanduser().resolve(),) if path is not None else ()

    def update_social_center(self):
        paths = self.social_center_tracking_paths()
        center = social_center_from_tracking_paths(paths)
        if center is None and not self.tracking.empty:
            center = (
                float(self.tracking["centroidX"].mean()),
                float(self.tracking["centroidY"].mean()),
            )
            self.social_center_file_count = 1
        else:
            self.social_center_file_count = len(paths)
        self.social_center = center

    def recompute_metrics(self):
        if self.tracking.empty:
            self.activity = pd.DataFrame()
            self.speed = pd.DataFrame()
            self.social_center_distance = pd.DataFrame()
            self.nearest_neighbor_distance = pd.DataFrame()
            self.interaction_count = pd.DataFrame()
            self.nest_component_distance = pd.DataFrame()
            self.nest_component_interaction_count = pd.DataFrame()
            self.social_center = None
            self.social_center_file_count = 0
            return
        self.update_social_center()
        self.activity, self.speed = compute_activity_tables(
            self.tracking,
            frame_rate=self.behavior_fps.value(),
            max_gap_seconds=self.max_gap.value(),
            speed_cutoff=self.speed_cutoff.value(),
        )
        self.social_center_distance = compute_social_center_distance_table(
            self.tracking,
            self.social_center,
        )
        self.nearest_neighbor_distance = compute_nearest_neighbor_distance_table(self.tracking)
        self.interaction_count = compute_interaction_count_table(
            self.tracking,
            cutoff=self.interaction_cutoff.value(),
        )
        self.recompute_nest_component_metrics()
        self.update_focus_plot()
        self.render_current_frame()

    def selected_bee(self):
        return self.focus_bee.currentData()

    def toggle_playback(self):
        if self.timer.isActive():
            self.stop_playback()
        else:
            interval = int(1000 / max(1.0, self.video_fps))
            self.timer.start(interval)
            self.play_button.setText("Pause")

    def stop_playback(self):
        self.timer.stop()
        self.play_button.setText("Play")

    def previous_session(self):
        if not self.sessions:
            return
        self.session_list.setCurrentRow(max(0, self.session_index - 1))

    def next_session(self):
        if not self.sessions:
            return
        self.session_list.setCurrentRow(min(len(self.sessions) - 1, self.session_index + 1))

    def previous_frame(self):
        self.seek_frame(max(0, self.current_frame - 1))

    def next_frame(self):
        if self.frame_count and self.current_frame >= self.frame_count - 1:
            self.stop_playback()
            return
        if not self.seek_frame(self.current_frame + 1):
            self.stop_playback()

    def seek_frame(self, frame: int) -> bool:
        target = max(0, int(frame))
        if self.frame_count:
            target = min(target, self.frame_count - 1)
        previous_frame = self.current_frame
        self.current_frame = target
        if not self.render_current_frame():
            self.current_frame = previous_frame
            return False
        return True

    def update_zoom_label(self, zoom: float):
        self.zoom_label.setText(f"Zoom {zoom * 100:.0f}%")

    def update_legend_button(self, visible: bool):
        self.legend_button.setText("Hide key" if visible else "Show key")
        self.legend_button.setToolTip("Hide the video overlay key" if visible else "Show the video overlay key")

    def current_legend_items(self) -> list[tuple[str, str, QColor]]:
        items: list[tuple[str, str, QColor]] = []
        if self.show_noid_tags.isChecked() and not self.noid_tags.empty:
            items.append(("diamond", "noID tag", NOID_TAG_COLOR))

        if self.tracking.empty:
            return items

        if self.show_activity.isChecked():
            items.extend(
                [
                    ("dot", "Active", QColor("#19b66a")),
                    ("dot", "Inactive", QColor("#d8b13f")),
                    ("dot", "Unknown", QColor("#9aa6b2")),
                ]
            )

        if self.show_tracks.isChecked():
            items.append(("ring", "Tracking point", QColor("#66d9ef")))

        if self.show_speed_labels.isChecked():
            items.append(("text", "Speed label", QColor("#f6f8fb")))

        if self.show_interactions.isChecked():
            items.append(("line", "Interaction", QColor(82, 180, 255, 180)))

        if self.show_nest_interactions.isChecked() and not self.brood_map.empty:
            items.append(("dot", "Nest interaction", QColor(247, 168, 49, 190)))

        if self.show_nest_distances.isChecked() and not self.brood_map.empty:
            items.append(("line", "Nest distance", QColor(220, 220, 255, 210)))

        if self.show_social_center.isChecked():
            items.append(("cross", "Social center", QColor("#ff5c8a")))

        if self.show_trails.isChecked():
            items.append(("line", "Trail", QColor("#66d9ef")))

        if self.show_nest_map.isChecked() and not self.brood_map.empty:
            labels = sorted(
                {
                    str(label)
                    for label in self.brood_map["label"].dropna().unique()
                    if str(label).strip()
                }
            )
            for label in labels[:5]:
                bgr = LABEL_COLORS_BGR.get(label, (255, 0, 255))
                items.append(("line", label, QColor(bgr[2], bgr[1], bgr[0], 185)))
            if len(labels) > 5:
                items.append(("text", f"+ {len(labels) - 5} more nest labels", QColor("#f6f8fb")))

        return items

    def reopen_capture(self) -> bool:
        session = self.current_session()
        if session is None:
            return False
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(str(session.video_path))
        self._capture_next_frame = 0
        return self.capture.isOpened()

    def read_frame_by_reopening(self, frame_index: int):
        if not self.reopen_capture():
            return None
        frame = None
        for _ in range(frame_index + 1):
            ok, frame = self.capture.read()
            if not ok:
                self._capture_next_frame = None
                return None
        return frame

    def read_video_frame(self, frame_index: int):
        if self._last_frame_index == frame_index and self._last_frame_bgr is not None:
            return self._last_frame_bgr.copy() if hasattr(self._last_frame_bgr, "copy") else self._last_frame_bgr
        if self.capture is None:
            return None

        frame = None
        ok = False
        if self._capture_next_frame == frame_index:
            ok, frame = self.capture.read()
        else:
            if self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                ok, frame = self.capture.read()
            if not ok:
                frame = self.read_frame_by_reopening(frame_index)
                ok = frame is not None

        if not ok:
            return None
        self._last_frame_bgr = frame
        self._last_frame_index = frame_index
        self._capture_next_frame = frame_index + 1
        return frame

    def render_current_frame(self) -> bool:
        if self.capture is None:
            return False

        frame = self.read_video_frame(self.current_frame)
        if frame is None:
            return False
        self._last_frame_bgr = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        pixmap = self.draw_overlays(pixmap, w, h)
        self.video_label.set_legend_items(self.current_legend_items())
        self.video_label.set_video_pixmap(pixmap)
        self.timeline.blockSignals(True)
        self.timeline.setValue(self.current_frame)
        self.timeline.blockSignals(False)
        self.frame_label.setText(f"Frame {self.current_frame} / {max(0, self.frame_count - 1)}")
        self.frame_marker.setValue(self.current_frame)
        return True

    def draw_overlays(self, pixmap: QPixmap, width: int, height: int) -> QPixmap:
        rows = self.rows_by_frame.get(self.current_frame)
        noid_rows = self.noid_rows_by_frame.get(self.current_frame)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        selected = self.selected_bee()

        if self.show_trails.isChecked():
            self.draw_trails(painter, selected)

        if self.show_nest_map.isChecked() and not self.brood_map.empty:
            self.draw_brood_map(painter)

        if self.show_noid_tags.isChecked() and noid_rows is not None:
            self.draw_noid_tags(painter, noid_rows)

        if rows is not None and not self.brood_map.empty:
            if self.show_nest_distances.isChecked():
                self.draw_nest_component_links(painter, rows, selected, interactions_only=False)
            if self.show_nest_interactions.isChecked():
                self.draw_active_nest_components(painter, rows, selected)

        if self.show_interactions.isChecked() and rows is not None:
            self.draw_interactions(painter, rows, selected)

        if self.show_social_center.isChecked() and self.social_center is not None:
            self.draw_social_center(painter, rows)

        stats = {"active": 0, "inactive": 0, "unknown": 0}
        if rows is not None:
            for _, row in rows.iterrows():
                bee_id = int(row.ID)
                if selected is not None and bee_id != selected:
                    dim = True
                else:
                    dim = False
                state = self.activity_state(self.current_frame, bee_id)
                stats[state] += 1
                speed_value = self.speed_value(self.current_frame, bee_id)
                self.draw_bee(painter, row, state, speed_value, dim)

        painter.end()
        total = sum(stats.values())
        noid_count_text = ""
        if not self.noid_tags.empty:
            noid_count = 0 if noid_rows is None else len(noid_rows)
            noid_count_text = f"\nnoID detections: {noid_count}"
        if self.tracking.empty:
            text = f"Frame {self.current_frame}\n{self.tracking_status_message}{noid_count_text}"
        else:
            social_center_text = (
                f"\nSocial center files: {self.social_center_file_count}"
                if self.social_center is not None and self.social_center_file_count
                else ""
            )
            text = (
                f"Frame {self.current_frame}\n"
                f"{self.tracking_status_message}\n"
                f"Detections: {total}\n"
                f"Active: {stats['active']}  Inactive: {stats['inactive']}  Unknown: {stats['unknown']}"
                f"{social_center_text}"
                f"{noid_count_text}"
            )
        self.stats_label.setText(text)
        return pixmap

    def activity_state(self, frame: int, bee_id: int) -> str:
        if self.activity.empty or frame not in self.activity.index or bee_id not in self.activity.columns:
            return "unknown"
        return classify_activity_value(self.activity.loc[frame, bee_id])

    def speed_value(self, frame: int, bee_id: int):
        if self.speed.empty or frame not in self.speed.index or bee_id not in self.speed.columns:
            return np.nan
        return self.speed.loc[frame, bee_id]

    def draw_bee(self, painter: QPainter, row, state: str, speed_value, dim: bool):
        x, y = float(row.centroidX), float(row.centroidY)
        base = stable_color(int(row.ID))
        activity_colors = {
            "active": QColor("#19b66a"),
            "inactive": QColor("#d8b13f"),
            "unknown": QColor("#9aa6b2"),
        }
        fill = activity_colors[state] if self.show_activity.isChecked() else base
        if dim:
            fill.setAlpha(70)

        marker_scale = 4.0
        radius = (9 if state == "active" else 7) * marker_scale
        ring_gap = 5 * marker_scale
        pen_width = max(2, int(2 * marker_scale))

        painter.save()
        painter.setPen(QPen(QColor("#111111"), pen_width))
        painter.setBrush(fill)
        painter.drawEllipse(QPointF(x, y), radius, radius)

        if self.show_tracks.isChecked():
            painter.setPen(QPen(base, pen_width))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(x, y), radius + ring_gap, radius + ring_gap)

        id_font = painter.font()
        id_font.setPixelSize(int(14 * marker_scale))
        id_font.setBold(True)
        painter.setFont(id_font)
        painter.setPen(QPen(QColor("#f6f8fb"), max(1, int(marker_scale / 2))))
        painter.drawText(
            QRect(
                int(x + 11 * marker_scale),
                int(y - 18 * marker_scale),
                int(80 * marker_scale),
                int(18 * marker_scale),
            ),
            Qt.AlignLeft | Qt.AlignVCenter,
            str(int(row.ID)),
        )

        if self.show_speed_labels.isChecked():
            speed_font = painter.font()
            speed_font.setPixelSize(int(12 * marker_scale))
            speed_font.setBold(False)
            painter.setFont(speed_font)
            text = "unknown" if pd.isna(speed_value) else f"{speed_value:.2f} px/fr"
            painter.drawText(
                QRect(
                    int(x + 11 * marker_scale),
                    int(y + 2 * marker_scale),
                    int(120 * marker_scale),
                    int(18 * marker_scale),
                ),
                Qt.AlignLeft | Qt.AlignVCenter,
                text,
            )

        painter.restore()

    def draw_noid_tags(self, painter: QPainter, rows: pd.DataFrame):
        marker_radius = 28
        pen_width = 5
        fill = QColor(NOID_TAG_COLOR)
        fill.setAlpha(95)
        outline = QColor(NOID_TAG_COLOR)
        outline.setAlpha(230)

        painter.save()
        font = painter.font()
        font.setPixelSize(30)
        font.setBold(True)
        painter.setFont(font)
        for _, row in rows.iterrows():
            x, y = float(row.centroidX), float(row.centroidY)
            diamond = QPolygonF(
                [
                    QPointF(x, y - marker_radius),
                    QPointF(x + marker_radius, y),
                    QPointF(x, y + marker_radius),
                    QPointF(x - marker_radius, y),
                ]
            )
            painter.setPen(QPen(outline, pen_width))
            painter.setBrush(QBrush(fill))
            painter.drawPolygon(diamond)
            painter.setPen(QPen(QColor("#f6f8fb"), 2))
            painter.drawText(
                QRect(int(x - marker_radius), int(y - marker_radius), marker_radius * 2, marker_radius * 2),
                Qt.AlignCenter,
                "?",
            )
        painter.restore()

    def draw_interactions(self, painter: QPainter, rows: pd.DataFrame, selected):
        cutoff = self.interaction_cutoff.value()
        values = rows[["ID", "centroidX", "centroidY"]].to_numpy()
        painter.setPen(QPen(QColor(82, 180, 255, 150), 2))
        for i in range(len(values)):
            bee_a, x1, y1 = values[i]
            for j in range(i + 1, len(values)):
                bee_b, x2, y2 = values[j]
                if selected is not None and selected not in {int(bee_a), int(bee_b)}:
                    continue
                d = float(np.hypot(x1 - x2, y1 - y2))
                if d <= cutoff:
                    painter.drawLine(QPointF(float(x1), float(y1)), QPointF(float(x2), float(y2)))

    def draw_social_center(self, painter: QPainter, rows: pd.DataFrame | None):
        cx, cy = self.social_center
        painter.setPen(QPen(QColor("#ff5c8a"), 3))
        painter.drawLine(QPointF(cx - 14, cy), QPointF(cx + 14, cy))
        painter.drawLine(QPointF(cx, cy - 14), QPointF(cx, cy + 14))
        if rows is not None:
            painter.setPen(QPen(QColor(255, 92, 138, 90), 1))
            for _, row in rows.iterrows():
                painter.drawLine(QPointF(cx, cy), QPointF(float(row.centroidX), float(row.centroidY)))

    def draw_trails(self, painter: QPainter, selected):
        if self.tracking.empty:
            return
        start = max(0, self.current_frame - self.trail_frames.value())
        trail = self.tracking[
            (self.tracking["frame"] >= start) &
            (self.tracking["frame"] <= self.current_frame)
        ]
        if selected is not None:
            trail = trail[trail["ID"] == selected]
        for bee_id, rows in trail.groupby("ID"):
            rows = rows.sort_values("frame")
            if len(rows) < 2:
                continue
            color = stable_color(int(bee_id))
            color.setAlpha(170)
            painter.setPen(QPen(color, TRAIL_LINE_WIDTH))
            points = [QPointF(float(r.centroidX), float(r.centroidY)) for _, r in rows.iterrows()]
            for a, b in zip(points[:-1], points[1:]):
                painter.drawLine(a, b)

    def draw_nest_component_links(self, painter: QPainter, rows: pd.DataFrame, selected, *, interactions_only: bool):
        components = nest_components_from_brood_map(self.brood_map)
        if not components:
            return

        color = QColor(220, 220, 255, 210)
        width = NEST_DISTANCE_LINE_WIDTH
        painter.setPen(QPen(color, width))
        painter.setBrush(Qt.NoBrush)

        for _, row in rows.iterrows():
            bee_id = int(row.ID)
            if selected is not None and bee_id != selected:
                continue
            point = (float(row.centroidX), float(row.centroidY))
            nearest = nearest_nest_component(point, components)
            if nearest is None:
                continue
            distance, nearest_point, _component = nearest

            start = QPointF(point[0], point[1])
            end = QPointF(nearest_point[0], nearest_point[1])
            if distance == 0:
                painter.drawEllipse(start, 14, 14)
            else:
                painter.drawLine(start, end)

    def draw_active_nest_components(self, painter: QPainter, rows: pd.DataFrame, selected):
        components = nest_components_from_brood_map(self.brood_map)
        if not components:
            return

        cutoff = self.interaction_cutoff.value()
        active_components: list[NestComponent] = []
        for _, row in rows.iterrows():
            bee_id = int(row.ID)
            if selected is not None and bee_id != selected:
                continue
            point = (float(row.centroidX), float(row.centroidY))
            for component in components:
                distance, _nearest = distance_to_nest_component(point, component)
                if distance <= cutoff and component not in active_components:
                    active_components.append(component)

        for component in active_components:
            self.draw_nest_component_shape(
                painter,
                component,
                fill_alpha=NEST_INTERACTION_HIGHLIGHT_ALPHA,
                line_width=NEST_INTERACTION_LINE_WIDTH,
            )

    def draw_nest_component_shape(
        self,
        painter: QPainter,
        component: NestComponent,
        *,
        fill_alpha: int = 0,
        line_width: int = NEST_MAP_LINE_WIDTH,
    ):
        bgr = LABEL_COLORS_BGR.get(component.label, (255, 0, 255))
        line_color = QColor(bgr[2], bgr[1], bgr[0], 230)
        fill_color = QColor(bgr[2], bgr[1], bgr[0], fill_alpha)
        points = [QPointF(x, y) for x, y in component.points]
        if not points:
            return

        painter.save()
        painter.setPen(QPen(line_color, line_width))
        painter.setBrush(QBrush(fill_color) if fill_alpha else Qt.NoBrush)

        shape = component.shape.lower()
        if shape == "circle":
            painter.drawEllipse(points[0], component.radius, component.radius)
        elif shape == "point":
            painter.setBrush(QBrush(fill_color if fill_alpha else line_color))
            painter.drawEllipse(points[0], max(6, line_width * 2), max(6, line_width * 2))
        elif shape == "polygon" and len(points) >= 3:
            painter.drawPolygon(QPolygonF(points))
        elif shape == "rectangle" and len(points) >= 2:
            p1, p2 = points[0], points[1]
            rect = QRect(
                int(min(p1.x(), p2.x())),
                int(min(p1.y(), p2.y())),
                int(abs(p2.x() - p1.x())),
                int(abs(p2.y() - p1.y())),
            )
            painter.drawRect(rect)
        elif len(points) >= 2:
            for a, b in zip(points[:-1], points[1:]):
                painter.drawLine(a, b)

        painter.restore()

    def draw_brood_map(self, painter: QPainter):
        for component in nest_components_from_brood_map(self.brood_map, metrics_only=False):
            self.draw_nest_component_shape(painter, component)

    def select_focus_bee(self, bee_id: int | None):
        for index in range(self.focus_bee.count()):
            if self.focus_bee.itemData(index) == bee_id:
                self.focus_bee.setCurrentIndex(index)
                return

    def restore_plot_defaults(self):
        for combo in (self.plot_metric, self.speed_units, self.plot_mode):
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self.select_focus_bee(None)
        self.update_focus_plot()
        self.render_current_frame()

    def current_plot_mode(self) -> str:
        return self.plot_mode.currentData() or "lines"

    def current_plot_metric(self) -> str:
        return self.plot_metric.currentData() or "speed"

    def plot_metric_table(self) -> tuple[pd.DataFrame, str, str]:
        metric = self.current_plot_metric()
        if metric == "activity":
            return self.activity, "Activity state", "0/1"
        if metric == "dist_sc":
            return self.social_center_distance, "Distance to social center", "px"
        if metric == "nearest_neighbor":
            return self.nearest_neighbor_distance, "Nearest neighbor distance", "px"
        if metric == "interactions":
            return self.interaction_count, "Interaction count", "bees"
        if metric == "nest_distance":
            return self.nest_component_distance, "Distance to nest component", "px"
        if metric == "nest_interactions":
            return self.nest_component_interaction_count, "Nest interaction count", "components"

        unit = self.speed_units.currentText()
        table = convert_speed_table_units(
            self.speed,
            unit,
            frame_rate=self.behavior_fps.value(),
            px_per_cm=pixels_per_cm,
        )
        return table, "Speed", unit

    def frame_ticks(self, first_frame: int, last_frame: int) -> list[tuple[int, str]]:
        if last_frame <= first_frame:
            return [(first_frame, str(first_frame))]
        n_ticks = min(6, last_frame - first_frame + 1)
        ticks = np.linspace(first_frame, last_frame, num=n_ticks)
        values = sorted({int(round(tick)) for tick in ticks})
        if values[0] != first_frame:
            values.insert(0, first_frame)
        if values[-1] != last_frame:
            values.append(last_frame)
        return [(value, str(value)) for value in values]

    def configure_plot_bounds(self, table: pd.DataFrame):
        if self.tracking.empty:
            return
        first_frame = int(self.tracking["frame"].min())
        last_frame = int(self.tracking["frame"].max())
        frame_span = max(1, last_frame - first_frame)
        self.speed_plot.setLimits(
            xMin=first_frame,
            xMax=last_frame,
            minXRange=1,
            maxXRange=frame_span,
        )
        self.speed_plot.setXRange(first_frame, last_frame, padding=0)
        self.speed_plot.getAxis("bottom").setTicks([self.frame_ticks(first_frame, last_frame)])

        if self.current_plot_metric() == "activity" and self.current_plot_mode() == "lines":
            self.speed_plot.setLimits(yMin=-0.1, yMax=1.1, minYRange=0.1, maxYRange=1.2)
            self.speed_plot.setYRange(-0.1, 1.1, padding=0)
        else:
            self.speed_plot.setLimits(yMin=None, yMax=None, minYRange=None, maxYRange=None)

        if self.current_plot_mode() in {"heatmap", "ethogram"} and not table.empty:
            self.speed_plot.setYRange(-0.5, max(0.5, len(table.columns) - 0.5), padding=0)

    def add_reference_lines(self):
        if self.current_plot_metric() != "speed":
            return
        cutoff = convert_speed_cutoff_units(
            float(self.speed_cutoff.value()),
            self.speed_units.currentText(),
            frame_rate=self.behavior_fps.value(),
            px_per_cm=pixels_per_cm,
        )
        line = pg.InfiniteLine(
            pos=cutoff,
            angle=0,
            movable=False,
            pen=pg.mkPen(QColor("#f05d5e"), width=1, style=Qt.DashLine),
        )
        line.setZValue(5)
        self.speed_plot.addItem(line)

    def metric_unknown_mask(self, table: pd.DataFrame, bee_id: int | None) -> pd.Series:
        if table.empty:
            return pd.Series(dtype=bool)
        if bee_id is not None and bee_id in table.columns:
            return table[bee_id].isna()
        return table.isna().all(axis=1)

    def add_unknown_metric_regions(self, table: pd.DataFrame, bee_id: int | None):
        for start, end in contiguous_true_ranges(self.metric_unknown_mask(table, bee_id)):
            region = pg.LinearRegionItem(
                values=(start - 0.5, end + 0.5),
                orientation="vertical",
                movable=False,
                brush=QColor(154, 166, 178, 42),
            )
            for line in region.lines:
                line.setPen(pg.mkPen(QColor(154, 166, 178, 0)))
            region.setZValue(-10)
            self.speed_plot.addItem(region)

    def plot_metric_series(self, bee_id: int, series: pd.Series, *, selected: bool = False):
        clean = series.dropna()
        if clean.empty:
            return
        color = stable_color(bee_id)
        color.setAlpha(240 if selected else 185)
        width = PLOT_SELECTED_LINE_WIDTH if selected else PLOT_LINE_WIDTH
        pen = pg.mkPen(color, width=width)
        self.speed_plot.plot(clean.index.to_numpy(), clean.to_numpy(), pen=pen)
        self.plot_series_by_bee[bee_id] = clean

    def plot_metric_heatmap(self, table: pd.DataFrame):
        if table.empty:
            return

        frames = np.arange(int(table.index.min()), int(table.index.max()) + 1)
        bees = [int(bee_id) for bee_id in table.columns]
        heat = table.reindex(frames).to_numpy(dtype=float)
        if heat.size == 0:
            return

        self.heatmap_bees = bees
        image = pg.ImageItem()
        image.setImage(heat, autoLevels=True)
        image.setRect(QRectF(frames[0] - 0.5, -0.5, len(frames), len(bees)))
        image.setZValue(-20)
        self.speed_plot.addItem(image)

        axis = self.speed_plot.getAxis("left")
        ticks = [(row, str(bee_id)) for row, bee_id in enumerate(bees)]
        axis.setTicks([ticks])

    def ethogram_table(self, table: pd.DataFrame) -> pd.DataFrame:
        return ethogram_state_table(
            table,
            self.current_plot_metric(),
            cutoff=self.interaction_cutoff.value(),
            activity=self.activity,
        )

    def ethogram_color(self, state: int | None) -> QColor:
        metric = self.current_plot_metric()
        if state is None:
            return QColor(154, 166, 178, 120)
        if metric in {"activity", "speed"}:
            return QColor("#19b66a") if state else QColor("#d8b13f")
        if metric in {"nest_interactions", "nest_distance"}:
            return QColor(247, 168, 49, 220) if state else QColor(60, 63, 67, 80)
        if metric in {"interactions", "nearest_neighbor"}:
            return QColor(82, 180, 255, 210) if state else QColor(60, 63, 67, 80)
        return QColor(183, 139, 255, 205) if state else QColor(60, 63, 67, 80)

    def plot_metric_ethogram(self, table: pd.DataFrame):
        states = self.ethogram_table(table)
        if states.empty:
            return

        bees = [int(bee_id) for bee_id in states.columns]
        self.heatmap_bees = bees
        self.speed_plot.setLabel("left", "Bee ID")
        axis = self.speed_plot.getAxis("left")
        axis.setTicks([[(row, str(bee_id)) for row, bee_id in enumerate(bees)]])

        for row_index, bee_id in enumerate(bees):
            for start, end, state in contiguous_value_ranges(states[bee_id]):
                rect = QGraphicsRectItem(start - 0.5, row_index - 0.38, end - start + 1, 0.76)
                rect.setPen(QPen(Qt.NoPen))
                rect.setBrush(QBrush(self.ethogram_color(state)))
                rect.setZValue(-20)
                self.speed_plot.addItem(rect)

    def nearest_plot_bee(self, frame_value: float, speed_value: float) -> tuple[int, int] | None:
        if not self.plot_series_by_bee:
            return None

        x_range, y_range = self.speed_plot.viewRange()
        frame_tolerance = max(2.0, (x_range[1] - x_range[0]) * 0.02)
        speed_tolerance = max(1.0, (y_range[1] - y_range[0]) * 0.12)
        best = None
        best_score = float("inf")

        for bee_id, series in self.plot_series_by_bee.items():
            frame_distances = np.abs(series.index.to_numpy(dtype=float) - frame_value)
            nearest_i = int(np.argmin(frame_distances))
            nearest_frame = int(series.index[nearest_i])
            nearest_speed = float(series.iloc[nearest_i])
            dx = float(frame_distances[nearest_i])
            dy = abs(nearest_speed - speed_value)
            if dx > frame_tolerance or dy > speed_tolerance:
                continue
            score = (dx / frame_tolerance) ** 2 + (dy / speed_tolerance) ** 2
            if score < best_score:
                best = (bee_id, nearest_frame)
                best_score = score

        return best

    def nearest_heatmap_bee(self, y_value: float) -> int | None:
        if not self.heatmap_bees:
            return None
        index = int(round(y_value))
        if 0 <= index < len(self.heatmap_bees):
            return self.heatmap_bees[index]
        return None

    def on_speed_plot_clicked(self, event):
        if event.button() != Qt.LeftButton:
            return
        view_box = self.speed_plot.plotItem.vb
        if not view_box.sceneBoundingRect().contains(event.scenePos()):
            return

        point = view_box.mapSceneToView(event.scenePos())
        frame = int(round(point.x()))
        is_double_click = getattr(event, "double", lambda: False)()
        if is_double_click:
            self.select_focus_bee(None)
            self.seek_frame(max(0, min(frame, max(0, self.frame_count - 1))))
            return

        if self.current_plot_mode() in {"heatmap", "ethogram"}:
            bee_id = self.nearest_heatmap_bee(point.y())
            if bee_id is not None:
                self.select_focus_bee(bee_id)
        else:
            selected = self.nearest_plot_bee(point.x(), point.y())
            if selected is not None:
                bee_id, frame = selected
                self.select_focus_bee(bee_id)
        self.seek_frame(max(0, min(frame, max(0, self.frame_count - 1))))

    def update_focus_plot(self):
        self.speed_plot.clear()
        self.plot_series_by_bee = {}
        self.heatmap_bees = []
        self.frame_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#e6c229", width=1))
        self.speed_plot.addItem(self.frame_marker)
        table, label, units = self.plot_metric_table()
        self.speed_units.setEnabled(self.current_plot_metric() == "speed" and self.current_plot_mode() != "ethogram")
        self.speed_plot.setLabel("left", label, units=units)
        self.speed_plot.setLabel("bottom", "Frame (CSV)")
        self.speed_plot.getAxis("left").setTicks(None)
        self.configure_plot_bounds(table)

        bee_id = self.selected_bee()
        if self.current_plot_mode() != "ethogram":
            self.add_unknown_metric_regions(table, bee_id)
            self.add_reference_lines()

        if table.empty:
            return

        if self.current_plot_mode() == "heatmap":
            self.plot_metric_heatmap(table)
        elif self.current_plot_mode() == "ethogram":
            self.plot_metric_ethogram(table)
        elif bee_id is None:
            for plot_bee_id in table.columns:
                self.plot_metric_series(int(plot_bee_id), table[plot_bee_id])
        elif bee_id in table.columns:
            self.plot_metric_series(int(bee_id), table[bee_id], selected=True)
        else:
            return

        self.frame_marker.setValue(self.current_frame)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", help="Optional video file or folder to open immediately")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = ReviewHub(Path(args.source) if args.source else None)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
