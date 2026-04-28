#!/usr/bin/env python3
"""Utilities for LabelMe-based nest labeling and brood-map export."""

from __future__ import annotations

import base64
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


BROOD_CSV_COLUMNS = [
    "object index",
    "label",
    "label ID",
    "vertex ID",
    "shape",
    "x",
    "y",
    "radius",
]

NEST_LABELS = [
    "Arena perimeter (polygon)",
    "Nest perimeter (polygon)",
    "Eggs perimeter (polygons)",
    "Eggs (points)",
    "Larvae (circles)",
    "Pupae (circles)",
    "Queen larva (circles)",
    "Queen pupae (circles)",
    "Wax pots (circles)",
    "full nectar pot (circles)",
    "empty wax pots (circles)",
    "pollen balls (circles)",
    "nectar source (circle)",
    "left temp probe (rectangle)",
    "right temp probe (rectangle)",
]

CALIBRATION_LABELS = ["Calibration A->B (line)"]

# OpenCV uses BGR colors.
LABEL_COLORS_BGR = {
    "Arena perimeter (polygon)": (0, 0, 128),
    "Nest perimeter (polygon)": (0, 128, 0),
    "Eggs perimeter (polygons)": (0, 128, 128),
    "Eggs (points)": (128, 0, 0),
    "Larvae (circles)": (128, 0, 128),
    "Pupae (circles)": (128, 128, 0),
    "Queen larva (circles)": (128, 128, 128),
    "Queen pupae (circles)": (0, 0, 64),
    "Wax pots (circles)": (0, 0, 192),
    "full nectar pot (circles)": (0, 128, 64),
    "empty wax pots (circles)": (0, 128, 192),
    "pollen balls (circles)": (128, 0, 64),
    "nectar source (circle)": (128, 0, 192),
    "left temp probe (rectangle)": (0, 0, 255),
    "right temp probe (rectangle)": (0, 165, 255),
    "Calibration A->B (line)": (0, 255, 0),
}

SUPPORTED_SHAPES = {"circle", "point", "polygon", "line", "rectangle"}
NEST_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def extract_colony_and_date(name: str | os.PathLike[str]) -> tuple[int, str] | None:
    """Parse colony number and YYYY-MM-DD date from a tracking/nest filename."""
    normalized = Path(name).name.replace("_", "-")
    match = re.search(r"col-?0*(\d+)-(\d{4}-\d{2}-\d{2})", normalized, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def _matches_target(path: Path, target: tuple[int, str]) -> bool:
    return extract_colony_and_date(path.name) == target


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted({path.expanduser().resolve() for path in paths})


def find_matching_brood_csvs(
    root: str | os.PathLike[str],
    target: tuple[int, str],
    brood_extension: str = "_nest_image.csv",
) -> list[Path]:
    """Find brood CSVs in *root* that match a colony/date target."""
    root = Path(root).expanduser()
    if not root.exists():
        return []

    ext_clean = brood_extension.lstrip("_-")
    valid_suffixes = {ext_clean, f"_{ext_clean}", f"-{ext_clean}"}
    matches = []
    for path in root.rglob("*.csv"):
        if path.name.startswith("."):
            continue
        if not any(path.name.endswith(suffix) for suffix in valid_suffixes):
            continue
        if _matches_target(path, target):
            matches.append(path)
    return _dedupe_paths(matches)


def find_matching_nest_images(
    root: str | os.PathLike[str],
    target: tuple[int, str],
) -> list[Path]:
    """Find nest images in *root* that match a colony/date target."""
    root = Path(root).expanduser()
    if not root.exists():
        return []

    matches = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in NEST_IMAGE_EXTENSIONS:
            continue
        if "nest" not in path.stem.lower():
            continue
        if _matches_target(path, target):
            matches.append(path)
    return _dedupe_paths(matches)


def load_labelme_json(json_path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_labelme_json(data: dict[str, Any], json_path: str | os.PathLike[str]) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def labelme_json_signature(json_path: str | os.PathLike[str]) -> str | None:
    """Return a stable signature of the saved LabelMe shapes, or None if absent."""
    path = Path(json_path)
    if not path.exists():
        return None
    data = load_labelme_json(path)
    return json.dumps(data.get("shapes", []), sort_keys=True, separators=(",", ":"))


def seeded_labelme_data(
    previous_json_path: str | os.PathLike[str],
    target_image_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Copy previous annotations onto a new image and refresh image metadata."""
    data = load_labelme_json(previous_json_path)
    target_image_path = Path(target_image_path)
    image = cv2.imread(str(target_image_path))
    if image is None:
        raise ValueError(f"Could not read target image: {target_image_path}")

    height, width = image.shape[:2]
    data["imagePath"] = target_image_path.name
    data["imageHeight"] = int(height)
    data["imageWidth"] = int(width)
    with open(target_image_path, "rb") as img_f:
        data["imageData"] = base64.b64encode(img_f.read()).decode("utf-8")
    return data


def write_seeded_labelme_json(
    previous_json_path: str | os.PathLike[str],
    target_image_path: str | os.PathLike[str],
    target_json_path: str | os.PathLike[str],
) -> dict[str, Any]:
    data = seeded_labelme_data(previous_json_path, target_image_path)
    dump_labelme_json(data, target_json_path)
    return data


def _row_for_point(
    object_index: int,
    label: str | None,
    label_id: Any,
    vertex_id: int,
    shape_type: str,
    point: Iterable[float],
):
    x, y = point
    return [object_index, label, label_id, vertex_id, shape_type, x, y, math.nan]


def labelme_json_to_rows(json_path: str | os.PathLike[str]) -> list[list[Any]]:
    """Convert a LabelMe JSON file into rows accepted by brood distance code."""
    nest = load_labelme_json(json_path)
    rows: list[list[Any]] = []
    for object_index, shape in enumerate(nest.get("shapes", [])):
        label = shape.get("label")
        shape_type = shape.get("shape_type")
        label_id = shape.get("group_id")
        points = shape.get("points", [])

        if shape_type not in SUPPORTED_SHAPES:
            continue

        if shape_type == "circle":
            if len(points) < 2:
                continue
            x_center, y_center = points[0]
            x_perimeter, y_perimeter = points[1]
            radius = ((x_center - x_perimeter) ** 2 + (y_center - y_perimeter) ** 2) ** 0.5
            rows.append([object_index, label, label_id, 1, shape_type, x_center, y_center, radius])
        elif shape_type == "point":
            if len(points) < 1:
                continue
            rows.append(_row_for_point(object_index, label, label_id, 1, shape_type, points[0]))
        else:
            for vertex_id, point in enumerate(points, start=1):
                rows.append(_row_for_point(object_index, label, label_id, vertex_id, shape_type, point))

    return rows


def convert_labelme_json_to_csv(
    json_path: str | os.PathLike[str],
    csv_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Write one brood-map CSV from one LabelMe JSON file."""
    json_path = Path(json_path)
    if csv_path is None:
        csv_path = json_path.with_suffix(".csv")
    csv_path = Path(csv_path)
    tmp_path = csv_path.with_name(f".{csv_path.name}.tmp")

    rows = labelme_json_to_rows(json_path)
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BROOD_CSV_COLUMNS)
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)
    return csv_path


def convert_folder_json_to_csv(folder_path: str | os.PathLike[str]) -> list[Path]:
    """Convert every LabelMe JSON file in a folder into same-stem CSV files."""
    folder = Path(folder_path)
    outputs = []
    for json_path in sorted(folder.glob("*.json")):
        outputs.append(convert_labelme_json_to_csv(json_path))
    return outputs


def generate_annotated_image(
    image_path: str | os.PathLike[str],
    json_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> bool:
    """Draw LabelMe shapes and a compact legend onto an image."""
    data = load_labelme_json(json_path)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: failed to load image for annotation overlay: {image_path}")
        return False

    used_labels = set()
    thickness = 10

    for shape in data.get("shapes", []):
        label = shape.get("label")
        pts = shape.get("points", [])
        s_type = shape.get("shape_type", "polygon")
        bgr = LABEL_COLORS_BGR.get(label, (255, 0, 255))
        used_labels.add((str(label), str(s_type), bgr))

        if s_type == "point":
            if len(pts) < 1:
                continue
            x, y = int(pts[0][0]), int(pts[0][1])
            cv2.circle(image, (x, y), 10, bgr, thickness=-1)
        elif s_type == "circle":
            if len(pts) < 2:
                continue
            cx, cy = int(pts[0][0]), int(pts[0][1])
            ex, ey = int(pts[1][0]), int(pts[1][1])
            radius = int(np.hypot(ex - cx, ey - cy))
            cv2.circle(image, (cx, cy), radius, bgr, thickness=thickness)
        elif s_type == "line":
            if len(pts) < 2:
                continue
            pt1 = (int(pts[0][0]), int(pts[0][1]))
            pt2 = (int(pts[1][0]), int(pts[1][1]))
            cv2.line(image, pt1, pt2, bgr, thickness=thickness)
        elif s_type == "polygon":
            if len(pts) < 3:
                continue
            arr = np.array(pts, dtype=np.int32)
            cv2.polylines(image, [arr], isClosed=True, color=bgr, thickness=thickness)
        elif s_type == "rectangle":
            if len(pts) < 2:
                continue
            p1 = (int(pts[0][0]), int(pts[0][1]))
            p2 = (int(pts[1][0]), int(pts[1][1]))
            cv2.rectangle(image, p1, p2, bgr, thickness=thickness)

    h, _, _ = image.shape
    legend_x = 10
    legend_y = h - 60
    rect_w, rect_h = 60, 50
    spacing = 10
    column_width = 430
    for lbl, _, bgr in sorted(used_labels):
        if legend_y < 10:
            legend_x += column_width
            legend_y = h - 60
        cv2.rectangle(
            image,
            (legend_x, legend_y),
            (legend_x + rect_w, legend_y + rect_h),
            bgr,
            thickness=-1,
        )
        cv2.putText(
            image,
            lbl,
            (legend_x + rect_w + 20, legend_y + rect_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        legend_y -= rect_h + spacing

    return bool(cv2.imwrite(str(output_path), image))
