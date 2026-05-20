import csv
import json

import cv2
import numpy as np

from nest_labeling import (
    BROOD_CSV_COLUMNS,
    convert_labelme_json_to_csv,
    extract_colony_and_date,
    labelme_json_to_rows,
    seeded_labelme_data,
)


def _write_json(path, shapes, **extra):
    payload = {
        "version": "5.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": "previous.png",
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 200,
    }
    payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_extract_colony_and_date_accepts_video_filename_variants():
    assert extract_colony_and_date("col_15-2021-06-12_00-00-01.mjpeg") == (15, "2021-06-12")
    assert extract_colony_and_date("bumblebox-17_2024-08-11_00_30_02.mp4") == (17, "2024-08-11")
    assert extract_colony_and_date("17_2024-08-11_00_30_02.mp4") == (17, "2024-08-11")


def test_labelme_json_to_rows_converts_supported_shapes(tmp_path):
    json_path = tmp_path / "col_15-2021-06-12-nest_image.json"
    _write_json(
        json_path,
        [
            {
                "label": "Larvae (circles)",
                "group_id": 7,
                "shape_type": "circle",
                "points": [[10, 20], [13, 24]],
            },
            {
                "label": "Eggs perimeter (polygons)",
                "group_id": None,
                "shape_type": "polygon",
                "points": [[1, 2], [3, 4], [5, 6]],
            },
            {
                "label": "ignored",
                "group_id": None,
                "shape_type": "unsupported",
                "points": [[0, 0]],
            },
        ],
    )

    rows = labelme_json_to_rows(json_path)

    assert rows[0] == [0, "Larvae (circles)", 7, 1, "circle", 10, 20, 5.0]
    assert rows[1][:7] == [1, "Eggs perimeter (polygons)", None, 1, "polygon", 1, 2]
    assert rows[2][:7] == [1, "Eggs perimeter (polygons)", None, 2, "polygon", 3, 4]
    assert rows[3][:7] == [1, "Eggs perimeter (polygons)", None, 3, "polygon", 5, 6]
    assert np.isnan(rows[1][7])
    assert np.isnan(rows[2][7])
    assert np.isnan(rows[3][7])


def test_convert_labelme_json_to_csv_writes_brood_map_schema(tmp_path):
    json_path = tmp_path / "col_15-2021-06-12-nest_image.json"
    csv_path = tmp_path / "col_15-2021-06-12-nest_image.csv"
    _write_json(
        json_path,
        [
            {
                "label": "Eggs (points)",
                "group_id": None,
                "shape_type": "point",
                "points": [[11, 12]],
            }
        ],
    )

    written = convert_labelme_json_to_csv(json_path)

    assert written == csv_path
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == BROOD_CSV_COLUMNS
    assert rows[1] == ["0", "Eggs (points)", "", "1", "point", "11", "12", "nan"]


def test_seeded_labelme_data_updates_target_image_metadata(tmp_path):
    prev_json = tmp_path / "previous.json"
    target_image = tmp_path / "col_15-2021-06-13-nest_image.png"
    cv2.imwrite(str(target_image), np.zeros((4, 7, 3), dtype=np.uint8))
    _write_json(
        prev_json,
        [
            {
                "label": "Nest perimeter (polygon)",
                "group_id": None,
                "shape_type": "polygon",
                "points": [[0, 0], [1, 0], [1, 1]],
            }
        ],
    )

    seeded = seeded_labelme_data(prev_json, target_image)

    assert seeded["imagePath"] == target_image.name
    assert seeded["imageHeight"] == 4
    assert seeded["imageWidth"] == 7
    assert seeded["imageData"]
    assert seeded["shapes"][0]["label"] == "Nest perimeter (polygon)"
