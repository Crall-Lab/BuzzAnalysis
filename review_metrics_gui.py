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
from PySide6.QtCore import QPointF, QRect, Qt, QProcess, QTimer
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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

from aux import movement_metrics
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
)


VIDEO_EXTENSIONS = {".mp4", ".mjpeg", ".avi", ".mov", ".mkv"}
REQUIRED_TRACKING_COLUMNS = {"frame", "ID", "centroidX", "centroidY"}


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


def classify_activity_value(value) -> str:
    if pd.isna(value):
        return "unknown"
    return "active" if int(value) == 1 else "inactive"


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
        self.setWindowTitle("Open review source")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose a video or a folder of videos to review."))
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


class VideoLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background: #101418; color: #d8e2e7;")
        self.setText("Open a video or folder to begin review")


class ReviewHub(QMainWindow):
    def __init__(self, source: Path | None = None):
        super().__init__()
        self.setWindowTitle("BuzzAnalysis Metric Review Hub")
        self.resize(1500, 900)

        self.sessions: list[ReviewSession] = []
        self.session_index = -1
        self.capture: cv2.VideoCapture | None = None
        self.current_frame = 0
        self.frame_count = 0
        self.video_fps = frame_per_sec
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self.tracking = pd.DataFrame()
        self.rows_by_frame: dict[int, pd.DataFrame] = {}
        self.activity = pd.DataFrame()
        self.speed = pd.DataFrame()
        self.bee_ids: list[int] = []
        self.social_center: tuple[float, float] | None = None
        self.active_tracking_path: Path | None = None
        self.tracking_status_message = "No tracking loaded"
        self.nest_status: NestMapStatus | None = None
        self.brood_map = pd.DataFrame()
        self.labeling_process: QProcess | None = None
        self.labeling_image_path: Path | None = None
        self.labeling_csv_target: Path | None = None
        self._last_frame_bgr = None

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
        left_layout.addWidget(QLabel("Sessions"))
        self.session_list = QListWidget()
        self.session_list.currentRowChanged.connect(self.load_session_at)
        left_layout.addWidget(self.session_list)
        nav_row = QHBoxLayout()
        prev_video = QPushButton("Previous")
        next_video = QPushButton("Next")
        prev_video.clicked.connect(self.previous_session)
        next_video.clicked.connect(self.next_session)
        nav_row.addWidget(prev_video)
        nav_row.addWidget(next_video)
        left_layout.addLayout(nav_row)
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
        controls.addWidget(self.play_button)
        controls.addWidget(prev_frame)
        controls.addWidget(next_frame)
        controls.addWidget(self.frame_label)
        controls.addStretch()
        center_layout.addLayout(controls)
        splitter.addWidget(center)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._make_layer_group())
        right_layout.addWidget(self._make_params_group())
        right_layout.addWidget(self._make_nest_group())
        right_layout.addWidget(self._make_inspector_group())
        splitter.addWidget(right)
        splitter.setSizes([260, 900, 340])

    def _make_layer_group(self) -> QWidget:
        box = QGroupBox("Metric Layers")
        layout = QVBoxLayout(box)
        self.show_tracks = QCheckBox("Tracking points")
        self.show_activity = QCheckBox("Activity state")
        self.show_speed_labels = QCheckBox("Speed labels")
        self.show_interactions = QCheckBox("Interactions")
        self.show_social_center = QCheckBox("Social center")
        self.show_trails = QCheckBox("Trails")
        self.show_nest_map = QCheckBox("Nest map")
        for checkbox, checked in [
            (self.show_tracks, True),
            (self.show_activity, True),
            (self.show_speed_labels, False),
            (self.show_interactions, False),
            (self.show_social_center, False),
            (self.show_trails, False),
            (self.show_nest_map, True),
        ]:
            checkbox.setChecked(checked)
            checkbox.stateChanged.connect(self.render_current_frame)
            layout.addWidget(checkbox)
        return box

    def _make_params_group(self) -> QWidget:
        box = QGroupBox("Metric Controls")
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
        self.interaction_cutoff.valueChanged.connect(self.render_current_frame)

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
        return box

    def _make_nest_group(self) -> QWidget:
        box = QGroupBox("Nest Map")
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
        return box

    def _make_inspector_group(self) -> QWidget:
        box = QGroupBox("Frame Inspector")
        layout = QVBoxLayout(box)
        self.stats_label = QLabel("No tracking loaded")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)
        self.speed_plot = pg.PlotWidget()
        self.speed_plot.setMinimumHeight(180)
        self.speed_plot.setLabel("left", "Speed", units="px/frame")
        self.speed_plot.setLabel("bottom", "Frame")
        self.frame_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#e6c229", width=1))
        self.speed_plot.addItem(self.frame_marker)
        layout.addWidget(self.speed_plot)
        return box

    def open_startup_dialog(self):
        dialog = StartupDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        if dialog.choice == "video":
            self.open_video()
        elif dialog.choice == "folder":
            self.open_folder()

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            str(Path.cwd()),
            "Videos (*.mp4 *.mjpeg *.avi *.mov *.mkv)",
        )
        if path:
            self.load_source(Path(path))

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open folder", str(Path.cwd()))
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
        for session in self.sessions:
            item = QListWidgetItem(session.label)
            if not session.tracking_paths:
                color = "#d8903f" if session.related_csv_paths else "#b64b4b"
                item.setForeground(QColor(color))
            self.session_list.addItem(item)

        if not self.sessions:
            QMessageBox.warning(self, "No videos found", "No supported video files were found.")
            return
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

        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.video_fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or frame_per_sec
        self.current_frame = 0
        self.timeline.setRange(0, max(0, self.frame_count - 1))
        self.populate_tracking_sources(session)
        self.load_selected_tracking(show_warning=True)

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

    def change_tracking_source(self, _index: int):
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
        self.bee_ids = []
        self.social_center = None
        self.active_tracking_path = None
        self.tracking_status_message = message
        self.focus_bee.blockSignals(True)
        self.focus_bee.clear()
        self.focus_bee.addItem("All bees", None)
        self.focus_bee.blockSignals(False)

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
        self.social_center = (
            float(self.tracking["centroidX"].mean()),
            float(self.tracking["centroidY"].mean()),
        )
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
            return
        try:
            brood = pd.read_csv(status.csv_path)
            required = {"object index", "label", "shape", "x", "y", "radius"}
            if not required.issubset(brood.columns):
                raise ValueError(f"missing columns: {sorted(required - set(brood.columns))}")
            for col in ("object index", "x", "y", "radius"):
                brood[col] = pd.to_numeric(brood[col], errors="coerce")
            self.brood_map = brood.dropna(subset=["object index", "x", "y"])
        except Exception as exc:
            self.brood_map = pd.DataFrame()
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

    def recompute_metrics(self):
        if self.tracking.empty:
            self.activity = pd.DataFrame()
            self.speed = pd.DataFrame()
            return
        self.activity, self.speed = compute_activity_tables(
            self.tracking,
            frame_rate=self.behavior_fps.value(),
            max_gap_seconds=self.max_gap.value(),
            speed_cutoff=self.speed_cutoff.value(),
        )
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
        self.seek_frame(self.current_frame + 1)

    def seek_frame(self, frame: int):
        self.current_frame = int(frame)
        self.render_current_frame()

    def render_current_frame(self):
        if self.capture is None:
            return

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ok, frame = self.capture.read()
        if not ok:
            return
        self._last_frame_bgr = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        pixmap = self.draw_overlays(pixmap, w, h)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.timeline.blockSignals(True)
        self.timeline.setValue(self.current_frame)
        self.timeline.blockSignals(False)
        self.frame_label.setText(f"Frame {self.current_frame} / {max(0, self.frame_count - 1)}")
        self.frame_marker.setValue(self.current_frame)

    def draw_overlays(self, pixmap: QPixmap, width: int, height: int) -> QPixmap:
        rows = self.rows_by_frame.get(self.current_frame)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        selected = self.selected_bee()

        if self.show_trails.isChecked():
            self.draw_trails(painter, selected)

        if self.show_nest_map.isChecked() and not self.brood_map.empty:
            self.draw_brood_map(painter)

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
        if self.tracking.empty:
            text = f"Frame {self.current_frame}\n{self.tracking_status_message}"
        else:
            text = (
                f"Frame {self.current_frame}\n"
                f"{self.tracking_status_message}\n"
                f"Detections: {total}\n"
                f"Active: {stats['active']}  Inactive: {stats['inactive']}  Unknown: {stats['unknown']}"
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
        radius = 9 if state == "active" else 7
        painter.setPen(QPen(QColor("#111111"), 2))
        painter.setBrush(fill)
        painter.drawEllipse(QPointF(x, y), radius, radius)

        if self.show_tracks.isChecked():
            painter.setPen(QPen(base, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(x, y), radius + 5, radius + 5)

        painter.setPen(QPen(QColor("#f6f8fb"), 2))
        painter.drawText(QRect(int(x + 11), int(y - 18), 80, 18), Qt.AlignLeft, str(int(row.ID)))

        if self.show_speed_labels.isChecked():
            text = "unknown" if pd.isna(speed_value) else f"{speed_value:.2f} px/fr"
            painter.drawText(QRect(int(x + 11), int(y + 2), 120, 18), Qt.AlignLeft, text)

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
            color.setAlpha(130)
            painter.setPen(QPen(color, 2))
            points = [QPointF(float(r.centroidX), float(r.centroidY)) for _, r in rows.iterrows()]
            for a, b in zip(points[:-1], points[1:]):
                painter.drawLine(a, b)

    def draw_brood_map(self, painter: QPainter):
        grouped = self.brood_map.groupby(["object index", "label", "shape"], dropna=False)
        for (_, label, shape), rows in grouped:
            label = "" if pd.isna(label) else str(label)
            shape = "" if pd.isna(shape) else str(shape)
            bgr = LABEL_COLORS_BGR.get(label, (255, 0, 255))
            color = QColor(bgr[2], bgr[1], bgr[0], 185)
            painter.setPen(QPen(color, 3))
            painter.setBrush(Qt.NoBrush)

            points = [QPointF(float(row.x), float(row.y)) for _, row in rows.iterrows()]
            if not points:
                continue

            if shape == "circle":
                row = rows.iloc[0]
                radius = float(row.radius) if not pd.isna(row.radius) else 0.0
                painter.drawEllipse(points[0], radius, radius)
            elif shape == "point":
                painter.setBrush(color)
                painter.drawEllipse(points[0], 6, 6)
                painter.setBrush(Qt.NoBrush)
            elif shape == "polygon" and len(points) >= 3:
                painter.drawPolygon(QPolygonF(points))
            elif shape == "line" and len(points) >= 2:
                for a, b in zip(points[:-1], points[1:]):
                    painter.drawLine(a, b)
            elif shape == "rectangle" and len(points) >= 2:
                p1, p2 = points[0], points[1]
                rect = QRect(
                    int(min(p1.x(), p2.x())),
                    int(min(p1.y(), p2.y())),
                    int(abs(p2.x() - p1.x())),
                    int(abs(p2.y() - p1.y())),
                )
                painter.drawRect(rect)

            painter.setPen(QPen(QColor(255, 255, 255, 190), 1))
            painter.drawText(QRect(int(points[0].x() + 6), int(points[0].y() + 6), 220, 20), Qt.AlignLeft, label)

    def update_focus_plot(self):
        self.speed_plot.clear()
        self.frame_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#e6c229", width=1))
        self.speed_plot.addItem(self.frame_marker)
        bee_id = self.selected_bee()
        if bee_id is None or self.speed.empty or bee_id not in self.speed.columns:
            return
        series = self.speed[bee_id].dropna()
        if series.empty:
            return
        self.speed_plot.plot(series.index.to_numpy(), series.to_numpy(), pen=pg.mkPen("#66d9ef", width=2))
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
