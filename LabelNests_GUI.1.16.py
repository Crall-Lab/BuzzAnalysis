import argparse
import sys
import os
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import QProcess, QProcessEnvironment
from PySide6.QtWidgets import QMessageBox

import shutil
import importlib.util

from nest_labeling import (
    convert_folder_json_to_csv,
    generate_annotated_image,
    labelme_json_signature,
    seeded_labelme_data,
    write_seeded_labelme_json,
)

class ZoomLabel(QtWidgets.QLabel):
    """
    A QLabel subclass that:
    1) Fits the pixmap to its QScrollArea on load or when requested.
    2) Supports Ctrl+wheel for zooming in/out (after fit).
    3) Supports click-and-drag panning by adjusting scrollbars in its QScrollArea.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.base_pixmap = None
        self.scale_factor = 1.0
        self._drag_start = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setScaledContents(False)

    def setPixmap(self, pixmap: QtGui.QPixmap):
        """
        Store base pixmap, then fit it to the viewport.
        """
        self.base_pixmap = pixmap
        self.scale_factor = 1.0
        self.fit_to_viewport()

    def fit_to_viewport(self):
        """
        Scale base_pixmap to exactly fill its QScrollArea viewport (keep aspect ratio),
        then adjust label size so no scrollbars until zoom.
        """
        if self.base_pixmap is None:
            print("self.base_pixmap is None")
            return
        scroll = self._find_scroll_area()
        if scroll is not None:
            avail = scroll.viewport().size()
            if avail.width() > 0 and avail.height() > 0:
                fitted = self.base_pixmap.scaled(
                    avail, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                )
                super().setPixmap(fitted)
                self.scale_factor = fitted.width() / self.base_pixmap.width()
                self.adjustSize()
                return
        else:
            print("In fit to viewport function: scroll is None")
        # fallback: display full
        super().setPixmap(self.base_pixmap)
        self.scale_factor = 1.0
        self.adjustSize()

    def apply_current_zoom(self):
        """
        Re‐render base_pixmap at whatever self.scale_factor currently is.
        (Used when swapping in a new base_pixmap but wanting to keep the previous zoom.)
        """
        if self.base_pixmap is None:
            return
        # Compute new size and re‐scale from the full‐res base_pixmap
        new_size = self.base_pixmap.size() * self.scale_factor
        scaled = self.base_pixmap.scaled(
            new_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        super().setPixmap(scaled)
        self.adjustSize()


    def wheelEvent(self, event: QtGui.QWheelEvent):
        """
        When Ctrl is held, zoom in/out. Otherwise, propagate to scrollbars.
        """
        if event.modifiers() & QtCore.Qt.ControlModifier and self.base_pixmap is not None:
            delta = event.angleDelta().y()
            if delta > 0:
                self.scale_factor *= 1.25
            else:
                self.scale_factor *= 0.8

            if self.scale_factor < 0.1:
                self.scale_factor = 0.1

            new_size = self.base_pixmap.size() * self.scale_factor
            scaled = self.base_pixmap.scaled(
                new_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            super().setPixmap(scaled)
            self.adjustSize()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """
        On left-click, record the position for panning.
        """
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        """
        On mouse move with left button, compute delta from last position and scroll.
        """
        if self._drag_start and (event.buttons() & QtCore.Qt.LeftButton):
            delta = event.pos() - self._drag_start
            scroll = self._find_scroll_area()
            if scroll:
                hbar = scroll.horizontalScrollBar()
                vbar = scroll.verticalScrollBar()
                hbar.setValue(hbar.value() - delta.x())
                vbar.setValue(vbar.value() - delta.y())
            self._drag_start = event.pos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self._drag_start = None
        super().mouseReleaseEvent(event)

    def _find_scroll_area(self):
        widget = self.parent()
        while widget and not isinstance(widget, QtWidgets.QScrollArea):
            widget = widget.parent()
        return widget


class LabelNestsApp(QtWidgets.QWidget):
    """
    A PySide6-based GUI that shows:
    1) A top status bar (QLabel)
    2) Row 1 of toggles [Toggle Prev Compare (left)     Toggle Next Compare (right)]
    3) A QSplitter with three containers: [ prev_container | main_scroll | next_container ]
       - prev_container and next_container each hold:
           * a “Toggle Annotations” button
           * a ZoomLabel inside a non-resizable QScrollArea
    4) Row 2 of navigation & actions [← Previous] [Open in LabelMe] [Toggle Annotations] [Fit to Window] [Next →]
       centered under the splitter.
    """

    def __init__(self, folder_path, start_image=None, labelmerc_path=None):
        super().__init__()

        # ------------- State & File List -------------
        self.folder_path = os.path.abspath(folder_path)
        self.image_list = sorted([
            f for f in os.listdir(self.folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.endswith('_annotated.png')
        ])
        self.current_index = 0
        if start_image and start_image in self.image_list:
            self.current_index = self.image_list.index(start_image)

        self.autoloaded_data = None
        self.viewing_annotated_current = False
        self.prev_compare_visible = False
        self.next_compare_visible = False
        self.viewing_annotated_prev = False
        self.viewing_annotated_next = False
        self.labelme_process = None
        self.labelme_image_path = None
        self.labelme_json_path = None
        self.labelme_initial_json_data = None
        self.labelme_post_json_data = None
        self.labelmerc_path_override = os.path.abspath(labelmerc_path) if labelmerc_path else None

        # ------------- Top Status Bar -------------
        self.message_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.message_label.setFixedHeight(60) # was 30
        self.message_label.setStyleSheet('background-color: black; color: yellow;')

        # ------------- MAIN IMAGE Pane -------------
        self.main_label = ZoomLabel()
        self.main_scroll_area = QtWidgets.QScrollArea()
        self.main_scroll_area.setWidget(self.main_label)
        self.main_scroll_area.setWidgetResizable(True)
        self.main_scroll_area.setAlignment(QtCore.Qt.AlignCenter)
        self.main_scroll_area.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.main_scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.main_scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        #self.main_scroll_area.setVisible(True)

        # ------------- PREVIOUS COMPARISON Pane -------------
        self.toggle_prev_btn = QtWidgets.QPushButton('Toggle Previous Image')
        self.toggle_prev_btn.clicked.connect(self.toggle_prev_compare)
        # Zoom‐in/out buttons for the previous pane
        self.prev_zoom_in_btn = QtWidgets.QPushButton("Zoom +")
        self.prev_zoom_in_btn.clicked.connect(lambda: self.manual_zoom(self.prev_label, 1.1))
        self.prev_zoom_out_btn = QtWidgets.QPushButton("Zoom −")
        self.prev_zoom_out_btn.clicked.connect(lambda: self.manual_zoom(self.prev_label, 0.9))
        self.prev_toggle_annot_btn = QtWidgets.QPushButton("Toggle Previous Annotations")
        self.prev_toggle_annot_btn.clicked.connect(self.toggle_prev_annotation)
        self.prev_label = ZoomLabel()
        self.prev_scroll_area = QtWidgets.QScrollArea()
        self.prev_scroll_area.setWidget(self.prev_label)
        self.prev_scroll_area.setWidgetResizable(False)
        self.prev_scroll_area.setAlignment(QtCore.Qt.AlignCenter)
        self.prev_scroll_area.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.prev_scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.prev_scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.prev_scroll_area.setVisible(True)

        #--------------MAIN Pane--------------------------

        self.main_zoom_in_btn = QtWidgets.QPushButton('Zoom +')
        self.main_zoom_in_btn.clicked.connect(lambda: self.manual_zoom(self.main_label, 1.1))
        self.toggle_current_btn = QtWidgets.QPushButton('Toggle Annotations for Current Image')
        self.toggle_current_btn.clicked.connect(self.toggle_current_annotation_view)
        self.main_zoom_out_btn = QtWidgets.QPushButton('Zoom −')
        self.main_zoom_out_btn.clicked.connect(lambda: self.manual_zoom(self.main_label, 0.9))

        # ------------- NEXT COMPARISON Pane -------------
        self.toggle_next_btn = QtWidgets.QPushButton('Toggle Next Image')
        self.toggle_next_btn.clicked.connect(self.toggle_next_compare)
        # Zoom‐in/out buttons for the next pane
        self.next_zoom_in_btn = QtWidgets.QPushButton("Zoom +")
        self.next_zoom_in_btn.clicked.connect(lambda: self.manual_zoom(self.next_label, 1.1))
        self.next_zoom_out_btn = QtWidgets.QPushButton("Zoom −")
        self.next_zoom_out_btn.clicked.connect(lambda: self.manual_zoom(self.next_label, 0.9))
        self.next_toggle_annot_btn = QtWidgets.QPushButton("Toggle Next Annotations")
        self.next_toggle_annot_btn.clicked.connect(self.toggle_next_annotation)
        self.next_label = ZoomLabel()
        self.next_scroll_area = QtWidgets.QScrollArea()
        self.next_scroll_area.setWidget(self.next_label)
        self.next_scroll_area.setWidgetResizable(False)
        self.next_scroll_area.setAlignment(QtCore.Qt.AlignCenter)
        self.next_scroll_area.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.next_scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.next_scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.next_scroll_area.setVisible(True)

        # Create two mini-layouts: one for all the left-pane buttons, and one for all the right-pane buttons
        left_group = QtWidgets.QHBoxLayout()
        left_group.addWidget(self.toggle_prev_btn)
        left_group.addWidget(self.prev_zoom_in_btn)
        left_group.addWidget(self.prev_zoom_out_btn)
        left_group.addWidget(self.prev_toggle_annot_btn)

        middle_group = QtWidgets.QHBoxLayout()
        middle_group.addWidget(self.main_zoom_in_btn)
        middle_group.addWidget(self.toggle_current_btn)
        middle_group.addWidget(self.main_zoom_out_btn)


        right_group = QtWidgets.QHBoxLayout()
        right_group.addWidget(self.next_toggle_annot_btn)
        right_group.addWidget(self.next_zoom_in_btn)
        right_group.addWidget(self.next_zoom_out_btn)
        right_group.addWidget(self.toggle_next_btn)

        # Now create the top-level row1 layout, inserting stretches to push the groups into the thirds
        row1_layout = QtWidgets.QHBoxLayout()
        #row1_layout.addStretch(1)            # left margin
        row1_layout.addLayout(left_group)    # all left-pane buttons stay together
        row1_layout.addStretch(1)            # bigger stretch to occupy the center third
        row1_layout.addLayout(middle_group)
        row1_layout.addStretch(1)
        row1_layout.addLayout(right_group)   # all right-pane buttons stay together
        #row1_layout.addStretch(1)            # right margin


        # ------------- Splitter Holding the Three Panes -------------
        self.image_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.image_splitter.addWidget(self.prev_scroll_area)
        self.image_splitter.addWidget(self.main_scroll_area)
        self.image_splitter.addWidget(self.next_scroll_area)

        # ------------- Row 2: ←Prev, Open LabelMe, Toggle Annotations, Fit to Window, Next→ (below splitter) -------------
        self.prev_nav_btn = QtWidgets.QPushButton('← Previous')
        self.prev_nav_btn.clicked.connect(self.go_to_previous_image)

        self.open_labelme_btn = QtWidgets.QPushButton('Open in LabelMe')
        self.open_labelme_btn.clicked.connect(self.launch_labelme_for_current_image)


        # … after creating self.toggle_current_btn …

        self.fit_btn = QtWidgets.QPushButton('Fit to Window')
        self.fit_btn.clicked.connect(self.reset_all_views)

        self.next_nav_btn = QtWidgets.QPushButton('Next →')
        self.next_nav_btn.clicked.connect(self.go_to_next_image)

        self.json_to_csv_btn = QtWidgets.QPushButton('Convert all JSON files in folder to CSV')
        self.json_to_csv_btn.clicked.connect(self.show_popup)
        #self.json_to_csv_btn.clicked.connect(self.convert_json_to_csv)

        row2_layout = QtWidgets.QHBoxLayout()
        row2_layout.addStretch(1)
        row2_layout.addWidget(self.prev_nav_btn)
        row2_layout.addWidget(self.open_labelme_btn)
        row2_layout.addWidget(self.fit_btn)
        row2_layout.addWidget(self.next_nav_btn)
        row2_layout.addStretch(1)
        row2_layout.addWidget(self.json_to_csv_btn)

        # ------------- Combined Layout -------------
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.message_label)
        main_layout.addLayout(row1_layout)       # Row 1
        main_layout.addWidget(self.image_splitter)
        main_layout.addLayout(row2_layout)       # Row 2
        self.setLayout(main_layout)

        self.setWindowTitle('LabelNests Annotation Interface')
        self.resize(1800, 800)
        self._set_controls_enabled(bool(self.image_list))
        if self.image_list:
            self.update_image()  # Load the first image on startup
        else:
            self.message_label.setText(
                "No images found.\nAdd .png/.jpg/.jpeg files to this folder and relaunch."
            )

    def _set_controls_enabled(self, enabled):
        controls = [
            self.toggle_prev_btn,
            self.prev_zoom_in_btn,
            self.prev_zoom_out_btn,
            self.prev_toggle_annot_btn,
            self.main_zoom_in_btn,
            self.toggle_current_btn,
            self.main_zoom_out_btn,
            self.next_toggle_annot_btn,
            self.next_zoom_in_btn,
            self.next_zoom_out_btn,
            self.toggle_next_btn,
            self.prev_nav_btn,
            self.open_labelme_btn,
            self.fit_btn,
            self.next_nav_btn,
            self.json_to_csv_btn,
        ]
        for control in controls:
            control.setEnabled(enabled)

    @staticmethod
    def _short_image_name(file_name, max_chars=28):
        if not file_name:
            return ""
        base_name, _ = os.path.splitext(os.path.basename(file_name))
        if len(base_name) <= max_chars:
            return base_name
        return f"{base_name[:max_chars - 3]}..."

    def _set_neighbor_toggle_text(self, side, target_name=None, annotated=False, missing_annotations=False):
        if side not in ("prev", "next"):
            return
        label = "Previous" if side == "prev" else "Next"
        button = self.toggle_prev_btn if side == "prev" else self.toggle_next_btn

        if not target_name:
            button.setText(f"Toggle {label} Image")
            return

        text = f"Toggle {label} Image: {self._short_image_name(target_name)}"
        if annotated:
            text += " annotated"
        if missing_annotations:
            text += " - no annotations found"
        button.setText(text)

    def _resolve_labelmerc_path(self):
        candidates = []
        if self.labelmerc_path_override:
            candidates.append(self.labelmerc_path_override)
        env_path = os.environ.get("BUMBLEBOX_LABELMERC")
        if env_path:
            candidates.append(env_path)
        candidates.append(os.path.join(self.folder_path, "labelmerc"))
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "labelmerc"))
        candidates.append(os.path.expanduser("~/.labelmerc"))

        for path in candidates:
            if path and os.path.exists(path):
                return os.path.abspath(path)
        return None

    @staticmethod
    def _labelme_module_available():
        return importlib.util.find_spec("labelme") is not None

    @staticmethod
    def _resolve_labelme_python():
        candidates = []
        override = os.environ.get("BUMBLEBOX_LABELME_PYTHON")
        if override:
            candidates.append(override)

        current_env = os.path.dirname(os.path.dirname(sys.executable))
        envs_root = os.path.dirname(current_env)
        candidates.append(os.path.join(envs_root, "buzzanalysis_labelme", "bin", "python"))
        candidates.append(sys.executable)

        for path in candidates:
            if path and os.path.exists(path) and os.access(path, os.X_OK):
                return os.path.abspath(path)
        return sys.executable

    def update_message_label(self):
        if not self.image_list:
            self.message_label.setText("No images found in the selected folder.")
            return
        
        self.autoloaded_data = None
        image_name = self.image_list[self.current_index]
        image_path = os.path.join(self.folder_path, image_name)
        base_name, _ = os.path.splitext(image_name)
        json_path = os.path.join(self.folder_path, f"{base_name}.json")
        annotated_path = os.path.join(self.folder_path, f"{base_name}_annotated.png")

        # --- Determine status banner text ---
        if os.path.exists(json_path) and os.path.exists(annotated_path):
            json_timestamp = QtCore.QDateTime.fromSecsSinceEpoch(
                int(os.path.getmtime(json_path)))

            annotated_image_timestamp = QtCore.QDateTime.fromSecsSinceEpoch(
                int(os.path.getmtime(annotated_path)))
            
            json_timestamp_str = json_timestamp.toString()
            annotated_image_timestamp_str = annotated_image_timestamp.toString()

            if json_timestamp > annotated_image_timestamp:
                if not self.viewing_annotated_current:
                    
                    print(f"json_timestamp: ", json_timestamp)
                    print(f"annotated_image_timestamp: ", annotated_image_timestamp)
                    print(type(json_timestamp), type(annotated_image_timestamp))

                    self.message_label.setText(
                    f"⚠️ {image_name}\nWARNING: Annotations exist but the annotated image might not be up to date.\nJSON last saved on: {json_timestamp_str}\nAnnotated image generated on: {annotated_image_timestamp_str}"
                )
                else:

                    print(f"json_timestamp: ", json_timestamp)
                    print(f"annotated_image_timestamp: ", annotated_image_timestamp)
                    print(type(json_timestamp), type(annotated_image_timestamp))
                    
                    self.message_label.setText(
                    f"⚠️ {image_name}\nWARNING: Viewing potentially old annotated image, actual annotation data has since been updated.\nJSON last saved on: {json_timestamp_str})\nAnnotated image generated on: {annotated_image_timestamp_str}"
                )
                    
            else:
                self.message_label.setText(
                    f"✅ {image_name}\nAnnotations already exist. JSON last saved on: {json_timestamp_str}\nAnnotated image generated on: {annotated_image_timestamp_str}"
                )


        elif os.path.exists(json_path):
            json_timestamp = QtCore.QDateTime.fromSecsSinceEpoch(
                int(os.path.getmtime(json_path))
                ).toString()
            self.message_label.setText(
                f"✅ ⚠️ {image_name}\nAnnotations created but no annotated image can be found (JSON last saved on: {json_timestamp})"
                )

        else:
            if self.current_index > 0:
                prev_name = self.image_list[self.current_index - 1]
                prev_base, _ = os.path.splitext(prev_name)
                prev_json = os.path.join(self.folder_path, f"{prev_base}.json")
                if os.path.exists(prev_json):
                    try:
                        self.autoloaded_data = seeded_labelme_data(prev_json, image_path)
                        self.message_label.setText(
                            f"📋 {image_name}\nNo annotations - previous day annotations will load for editing in LabelMe"
                        )
                    except Exception as exc:
                        self.message_label.setText(
                            f"⚠️ {image_name}\nPrevious annotations exist, but could not be prepared: {exc}"
                        )
                else:
                    self.message_label.setText(
                        f"📋 {image_name}\nNo current or previous image annotations found"
                    )
            else:
                self.message_label.setText(
                    f"📋 {image_name}\nFirst image in folder - no annotations found"
                )



    def update_image(self):
        """
        (Re)load the "current_index" image. Then:
        1) Update the status bar (green‐check if .json exists; or “loaded from prev” if not).
        2) Fit the main image to the viewport.
        3) If prev_compare_visible == True, immediately load & fit that pane’s image (raw).
           Likewise for next.
        4) Redistribute splitter sizes so that visible panes share the space.
        """
        if not self.image_list:
            self.message_label.setText("No images found in the selected folder.")
            return

        self.autoloaded_data = None
        image_name = self.image_list[self.current_index]
        image_path = os.path.join(self.folder_path, image_name)
        base_name, _ = os.path.splitext(image_name)
        json_path = os.path.join(self.folder_path, f"{base_name}.json")
        annotated_path = os.path.join(self.folder_path, f"{base_name}_annotated.png")

        if not os.path.exists(annotated_path) and os.path.exists(json_path):
            self.generate_annotated_image(image_path, json_path, annotated_path)
            
        self.update_message_label()

    
        old_scale = self.main_label.scale_factor   # remember whatever zoom was set
        old_hscroll = self.main_scroll_area.horizontalScrollBar().value()
        old_vscroll = self.main_scroll_area.verticalScrollBar().value()
        # --- Load (and fit) the MAIN image, raw vs. annotated ---
        if self.viewing_annotated_current and os.path.exists(annotated_path):
            # If the user has toggled annotations on, and an _annotated.png exists,
            # load that instead of the raw image.
            pixmap = QtGui.QPixmap(annotated_path)
            #super(ZoomLabel, self.main_label).setPixmap(pixmap)
            self.main_label.setPixmap(pixmap)
            self.main_label.scale_factor = old_scale
            if old_scale == 1.0:
                self.main_label.fit_to_viewport()
            else:
                self.main_label.apply_current_zoom()
                self.main_scroll_area.horizontalScrollBar().setValue(old_hscroll)
                self.main_scroll_area.verticalScrollBar().setValue(old_vscroll)
            #self.message_label.setText(f"👁 Current: {base_name}_annotated.png")
        else:
            # Otherwise, load the raw image and reset the flag if no annotation file is found
            pixmap = QtGui.QPixmap(image_path)
            #self.main_label.setPixmap(pixmap)
            self.main_label.base_pixmap = pixmap
            self.main_label.scale_factor = old_scale
            if old_scale == 1.0:
                self.main_label.fit_to_viewport()
            else:
                self.main_label.apply_current_zoom()
                self.main_scroll_area.horizontalScrollBar().setValue(old_hscroll)
                self.main_scroll_area.verticalScrollBar().setValue(old_vscroll)
            #self.message_label.setText(f"👁 Current: {image_name}")
            # If viewing_annotated_current was True but the annotated file doesn't exist,
            # immediately turn the flag back off so we don't get stuck in a bad state.
            if self.viewing_annotated_current and not os.path.exists(annotated_path):
                self.viewing_annotated_current = False
            
            
        # Either way, once the pixmap is set we still need to fit it (unless
        # we just loaded the annotated file at its own size, in which case
        # .adjustSize() above already sized it correctly).
        #if not (self.viewing_annotated_current and os.path.exists(annotated_path)):
        #    self.main_label.fit_to_viewport()

        if old_scale == 1.0:
            self.main_label.fit_to_viewport()

        #self.viewing_annotated_current = False

        # --- If comparison panes are visible, load & fit their raw images ---
        if self.prev_compare_visible:
            self.load_prev_comparison(fit=False)
        if self.next_compare_visible:
            self.load_next_comparison(fit=False)

        # --- Set container visibility and redistribute splitter sizes ---
        self.prev_scroll_area.setVisible(self.prev_compare_visible)
        self.next_scroll_area.setVisible(self.next_compare_visible)
        self.update_splitter_sizes()



    #Note to self: Need to update this to check the time stamps of the json file vs. the annotated image path, to check whether one is newer than the other
    #If the json is newer, we need to regenerate the annotated image path. Do this for this one and for the load_next_comparison
    def load_prev_comparison(self, fit=True):
        """
        Load raw (or annotated, if toggled) into prev_label, then fit.
        Update toggle_prev_btn text to reflect filename.
        """
        target_index = self.current_index - 1
        if target_index < 0 or target_index >= len(self.image_list):
            self._set_neighbor_toggle_text("prev", None)
            return
        target_name = self.image_list[target_index]
        base_name, _ = os.path.splitext(target_name)
        image_path = os.path.join(self.folder_path, target_name)
        annotated_path = os.path.join(self.folder_path, f"{base_name}_annotated.png")
        json_path = os.path.join(self.folder_path, f"{base_name}.json")
        old_scale = self.prev_label.scale_factor

        if self.viewing_annotated_prev:
            if not os.path.exists(annotated_path) and os.path.exists(json_path):
                self.generate_annotated_image(image_path, json_path, annotated_path)
            if os.path.exists(annotated_path) and os.path.exists(json_path):
                json_modification_time = os.path.getmtime(json_path)
                annotated_image_modification_time = os.path.getmtime(annotated_path)
                if json_modification_time > annotated_image_modification_time:
                    print("json file has been modified, generating new annotated image")
                    self.generate_annotated_image(image_path, json_path, annotated_path)
                pixmap = QtGui.QPixmap(annotated_path)
                self._set_neighbor_toggle_text("prev", target_name, annotated=True)
            else:
                pixmap = QtGui.QPixmap(image_path)
                self._set_neighbor_toggle_text("prev", target_name, missing_annotations=True)
        else:
            pixmap = QtGui.QPixmap(image_path)
            self._set_neighbor_toggle_text("prev", target_name)

        self.prev_label.base_pixmap = pixmap
        self.prev_label.scale_factor = old_scale
        if fit:
            self.prev_label.fit_to_viewport()
        elif old_scale == 1.0:
            self.prev_label.fit_to_viewport()
        else:
            self.prev_label.apply_current_zoom()

    def load_next_comparison(self, fit=True):
        """
        Load raw (or annotated, if toggled) into next_label, then fit.
        Update toggle_next_btn text to reflect filename.
        """
        target_index = self.current_index + 1
        if target_index < 0 or target_index >= len(self.image_list):
            self._set_neighbor_toggle_text("next", None)
            return
        target_name = self.image_list[target_index]
        base_name, _ = os.path.splitext(target_name)
        image_path = os.path.join(self.folder_path, target_name)
        annotated_path = os.path.join(self.folder_path, f"{base_name}_annotated.png")
        json_path = os.path.join(self.folder_path, f"{base_name}.json")
        old_scale = self.next_label.scale_factor

        if self.viewing_annotated_next:
            if not os.path.exists(annotated_path) and os.path.exists(json_path):
                self.generate_annotated_image(image_path, json_path, annotated_path)
            if os.path.exists(annotated_path) and os.path.exists(json_path):
                json_modification_time = os.path.getmtime(json_path)
                annotated_image_modification_time = os.path.getmtime(annotated_path)
                if json_modification_time > annotated_image_modification_time:
                    print("json file has been modified, generating new annotated image")
                    self.generate_annotated_image(image_path, json_path, annotated_path)
                pixmap = QtGui.QPixmap(annotated_path)
                self._set_neighbor_toggle_text("next", target_name, annotated=True)
            else:
                pixmap = QtGui.QPixmap(image_path)
                self._set_neighbor_toggle_text("next", target_name, missing_annotations=True)
        else:
            pixmap = QtGui.QPixmap(image_path)
            self._set_neighbor_toggle_text("next", target_name)

        self.next_label.base_pixmap = pixmap
        self.next_label.scale_factor = old_scale
        if fit:
            self.next_label.fit_to_viewport()
        elif old_scale == 1.0:
            self.next_label.fit_to_viewport()
        else:
            self.next_label.apply_current_zoom()

    def reset_all_views(self):
        """
        Fit every visible pane back to its viewport.
        Also reset toggle buttons to generic text if comparison hidden.
        """
        if not self.image_list:
            return
        target_prev_name = self.image_list[self.current_index - 1] if self.current_index > 0 else None
        target_next_name = self.image_list[self.current_index + 1] if self.current_index + 1 < len(self.image_list) else None

        self.main_label.fit_to_viewport()
        if self.prev_compare_visible and target_prev_name:
            self.prev_scroll_area.setVisible(True)
            self.prev_label.fit_to_viewport()
        else:
            self._set_neighbor_toggle_text("prev", target_prev_name)
            self.prev_scroll_area.setVisible(False)
            if not target_prev_name:
                self.prev_compare_visible = False

        if self.next_compare_visible and target_next_name:
            self.next_scroll_area.setVisible(True)
            self.next_label.fit_to_viewport()
        else:
            self._set_neighbor_toggle_text("next", target_next_name)
            self.next_scroll_area.setVisible(False)
            if not target_next_name:
                self.next_compare_visible = False
        self.update_splitter_sizes()

    def toggle_prev_compare(self):
        """Show/hide the previous-image pane."""
        if not self.image_list:
            return
        self.prev_compare_visible = not self.prev_compare_visible
        self.prev_scroll_area.setVisible(self.prev_compare_visible)
        if self.prev_compare_visible:
            if self.current_index == 0:
                self.prev_compare_visible = False
                self.prev_scroll_area.setVisible(False)
                self._set_neighbor_toggle_text("prev", None)
                self.message_label.setText("No previous image available for comparison.")
                self.update_splitter_sizes()
                return
            #self.viewing_annotated_prev = False
            self.prev_scroll_area.setVisible(True)
            self.load_prev_comparison(fit=True)
            QtCore.QTimer.singleShot(0, lambda: self.prev_label.fit_to_viewport())
        else:
            target_prev_name = self.image_list[self.current_index - 1] if self.current_index > 0 else None
            self._set_neighbor_toggle_text("prev", target_prev_name)
            self.prev_scroll_area.setVisible(False)
        self.update_splitter_sizes()

    def toggle_next_compare(self):
        """Show/hide the next-image pane."""
        if not self.image_list:
            return
        self.next_compare_visible = not self.next_compare_visible
        self.next_scroll_area.setVisible(self.next_compare_visible)
        if self.next_compare_visible:
            if self.current_index >= len(self.image_list) - 1:
                self.next_compare_visible = False
                self.next_scroll_area.setVisible(False)
                self._set_neighbor_toggle_text("next", None)
                self.message_label.setText("No next image available for comparison.")
                self.update_splitter_sizes()
                return
            #self.viewing_annotated_next = False
            self.next_scroll_area.setVisible(True)
            self.load_next_comparison(fit=True)
            QtCore.QTimer.singleShot(0, lambda: self.next_label.fit_to_viewport())
        else:
            target_next_name = self.image_list[self.current_index + 1] if self.current_index + 1 < len(self.image_list) else None
            self._set_neighbor_toggle_text("next", target_next_name)
            self.next_scroll_area.setVisible(False)
        self.update_splitter_sizes()


    def generate_annotated_image(self, image_path, json_path, output_path):
        """
        Read LabelMe JSON, draw shapes onto the raw image via OpenCV,
        add a legend in the bottom-left corner, and save to <image>_annotated.png.
        """
        try:
            ok = generate_annotated_image(image_path, json_path, output_path)
            if ok:
                print(f"Annotated image saved to: {output_path}")
            return ok
        except Exception as exc:
            print(f"Error generating annotated image for {image_path}: {exc}")
            return False


    def toggle_current_annotation_view(self):
        """
        Switch between raw main image ↔ annotated overlay.
        If annotation .png doesn’t exist but .json does, generate the overlay first.
        After toggling, fit the result to viewport.
        """
        image_name = self.image_list[self.current_index]
        base_name, _ = os.path.splitext(image_name)
        image_path = os.path.join(self.folder_path, image_name)
        annotated_path = os.path.join(self.folder_path, f"{base_name}_annotated.png")
        json_path = os.path.join(self.folder_path, f"{base_name}.json")

        old_scale = self.main_label.scale_factor
        old_hscroll = self.main_scroll_area.horizontalScrollBar().value()
        old_vscroll = self.main_scroll_area.verticalScrollBar().value()

        self.viewing_annotated_current = not self.viewing_annotated_current
        if self.viewing_annotated_current:
            if not os.path.exists(annotated_path) and os.path.exists(json_path):
                self.generate_annotated_image(image_path, json_path, annotated_path)
            if os.path.exists(annotated_path):
                new_pixmap = QtGui.QPixmap(annotated_path)
                self.main_label.base_pixmap = new_pixmap
                self.main_label.scale_factor = old_scale
                if old_scale == 1.0:
                    self.main_label.fit_to_viewport()
                else:
                    self.main_label.apply_current_zoom()
                    self.main_scroll_area.horizontalScrollBar().setValue(old_hscroll)
                    self.main_scroll_area.verticalScrollBar().setValue(old_vscroll)
                #self.message_label.setText(f"👁 Viewing annotated: {image_name}")
            else:
                self.viewing_annotated_current = False
                new_pixmap = QtGui.QPixmap(image_path)
                self.main_label.base_pixmap = new_pixmap
                self.main_label.scale_factor = old_scale
                if old_scale == 1.0:
                    self.main_label.fit_to_viewport()
                else:
                    self.main_label.apply_current_zoom()
                    self.main_scroll_area.horizontalScrollBar().setValue(old_hscroll)
                    self.main_scroll_area.verticalScrollBar().setValue(old_vscroll)
                #self.message_label.setText(f"⚠ No annotated image found; showing original.")
        else:
            # turn annotations off: display raw image but keep zoom+pan
            new_pixmap = QtGui.QPixmap(image_path)
            self.main_label.base_pixmap = new_pixmap
            self.main_label.scale_factor = old_scale
            if old_scale == 1.0:
                self.main_label.fit_to_viewport()
            else:
                self.main_label.apply_current_zoom()
                self.main_scroll_area.horizontalScrollBar().setValue(old_hscroll)
                self.main_scroll_area.verticalScrollBar().setValue(old_vscroll)
            #self.message_label.setText(f"👁 Viewing original: {image_name}")
        self.update_message_label()
        
    def toggle_prev_annotation(self):
        """
        Toggle annotated overlay for the *previous* comparison pane.
        Re-load that pane accordingly.
        """
        self.viewing_annotated_prev = not self.viewing_annotated_prev
        self.load_prev_comparison(fit=False)

    def toggle_next_annotation(self):
        """
        Toggle annotated overlay for the *next* comparison pane.
        Re-load that pane accordingly.
        """
        self.viewing_annotated_next = not self.viewing_annotated_next
        self.load_next_comparison(fit=False)

    def store_pre_labelme_json_state(self):
        self.labelme_initial_json_data = labelme_json_signature(self.labelme_json_path)


    def store_post_labelme_json_state(self):
        self.labelme_post_json_data = labelme_json_signature(self.labelme_json_path)


    def launch_labelme_for_current_image(self):
        """
        Open LabelMe on the current image using QProcess. If we previously “seeded” it
        from the last frame, write that .json first so LabelMe picks it up.
        """
        if not self.image_list:
            QMessageBox.warning(self, "No Images", "No images found to label.")
            return

        # --- Check if a labelme instance is already running ---
        if self.labelme_process is not None and self.labelme_process.state() == QProcess.Running:
            print("LabelMe is already running.")
            # Optional: Bring the window to the front if possible, or just inform the user.
            return

        # --- Your existing logic to prepare files ---
        self.autoloaded_data = None
        self.labelme_image_path = None
        self.labelme_json_path = None
        self.labelme_initial_json_data = None

        image_name = self.image_list[self.current_index]
        image_path = os.path.join(self.folder_path, image_name)
        base_name, _ = os.path.splitext(image_name)
        json_path = os.path.join(self.folder_path, f"{base_name}.json")

        self.labelme_image_path = image_path
        self.labelme_json_path = json_path

        if not os.path.exists(json_path):
            print("No JSON file yet exists, checking if the previous day has annotations to load")
            if self.current_index > 0:
                prev_name = self.image_list[self.current_index - 1]
                prev_base, _ = os.path.splitext(prev_name)
                prev_json = os.path.join(self.folder_path, f"{prev_base}.json")
                if os.path.exists(prev_json):
                    try:
                        write_seeded_labelme_json(prev_json, image_path, json_path)
                        print("Successfully loaded the previous day's annotations")
                    except Exception as exc:
                        QMessageBox.warning(
                            self,
                            "Could Not Seed Labels",
                            f"Previous annotations exist, but could not be prepared:\n{exc}",
                        )
                        return
            else:
                print("No previous image to pull JSON data from")

        if os.path.exists(json_path):
            self.store_pre_labelme_json_state()

        labelmerc_path = self._resolve_labelmerc_path()
        if self.labelmerc_path_override and not os.path.exists(self.labelmerc_path_override):
            QMessageBox.warning(
                self,
                "Missing labelmerc",
                (
                    "The provided labelmerc path does not exist.\n"
                    f"{self.labelmerc_path_override}\n\n"
                    "Continuing with LabelMe defaults."
                ),
            )

        # --- The QProcess implementation ---
        # 1. Create the QProcess instance
        self.labelme_process = QProcess(self)
        python_exe = self._resolve_labelme_python()
        use_python_module = python_exe != sys.executable or self._labelme_module_available()
        labelme_executable = shutil.which("labelme")
        if not use_python_module and not labelme_executable:
            QMessageBox.critical(
                self,
                "LabelMe Not Found",
                (
                    "LabelMe is not installed in this environment.\n\n"
                    "Install into BuzzAnalysis with:\n"
                    "  ./install_gui_dependencies.sh\n\n"
                    "or directly with:\n"
                    "  conda install -n BuzzAnalysis -c conda-forge labelme"
                ),
            )
            self.labelme_process = None
            return

        if use_python_module:
            program = python_exe
            arguments = ["-m", "labelme", image_path]
        else:
            program = labelme_executable
            arguments = [image_path]

        if labelmerc_path:
            arguments.extend(["--config", labelmerc_path])
        else:
            print("LabelMe config file (labelmerc) not found; using LabelMe defaults.")

        self.labelme_process.setProgram(program)
        self.labelme_process.setArguments(arguments)
        env = QProcessEnvironment.systemEnvironment()
        venv_bin = os.path.dirname(python_exe)
        labelme_env = os.path.dirname(venv_bin)
        env.insert("PATH", venv_bin + os.pathsep + env.value("PATH"))
        if python_exe == sys.executable:
            env.insert("QT_API", "pyside6")
        else:
            env.insert("QT_API", "pyqt5")
            env.insert("CONDA_PREFIX", labelme_env)
            env.insert("CONDA_DEFAULT_ENV", os.path.basename(labelme_env))
        self.labelme_process.setProcessEnvironment(env)

        # 5. (Optional) capture stderr so you can poke at hidden Qt errors
        self.labelme_process.setProcessChannelMode(QProcess.SeparateChannels)
        self.labelme_process.readyReadStandardError.connect(self._dump_labelme_error)
        self.labelme_process.readyReadStandardOutput.connect(self._dump_labelme_output)
        self.labelme_process.errorOccurred.connect(self._on_labelme_process_error)
        # 2. (Optional but Recommended) Connect to the 'finished' signal
        # This will be called when the user closes the LabelMe window.
        self.labelme_process.finished.connect(self.on_labelme_finished)
        self.labelme_process.start()


    def _dump_labelme_error(self):
        err = bytes(self.labelme_process.readAllStandardError()).decode(errors="ignore")
        if err.strip():
            print("╔══ LabelMe stderr ═════════════╗")
            print(err.strip())
            print("╚═══════════════════════════════╝")

    def _dump_labelme_output(self):
        out = bytes(self.labelme_process.readAllStandardOutput()).decode(errors="ignore")
        if out.strip():
            print("LabelMe:", out.strip())

    def _on_labelme_process_error(self, process_error):
        if process_error == QProcess.FailedToStart:
            QMessageBox.critical(
                self,
                "Failed to launch LabelMe",
                "LabelMe process failed to start. Check that LabelMe and Qt are installed correctly.",
            )
            
    
    def on_labelme_finished(self, exitCode, exitStatus):
        """
        Handles the termination of the LabelMe process.
        """
        if exitStatus == QProcess.CrashExit:
            print("LabelMe crashed or was terminated unexpectedly.")
            # Handle the crash - maybe show an error message to the user.
        else:
            # The process exited normally, now we can check the exit code.
            if exitCode == 0:
                print("LabelMe finished successfully.")
                # Trigger refresh of data
                image_path = self.labelme_image_path
                json_path = self.labelme_json_path
                annotated_path = os.path.splitext(image_path)[0] + '_annotated.png'
            
                if os.path.exists(json_path):
                    self.store_post_labelme_json_state()
                    if self.labelme_initial_json_data != self.labelme_post_json_data:
                        print("JSON file content has been changed, generating annotated image")
                        self.generate_annotated_image(image_path, json_path, annotated_path)
                    else:
                        print("JSON file content has not changed!")
                else:
                    print("No JSON file generated while in LabelMe, nothing to do now!")
                
                '''
                if os.path.exists(json_path) and os.path.exists(annotated_path):
                    json_timestamp = QtCore.QDateTime.fromSecsSinceEpoch(
                    int(os.path.getmtime(json_path)))
                    #).toString()

                    #for debugging
                    print(json_timestamp)

                    annotated_image_timestamp = QtCore.QDateTime.fromSecsSinceEpoch(
                    int(os.path.getmtime(annotated_path)))
                    #).toString()
                    #for debugging
                    print(annotated_image_timestamp)

                    if json_timestamp > annotated_image_timestamp:
                        print("JSON file has been modified, generating new annotated image")
                        self.generate_annotated_image(image_path, json_path, annotated_path)
                elif os.path.exists(json_path):
                    print("JSON file exists but annotated image does not, generating annotated image")
                    self.generate_annotated_image(image_path, json_path, annotated_path)
                '''
            else:
                print(f"LabelMe finished with a non-zero exit code: {exitCode}")
                out = bytes(self.labelme_process.readAllStandardOutput()).decode(errors="ignore")
                if out.strip():
                    print("LabelMe stdout:", out)
                # The program ran but reported an error.
                # You might want to check the LabelMe documentation for what 'exitCode' means.

        # Reset the process handle so it can be launched again.
        self.labelme_process = None

    def update_splitter_sizes(self):
        """
        Redistribute widths in the QSplitter:
         - both prev & next visible → [⅓, ⅓, ⅓]
         - only prev visible      → [½, ½, 0]
         - only next visible      → [0, ½, ½]
         - neither visible        → [0, 1, 0]
        """
        total = self.image_splitter.width()
        if self.prev_compare_visible and self.next_compare_visible:
            self.image_splitter.setSizes([total // 3, total // 3, total // 3])
        elif self.prev_compare_visible:
            self.image_splitter.setSizes([total // 2, total // 2, 0])
        elif self.next_compare_visible:
            self.image_splitter.setSizes([0, total // 2, total // 2])
        else:
            self.image_splitter.setSizes([0, total, 0])

    def go_to_next_image(self):
        """Advance current_index, keep comparison panes visible & updated."""
        if not self.image_list:
            return
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.update_image()

    def go_to_previous_image(self):
        """Decrement current_index, keep comparison panes visible & updated."""
        if not self.image_list:
            return
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image()

    def resizeEvent(self, event):
        """
        Whenever the window resizes, re‐distribute splitter sizes so each
        visible pane resizes appropriately.
        """
        super().resizeEvent(event)
        self.update_splitter_sizes()

    def manual_zoom(self, zoom_label: ZoomLabel, factor: float):
        """
        Multiply the given ZoomLabel's scale_factor by `factor`, then re-render.
        """
        if zoom_label.base_pixmap is None:
            return

        # Clamp factor so it never goes below a tiny threshold
        new_sf = zoom_label.scale_factor * factor
        if new_sf < 0.05:
            new_sf = 0.05
        zoom_label.scale_factor = new_sf

        # Build a new scaled pixmap and display it
        new_size = zoom_label.base_pixmap.size() * zoom_label.scale_factor
        scaled = zoom_label.base_pixmap.scaled(
            new_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        zoom_label._drag_start = None
        super(ZoomLabel, zoom_label).setPixmap(scaled)
        zoom_label.adjustSize()


    def convert_json_to_csv(self):
        print("starting the convert function...")
        try:
            csv_paths = convert_folder_json_to_csv(self.folder_path)
            if not csv_paths:
                print("No JSON files found to convert.")
                self.message_label.setText("No JSON files found to convert.")
                return
            message = (
                f"Successfully converted {len(csv_paths)} JSON files to CSVs - "
                "you should see them in your data folder!"
            )
            print(message)
            self.message_label.setText(message)
        except Exception as exc:
            print(f"WARNING: Error converting JSON files to CSVs: {exc}")
            self.message_label.setText(f"WARNING: Error converting JSON files to CSVs: {exc}")

    def show_popup(self):
        msg = QMessageBox()
        msg.setWindowTitle("Generate CSV Files From Current JSON Files?")
        msg.setText("By clicking Yes, you will convert all JSON files to CSVs!")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        msg.setInformativeText("JSON files store the nest data for each image. By clicking \"Yes\", for every JSON file in the folder, a CSV file of the same name will be generated. Doing so will overwrite any previously-generated CSV files with the same filenames (see \"Details\" for more information)")
        
        msg.setDetailedText("Make sure that:\n\n1) You haven't made any modifications to the CSV files that you want to save \n\n2) That you haven't made any changes to the JSON files that you DON'T want to save! (i.e. using LabelMe to change nest component labels or locations)\n\nIf you have, make sure to change the CSV filenames manually to stop them from being over-written!")
        choice = msg.exec()
        if choice == QMessageBox.Yes:
            self.convert_json_to_csv()


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Launch the BumbleBox LabelNests GUI."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Folder containing composite nest images and optional .json annotations.",
    )
    parser.add_argument(
        "--start-image",
        help="Optional image filename to open first if present in folder.",
    )
    parser.add_argument(
        "--labelmerc",
        help=(
            "Optional path to LabelMe config file. If omitted, the app checks "
            "BUMBLEBOX_LABELMERC, ./labelmerc, script_dir/labelmerc, and ~/.labelmerc."
        ),
    )
    return parser


if __name__ == '__main__':
    args, qt_args = _build_arg_parser().parse_known_args()
    app = QtWidgets.QApplication([sys.argv[0], *qt_args])
    folder = args.folder
    if not folder:
        folder = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder with Images")
    if folder:
        if not os.path.isdir(folder):
            QMessageBox.critical(None, "Invalid Folder", f"Folder does not exist:\n{folder}")
            sys.exit(1)
        win = LabelNestsApp(
            folder,
            start_image=args.start_image,
            labelmerc_path=args.labelmerc,
        )
        win.show()
        sys.exit(app.exec())
