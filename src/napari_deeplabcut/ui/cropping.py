# src/napari_deeplabcut/ui/cropping.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from napari.layers import Image, Points, Shapes
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import QCheckBox, QGroupBox, QLabel, QMessageBox, QPushButton, QVBoxLayout

import napari_deeplabcut.core.io as io
from napari_deeplabcut._writer import _write_image
from napari_deeplabcut.core.dataframes import guarantee_multiindex_rows
from napari_deeplabcut.core.project_paths import (
    canonicalize_path,
    infer_dlc_project,
    infer_dlc_project_from_image_layer,
)


@dataclass(frozen=True)
class FrameExtractionPlan:
    image_layer: Image
    points_layer: Points | None
    frame_index: int
    output_root: Path
    output_path: Path
    labels_path: Path | None
    export_labels: bool
    crop: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class CropSavePlan:
    config_path: Path
    project_root: Path
    video_key: str
    crop: tuple[int, int, int, int]


class VideoActionPanel(QGroupBox):
    """Small video tools panel with lightweight user-facing context."""

    def __init__(self, *, on_extract_frame, on_store_crop):
        super().__init__("Video")

        layout = QVBoxLayout(self)

        self.extract_button = QPushButton("Extract current frame")
        self.extract_button.clicked.connect(on_extract_frame)
        layout.addWidget(self.extract_button)

        self.export_labels_cb = QCheckBox("Also export labels")
        self.export_labels_cb.setToolTip(
            "Write the current frame labels to machinelabels-iter0.h5 if a DLC points layer is available."
        )
        layout.addWidget(self.export_labels_cb)

        self.apply_crop_cb = QCheckBox("Apply selected rectangle")
        self.apply_crop_cb.setToolTip("Crop the extracted frame to the selected rectangle before writing it.")
        layout.addWidget(self.apply_crop_cb)

        self.crop_button = QPushButton("Save crop to config")
        self.crop_button.clicked.connect(on_store_crop)
        layout.addWidget(self.crop_button)

        self.help_label = QLabel(
            "Extract the current frame from the active video/image layer. "
            "Optionally export labels and/or apply the selected crop rectangle."
        )
        self.help_label.setWordWrap(True)
        layout.addWidget(self.help_label)

        self.context_label = QLabel("")
        self.context_label.setWordWrap(True)
        layout.addWidget(self.context_label)

    def set_context_text(self, text: str) -> None:
        self.context_label.setText(text or "")


def build_video_action_menu(*, on_extract_frame, on_store_crop) -> VideoActionPanel:
    return VideoActionPanel(
        on_extract_frame=on_extract_frame,
        on_store_crop=on_store_crop,
    )


def _resolve_video_key(image_layer: Image, project_path: str, fallback_video_name: str | None = None) -> str | None:
    """
    Determine the key used under cfg['video_sets'].

    Prefer the actual loaded video source path if available, otherwise fall back
    to <project>/videos/<video_name>.
    """
    try:
        src = getattr(getattr(image_layer, "source", None), "path", None)
    except Exception:
        src = None

    if src:
        try:
            return str(Path(src).expanduser().resolve())
        except Exception:
            return str(Path(src))

    video_name = fallback_video_name or getattr(image_layer, "name", None)
    if not video_name:
        return None

    return os.path.join(project_path, "videos", video_name)


def _rectangle_to_crop_tuple(viewer, layer: Shapes, rect_index: int) -> tuple[int, int, int, int] | None:
    try:
        bbox = np.asarray(layer.data[rect_index])[:, 1:].copy()
    except Exception:
        return None

    if bbox.ndim != 2 or bbox.shape[1] != 2:
        return None

    try:
        # Preserve the existing convention from _widgets.py
        h = viewer.dims.range[2][1]
    except Exception:
        return None

    bbox[:, 0] = h - bbox[:, 0]
    bbox = np.clip(bbox, 0, a_max=None).astype(int)

    y1, x1 = bbox.min(axis=0)
    y2, x2 = bbox.max(axis=0)

    if x2 <= x1 or y2 <= y1:
        return None

    return int(x1), int(x2), int(y1), int(y2)


def _selected_rectangle_indices(layer: Shapes) -> list[int]:
    try:
        selected = list(getattr(layer, "selected_data", set()) or set())
    except Exception:
        return []

    out: list[int] = []
    for idx in selected:
        try:
            if layer.shape_type[idx] == "rectangle":
                out.append(idx)
        except Exception:
            continue
    return out


def find_crop_rectangle(viewer, *, prefer_selected: bool = True) -> tuple[int, int, int, int] | None:
    """
    Resolve the crop rectangle in a deterministic order:

    1) selected rectangle in active Shapes layer
    2) selected rectangle in any Shapes layer
    3) latest rectangle in active Shapes layer
    4) latest rectangle globally
    """
    shape_layers = [layer for layer in viewer.layers if isinstance(layer, Shapes)]
    if not shape_layers:
        return None

    active = viewer.layers.selection.active

    # 1) selected rectangle in active Shapes layer
    if prefer_selected and isinstance(active, Shapes):
        for idx in _selected_rectangle_indices(active):
            crop = _rectangle_to_crop_tuple(viewer, active, idx)
            if crop is not None:
                return crop

    # 2) selected rectangle in any Shapes layer
    if prefer_selected:
        for layer in shape_layers:
            for idx in _selected_rectangle_indices(layer):
                crop = _rectangle_to_crop_tuple(viewer, layer, idx)
                if crop is not None:
                    return crop

    # 3) latest rectangle in active Shapes layer
    if isinstance(active, Shapes):
        try:
            rect_indices = [i for i, shape in enumerate(active.shape_type) if shape == "rectangle"]
        except Exception:
            rect_indices = []
        for idx in reversed(rect_indices):
            crop = _rectangle_to_crop_tuple(viewer, active, idx)
            if crop is not None:
                return crop

    # 4) latest rectangle globally
    for layer in reversed(shape_layers):
        try:
            rect_indices = [i for i, shape in enumerate(layer.shape_type) if shape == "rectangle"]
        except Exception:
            rect_indices = []
        for idx in reversed(rect_indices):
            crop = _rectangle_to_crop_tuple(viewer, layer, idx)
            if crop is not None:
                return crop

    return None


def _frame_digits(n_frames: int) -> int:
    return max(1, len(str(max(n_frames - 1, 0))))


def plan_frame_extraction(
    viewer,
    *,
    image_layer: Image | None,
    points_layer: Points | None = None,
    explicit_output_root: str | Path | None = None,
    export_labels: bool = False,
    apply_crop: bool = False,
) -> tuple[FrameExtractionPlan | None, str | None]:
    """
    Validate and plan a single-frame extraction before doing any I/O.
    """
    if image_layer is None:
        return None, "No image/video layer is active."

    data = getattr(image_layer, "data", None)
    if data is None or not hasattr(data, "shape") or len(data.shape) < 3:
        return None, "Active image layer does not look like a video/image stack."

    try:
        frame_index = int(viewer.dims.current_step[0])
    except Exception:
        return None, "Could not determine the current frame index."

    n_frames = int(data.shape[0])
    if frame_index < 0 or frame_index >= n_frames:
        return None, f"Current frame index {frame_index} is out of bounds for a stack of length {n_frames}."

    root_value = explicit_output_root or (image_layer.metadata or {}).get("root")
    if not root_value:
        return None, "Could not determine the output folder for extracted frames."

    try:
        output_root = Path(root_value).expanduser().resolve()
    except Exception:
        output_root = Path(root_value)

    crop = None
    if apply_crop:
        crop = find_crop_rectangle(viewer, prefer_selected=True)
        if crop is None:
            return None, "Apply selected rectangle is enabled, but no valid rectangle is selected."

    frame_name = f"img{str(frame_index).zfill(_frame_digits(n_frames))}.png"
    output_path = output_root / frame_name

    labels_path = None
    if export_labels:
        if points_layer is None:
            return None, "Also export labels is enabled, but no Points layer is available."
        labels_path = output_root / "machinelabels-iter0.h5"

    return (
        FrameExtractionPlan(
            image_layer=image_layer,
            points_layer=points_layer,
            frame_index=frame_index,
            output_root=output_root,
            output_path=output_path,
            labels_path=labels_path,
            export_labels=export_labels,
            crop=crop,
        ),
        None,
    )


def execute_frame_extraction(plan: FrameExtractionPlan) -> tuple[list[Path], str | None]:
    """
    Execute a previously validated extraction plan.

    Returns
    -------
    (written_paths, note)
        written_paths: all files written
        note: optional non-fatal note (e.g. labels skipped)
    """
    frame = np.asarray(plan.image_layer.data[plan.frame_index])

    if plan.crop is not None:
        x1, x2, y1, y2 = plan.crop
        frame = frame[y1:y2, x1:x2]

    plan.output_root.mkdir(parents=True, exist_ok=True)
    _write_image(frame, str(plan.output_path))

    written = [plan.output_path]
    note = None

    if plan.export_labels and plan.points_layer is not None and plan.labels_path is not None:
        df = io.form_df(
            plan.points_layer.data,
            layer_metadata=plan.points_layer.metadata,
            layer_properties=plan.points_layer.properties,
        )

        if plan.frame_index >= len(df):
            note = "Frame image was extracted, "
            "but labels were skipped because the points layer did not contain the current frame."
            return written, note

        df = df.iloc[plan.frame_index : plan.frame_index + 1]

        canon = canonicalize_path(plan.output_path, 3)
        df.index = pd.MultiIndex.from_tuples([tuple(canon.split("/"))])

        if plan.labels_path.is_file():
            try:
                df_prev = pd.read_hdf(plan.labels_path, key="df_with_missing")
            except Exception:
                df_prev = pd.read_hdf(plan.labels_path)
            guarantee_multiindex_rows(df_prev)
            df = pd.concat([df_prev, df])
            df = df[~df.index.duplicated(keep="first")]

        df.to_hdf(plan.labels_path, key="df_with_missing")
        written.append(plan.labels_path)

    return written, note


def plan_crop_save(
    viewer,
    *,
    image_layer: Image,
    explicit_project_path: str | None = None,
    fallback_video_name: str | None = None,
) -> tuple[CropSavePlan | None, str | None]:
    """
    Validate and plan a crop save to config.yaml before doing any I/O.
    """
    if image_layer is None:
        return None, "No image/video layer is active."

    if explicit_project_path:
        ctx = infer_dlc_project(
            explicit_root=explicit_project_path,
            anchor_candidates=[getattr(getattr(image_layer, "source", None), "path", None)],
            prefer_project_root=True,
        )
    else:
        ctx = infer_dlc_project_from_image_layer(image_layer, prefer_project_root=True)

    config_path = ctx.config_path
    project_root = ctx.project_root or ctx.root_anchor

    if config_path is None or project_root is None:
        return None, "Could not determine a DLC config.yaml for this video."

    crop = find_crop_rectangle(viewer, prefer_selected=True)
    if crop is None:
        return None, "Select or draw a valid rectangle before saving a crop."

    video_key = _resolve_video_key(image_layer, str(project_root), fallback_video_name=fallback_video_name)
    if not video_key:
        return None, "Could not resolve the current video key for config.yaml."

    return (
        CropSavePlan(
            config_path=Path(config_path),
            project_root=Path(project_root),
            video_key=video_key,
            crop=crop,
        ),
        None,
    )


def execute_crop_save(plan: CropSavePlan) -> str:
    """
    Persist crop coordinates into the project's config.yaml.

    Returns
    -------
    str
        Success message suitable for user-facing notifications.
    """
    cfg = io.load_config(str(plan.config_path))
    video_sets = cfg.setdefault("video_sets", {})
    existing = dict(video_sets.get(plan.video_key, {}))
    existing["crop"] = ", ".join(map(str, plan.crop))
    video_sets[plan.video_key] = existing
    io.write_config(str(plan.config_path), cfg)

    return f"Saved crop {plan.crop} to {plan.config_path.name} for video key {plan.video_key}"


def store_crop_coordinates(
    viewer,
    *,
    image_layer: Image,
    explicit_project_path: str | None = None,
    fallback_video_name: str | None = None,
) -> tuple[bool, str]:
    """
    Compatibility wrapper used by the widget layer.
    """
    plan, error = plan_crop_save(
        viewer,
        image_layer=image_layer,
        explicit_project_path=explicit_project_path,
        fallback_video_name=fallback_video_name,
    )
    if plan is None:
        return False, (error or "Could not save crop coordinates.")

    msg = execute_crop_save(plan)
    return True, msg


# ---------------------------------
# UI helpers
# ---------------------------------
def get_active_or_last_layer(viewer, layer_type):
    """Prefer the active layer when it matches `layer_type`, else fall back to the last layer of that type."""
    active = viewer.layers.selection.active
    if isinstance(active, layer_type):
        return active

    for layer in reversed(viewer.layers):
        if isinstance(layer, layer_type):
            return layer
    return None


def resolve_project_path_from_image_layer(layer: Image) -> str | None:
    """Best-effort project path string from DLC project context."""
    try:
        ctx = infer_dlc_project_from_image_layer(layer, prefer_project_root=True)
    except Exception:
        return None

    project_path = ctx.project_root or ctx.root_anchor
    return str(project_path) if project_path is not None else None


def update_video_panel_context(viewer, panel) -> None:
    """Refresh lightweight user-facing context shown in the video action panel."""
    if panel is None:
        return

    image_layer = get_active_or_last_layer(viewer, Image)
    if image_layer is None:
        panel.set_context_text("Load a video/image layer to enable extraction and crop tools.")
        return

    try:
        n_frames = int(image_layer.data.shape[0])
        frame_index = int(viewer.dims.current_step[0])
        frame_text = f"Frame {frame_index + 1}/{n_frames}"
    except Exception:
        frame_text = "Frame ?/?"

    root_value = (image_layer.metadata or {}).get("root")
    root_text = str(root_value) if root_value else "unresolved"

    crop = find_crop_rectangle(viewer, prefer_selected=True)
    crop_text = f"Selected crop: {crop}" if crop is not None else "Selected crop: none"

    panel.set_context_text(f"{frame_text}\nOutput folder: {root_text}\n{crop_text}")


def run_extract_current_frame(
    viewer,
    panel,
    *,
    validate_points_layer=None,
) -> tuple[bool, str]:
    """
    End-to-end frame extraction workflow for the video panel.

    Returns
    -------
    (ok, message)
        ok: whether the extraction completed
        message: user-facing status text
    """
    image_layer = get_active_or_last_layer(viewer, Image)
    export_labels = bool(getattr(panel, "export_labels_cb", None) and panel.export_labels_cb.isChecked())
    apply_crop = bool(getattr(panel, "apply_crop_cb", None) and panel.apply_crop_cb.isChecked())

    points_layer = get_active_or_last_layer(viewer, Points) if export_labels else None

    if export_labels and points_layer is not None and callable(validate_points_layer):
        if not validate_points_layer(points_layer):
            msg = "The selected Points layer is not a valid DLC keypoints layer for label export."
            show_warning(msg)
            return False, msg

    plan, error = plan_frame_extraction(
        viewer,
        image_layer=image_layer,
        points_layer=points_layer,
        explicit_output_root=(image_layer.metadata or {}).get("root") if image_layer is not None else None,
        export_labels=export_labels,
        apply_crop=apply_crop,
    )

    if plan is None:
        msg = error or "Could not plan frame extraction."
        show_warning(msg)
        return False, msg

    if plan.output_path.exists():
        answer = QMessageBox.question(
            panel,
            "Overwrite extracted frame?",
            f"The extracted frame already exists:\n\n{plan.output_path}\n\nDo you want to overwrite it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return False, "Frame extraction cancelled."

    written_paths, note = execute_frame_extraction(plan)

    msg = f"Extracted frame {plan.frame_index} to {plan.output_path}"
    if len(written_paths) > 1:
        msg += f" and updated {written_paths[-1].name}"

    if note:
        show_warning(note)
        msg = f"{msg} ({note})"
    else:
        show_info(msg)

    return True, msg


def run_store_crop_coordinates(
    viewer,
    panel,
    *,
    explicit_project_path: str | None = None,
    fallback_video_name: str | None = None,
) -> tuple[bool, str, str | None]:
    """
    End-to-end crop-save workflow for the video panel.

    Returns
    -------
    (ok, message, project_path)
        ok: whether crop save succeeded
        message: user-facing status text
        project_path: resolved best-effort project path (if any)
    """
    image_layer = get_active_or_last_layer(viewer, Image)
    if image_layer is None:
        msg = "No image/video layer is active."
        show_warning(msg)
        return False, msg, explicit_project_path

    resolved_project_path = explicit_project_path or resolve_project_path_from_image_layer(image_layer)

    ok, msg = store_crop_coordinates(
        viewer,
        image_layer=image_layer,
        explicit_project_path=resolved_project_path,
        fallback_video_name=fallback_video_name,
    )

    if ok:
        show_info(msg)
    else:
        show_warning(msg)

    return ok, msg, resolved_project_path
