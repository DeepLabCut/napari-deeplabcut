# src/napari_deeplabcut/ui/cropping.py
from __future__ import annotations

import os
from math import ceil, floor
from pathlib import Path

import numpy as np
import pandas as pd
from napari.layers import Image, Points, Shapes
from napari.utils.notifications import show_info, show_warning
from pydantic import BaseModel, ConfigDict, field_validator
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QCheckBox, QGroupBox, QLabel, QMessageBox, QPushButton, QVBoxLayout

import napari_deeplabcut.core.io as io
from napari_deeplabcut._writer import _write_image
from napari_deeplabcut.core.dataframes import guarantee_multiindex_rows
from napari_deeplabcut.core.project_paths import (
    canonicalize_path,
    infer_dlc_project,
    infer_dlc_project_from_image_layer,
)

DLC_CROP_LAYER_NAME = "DLC crop"
DLC_CROP_LAYER_META_KEY = "_dlc_crop_layer"


class _CropModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class ViewerCropCoords(_CropModel):
    """
    Rectangle coordinates in napari/image-data space.
    Used only for numpy slicing during extraction.
    """

    values: tuple[int, int, int, int]

    @field_validator("values")
    @classmethod
    def _validate_values(cls, v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if len(v) != 4:
            raise ValueError(f"ViewerCropCoords must be a 4-tuple (x1, x2, y1, y2). Got: {v!r}")
        x1, x2, y1, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"ViewerCropCoords must satisfy x2>x1 and y2>y1. Got: {v!r}")
        return v


class DLCConfigCropCoords(_CropModel):
    """
    Rectangle coordinates in the legacy DLC-compatible config.yaml convention.
    Used only for writing cfg['video_sets'][...]['crop'].
    """

    values: tuple[int, int, int, int]

    @field_validator("values")
    @classmethod
    def _validate_values(cls, v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if len(v) != 4:
            raise ValueError(f"DLCConfigCropCoords must be a 4-tuple (x1, x2, y1, y2). Got: {v!r}")
        x1, x2, y1, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"DLCConfigCropCoords must satisfy x2>x1 and y2>y1. Got: {v!r}")
        return v


class FrameExtractionPlan(_CropModel):
    image_layer: Image
    points_layer: Points | None
    frame_index: int
    output_root: Path
    output_path: Path
    labels_path: Path | None
    export_labels: bool
    viewer_crop: ViewerCropCoords | None = None

    @field_validator("frame_index")
    @classmethod
    def _validate_frame_index(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"frame_index must be >= 0. Got: {v}")
        return v


class CropRectangleSpec(_CropModel):
    """
    A single resolved rectangle with two explicit coordinate conventions.
    """

    viewer_crop: ViewerCropCoords
    config_crop: DLCConfigCropCoords


class CropSavePlan(_CropModel):
    config_path: Path
    project_root: Path
    video_key: str
    config_crop: DLCConfigCropCoords

    @field_validator("config_crop", mode="before")
    @classmethod
    def _reject_viewer_coords_for_config(cls, v):
        if isinstance(v, ViewerCropCoords):
            raise ValueError(
                "Refusing to write napari/viewer crop coordinates to DLC config.yaml. "
                "Use CropRectangleSpec.config_crop or plan_crop_save(...), not viewer_crop."
            )
        return v


def _format_crop(coords: ViewerCropCoords | DLCConfigCropCoords | None) -> str:
    return str(coords.values) if coords is not None else "none"


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

        self.apply_crop_cb = QCheckBox("Crop to rectangle")
        self.apply_crop_cb.setToolTip(
            "Use a rectangle crop for extraction. Viewer coords are used for extraction; "
            "DLC config coords are saved separately."
        )
        layout.addWidget(self.apply_crop_cb)

        self.crop_button = QPushButton("Save crop to config")
        self.crop_button.clicked.connect(on_store_crop)
        layout.addWidget(self.crop_button)

        self.context_label = QLabel("No crop or video context yet.")
        self.context_label.setWordWrap(True)
        layout.addWidget(self.context_label)

    def set_context_text(self, text: str) -> None:
        self.context_label.setText(text or "")


def build_video_action_menu(*, on_extract_frame, on_store_crop) -> VideoActionPanel:
    return VideoActionPanel(
        on_extract_frame=on_extract_frame,
        on_store_crop=on_store_crop,
    )


def is_dlc_crop_layer(layer) -> bool:
    return isinstance(layer, Shapes) and bool((layer.metadata or {}).get(DLC_CROP_LAYER_META_KEY, False))


def get_dlc_crop_layer(viewer) -> Shapes | None:
    for layer in viewer.layers:
        if is_dlc_crop_layer(layer):
            return layer
    return None


def ensure_dlc_crop_layer(viewer) -> Shapes:
    """
    Ensure a dedicated crop Shapes layer exists and is ready for rectangle drawing.
    """
    existing = get_dlc_crop_layer(viewer)
    if existing is not None:
        existing.visible = True
        try:
            viewer.layers.selection.active = existing
        except Exception:
            pass
        try:
            existing.mode = "add_rectangle"
        except Exception:
            pass
        return existing

    layer = viewer.add_shapes(
        name=DLC_CROP_LAYER_NAME,
        metadata={DLC_CROP_LAYER_META_KEY: True},
    )
    try:
        layer.mode = "add_rectangle"
    except Exception:
        pass

    try:
        viewer.layers.selection.active = layer
    except Exception:
        pass

    return layer


def handle_apply_crop_toggled(viewer, panel, checked: bool) -> None:
    """
    When crop application is enabled, create/select the dedicated crop layer and
    make it the preferred source of truth for crop rectangles.

    Auto-refresh wiring is synchronized by update_video_panel_context(...),
    so this function only needs to ensure the layer exists when enabled.
    """
    if checked:
        ensure_dlc_crop_layer(viewer)

    update_video_panel_context(viewer, panel)


# ---------------------------------
# Dedicated crop-layer auto-refresh
# ---------------------------------


def _ensure_crop_refresh_timer(panel, refresh_callback) -> QTimer:
    """
    Create (once) a lightweight debounce timer stored on the panel.

    We debounce crop refreshes because Shapes data can emit repeatedly while
    the user drags/resizes a rectangle.
    """
    timer = getattr(panel, "_dlc_crop_refresh_timer", None)
    if timer is not None:
        return timer

    timer = QTimer(panel)
    timer.setSingleShot(True)
    timer.setInterval(30)  # small debounce; coalesces drag/resize bursts
    timer.timeout.connect(refresh_callback)
    panel._dlc_crop_refresh_timer = timer
    return timer


def _schedule_crop_refresh(panel, refresh_callback) -> None:
    timer = _ensure_crop_refresh_timer(panel, refresh_callback)
    timer.start()


def sync_crop_layer_autorefresh(viewer, panel, refresh_callback) -> None:
    """
    Keep a debounced data-change listener attached only to the dedicated DLC crop layer.

    This is intentionally a no-op in all other contexts:
    - no dedicated crop layer -> nothing connected
    - same layer already connected -> nothing changes
    - crop layer removed/replaced -> disconnect old, connect new
    """
    current_layer = get_dlc_crop_layer(viewer)

    prev_layer = getattr(panel, "_dlc_crop_refresh_layer", None)
    prev_handler = getattr(panel, "_dlc_crop_refresh_handler", None)

    # Fast no-op path: already synced to the current dedicated crop layer
    if current_layer is not None and prev_layer is current_layer and prev_handler is not None:
        return

    # Disconnect previous listener if present
    if prev_layer is not None and prev_handler is not None:
        try:
            prev_layer.events.data.disconnect(prev_handler)
        except Exception:
            pass
        try:
            prev_layer.events.selected_data.disconnect(prev_handler)
        except Exception:
            pass

    panel._dlc_crop_refresh_layer = None
    panel._dlc_crop_refresh_handler = None

    timer = getattr(panel, "_dlc_crop_refresh_timer", None)
    if current_layer is None and timer is not None:
        try:
            timer.stop()
        except Exception:
            pass

    if current_layer is None:
        return

    def _on_crop_layer_change(event=None):
        _schedule_crop_refresh(panel, refresh_callback)

    try:
        current_layer.events.data.connect(_on_crop_layer_change)
    except Exception:
        return

    try:
        current_layer.events.selected_data.connect(_on_crop_layer_change)
    except Exception:
        pass

    panel._dlc_crop_refresh_layer = current_layer
    panel._dlc_crop_refresh_handler = _on_crop_layer_change


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
            return str(Path(src).expanduser())
        except Exception:
            return str(Path(src))

    video_name = fallback_video_name or getattr(image_layer, "name", None)
    if not video_name:
        return None

    return os.path.join(project_path, "videos", video_name)


def _is_rectangle_shape(layer: Shapes, index: int) -> bool:
    try:
        return layer.shape_type[index] == "rectangle"
    except Exception:
        return False


def _rectangle_indices(layer: Shapes) -> list[int]:
    try:
        return [i for i, _shape in enumerate(layer.shape_type) if _is_rectangle_shape(layer, i)]
    except Exception:
        return []


def _selected_rectangle_indices(layer: Shapes) -> list[int]:
    try:
        selected = list(getattr(layer, "selected_data", set()) or set())
    except Exception:
        return []

    return [idx for idx in selected if _is_rectangle_shape(layer, idx)]


def _rectangle_spec(viewer, layer: Shapes, rect_index: int) -> CropRectangleSpec | None:
    """
    Resolve both coordinate conventions from the same rectangle:

    - viewer_crop:
        raw napari/image-data coordinates used for extraction
    - config_crop:
        legacy backwards-compatible coordinates written to config.yaml
    """
    try:
        data = np.asarray(layer.data[rect_index], dtype=float)
    except Exception:
        return None

    if data.ndim != 2 or data.shape[1] < 2:
        return None

    # Use the last two columns consistently as [y, x]
    yx = data[:, -2:]
    if yx.ndim != 2 or yx.shape[1] != 2:
        return None

    y_vals = yx[:, 0]
    x_vals = yx[:, 1]

    if len(x_vals) == 0 or len(y_vals) == 0:
        return None

    # ----------------------------
    # 1) napari/image-data coords
    # ----------------------------
    vx1 = max(0, int(floor(np.min(x_vals))))
    vx2 = max(0, int(ceil(np.max(x_vals))))
    vy1 = max(0, int(floor(np.min(y_vals))))
    vy2 = max(0, int(ceil(np.max(y_vals))))

    if vx2 <= vx1 or vy2 <= vy1:
        return None

    viewer_crop = ViewerCropCoords(values=(vx1, vx2, vy1, vy2))

    # -----------------------------------------
    # 2) legacy DLC-compatible config.yaml crop
    # -----------------------------------------
    try:
        h = viewer.dims.range[2][1]
    except Exception:
        return None

    legacy_y = h - y_vals
    legacy = np.column_stack([legacy_y, x_vals])
    legacy = np.clip(legacy, 0, a_max=None).astype(int)

    y1_cfg, x1_cfg = legacy.min(axis=0)
    y2_cfg, x2_cfg = legacy.max(axis=0)

    if x2_cfg <= x1_cfg or y2_cfg <= y1_cfg:
        return None

    config_crop = DLCConfigCropCoords(values=(int(x1_cfg), int(x2_cfg), int(y1_cfg), int(y2_cfg)))

    return CropRectangleSpec(
        viewer_crop=viewer_crop,
        config_crop=config_crop,
    )


def _find_rectangle_in_layer(
    viewer,
    layer: Shapes,
    *,
    prefer_selected: bool = True,
) -> CropRectangleSpec | None:
    """
    Resolve a rectangle spec from a specific Shapes layer.

    Preference:
    1) selected rectangle(s)
    2) latest rectangle in that layer
    """
    if prefer_selected:
        for idx in _selected_rectangle_indices(layer):
            spec = _rectangle_spec(viewer, layer, idx)
            if spec is not None:
                return spec

    for idx in reversed(_rectangle_indices(layer)):
        spec = _rectangle_spec(viewer, layer, idx)
        if spec is not None:
            return spec

    return None


def find_crop_rectangle(viewer, *, prefer_selected: bool = True) -> CropRectangleSpec | None:
    """
    Resolve the crop rectangle spec in a deterministic order:

    1) dedicated DLC crop layer (source of truth) if it exists and contains a rectangle
    2) active Shapes layer
    3) next available Shapes layer(s)

    Returns
    -------
    CropRectangleSpec | None
        Contains both:
        - viewer_crop: for extraction
        - config_crop: for config.yaml saving
    """
    crop_layer = get_dlc_crop_layer(viewer)
    if crop_layer is not None:
        spec = _find_rectangle_in_layer(viewer, crop_layer, prefer_selected=prefer_selected)
        if spec is not None:
            return spec

    active = viewer.layers.selection.active
    if isinstance(active, Shapes) and not is_dlc_crop_layer(active):
        spec = _find_rectangle_in_layer(viewer, active, prefer_selected=prefer_selected)
        if spec is not None:
            return spec

    for layer in viewer.layers:
        if isinstance(layer, Shapes) and not is_dlc_crop_layer(layer):
            spec = _find_rectangle_in_layer(viewer, layer, prefer_selected=prefer_selected)
            if spec is not None:
                return spec

    return None


def get_crop_source_summary(viewer) -> tuple[str, CropRectangleSpec | None]:
    """
    Return a human-friendly crop source label plus the resolved rectangle spec.
    """
    crop_layer = get_dlc_crop_layer(viewer)
    if crop_layer is not None:
        spec = _find_rectangle_in_layer(viewer, crop_layer, prefer_selected=True)
        if spec is not None:
            return f"{DLC_CROP_LAYER_NAME} layer", spec

    active = viewer.layers.selection.active
    if isinstance(active, Shapes) and not is_dlc_crop_layer(active):
        spec = _find_rectangle_in_layer(viewer, active, prefer_selected=True)
        if spec is not None:
            return f"active Shapes layer ({active.name})", spec

    for layer in viewer.layers:
        if isinstance(layer, Shapes) and not is_dlc_crop_layer(layer):
            spec = _find_rectangle_in_layer(viewer, layer, prefer_selected=True)
            if spec is not None:
                return f"Shapes layer ({layer.name})", spec

    return "none", None


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
        crop_spec = find_crop_rectangle(viewer, prefer_selected=True)
        if crop_spec is None:
            return None, (
                "Crop to rectangle is enabled, but no valid rectangle was found. "
                "Use the dedicated DLC crop layer or select a rectangle in a Shapes layer."
            )
        crop = crop_spec.viewer_crop

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
            viewer_crop=crop,
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

    if plan.viewer_crop is not None:
        x1, x2, y1, y2 = plan.viewer_crop.values
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
            note = (
                "Frame image was extracted, "
                "but labels were skipped because the points layer did not contain the current frame."
            )
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
    image_layer: Image | None,
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

    crop_spec = find_crop_rectangle(viewer, prefer_selected=True)
    if crop_spec is None:
        return None, (
            "No valid rectangle was found for crop saving. "
            "Use the dedicated DLC crop layer or select a rectangle in a Shapes layer."
        )
    crop = crop_spec.config_crop

    video_key = _resolve_video_key(image_layer, str(project_root), fallback_video_name=fallback_video_name)
    if not video_key:
        return None, "Could not resolve the current video key for config.yaml."

    return (
        CropSavePlan(
            config_path=Path(config_path),
            project_root=Path(project_root),
            video_key=video_key,
            config_crop=crop,
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
    existing["crop"] = ", ".join(map(str, plan.config_crop.values))
    video_sets[plan.video_key] = existing
    io.write_config(str(plan.config_path), cfg)

    return f"Saved crop {plan.config_crop.values} to {plan.config_path.name} for video key {plan.video_key}"


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

    sync_crop_layer_autorefresh(
        viewer,
        panel,
        refresh_callback=lambda: update_video_panel_context(viewer, panel),
    )

    image_layer = get_active_or_last_layer(viewer, Image)
    if image_layer is None:
        panel.set_context_text("No active video/image layer.")
        return

    try:
        n_frames = int(image_layer.data.shape[0])
        frame_index = int(viewer.dims.current_step[0])
        frame_text = f"Frame {frame_index + 1}/{n_frames}"
    except Exception:
        frame_text = "Frame ?/?"

    root_value = (image_layer.metadata or {}).get("root")
    root_text = str(root_value) if root_value else "unresolved"

    crop_source, crop_spec = get_crop_source_summary(viewer)

    if crop_spec is None:
        panel.set_context_text(
            f"{frame_text}\nOutput folder: {root_text}\nCrop source: {crop_source}\nNo valid rectangle selected yet."
        )
        return

    viewer_crop_text = str(crop_spec.viewer_crop.values)
    config_crop_text = str(crop_spec.config_crop.values)

    panel.set_context_text(
        f"{frame_text}\n"
        f"Output folder: {root_text}\n"
        f"Crop source: {crop_source}\n"
        f"Viewer crop: {viewer_crop_text}\n"
        f"Config crop: {config_crop_text}"
    )


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
