# src/napari_deeplabcut/core/layers.py
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from napari.layers import Image, Points, Shapes, Tracks
from qtpy.QtCore import QTimer

from napari_deeplabcut.config.models import AnnotationKind, DLCHeaderModel
from napari_deeplabcut.core.keypoints import build_color_cycles
from napari_deeplabcut.utils.deprecations import deprecated

T = TypeVar("T")

# TODO move to a layers/ folder?
logger = logging.getLogger(__name__)


# Helper to populate keypoint layer properties
def populate_keypoint_layer_properties(
    header: DLCHeaderModel,
    *,
    labels: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
    likelihood: Sequence[float] | None = None,
    paths: list[str] | None = None,
    size: int | None = 8,
    pcutoff: float | None = 0.6,
    colormap: str | None = "viridis",
) -> dict:
    """
    Populate metadata and display properties for a keypoint Points layer.

    Notes
    -----
    - Single-animal DLC: "individuals" level is effectively absent; we represent
      that as ids[0] == "" (falsy) => color/text by label.
    - Multi-animal DLC: ids[0] is a non-empty individual identifier => color/text by id.
    - Must accept empty labels/ids/likelihood and must not assume ≥ 1 entry.
    """

    if labels is None:
        labels = header.bodyparts
    if ids is None:
        ids = header.individuals
    if likelihood is None:
        likelihood_arr = np.ones(len(labels), dtype=float)
    else:
        likelihood_arr = np.asarray(likelihood, dtype=float)

    # 1) Normalize inputs to plain lists (Series-safe)
    #    This prevents pandas Series truthiness errors.
    labels_list = list(labels) if labels is not None else []
    ids_list = list(ids) if ids is not None else []

    # 2) Likelihood: always numeric ndarray for vector ops
    if likelihood is None:
        likelihood_arr = np.ones(len(labels_list), dtype=float)
    else:
        likelihood_arr = np.asarray(list(likelihood), dtype=float)

    # 3) Determine single vs multi animal:
    #    - empty ids => treat as single-animal (label-based)
    #    - ids[0] == "" => also single-animal (label-based)
    first_id = ids_list[0] if len(ids_list) > 0 else ""
    use_id = bool(first_id)

    face_color_cycle_maps = build_color_cycles(header, colormap)
    face_color_prop = "id" if use_id else "label"

    return {
        "name": "keypoints",
        "text": "{id}–{label}" if use_id else "label",
        "properties": {
            "label": list(labels),
            "id": list(ids),
            "likelihood": likelihood_arr,
            "valid": likelihood_arr > pcutoff,
        },
        "face_color_cycle": face_color_cycle_maps[face_color_prop],
        "face_color": face_color_prop,
        "face_colormap": colormap,
        "border_color": "valid",
        "border_color_cycle": ["black", "red"],
        "border_width": 0,
        "border_width_is_relative": False,
        "size": size,
        "metadata": {
            "header": header,
            "face_color_cycles": face_color_cycle_maps,
            "colormap_name": colormap,
            "paths": paths or [],
        },
    }


def is_machine_layer(layer) -> bool:
    md = getattr(layer, "metadata", {}) or {}
    io = md.get("io") or {}
    k = io.get("kind")
    # allow enum or string
    if k is AnnotationKind.MACHINE:
        return True
    is_machine = str(k).lower() == "machine"
    if is_machine:
        logger.info(
            "A literal 'machine' str was used for io.kind; please use AnnotationKind.MACHINE for better validation."
        )
    return is_machine


# -----------------------------------------------
#  Layer-finding utilities
# -----------------------------------------------
def iter_layers(viewer_or_layers: Any) -> Iterable[Any]:
    """Yield layers from a napari Viewer or an iterable of layers."""
    layers = getattr(viewer_or_layers, "layers", viewer_or_layers)
    return layers


def find_first_layer(
    viewer_or_layers: Any,
    layer_type: type[T],
    predicate: Callable[[T], bool] | None = None,
) -> T | None:
    """Return the first layer of type ``layer_type`` that matches ``predicate``.

    Parameters
    ----------
    viewer_or_layers:
        A napari Viewer, LayerList, or any iterable of layers.
    layer_type:
        The desired layer type (e.g., napari.layers.Points).
    predicate:
        Optional function to further filter matching layers.

    Notes
    -----
    This intentionally mirrors the common pattern used throughout the plugin:
    "iterate viewer.layers in order and pick the first match".
    """
    pred = predicate or (lambda _ly: True)
    for ly in iter_layers(viewer_or_layers):
        if isinstance(ly, layer_type) and pred(ly):
            return ly
    return None


def find_last_layer(
    viewer_or_layers: Any,
    layer_type: type[T],
    predicate: Callable[[T], bool] | None = None,
) -> T | None:
    """Return the last layer of type ``layer_type`` that matches ``predicate``."""
    pred = predicate or (lambda _ly: True)
    last: T | None = None
    for ly in iter_layers(viewer_or_layers):
        if isinstance(ly, layer_type) and pred(ly):
            last = ly
    return last


# --------------------
# Convenience wrappers
# --------------------


def get_first_points_layer(viewer_or_layers: Any) -> Any | None:
    return find_first_layer(viewer_or_layers, Points)


def get_first_image_layer(viewer_or_layers: Any) -> Any | None:
    return find_first_layer(viewer_or_layers, Image)


def get_first_video_image_layer(viewer_or_layers: Any) -> Any | None:
    """First Image layer that looks like a video (>=3D data)."""

    def _is_video(img: Any) -> bool:
        try:
            return hasattr(img, "data") and getattr(img.data, "ndim", 0) >= 3
        except Exception:
            return False

    return find_first_layer(viewer_or_layers, Image, _is_video)


def get_points_layer_with_tables(viewer_or_layers: Any) -> Any | None:
    """First Points layer whose metadata has a non-empty 'tables' entry."""

    def _has_tables(pts: Any) -> bool:
        try:
            md = getattr(pts, "metadata", None) or {}
            return bool(md.get("tables"))
        except Exception:
            return False

    return find_first_layer(viewer_or_layers, Points, _has_tables)


def get_first_shapes_layer(viewer_or_layers: Any) -> Any | None:
    return find_first_layer(viewer_or_layers, Shapes)


def get_first_tracks_layer(viewer_or_layers: Any) -> Any | None:
    return find_first_layer(viewer_or_layers, Tracks)


@dataclass(frozen=True)
class LabelProgress:
    labeled_points: int
    total_points: int
    labeled_percent: float
    remaining_percent: float
    frame_count: int
    bodypart_count: int
    individual_count: int
    completed_frames: int
    completed_percent: float
    incomplete_frames: tuple[int, ...]
    incomplete_frames_by_individual: dict[str, int]
    missing_points_by_individual: dict[str, int]


def _get_header_model_from_metadata(md: dict) -> DLCHeaderModel | None:
    if not isinstance(md, dict):
        return None

    hdr = md.get("header")
    if hdr is None:
        return None

    if isinstance(hdr, DLCHeaderModel):
        return hdr

    if isinstance(hdr, dict):
        try:
            return DLCHeaderModel.model_validate(hdr)
        except Exception:
            return None

    try:
        return DLCHeaderModel(columns=hdr)
    except Exception:
        return None


def get_uniform_point_size(layer: Points, *, default: int = 6) -> int:
    size = getattr(layer, "size", default)
    try:
        arr = np.asarray(size, dtype=float).ravel()
        if arr.size == 0:
            return default
        return int(round(float(np.nanmean(arr))))
    except Exception:
        try:
            return int(round(float(size)))
        except Exception:
            return default


def set_uniform_point_size(layer: Points, size: int) -> None:
    # Scalar assignment keeps it lightweight and applies uniformly.
    layer.size = float(size)


def infer_frame_count(
    layer: Points, *, preferred_paths: list[str] | None = None, fallback_n_frames: int | None = None
) -> int:
    md = getattr(layer, "metadata", {}) or {}

    paths = preferred_paths or md.get("paths") or []
    if paths:
        return len(paths)
    if fallback_n_frames is not None:
        return fallback_n_frames

    data = np.asarray(getattr(layer, "data", []))
    if data.size == 0:
        return 0

    try:
        # Points layers use frame/time in first column
        return int(np.nanmax(data[:, 0])) + 1
    except Exception:
        return 0


def infer_bodypart_count(layer: Points) -> int:
    hdr = _get_header_model_from_metadata(getattr(layer, "metadata", {}) or {})
    if hdr is None:
        return 0

    try:
        return len([bp for bp in hdr.bodyparts if str(bp) != ""])
    except Exception:
        return 0


def infer_individual_count(layer: Points) -> int:
    """
    Returns the number of valid DLC individuals.

    Single-animal convention:
    - if no individuals are defined
    - or individuals are empty / blank
    => returns 1
    """
    hdr = _get_header_model_from_metadata(getattr(layer, "metadata", {}) or {})
    if hdr is None:
        return 1

    try:
        inds = [str(ind) for ind in hdr.individuals if str(ind) != ""]
        return max(1, len(inds))
    except Exception:
        return 1


def _normalized_slot_id(value) -> str:
    if value in ("", None):
        return ""
    try:
        if np.isnan(value):
            return ""
    except Exception:
        pass
    text = str(value)
    return "" if text.lower() == "nan" else text


def infer_observed_bodypart_names(layer: Points) -> list[str]:
    """
    Ordered unique bodypart labels actually present in the active napari layer.
    """
    props = getattr(layer, "properties", {}) or {}
    labels = np.asarray(props.get("label", []), dtype=object).ravel()

    out: list[str] = []
    seen: set[str] = set()
    for v in labels:
        if v in ("", None):
            continue
        text = str(v)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def infer_observed_individual_names(layer: Points) -> list[str]:
    """
    Ordered unique individual ids actually present in the active napari layer.

    Single-animal convention:
    - no ids / blank ids -> ['']
    """
    props = getattr(layer, "properties", {}) or {}
    ids_raw = props.get("id", None)
    if ids_raw is None:
        return [""]

    ids = np.asarray(ids_raw, dtype=object).ravel()

    out: list[str] = []
    seen: set[str] = set()
    for v in ids:
        text = _normalized_slot_id(v)
        if text == "":
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)

    return out if out else [""]


def _iter_labeled_slots(layer: Points):
    """
    Yield unique annotatable slots currently represented in the active napari layer.

    Each slot is keyed by:
    - frame index
    - individual id ('' for single-animal / blank ids)
    - bodypart label
    """
    data = np.asarray(getattr(layer, "data", []))
    if data.ndim < 2 or data.shape[1] == 0:
        return

    props = getattr(layer, "properties", {}) or {}
    labels = np.asarray(props.get("label", []), dtype=object).ravel()
    ids_raw = props.get("id", None)

    if ids_raw is None:
        ids = np.array([""] * len(labels), dtype=object)
    else:
        ids = np.asarray(ids_raw, dtype=object).ravel()

    n = min(len(data), len(labels), len(ids))
    for i in range(n):
        try:
            frame = int(data[i, 0])
        except Exception:
            continue

        label_val = labels[i]
        if label_val in ("", None):
            continue
        label = str(label_val)
        if not label:
            continue

        id_text = _normalized_slot_id(ids[i])
        yield (frame, id_text, label)


def compute_label_progress(
    layer: Points, *, fallback_paths: list[str] | None = None, fallback_n_frames: int | None = None
) -> LabelProgress:
    """
    Compute progress for the active napari layer.

    Semantics:
    - Main percentage remains theoretical:
        labeled_points / (frame_count * bodypart_count * individual_count)
    - Richer frame-completion details are computed from the observed slot universe
      currently represented in napari:
        observed_ids × observed_labels
    """
    frame_count = infer_frame_count(layer, preferred_paths=fallback_paths, fallback_n_frames=fallback_n_frames)
    bodypart_count = infer_bodypart_count(layer)
    individual_count = infer_individual_count(layer)

    total_points = frame_count * bodypart_count * individual_count

    # Keep the top-line point percentage as before.
    data = np.asarray(getattr(layer, "data", []))
    labeled_points = int(data.shape[0]) if data.ndim >= 2 else 0

    if total_points > 0:
        labeled_points = min(labeled_points, total_points)
        labeled_percent = 100.0 * labeled_points / total_points
    else:
        labeled_percent = 0.0

    remaining_percent = max(0.0, 100.0 - labeled_percent)

    # Richer completion details based on what is actually represented in napari.
    slots = set(_iter_labeled_slots(layer) or [])

    observed_labels = infer_observed_bodypart_names(layer)
    observed_ids = infer_observed_individual_names(layer)
    expected_ids = observed_ids if observed_ids else [""]

    expected_pairs = {(id_text, label) for id_text in expected_ids for label in observed_labels}
    expected_per_frame = len(expected_pairs)

    frame_to_pairs: dict[int, set[tuple[str, str]]] = {}
    for frame, id_text, label in slots:
        frame_to_pairs.setdefault(frame, set()).add((id_text, label))

    # Count-based frame completion: match the diagnostic / user-facing intuition.
    frame_slot_counts: dict[int, int] = {frame: len(pairs) for frame, pairs in frame_to_pairs.items()}

    completed_frames = 0
    incomplete_frames: list[int] = []
    incomplete_frames_by_individual: dict[str, int] = {id_text: 0 for id_text in expected_ids}
    missing_points_by_individual: dict[str, int] = {id_text: 0 for id_text in expected_ids}

    if frame_count > 0 and expected_per_frame > 0:
        for frame in range(frame_count):
            present = frame_to_pairs.get(frame, set())
            present_count = frame_slot_counts.get(frame, 0)

            # A frame is considered complete if it has the expected number of unique slots.
            if present_count >= expected_per_frame:
                completed_frames += 1
                continue

            incomplete_frames.append(frame)

            # Still compute missing details by comparing against expected pairs.
            # This is now only used for richer tooltip details on frames that are
            # count-incomplete, which keeps the user-facing summary intuitive.
            missing = expected_pairs - present

            missing_by_individual: dict[str, int] = {}
            for id_text, _label in missing:
                missing_by_individual[id_text] = missing_by_individual.get(id_text, 0) + 1

            for id_text, missing_count in missing_by_individual.items():
                incomplete_frames_by_individual[id_text] = incomplete_frames_by_individual.get(id_text, 0) + 1
                missing_points_by_individual[id_text] = missing_points_by_individual.get(id_text, 0) + missing_count

        completed_percent = 100.0 * completed_frames / frame_count
    else:
        completed_percent = 0.0

    return LabelProgress(
        labeled_points=labeled_points,
        total_points=total_points,
        labeled_percent=labeled_percent,
        remaining_percent=remaining_percent,
        frame_count=frame_count,
        bodypart_count=bodypart_count,
        individual_count=individual_count,
        completed_frames=completed_frames,
        completed_percent=completed_percent,
        incomplete_frames=tuple(incomplete_frames),
        incomplete_frames_by_individual=incomplete_frames_by_individual,
        missing_points_by_individual=missing_points_by_individual,
    )


def infer_folder_display_name(
    active_layer,
    *,
    fallback_root: str | None = None,
) -> str:
    """
    Best-effort label for the current image/video folder context.
    """
    if active_layer is None:
        return "—"

    md = getattr(active_layer, "metadata", {}) or {}

    paths = md.get("paths") or []
    if paths:
        try:
            return Path(paths[0]).expanduser().parent.name or "—"
        except Exception:
            pass

    root = md.get("root") or fallback_root
    if root:
        try:
            return Path(root).expanduser().name or "—"
        except Exception:
            pass

    try:
        src = getattr(getattr(active_layer, "source", None), "path", None)
        if src:
            p = Path(str(src))
            if p.is_file():
                # video source: show parent folder name
                return p.parent.name or p.stem or "—"
            return p.name or "—"
    except Exception:
        pass

    return "—"


@deprecated(details="Use LayerLifecycleManager.active_dlc_image_layer instead")
def find_relevant_image_layer(viewer) -> Image | None:
    active = viewer.layers.selection.active
    if isinstance(active, Image):
        return active

    for layer in viewer.layers:
        if isinstance(layer, Image):
            return layer

    return None


# -----------------------------------------------
#  Points interaction observer
# -----------------------------------------------


@dataclass(frozen=True)
class PointsInteractionEvent:
    """
    Structured points-layer interaction event.

    Parameters
    ----------
    viewer:
        The napari viewer.
    layer:
        The active Points layer at the time the event flushes, or None.
    reasons:
        A normalized set of reasons that triggered the event. Typical values
        include {"install"}, {"selection"}, {"active_layer"}, {"layers"},
        and {"content"}, depending on which observer hooks fired.
    """

    viewer: Any
    layer: Points | None
    reasons: frozenset[str]


def _iter_event_emitters(event_group: Any, names: tuple[str, ...]):
    """
    Yield (name, emitter) pairs for names that exist on an event group.
    """
    if event_group is None:
        return
    for name in names:
        emitter = getattr(event_group, name, None)
        if emitter is not None:
            yield name, emitter


def capture_points_state(
    layer: Points,
    *,
    include_data: bool = False,
    include_properties: bool = False,
) -> dict[str, Any]:
    """
    Best-effort snapshot helper for future history/undo systems.

    This is intentionally separate from the observer so callers can choose
    whether they want lightweight interaction events or heavier snapshots.
    """
    data = getattr(layer, "data", None)
    try:
        n_points = 0 if data is None else len(data)
    except Exception:
        n_points = 0

    state: dict[str, Any] = {
        "name": getattr(layer, "name", None),
        "selected_data": tuple(sorted(int(i) for i in getattr(layer, "selected_data", set()) or set())),
        "n_points": n_points,
    }

    if include_data:
        try:
            state["data"] = getattr(layer, "data", None).copy()
        except Exception:
            state["data"] = None

    if include_properties:
        try:
            props = getattr(layer, "properties", {}) or {}
            state["properties"] = {k: v.copy() if hasattr(v, "copy") else v for k, v in dict(props).items()}
        except Exception:
            state["properties"] = None

    return state


class PointsInteractionObserver:
    """
    Observe the active napari Points layer and emit coalesced interaction events.

    Public / stable anchors used
    ----------------------------
    - viewer.layers.selection.events.active
    - layer.selected_data.events.changed / active
    - layer.selected_data.events.items_changed (if present; useful in practice)
    - viewer.layers.events.inserted / removed / reordered (if present)

    Notes
    -----
    This is intentionally conservative:
    - it avoids private napari APIs
    - it tolerates event-name differences by connecting only to emitters that exist
    - it coalesces bursts of events into one callback using a QTimer
    """

    def __init__(
        self,
        viewer: Any,
        callback: Callable[[PointsInteractionEvent], None],
        *,
        debounce_ms: int = 0,
        watch_content: bool = False,
    ) -> None:
        self.viewer = viewer
        self.callback = callback
        self.debounce_ms = max(0, int(debounce_ms))
        self.watch_content = watch_content

        self._active_layer: Points | None = None
        self._viewer_connections: list[tuple[Any, Callable]] = []
        self._layer_connections: list[tuple[Any, Callable]] = []
        self._pending_reasons: set[str] = set()

        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._flush)

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def install(self) -> None:
        """
        Install the observer onto the viewer.
        """
        self._connect_viewer_events()
        self._rebind_active_points_layer()
        self._schedule("install")

    def close(self) -> None:
        """
        Disconnect all emitters and stop the timer.
        """
        self._timer.stop()
        self._disconnect_all(self._layer_connections)
        self._disconnect_all(self._viewer_connections)
        self._active_layer = None
        self._pending_reasons.clear()

    # ------------------------------------------------------------------
    # Internal connection helpers
    # ------------------------------------------------------------------

    def _connect(self, emitter: Any, callback: Callable, bucket: list[tuple[Any, Callable]]) -> None:
        emitter.connect(callback)
        bucket.append((emitter, callback))

    def _disconnect_all(self, bucket: list[tuple[Any, Callable]]) -> None:
        while bucket:
            emitter, callback = bucket.pop()
            try:
                emitter.disconnect(callback)
            except Exception:
                pass

    def _connect_viewer_events(self) -> None:
        # Active layer changes are the most important public hook.
        active_emitter = getattr(self.viewer.layers.selection.events, "active", None)
        if active_emitter is not None:
            self._connect(active_emitter, self._on_active_layer_changed, self._viewer_connections)

        layer_events = getattr(self.viewer.layers, "events", None)
        for _name, emitter in _iter_event_emitters(layer_events, ("inserted", "removed", "reordered")):
            self._connect(emitter, self._on_layers_changed, self._viewer_connections)

    def _rebind_active_points_layer(self) -> None:
        self._disconnect_all(self._layer_connections)

        active = getattr(self.viewer.layers.selection, "active", None)
        if not isinstance(active, Points):
            self._active_layer = None
            return

        self._active_layer = active

        # Primary selection hooks: Selection model events
        selection = getattr(active, "selected_data", None)
        selection_events = getattr(selection, "events", None)

        for _name, emitter in _iter_event_emitters(selection_events, ("changed", "active", "items_changed")):
            self._connect(emitter, self._on_selection_changed, self._layer_connections)

        # Optional content hooks, useful for future history/versioning use cases.
        if self.watch_content:
            layer_events = getattr(active, "events", None)
            for event_name in ("data", "properties", "current_properties", "mode"):
                for _name, emitter in _iter_event_emitters(layer_events, (event_name,)):
                    self._connect(emitter, self._on_content_changed, self._layer_connections)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_active_layer_changed(self, event=None) -> None:
        self._rebind_active_points_layer()
        self._schedule("active_layer")

    def _on_layers_changed(self, event=None) -> None:
        # The active layer may or may not have changed, but rebinding is cheap and safe.
        self._rebind_active_points_layer()
        self._schedule("layers")

    def _on_selection_changed(self, event=None) -> None:
        self._schedule("selection")

    def _on_content_changed(self, event=None) -> None:
        self._schedule("content")

    # ------------------------------------------------------------------
    # Coalescing
    # ------------------------------------------------------------------

    def _schedule(self, reason: str) -> None:
        self._pending_reasons.add(reason)
        if not self._timer.isActive():
            self._timer.start(self.debounce_ms)

    def _flush(self) -> None:
        reasons = frozenset(self._pending_reasons)
        self._pending_reasons.clear()

        event = PointsInteractionEvent(
            viewer=self.viewer,
            layer=self._active_layer,
            reasons=reasons,
        )
        self.callback(event)
