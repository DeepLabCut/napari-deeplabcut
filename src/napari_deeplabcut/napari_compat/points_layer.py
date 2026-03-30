# src/napari_deeplabcut/napari_compat/points_layer.py
from __future__ import annotations

import logging
from copy import deepcopy
from types import MethodType
from typing import Any

import numpy as np

from napari_deeplabcut.core.keypoints import Keypoint

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Optional napari private import
# -----------------------------------------------------------------------------

try:
    from napari.layers.points._points_key_bindings import register_points_action
except Exception:

    def register_points_action(*args, **kwargs):
        def deco(fn):
            return fn

        return deco

    logger.debug("napari private register_points_action unavailable; skipping action registration.")


# -----------------------------------------------------------------------------
# Compat constants
# -----------------------------------------------------------------------------

_WIDTH_KEYS = ("border_width", "edge_width")
_BORDER_COLOR_KEYS = ("border_color", "edge_color")
_BORDER_MANAGERS = ("_border", "_edge")
_WIDTH_ATTRS = ("_border_width", "_edge_width")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _first_present(mapping: dict[str, Any], *names: str) -> str | None:
    """Return the first key present in a mapping."""
    for name in names:
        if name in mapping:
            return name
    return None


def _first_attr(obj: Any, *names: str) -> Any | None:
    """Return the first existing attribute value from an object."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _normalize_id(id_: Any) -> Any:
    """Normalize clipboard/stored ids so Keypoint comparisons are stable."""
    if id_ is None:
        return id_

    try:
        if np.isnan(id_):
            return ""
    except Exception:
        pass

    try:
        return id_.item() if hasattr(id_, "item") else id_
    except Exception:
        return id_


def _get_current_slice_point(controls: Any, layer: Any) -> tuple | Any | None:
    """
    Best-effort current slice position.

    Preference order:
    1. viewer dims (authoritative user-visible state)
    2. old napari private layer state: _slice_indices
    3. newer napari private layer state: _slice_input.point
    """
    try:
        current_step = tuple(controls.viewer.dims.current_step)
        if current_step:
            return current_step
    except Exception:
        pass

    if hasattr(layer, "_slice_indices"):
        return layer._slice_indices

    slice_input = getattr(layer, "_slice_input", None)
    if slice_input is not None and hasattr(slice_input, "point"):
        return slice_input.point

    return None


def _get_clipboard_slice_point(indices: Any) -> Any:
    """Extract the copied slice point from old/new napari clipboard payloads."""
    if hasattr(indices, "point"):
        return indices.point
    return indices


def _get_not_displayed_axes(controls: Any, layer: Any, data: np.ndarray) -> list[int]:
    """
    Best-effort non-displayed axes.

    Preference order:
    1. newer napari private layer state: _slice_input.not_displayed
    2. infer from ndim / viewer ndisplay
    """
    slice_input = getattr(layer, "_slice_input", None)
    if slice_input is not None and hasattr(slice_input, "not_displayed"):
        try:
            return list(slice_input.not_displayed)
        except Exception:
            pass

    try:
        ndim = int(data.shape[1])
    except Exception:
        return []

    try:
        ndisplay = int(controls.viewer.dims.ndisplay)
    except Exception:
        ndisplay = 2

    n_not_displayed = max(0, ndim - ndisplay)
    return list(range(n_not_displayed))


def _filter_clipboard_payload(clipboard: dict[str, Any], mask: list[bool]) -> dict[str, Any]:
    """Return a filtered clipboard payload keeping only rows selected by mask."""
    return {k: v[mask] for k, v in clipboard.items()}


def _paste_text_payload(layer: Any, clipboard: dict[str, Any]) -> None:
    """Paste text payload if present."""
    text_payload = clipboard.get("text")
    if text_payload is None:
        return

    try:
        layer.text._paste(**text_payload)
    except Exception:
        logger.debug("Failed to paste text payload", exc_info=True)


def _append_widths(layer: Any, clipboard: dict[str, Any]) -> None:
    """Append border/edge widths compatibly across napari versions."""
    width_key = _first_present(clipboard, *_WIDTH_KEYS)
    if width_key is None:
        return

    current_width = _first_attr(layer, "border_width", "edge_width")
    if current_width is None:
        return

    new_width = np.append(current_width, deepcopy(clipboard[width_key]), axis=0)

    for private_name in _WIDTH_ATTRS:
        if hasattr(layer, private_name):
            setattr(layer, private_name, new_width)
            return


def _paste_colors(layer: Any, clipboard: dict[str, Any]) -> None:
    """Paste face/border colors compatibly across napari versions."""
    from napari.layers.utils.layer_utils import _features_to_properties

    props = _features_to_properties(clipboard["features"])

    border_manager = _first_attr(layer, *_BORDER_MANAGERS)
    border_color_key = _first_present(clipboard, *_BORDER_COLOR_KEYS)
    if border_manager is not None and border_color_key is not None:
        border_manager._paste(colors=clipboard[border_color_key], properties=props)

    face_manager = getattr(layer, "_face", None)
    if face_manager is not None and "face_color" in clipboard:
        face_manager._paste(colors=clipboard["face_color"], properties=props)


def _offset_pasted_data(
    controls: Any,
    layer: Any,
    clipboard: dict[str, Any],
    data: np.ndarray,
) -> np.ndarray:
    """
    Shift pasted coordinates along non-displayed axes so paste occurs
    on the viewer's current slice/frame.
    """
    not_disp = _get_not_displayed_axes(controls, layer, data)
    if not not_disp:
        return data

    current_point = _get_current_slice_point(controls, layer)
    copied_point = _get_clipboard_slice_point(clipboard["indices"])
    if current_point is None or copied_point is None:
        return data

    try:
        offset = [float(current_point[i]) - float(copied_point[i]) for i in not_disp]
    except Exception:
        logger.debug("Failed to compute paste offset", exc_info=True)
        return data

    shifted = deepcopy(data)
    shifted[:, not_disp] = shifted[:, not_disp] + np.asarray(offset, dtype=float)
    return shifted


# -----------------------------------------------------------------------------
# Public compat helpers
# -----------------------------------------------------------------------------


def apply_points_layer_ui_tweaks(viewer, layer, *, dropdown_cls, plt_module) -> object | None:
    """
    Best-effort private napari UI wiring.

    Returns
    -------
    object | None
        The created colormap selector, or None if unavailable.
    """
    try:
        controls = viewer.window.qt_viewer.dockLayerControls
        point_controls = controls.widget().widgets[layer]
    except Exception:
        return None

    widgets_to_hide = [
        ("_face_color_control", "face_color_edit"),
        ("_face_color_control", "face_color_label"),
        ("_border_color_control", "border_color_edit"),
        ("_border_color_control", "border_color_edit_label"),
        ("_out_slice_checkbox_control", "out_of_slice_checkbox"),
        ("_out_slice_checkbox_control", "out_of_slice_checkbox_label"),
    ]

    for parent_attr, widget_attr in widgets_to_hide:
        try:
            parent = getattr(point_controls, parent_attr)
            widget = getattr(parent, widget_attr)
            widget.hide()
        except Exception:
            pass

    try:
        cmap_source = plt_module.colormaps
        if callable(cmap_source):
            cmap_source = cmap_source()

        colormap_selector = dropdown_cls(cmap_source, point_controls)
        colormap_selector.update_to(layer.metadata.get("colormap_name", "viridis"))
        point_controls.layout().addRow("colormap", colormap_selector)
        return colormap_selector
    except Exception as e:
        logger.debug("Failed to add colormap selector: %r", e, exc_info=True)
        return None


def install_add_wrapper(layer, *, add_impl, schedule_recolor) -> None:
    """
    Wrap layer.add to schedule recolor after add.

    Parameters
    ----------
    add_impl
        Callable like keypoints._add bound to store.
    schedule_recolor
        Callable(layer) -> None.
    """
    try:

        def add_and_recolor(this, *args, **kwargs):
            res = add_impl(*args, **kwargs)
            try:
                schedule_recolor(this)
            except Exception:
                pass
            return res

        layer.add = MethodType(add_and_recolor, layer)
    except Exception as e:
        logger.debug("Skipping add wrapper install: %r", e)


def install_paste_patch(layer, *, paste_func) -> None:
    """
    Patch napari Points._paste_data with our safe implementation.
    """
    try:
        layer._paste_data = MethodType(paste_func, layer)
    except Exception as e:
        logger.debug("Skipping paste patch install: %r", e)


def make_paste_data(controls, *, store):
    """
    Build a compat paste handler for napari Points layers.

    Behavior:
    - pastes only keypoints not already annotated in the target frame
    - shifts pasted coordinates to the viewer's current non-displayed slice(s)
    - supports old/new napari clipboard/style internals
    """
    keypoint_cls = getattr(getattr(controls, "keypoints", None), "Keypoint", Keypoint)

    def _paste_data(layer_self, _store=store):
        clipboard = getattr(layer_self, "_clipboard", None) or {}
        features = clipboard.get("features")
        if features is None:
            return

        unannotated = [
            keypoint_cls(label, _normalize_id(id_)) not in _store.annotated_keypoints
            for label, id_ in zip(features["label"], features["id"], strict=False)
        ]
        if not any(unannotated):
            return

        # Only mutate clipboard once we know there is something to paste.
        features = clipboard.pop("features")
        indices = clipboard.pop("indices")
        text = clipboard.pop("text", None)

        new_features = features.iloc[unannotated]
        filtered_clipboard = _filter_clipboard_payload(clipboard, unannotated)
        filtered_clipboard["features"] = new_features
        filtered_clipboard["indices"] = indices

        if text is not None:
            filtered_clipboard["text"] = {
                "string": text["string"][unannotated],
                "color": text["color"],
            }

        layer_self._clipboard = filtered_clipboard

        if not filtered_clipboard:
            return

        npoints = len(layer_self._view_data)
        totpoints = len(layer_self.data)

        data = _offset_pasted_data(
            controls=controls,
            layer=layer_self,
            clipboard=filtered_clipboard,
            data=deepcopy(filtered_clipboard["data"]),
        )

        layer_self._data = np.append(layer_self.data, data, axis=0)
        layer_self._shown = np.append(layer_self.shown, deepcopy(filtered_clipboard["shown"]), axis=0)
        layer_self._size = np.append(layer_self.size, deepcopy(filtered_clipboard["size"]), axis=0)
        layer_self._symbol = np.append(layer_self.symbol, deepcopy(filtered_clipboard["symbol"]), axis=0)

        layer_self._feature_table.append(filtered_clipboard["features"])
        _paste_text_payload(layer_self, filtered_clipboard)
        _append_widths(layer_self, filtered_clipboard)
        _paste_colors(layer_self, filtered_clipboard)

        n_new = len(filtered_clipboard["data"])
        layer_self._selected_view = list(range(npoints, npoints + n_new))
        layer_self._selected_data = set(range(totpoints, totpoints + n_new))
        layer_self.refresh()

        try:
            controls._schedule_recolor(_store.layer)
        except Exception:
            pass

    return _paste_data
