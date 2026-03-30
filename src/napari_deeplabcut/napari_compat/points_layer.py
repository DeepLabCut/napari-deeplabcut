# src/napari_deeplabcut/napari_compat/points_layer.py
from __future__ import annotations

import logging
from copy import deepcopy
from types import MethodType

import numpy as np

from napari_deeplabcut.core.keypoints import Keypoint

logger = logging.getLogger(__name__)

try:
    from napari.layers.points._points_key_bindings import register_points_action
except Exception:

    def register_points_action(*args, **kwargs):
        def deco(fn):
            return fn

        return deco

    logger.debug("napari private register_points_action unavailable; skipping action registration.")


def apply_points_layer_ui_tweaks(viewer, layer, *, dropdown_cls, plt_module) -> object | None:
    """
    Private napari UI wiring (best-effort).
    If napari changes internals, degrade gracefully.
    """
    try:
        controls = viewer.window.qt_viewer.dockLayerControls
        point_controls = controls.widget().widgets[layer]
    except Exception:
        return None

    # Hide face/border editors/out-of-slice toggle (guarded)
    for attr_path in [
        ("_face_color_control", "face_color_edit"),
        ("_face_color_control", "face_color_label"),
        ("_border_color_control", "border_color_edit"),
        ("_border_color_control", "border_color_edit_label"),
        ("_out_slice_checkbox_control", "out_of_slice_checkbox"),
        ("_out_slice_checkbox_control", "out_of_slice_checkbox_label"),
    ]:
        try:
            obj = getattr(point_controls, attr_path[0])
            widget = getattr(obj, attr_path[1])
            widget.hide()
        except Exception:
            pass

    # Add colormap selector (guarded)
    try:
        cmap_source = plt_module.colormaps
        if callable(cmap_source):
            cmap_source = cmap_source()
        colormap_selector = dropdown_cls(cmap_source, point_controls)
        colormap_selector.update_to(layer.metadata.get("colormap_name", "viridis"))
        # caller wires signal to its handler; keep this compat layer minimal
        point_controls.layout().addRow("colormap", colormap_selector)
        return colormap_selector
    except Exception as e:
        logger.debug("Failed to add colormap selector: %r", e, exc_info=True)
        return None


def install_add_wrapper(layer, *, add_impl, schedule_recolor) -> None:
    """
    Wrap layer.add to schedule recolor after add.

    add_impl: callable like keypoints._add bound to store
    schedule_recolor: callable(layer) -> None
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

    paste_func: callable(layer, store) -> None or callable with signature used by original patch
    """
    try:
        layer._paste_data = MethodType(paste_func, layer)
    except Exception as e:
        logger.debug("Skipping paste patch install: %r", e)


def make_paste_data(controls, *, store):
    """
    Build a paste handler that mimics the previous KeypointControls._paste_data behavior,
    but lives in compat.

    Compat goals:
    - old/new napari slice APIs (_slice_indices vs _slice_input.point)
    - old/new napari points style keys (edge_* vs border_*)
    """

    def _normalize_id(id_):
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

    def _get_current_slice_point(layer_self):
        # Best source for "where the user currently is" is the viewer dims.
        try:
            current_step = tuple(controls.viewer.dims.current_step)
            if len(current_step) == layer_self.data.shape[1]:
                return current_step
        except Exception:
            pass

        # Older napari private API fallback
        if hasattr(layer_self, "_slice_indices"):
            return layer_self._slice_indices

        # Newer napari private API fallback
        slice_input = getattr(layer_self, "_slice_input", None)
        if slice_input is not None and hasattr(slice_input, "point"):
            return slice_input.point

        return None

    def _get_clipboard_slice_point(clipboard_indices):
        # Newer napari stores a _ThickNDSlice-like object with .point
        if hasattr(clipboard_indices, "point"):
            return clipboard_indices.point
        return clipboard_indices

    def _get_clipboard_key(clipboard, *names):
        for name in names:
            if name in clipboard:
                return name
        return None

    def _get_private_color_manager(layer_self, *names):
        for name in names:
            obj = getattr(layer_self, name, None)
            if obj is not None:
                return obj
        return None

    def _paste_data(layer_self, _store=store):
        features = layer_self._clipboard.get("features")
        if features is None:
            return

        unannotated = [
            Keypoint(label, _normalize_id(id_)) not in _store.annotated_keypoints
            for label, id_ in zip(features["label"], features["id"], strict=False)
        ]
        if not any(unannotated):
            return

        # Only mutate clipboard after we know we want to paste
        features = layer_self._clipboard.pop("features")
        new_features = features.iloc[unannotated]

        indices_ = layer_self._clipboard.pop("indices")
        text_ = layer_self._clipboard.pop("text", None)

        layer_self._clipboard = {k: v[unannotated] for k, v in layer_self._clipboard.items()}
        layer_self._clipboard["features"] = new_features
        layer_self._clipboard["indices"] = indices_

        if text_ is not None:
            new_text = {
                "string": text_["string"][unannotated],
                "color": text_["color"],
            }
            layer_self._clipboard["text"] = new_text

        npoints = len(layer_self._view_data)
        totpoints = len(layer_self.data)

        if len(layer_self._clipboard) > 0:
            not_disp = layer_self._slice_input.not_displayed
            data = deepcopy(layer_self._clipboard["data"])

            # ---- compat: current slice point / copied slice point ----
            current_point = _get_current_slice_point(layer_self)
            copied_point = _get_clipboard_slice_point(layer_self._clipboard["indices"])

            if current_point is not None and copied_point is not None and len(not_disp) > 0:
                offset = []
                for i in not_disp:
                    cur = current_point[i]
                    old = copied_point[i]
                    offset.append(float(cur) - float(old))
                data[:, not_disp] = data[:, not_disp] + np.asarray(offset, dtype=float)

            layer_self._data = np.append(layer_self.data, data, axis=0)
            layer_self._shown = np.append(
                layer_self.shown,
                deepcopy(layer_self._clipboard["shown"]),
                axis=0,
            )
            layer_self._size = np.append(
                layer_self.size,
                deepcopy(layer_self._clipboard["size"]),
                axis=0,
            )
            layer_self._symbol = np.append(
                layer_self.symbol,
                deepcopy(layer_self._clipboard["symbol"]),
                axis=0,
            )

            layer_self._feature_table.append(layer_self._clipboard["features"])

            text_payload = layer_self._clipboard.get("text")
            if text_payload is not None:
                layer_self.text._paste(**text_payload)

            # ---- compat: edge_width vs border_width ----
            width_key = _get_clipboard_key(layer_self._clipboard, "border_width", "edge_width")
            if width_key is not None:
                current_width = None
                for public_name in ("border_width", "edge_width"):
                    if hasattr(layer_self, public_name):
                        current_width = getattr(layer_self, public_name)
                        break

                if current_width is not None:
                    new_width = np.append(
                        current_width,
                        deepcopy(layer_self._clipboard[width_key]),
                        axis=0,
                    )
                    for private_name in ("_border_width", "_edge_width"):
                        if hasattr(layer_self, private_name):
                            setattr(layer_self, private_name, new_width)
                            break

            from napari.layers.utils.layer_utils import _features_to_properties

            props = _features_to_properties(layer_self._clipboard["features"])

            # ---- compat: edge/border color managers ----
            border_manager = _get_private_color_manager(layer_self, "_border", "_edge")
            border_color_key = _get_clipboard_key(layer_self._clipboard, "border_color", "edge_color")

            if border_manager is not None and border_color_key is not None:
                border_manager._paste(
                    colors=layer_self._clipboard[border_color_key],
                    properties=props,
                )

            face_manager = _get_private_color_manager(layer_self, "_face")
            if face_manager is not None and "face_color" in layer_self._clipboard:
                face_manager._paste(
                    colors=layer_self._clipboard["face_color"],
                    properties=props,
                )

            layer_self._selected_view = list(range(npoints, npoints + len(layer_self._clipboard["data"])))
            layer_self._selected_data = set(range(totpoints, totpoints + len(layer_self._clipboard["data"])))
            layer_self.refresh()

            try:
                controls._schedule_recolor(_store.layer)
            except Exception:
                pass

    return _paste_data
