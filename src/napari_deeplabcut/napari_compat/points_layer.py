# src/napari_deeplabcut/napari_compat/points_layer.py
from __future__ import annotations

import logging
from copy import deepcopy
from types import MethodType

import numpy as np

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
    """

    def _paste_data(layer_self, _store=store):
        features = layer_self._clipboard.pop("features", None)
        if features is None:
            return

        unannotated = [
            controls.keypoints.Keypoint(label, id_) not in _store.annotated_keypoints
            for label, id_ in zip(features["label"], features["id"], strict=False)
        ]
        if not any(unannotated):
            return

        new_features = features.iloc[unannotated]
        indices_ = layer_self._clipboard.pop("indices")
        text_ = layer_self._clipboard.pop("text", None)  # <- guard missing key
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

        if len(layer_self._clipboard.keys()) > 0:
            not_disp = layer_self._slice_input.not_displayed
            data = deepcopy(layer_self._clipboard["data"])
            offset = [layer_self._slice_indices[i] - layer_self._clipboard["indices"][i] for i in not_disp]
            data[:, not_disp] = data[:, not_disp] + np.array(offset)
            layer_self._data = np.append(layer_self.data, data, axis=0)
            layer_self._shown = np.append(layer_self.shown, deepcopy(layer_self._clipboard["shown"]), axis=0)
            layer_self._size = np.append(layer_self.size, deepcopy(layer_self._clipboard["size"]), axis=0)
            layer_self._symbol = np.append(layer_self.symbol, deepcopy(layer_self._clipboard["symbol"]), axis=0)

            layer_self._feature_table.append(layer_self._clipboard["features"])

            text_payload = layer_self._clipboard.get("text")
            if text_payload is not None:  # <- guard missing / None payload
                layer_self.text._paste(**text_payload)

            layer_self._edge_width = np.append(
                layer_self.edge_width,
                deepcopy(layer_self._clipboard["edge_width"]),
                axis=0,
            )

            # private napari helpers: guarded by compat usage
            from napari.layers.utils.layer_utils import _features_to_properties

            layer_self._edge._paste(
                colors=layer_self._clipboard["edge_color"],
                properties=_features_to_properties(layer_self._clipboard["features"]),
            )
            layer_self._face._paste(
                colors=layer_self._clipboard["face_color"],
                properties=_features_to_properties(layer_self._clipboard["features"]),
            )

            layer_self._selected_view = list(range(npoints, npoints + len(layer_self._clipboard["data"])))
            layer_self._selected_data = set(range(totpoints, totpoints + len(layer_self._clipboard["data"])))
            layer_self.refresh()

            try:
                controls._schedule_recolor(_store.layer)
            except Exception:
                pass

    return _paste_data
