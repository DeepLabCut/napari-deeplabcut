from collections import namedtuple
from collections.abc import Sequence
from enum import auto

import numpy as np
from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari.layers import Points
from napari.layers.points._points_constants import SYMBOL_TRANSLATION_INVERTED
from napari.layers.points._points_utils import coerce_symbols
from scipy.spatial import cKDTree

from napari_deeplabcut.misc import CycleEnum


# Monkeypatch the point size slider
def _change_size(self, value):
    """Resize all points at once regardless of the current selection."""
    self.layer._current_size = value
    if self.layer._update_properties:
        self.layer.size = (self.layer.size > 0) * value
        self.layer.refresh()
        self.layer.events.size()


def _change_symbol(self, text):
    symbol = coerce_symbols(np.array([SYMBOL_TRANSLATION_INVERTED[text]]))[0]
    self.layer._current_symbol = symbol
    if self.layer._update_properties:
        self.layer.symbol = symbol
        self.layer.events.symbol()
    self.layer.events.current_symbol()


QtPointsControls.changeCurrentSize = _change_size
QtPointsControls.changeCurrentSymbol = _change_symbol


class ColorMode(CycleEnum):
    """Modes in which keypoints can be colored

    BODYPART: the keypoints are grouped by bodypart (all bodyparts have the same color)
    INDIVIDUAL: the keypoints are grouped by individual (all keypoints for the same
        individual have the same color)
    """

    BODYPART = auto()
    INDIVIDUAL = auto()

    @classmethod
    def default(cls):
        return cls.BODYPART


class LabelMode(CycleEnum):
    """
    Labeling modes.
    SEQUENTIAL: points are placed in sequence, then frame after frame;
        clicking to add an already annotated point has no effect.
    QUICK: similar to SEQUENTIAL, but trying to add an already
        annotated point actually moves it to the cursor location.
    LOOP: the currently selected point is placed frame after frame,
        before wrapping at the end to frame 1, etc.
    """

    SEQUENTIAL = auto()
    QUICK = auto()
    LOOP = auto()

    @classmethod
    def default(cls):
        return cls.SEQUENTIAL


# Description tooltips for the labeling modes radio buttons.
TOOLTIPS = {
    "SEQUENTIAL": "Points are placed in sequence, then frame after frame;\n"
    "clicking to add an already annotated point has no effect.",
    "QUICK": "Similar to SEQUENTIAL, but trying to add an already\n"
    "annotated point actually moves it to the cursor location.",
    "LOOP": "The currently selected point is placed frame after frame,\nbefore wrapping at the end to frame 1, etc.",
}


Keypoint = namedtuple("Keypoint", ["label", "id"])


class KeypointStore:
    def __init__(self, viewer, layer: Points):
        self.viewer = viewer
        self._keypoints = []
        self.layer = layer
        self.viewer.dims.set_current_step(0, 0)

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layer):
        self._layer = layer
        all_pairs = self.layer.metadata["header"].form_individual_bodypart_pairs()
        self._keypoints = [
            Keypoint(label, id_) for id_, label in all_pairs
        ]  # Ordered references to all possible keypoints

    @property
    def current_step(self):
        return self.viewer.dims.current_step[0]

    @property
    def n_steps(self):
        return self.viewer.dims.nsteps[0]

    @property
    def annotated_keypoints(self) -> list[Keypoint]:
        mask = self.current_mask
        labels = self.layer.properties["label"][mask]
        ids = self.layer.properties["id"][mask]
        return [Keypoint(label, id_) for label, id_ in zip(labels, ids, strict=False)]

    @property
    def current_mask(self) -> Sequence[bool]:
        return np.asarray(self.layer.data[:, 0] == self.current_step)

    @property
    def current_keypoint(self) -> Keypoint:
        props = self.layer.current_properties
        return Keypoint(label=props["label"][0], id=props["id"][0])

    @current_keypoint.setter
    def current_keypoint(self, keypoint: Keypoint):
        # Avoid changing the properties of a selected point
        if not len(self.layer.selected_data):
            current_properties = self.layer.current_properties
            current_properties["label"] = np.asarray([keypoint.label])
            current_properties["id"] = np.asarray([keypoint.id])
            self.layer.current_properties = current_properties

    def next_keypoint(self, *args):
        ind = self._keypoints.index(self.current_keypoint) + 1
        if ind <= len(self._keypoints) - 1:
            self.current_keypoint = self._keypoints[ind]

    def prev_keypoint(self, *args):
        ind = self._keypoints.index(self.current_keypoint) - 1
        if ind >= 0:
            self.current_keypoint = self._keypoints[ind]

    @property
    def labels(self) -> list[str]:
        return self.layer.metadata["header"].bodyparts

    @property
    def current_label(self) -> str:
        return self.layer.current_properties["label"][0]

    @current_label.setter
    def current_label(self, label: str):
        if not len(self.layer.selected_data):
            current_properties = self.layer.current_properties
            current_properties["label"] = np.asarray([label])
            self.layer.current_properties = current_properties

    @property
    def ids(self) -> list[str]:
        return self.layer.metadata["header"].individuals

    @property
    def current_id(self) -> str:
        return self.layer.current_properties["id"][0]

    @current_id.setter
    def current_id(self, id_: str):
        if not len(self.layer.selected_data):
            current_properties = self.layer.current_properties
            current_properties["id"] = np.asarray([id_])
            self.layer.current_properties = current_properties

    def _advance_step(self, event):
        ind = (self.current_step + 1) % self.n_steps
        self.viewer.dims.set_current_step(0, ind)

    def _find_first_unlabeled_frame(self, event):
        inds = set(range(self.n_steps))
        unlabeled_inds = inds.difference(self.layer.data[:, 0].astype(int))
        if not unlabeled_inds:
            self.viewer.dims.set_current_step(0, self.n_steps - 1)
        else:
            self.viewer.dims.set_current_step(0, min(unlabeled_inds))


def _add(store, coord):
    coord = np.atleast_2d(coord)

    # Controls may be missing if metadata was replaced/filtered.
    controls = (store.layer.metadata or {}).get("controls", None)
    label_mode = getattr(controls, "_label_mode", None)

    if store.current_keypoint not in store.annotated_keypoints:
        # 1) append data
        store.layer.data = np.append(store.layer.data, coord, axis=0)

        # 2) append/align properties to match number of points
        kp = store.current_keypoint
        n_new = coord.shape[0]
        n_total = len(store.layer.data)
        n_old = n_total - n_new

        props = store.layer.properties.copy()

        def _as_array(key, dtype):
            arr = props.get(key, None)
            if arr is None:
                return np.array([], dtype=dtype)
            return np.asarray(arr, dtype=dtype)

        # Existing values truncated/padded to n_old, then append new rows
        label_arr = _as_array("label", object)[:n_old]
        id_arr = _as_array("id", object)[:n_old]
        lik_arr = _as_array("likelihood", float)[:n_old]

        # If any are shorter than n_old, pad (rare but safe)
        if label_arr.size < n_old:
            label_arr = np.concatenate([label_arr, np.array([kp.label] * (n_old - label_arr.size), dtype=object)])
        if id_arr.size < n_old:
            id_arr = np.concatenate([id_arr, np.array([kp.id] * (n_old - id_arr.size), dtype=object)])
        if lik_arr.size < n_old:
            lik_arr = np.concatenate([lik_arr, np.ones(n_old - lik_arr.size, dtype=float)])

        props["label"] = np.concatenate([label_arr, np.array([kp.label] * n_new, dtype=object)])
        props["id"] = np.concatenate([id_arr, np.array([kp.id] * n_new, dtype=object)])
        props["likelihood"] = np.concatenate([lik_arr, np.ones(n_new, dtype=float)])

        store.layer.properties = props

    elif label_mode is LabelMode.QUICK:
        ind = store.annotated_keypoints.index(store.current_keypoint)
        data = store.layer.data
        data[np.flatnonzero(store.current_mask)[ind]] = coord.squeeze()
        store.layer.data = data

    store.layer.selected_data = set()

    # If controls are missing, behave like the default mode (advance keypoint)
    if label_mode is LabelMode.LOOP:
        store.layer.events.query_next_frame()
    else:
        store.next_keypoint()


def _find_nearest_neighbors(xy_true, xy_pred, k=5):
    n_preds = xy_pred.shape[0]
    tree = cKDTree(xy_pred)
    dist, inds = tree.query(xy_true, k=k)
    idx = np.argsort(dist[:, 0])
    neighbors = np.full(len(xy_true), -1, dtype=int)
    picked = set()
    for i, ind in enumerate(inds[idx]):
        for j in ind:
            if j not in picked:
                picked.add(j)
                neighbors[idx[i]] = j
                break
        if len(picked) == n_preds:
            break
    return neighbors
