# src/napari_deeplabcut/keypoints.py
import logging
import weakref
from collections import namedtuple
from collections.abc import Callable, Sequence
from enum import auto

import numpy as np
from matplotlib import colormaps as mpl_colormaps
from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari.layers import Points
from napari.layers.points._points_constants import SYMBOL_TRANSLATION_INVERTED
from napari.layers.points._points_utils import coerce_symbols
from napari.utils import colormaps
from pydantic import ValidationError
from scipy.spatial import cKDTree

from napari_deeplabcut.config.models import DLCHeaderModel
from napari_deeplabcut.core.metadata import read_points_meta
from napari_deeplabcut.misc import CycleEnum, HeaderLike
from napari_deeplabcut.utils.deprecations import deprecated

logger = logging.getLogger(__name__)


class LayerUnavailableError(RuntimeError):
    """Raised when a KeypointStore can no longer resolve its backing layer."""


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


@deprecated(details="Unused currently, should be removed.")
def _validate_points_meta_best_effort(layer) -> bool:
    """
    We drop header + controls during validation to avoid runtime-object issues.
    """
    res = read_points_meta(layer, migrate_legacy=True, drop_controls=True, drop_header=True)
    if isinstance(res, ValidationError):
        logger.debug("Points metadata invalid for layer=%r: %s", getattr(layer, "name", layer), res)
        return False
    return True


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
    def __init__(
        self,
        viewer,
        layer: Points,
        *,
        resolve_layer_by_id: Callable[[int], Points] | None = None,
    ):
        self.viewer = viewer
        self._keypoints = []
        self._header: DLCHeaderModel | None = None

        self._layer_id: int | None = None
        self._resolve_layer_by_id = resolve_layer_by_id

        # Fallback if no resolver is provided
        self._layer_ref: weakref.ReferenceType[Points] | None = None
        self._strong_layer_ref: Points | None = None  # Used to keep the layer alive if no resolver is provided

        self.layer = layer  # Use the setter to initialize keypoints and header

        self.viewer.dims.set_current_step(0, 0)

    def set_label_mode_getter(self, getter: Callable[[], LabelMode]):
        self._get_label_mode = getter

    @property
    def layer_id(self) -> int | None:
        return self._layer_id

    def attach_layer_resolver(self, resolve_layer_by_id: Callable[[int], Points | None]) -> None:
        """Attach a narrow lifecycle-owned resolver.

        The resolver should accept a layer_id and return the currently live layer,
        or None if the layer is no longer available.
        """
        self._resolve_layer_by_id = resolve_layer_by_id

    def maybe_layer(self) -> Points | None:
        """Resolve the current layer if still available, else return None."""
        if self._layer_id is None:
            return None

        # Lifecycle resolver is authoritative when present.
        if self._resolve_layer_by_id is not None:
            try:
                return self._resolve_layer_by_id(self._layer_id)
            except Exception:
                logger.debug("Layer resolver failed for layer_id=%r", self._layer_id, exc_info=True)
                return None

        # Fallback for tests / legacy contexts.
        if self._layer_ref is not None:
            return self._layer_ref()
        return self._strong_layer_ref

    def require_layer(self) -> Points:
        layer = self.maybe_layer()
        if layer is None:
            raise LayerUnavailableError(f"Layer is no longer available for KeypointStore layer_id={self._layer_id}")
        return layer

    @property
    def layer(self) -> Points:
        return self.require_layer()

    @layer.setter
    def layer(self, layer: Points):
        same_layer = self._layer_id == id(layer)

        self._layer_id = id(layer)

        try:
            self._layer_ref = weakref.ref(layer)
            self._strong_layer_ref = None
        except TypeError:
            # Fallback if a given object cannot be weak-referenced.
            self._layer_ref = None
            self._strong_layer_ref = layer

        # Avoid repeated validated metadata reads when rebinding the same live layer.
        if same_layer:
            logger.debug(
                "Skipping KeypointStore header refresh for same layer_id=%r name=%r",
                self._layer_id,
                getattr(layer, "name", layer),
            )
            return

        self._refresh_header_from_layer(layer)

    def _refresh_header_from_layer(self, layer: Points) -> None:
        res = read_points_meta(layer, migrate_legacy=True, drop_controls=True, drop_header=False)
        if isinstance(res, ValidationError) or res.header is None:
            self._header = None
            self._keypoints = []
            return

        self._header = res.header
        pairs = self._header.form_individual_bodypart_pairs()
        self._keypoints = [Keypoint(label, id_) for id_, label in pairs]

    @property
    def labels(self) -> list[str]:
        return self._header.bodyparts if self._header is not None else []

    @property
    def ids(self) -> list[str]:
        return self._header.individuals if self._header is not None else []

    @property
    def current_step(self):
        return self.viewer.dims.current_step[0]

    @property
    def n_steps(self):
        return self.viewer.dims.nsteps[0]

    @property
    def annotated_keypoints(self) -> list[Keypoint]:
        layer = self.layer
        mask = self.current_mask
        labels = layer.properties["label"][mask]
        ids = layer.properties["id"][mask]
        return [Keypoint(label, id_) for label, id_ in zip(labels, ids, strict=False)]

    @property
    def current_mask(self) -> Sequence[bool]:
        layer = self.layer
        return np.asarray(layer.data[:, 0] == self.current_step)

    @property
    def current_keypoint(self) -> Keypoint:
        props = getattr(self.layer, "current_properties", {}) or {}
        try:
            label = props.get("label", [""])[0]
        except Exception:
            label = ""
        try:
            id_ = props.get("id", [""])[0]
        except Exception:
            id_ = ""
        return Keypoint(label=label, id=id_)

    @current_keypoint.setter
    def current_keypoint(self, keypoint: Keypoint):
        layer = self.layer
        # Avoid changing the properties of a selected point
        if not len(layer.selected_data):
            current_properties = layer.current_properties
            current_properties["label"] = np.asarray([keypoint.label])
            current_properties["id"] = np.asarray([keypoint.id])
            layer.current_properties = current_properties

    def next_keypoint(self, *args):
        ind = self._keypoints.index(self.current_keypoint) + 1
        if ind <= len(self._keypoints) - 1:
            self.current_keypoint = self._keypoints[ind]

    def prev_keypoint(self, *args):
        ind = self._keypoints.index(self.current_keypoint) - 1
        if ind >= 0:
            self.current_keypoint = self._keypoints[ind]

    @property
    def current_label(self) -> str:
        return self.layer.current_properties["label"][0]

    @current_label.setter
    def current_label(self, label: str):
        layer = self.layer
        if not len(layer.selected_data):
            current_properties = layer.current_properties
            current_properties["label"] = np.asarray([label])
            layer.current_properties = current_properties

    @property
    def current_id(self) -> str:
        return self.layer.current_properties["id"][0]

    @current_id.setter
    def current_id(self, id_: str):
        layer = self.layer
        if not len(layer.selected_data):
            current_properties = layer.current_properties
            current_properties["id"] = np.asarray([id_])
            layer.current_properties = current_properties

    def _advance_step(self, event):
        ind = (self.current_step + 1) % self.n_steps
        self.viewer.dims.set_current_step(0, ind)

    def _find_first_unlabeled_frame(self, event):
        layer = self.layer
        inds = set(range(self.n_steps))
        unlabeled_inds = inds.difference(layer.data[:, 0].astype(int))
        if not unlabeled_inds:
            self.viewer.dims.set_current_step(0, self.n_steps - 1)
        else:
            self.viewer.dims.set_current_step(0, min(unlabeled_inds))

    def add(self, coord):
        coord = np.atleast_2d(coord)

        get_mode = getattr(self, "_get_label_mode", None)
        label_mode = get_mode() if callable(get_mode) else None

        if self.current_keypoint not in self.annotated_keypoints:
            layer = self.layer

            # 1) append data
            layer.data = np.append(layer.data, coord, axis=0)

            # 2) append/align properties to match number of points
            kp = self.current_keypoint
            n_new = coord.shape[0]
            n_total = len(layer.data)
            n_old = n_total - n_new

            props = layer.properties.copy()

            def _as_array(key, dtype):
                arr = props.get(key, None)
                if arr is None:
                    return np.array([], dtype=dtype)
                return np.asarray(arr, dtype=dtype)

            label_arr = _as_array("label", object)[:n_old]
            id_arr = _as_array("id", object)[:n_old]
            lik_arr = _as_array("likelihood", float)[:n_old]

            if label_arr.size < n_old:
                label_arr = np.concatenate([label_arr, np.array([kp.label] * (n_old - label_arr.size), dtype=object)])
            if id_arr.size < n_old:
                id_arr = np.concatenate([id_arr, np.array([kp.id] * (n_old - id_arr.size), dtype=object)])
            if lik_arr.size < n_old:
                lik_arr = np.concatenate([lik_arr, np.ones(n_old - lik_arr.size, dtype=float)])

            props["label"] = np.concatenate([label_arr, np.array([kp.label] * n_new, dtype=object)])
            props["id"] = np.concatenate([id_arr, np.array([kp.id] * n_new, dtype=object)])
            props["likelihood"] = np.concatenate([lik_arr, np.ones(n_new, dtype=float)])

            layer.properties = props

        elif label_mode is LabelMode.QUICK:
            layer = self.layer
            ind = self.annotated_keypoints.index(self.current_keypoint)
            data = layer.data
            data[np.flatnonzero(self.current_mask)[ind]] = coord.squeeze()
            layer.data = data

        self.layer.selected_data = set()

        if label_mode is LabelMode.LOOP:
            self.layer.events.query_next_frame()
        else:
            self.next_keypoint()


@deprecated(details="Temporary compat shim, remove once KeypointStore.add is properly integrated.")
def add(store: KeypointStore, coord):
    return store.add(coord)


def find_nearest_neighbors(xy_true, xy_pred, k=5):
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


# ----------------------------
# Colormap functions
# ----------------------------
def _rgba_array(colors) -> np.ndarray:
    """Normalize a color list/array to float RGBA shape (N, 4)."""
    arr = np.asarray(colors, dtype=float)

    if arr.size == 0:
        return np.empty((0, 4), dtype=float)

    if arr.ndim == 1:
        if arr.shape[0] == 3:
            arr = np.r_[arr, 1.0][None, :]
        elif arr.shape[0] == 4:
            arr = arr[None, :]
        else:
            raise ValueError(f"Unexpected color shape: {arr.shape!r}")
    elif arr.ndim == 2 and arr.shape[1] == 3:
        arr = np.c_[arr, np.ones(len(arr), dtype=float)]
    elif arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Unexpected colors array shape: {arr.shape!r}")

    return np.asarray(arr, dtype=float)


def _repeat_or_trim(colors: np.ndarray, n_colors: int) -> np.ndarray:
    """Return exactly n_colors rows by trimming or cycling a palette."""
    if n_colors <= 0:
        return np.empty((0, 4), dtype=float)

    colors = _rgba_array(colors)
    if len(colors) == 0:
        return np.empty((0, 4), dtype=float)

    if len(colors) >= n_colors:
        return colors[:n_colors]

    reps = int(np.ceil(n_colors / len(colors)))
    out = np.tile(colors, (reps, 1))[:n_colors]

    logger.debug(
        "Requested %d colors from a listed palette of length %d; cycling palette.",
        n_colors,
        len(colors),
    )
    return out


def _try_matplotlib_listed_colors(colormap: str | None) -> np.ndarray | None:
    """
    Return listed RGBA colors from a matplotlib colormap when available.

    This is the preferred path for qualitative palettes like Set3, tab10, tab20,
    Dark2, etc., because they should be treated as discrete palettes, not sampled
    continuously.
    """
    if not colormap:
        return None

    try:
        mpl_cmap = mpl_colormaps.get_cmap(colormap)
    except Exception:
        return None

    listed = getattr(mpl_cmap, "colors", None)
    if listed is None:
        return None

    try:
        return _rgba_array(listed)
    except Exception:
        logger.debug("Failed to normalize matplotlib listed colors for %r", colormap, exc_info=True)
        return None


def _sample_continuous_colormap(cmap, n_colors: int) -> np.ndarray:
    """
    Sample a continuous colormap at bin centers.

    Using centers instead of endpoints avoids repeated-looking adjacent colors and
    behaves better for categorical assignment.
    """
    if n_colors <= 0:
        return np.empty((0, 4), dtype=float)

    values = (np.arange(n_colors, dtype=float) + 0.5) / n_colors
    return _rgba_array(cmap.map(values))


def build_color_cycle(n_colors: int, colormap: str | None = "viridis") -> np.ndarray:
    """
    Build a robust RGBA color cycle.

    Policy
    ------
    1) If `colormap` is a listed matplotlib palette (e.g. Set3, tab10, tab20,
       Dark2...), use its listed colors directly.
    2) Otherwise, resolve via napari and sample at bin centers.
    """
    if n_colors <= 0:
        return np.empty((0, 4), dtype=float)

    # Prefer discrete/listed matplotlib palettes directly.
    listed = _try_matplotlib_listed_colors(colormap)
    if listed is not None and len(listed) > 0:
        return _repeat_or_trim(listed, n_colors)

    # Fall back to napari colormap resolution.
    cmap = colormaps.ensure_colormap(colormap)

    # If napari resolved something that itself behaves like a listed palette,
    # prefer those colors directly as well.
    try:
        cmap_colors = getattr(cmap, "colors", None)
        if cmap_colors is not None:
            cmap_colors = _rgba_array(cmap_colors)
            interp = str(getattr(cmap, "interpolation", "")).lower()
            if len(cmap_colors) > 0 and interp == "zero":
                return _repeat_or_trim(cmap_colors, n_colors)
    except Exception:
        logger.debug("Failed to inspect napari colormap %r for listed colors", colormap, exc_info=True)

    return _sample_continuous_colormap(cmap, n_colors)


def build_color_cycles(header: HeaderLike, colormap: str | None = "viridis"):
    """
    Build categorical label/id color mappings from a DLC-style header.

    Notes
    -----
    - bodyparts always preserve header order
    - individuals preserve header order, excluding blank single-animal placeholders
    """
    bodyparts = [str(x) for x in header.bodyparts]
    individuals = [str(x) for x in header.individuals if str(x) != ""]

    label_colors = build_color_cycle(len(bodyparts), colormap)
    id_colors = build_color_cycle(len(individuals), colormap)

    return {
        "label": dict(zip(bodyparts, label_colors, strict=False)),
        "id": dict(zip(individuals, id_colors, strict=False)),
    }
