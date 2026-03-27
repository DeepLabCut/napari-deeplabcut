"""Helper functions for building trails layers from points layers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from napari.layers import Points, Tracks
from napari.utils.colormaps import Colormap

from napari_deeplabcut.config.models import TrailsDisplayConfig
from napari_deeplabcut.core import keypoints
from napari_deeplabcut.core.layer_versioning import layer_change_generations
from napari_deeplabcut.core.metadata import parse_points_metadata
from napari_deeplabcut.core.sidecar import get_trails_config, set_trails_config


@dataclass(frozen=True)
class TrailsPayload:
    tracks_data: np.ndarray
    properties: dict[str, np.ndarray]
    color_by: str
    colormaps_dict: dict[str, Colormap]
    signature: tuple
    geometry_signature: tuple


def trails_signature(layer: Points, color_mode: str | keypoints.ColorMode) -> tuple:
    """
    Full signature used to decide whether trails style needs refreshing.

    Includes:
    - source layer identity
    - active color mode
    - configured colormap name
    - content/presentation change generations
    """
    generations = layer_change_generations(layer)

    return (
        id(layer),
        str(color_mode),
        (layer.metadata or {}).get("colormap_name"),
        generations.content,
        generations.presentation,
    )


def trails_geometry_signature(layer: Points) -> tuple:
    """
    Signature used to decide whether trails geometry must be rebuilt.

    Includes:
    - source layer identity
    - content change generation
    """
    generations = layer_change_generations(layer)

    return (
        id(layer),
        generations.content,
    )


def is_multianimal_points_layer(layer: Points) -> bool:
    props = getattr(layer, "properties", {}) or {}
    ids = props.get("id")
    if ids is None or len(ids) == 0:
        return False

    try:
        first = ids[0]
    except Exception:
        return False

    return isinstance(first, str) and first != ""


def active_trails_color_property(
    layer: Points,
    color_mode: str | keypoints.ColorMode,
) -> tuple[str, np.ndarray, bool]:
    """
    Return:
        color_prop: 'id' or 'label'
        categories_for_color: values to color by
        is_multi: whether the layer is multi-animal
    """
    props = getattr(layer, "properties", {}) or {}
    labels = props.get("label")
    ids = props.get("id")

    if labels is None or len(labels) == 0:
        raise ValueError("Points layer has no 'label' property; cannot build trails.")

    is_multi = is_multianimal_points_layer(layer)

    if str(color_mode) == str(keypoints.ColorMode.INDIVIDUAL) and is_multi:
        return "id", ids, True

    return "label", labels, is_multi


def _trails_rgba_array(colors: list) -> np.ndarray:
    out = []
    for c in colors:
        c = np.asarray(c, dtype=float)
        if c.ndim == 0:
            c = np.array([0.5, 0.5, 0.5, 1.0], dtype=float)
        if c.shape[0] == 3:
            c = np.r_[c, 1.0]
        out.append(c)
    return np.asarray(out, dtype=float)


def categorical_colormap_from_points_layer(
    layer: Points,
    color_prop: str,
    categories_for_color: np.ndarray,
    *,
    cycle_override: dict | None = None,
) -> tuple[Colormap, list[str], np.ndarray]:
    """
    Build a categorical zero-interpolation napari Colormap matching the points layer cycles.

    Parameters
    ----------
    layer
        Source points layer.
    color_prop
        'id' or 'label'.
    categories_for_color
        Property values used for color coding.
    cycle_override
        Optional resolved category->color mapping to use instead of raw metadata.
    """
    uniq_color = list(dict.fromkeys(map(str, categories_for_color)))
    n_color = len(uniq_color)

    if n_color == 0:
        raise ValueError("No categories found for trails coloring.")

    if cycle_override is not None:
        cycle_dict = {str(k): np.asarray(v, dtype=float) for k, v in cycle_override.items()}
    else:
        face_cycles = (layer.metadata or {}).get("face_color_cycles", {})
        cycle_dict = {str(k): np.asarray(v, dtype=float) for k, v in (face_cycles.get(color_prop, {}) or {}).items()}

    color_list = []
    for u in uniq_color:
        c = cycle_dict.get(u)
        if c is None:
            color_list = []
            break
        color_list.append(c)

    if not color_list:
        palette = plt.get_cmap("tab20").colors
        color_list = [palette[i % len(palette)] for i in range(n_color)]

    colors_rgba = _trails_rgba_array(color_list)

    color_index = {u: i for i, u in enumerate(uniq_color)}
    codes = np.array([color_index[str(u)] for u in categories_for_color], dtype=int)

    if n_color == 1:
        # zero-interpolation colormap still expects len(controls) == len(colors) + 1
        colors_rgba = np.vstack([colors_rgba[0], colors_rgba[0]])
        controls = np.array([0.0, 0.5, 1.0], dtype=float)
        codes_norm = np.full(len(categories_for_color), 0.5, dtype=float)
    else:
        controls = np.linspace(0.0, 1.0, n_color + 1, dtype=float)
        codes_norm = ((codes + 0.5) / n_color).astype(float)

    cmap = Colormap(
        colors=colors_rgba,
        controls=controls,
        name=f"{color_prop}_categorical",
        interpolation="zero",
    )

    return cmap, uniq_color, codes_norm


def trails_track_ids(layer: Points, *, is_multi: bool) -> np.ndarray:
    """
    Group trajectories by:
      - (id, label) for multi-animal
      - label for single-animal
    """
    props = getattr(layer, "properties", {}) or {}
    labels = props.get("label")
    ids = props.get("id")

    if labels is None or len(labels) == 0:
        raise ValueError("Points layer has no 'label' property; cannot build trails.")

    if is_multi:
        group_keys = [f"{i}|{l}" for i, l in zip(ids, labels, strict=False)]
    else:
        group_keys = [str(l) for l in labels]

    uniq_group = list(dict.fromkeys(group_keys))
    gid_map = {g: k for k, g in enumerate(uniq_group)}
    return np.array([gid_map[g] for g in group_keys], dtype=int)


def build_trails_payload(
    layer: Points,
    color_mode: str | keypoints.ColorMode,
    *,
    cycle_override: dict | None = None,
) -> TrailsPayload:
    color_prop, categories_for_color, is_multi = active_trails_color_property(layer, color_mode)
    cmap, _, codes_norm = categorical_colormap_from_points_layer(
        layer,
        color_prop,
        categories_for_color,
        cycle_override=cycle_override,
    )
    track_ids = trails_track_ids(layer, is_multi=is_multi)

    tracks_data = np.c_[track_ids, layer.data]
    color_key = f"{color_prop}_codes"

    return TrailsPayload(
        tracks_data=tracks_data,
        properties={color_key: codes_norm},
        color_by=color_key,
        colormaps_dict={color_key: cmap},
        signature=trails_signature(layer, color_mode),
        geometry_signature=trails_geometry_signature(layer),
    )


def tracks_kwargs_from_display_config(cfg: TrailsDisplayConfig) -> dict:
    """
    Convert persistent trails display config into kwargs suitable for viewer.add_tracks().

    Notes
    -----
    We intentionally do NOT include `visible` here, because visibility is UI state:
    when the user explicitly checks "Show trails", the created layer should be shown.
    """
    return {
        "tail_length": int(cfg.tail_length),
        "head_length": int(cfg.head_length),
        "tail_width": float(cfg.tail_width),
        "opacity": float(cfg.opacity),
        "blending": str(cfg.blending),
    }


def display_config_from_tracks_layer(layer: Tracks, *, visible: bool | None = None) -> TrailsDisplayConfig:
    """
    Extract persistent trails display config from an existing Tracks layer.

    Parameters
    ----------
    layer
        Tracks layer to snapshot.
    visible
        Optional visibility override. Useful when persisting a deliberate hide/show
        action without mutating the layer first.
    """
    return TrailsDisplayConfig(
        tail_length=int(getattr(layer, "tail_length", 50)),
        head_length=int(getattr(layer, "head_length", 50)),
        tail_width=float(getattr(layer, "tail_width", 6.0)),
        opacity=float(getattr(layer, "opacity", 1.0)),
        blending=str(getattr(layer, "blending", "translucent")),
        visible=bool(layer.visible if visible is None else visible),
    )


def safe_folder_anchor_from_points_layer(layer: Points) -> str | None:
    """Best-effort anchor folder for folder-scoped sidecar state."""
    md = layer.metadata or {}

    try:
        pts = parse_points_metadata(md)
        if pts.io and pts.io.project_root:
            return str(pts.io.project_root)
    except Exception:
        pass

    root = md.get("root")
    if isinstance(root, str) and root:
        return root

    src = md.get("source_h5")
    if isinstance(src, str) and src:
        try:
            return str(Path(src).expanduser().resolve().parent)
        except Exception:
            return str(Path(src).parent)

    return None


# ------------------------------------------
# Trails layer controller
# ------------------------------------------
class TrailsController:
    """
    Owns trails layer lifecycle, signature tracking, and folder-scoped persistence.

    The widget remains responsible only for wiring user/viewer events into this
    controller.
    """

    def __init__(
        self,
        viewer,
        *,
        managed_points_layers_getter: Callable[[], Iterable[Points]],
        color_mode_getter: Callable[[], str],
        resolved_cycle_getter: Callable[[Points], dict],
    ) -> None:
        self._viewer = viewer
        self._managed_points_layers_getter = managed_points_layers_getter
        self._color_mode_getter = color_mode_getter
        self._resolved_cycle_getter = resolved_cycle_getter

        self._trails: Tracks | None = None
        self._trails_geom_sig: tuple | None = None
        self._trails_style_sig: tuple | None = None
        self._refreshing_trails = False

    @property
    def layer(self) -> Tracks | None:
        return self._trails

    def has_live_trails(self) -> bool:
        return self._trails is not None

    def current_source_layer(self) -> Points | None:
        active = self._viewer.layers.selection.active
        managed = list(self._managed_points_layers_getter())

        if isinstance(active, Points) and active in managed:
            return active

        if self._trails is not None:
            src_id = (self._trails.metadata or {}).get("_source_points_layer_id")
            if src_id is not None:
                for layer in managed:
                    if id(layer) == src_id:
                        return layer

        return managed[0] if managed else None

    def current_anchor(self, layer: Points | None = None) -> str | None:
        pts_layer = layer if layer is not None else self.current_source_layer()
        if pts_layer is None:
            return None
        return safe_folder_anchor_from_points_layer(pts_layer)

    def persist_current_config(self, *, visible: bool | None = None) -> None:
        """
        Persist the current live trails display config to the folder sidecar.
        """
        if self._trails is None:
            return

        anchor = (self._trails.metadata or {}).get("_dlc_trails_anchor")
        if not anchor:
            anchor = self.current_anchor()

        if not anchor:
            return

        try:
            cfg = display_config_from_tracks_layer(self._trails, visible=visible)
            set_trails_config(anchor, cfg)
        except Exception:
            # best-effort only
            pass

    def persist_folder_ui_state_for_points_layer(
        self,
        layer: Points,
        *,
        checkbox_checked: bool,
    ) -> None:
        """
        Best-effort persistence of folder-scoped UI state for a specific points layer.

        Policy:
        - If live trails belong to this layer/folder, persist their full display config.
        - Otherwise preserve stored config and mark visible=False.
        """
        anchor = safe_folder_anchor_from_points_layer(layer)
        if not anchor:
            return

        try:
            if self._trails is not None:
                trails_md = self._trails.metadata or {}
                trails_anchor = trails_md.get("_dlc_trails_anchor")
                trails_src_id = trails_md.get("_source_points_layer_id")

                same_source = trails_src_id == id(layer)
                same_anchor = trails_anchor is not None and str(trails_anchor) == str(anchor)

                if same_source or same_anchor:
                    cfg = display_config_from_tracks_layer(
                        self._trails,
                        visible=bool(checkbox_checked and self._trails.visible),
                    )
                    set_trails_config(anchor, cfg)
                    return

            cfg = get_trails_config(anchor)
            if cfg is None:
                return

            cfg = cfg.model_copy(update={"visible": False})
            set_trails_config(anchor, cfg)
        except Exception:
            pass

    def toggle(self, checked: bool) -> None:
        """
        Main entry point from the widget checkbox.
        """
        if not checked:
            self._hide_and_persist()
            return

        if not list(self._managed_points_layers_getter()):
            return

        pts_layer = self.current_source_layer()
        if pts_layer is None:
            return

        geom_sig = trails_geometry_signature(pts_layer)
        style_sig = trails_signature(pts_layer, self._color_mode_getter())

        if self._trails is None:
            self.create()
            return

        if self._trails_geom_sig != geom_sig:
            self.refresh()
            return

        if self._trails_style_sig != style_sig:
            self.update_style()
            return

        self._trails.visible = True

    def _hide_and_persist(self) -> None:
        if self._trails is None:
            return

        self._trails.visible = False

        try:
            pts_layer = self.current_source_layer()
            if pts_layer is None:
                return

            anchor = self.current_anchor(pts_layer)
            if not anchor:
                return

            cfg = get_trails_config(anchor)
            if cfg is None and self._trails is not None:
                cfg = display_config_from_tracks_layer(self._trails)

            if cfg is not None:
                cfg = cfg.model_copy(update={"visible": False})
                set_trails_config(anchor, cfg)
        except Exception:
            pass

    def refresh(self) -> None:
        selected, active = self._snapshot_selection()
        try:
            self.remove()
            self.create()
        finally:
            self._restore_selection(selected, active)

    def remove(self) -> None:
        if self._trails is None:
            return

        self._refreshing_trails = True
        try:
            self._viewer.layers.remove(self._trails)
        except Exception:
            pass
        finally:
            self._trails = None
            self._trails_geom_sig = None
            self._trails_style_sig = None
            self._refreshing_trails = False

    def create(self) -> None:
        if self._trails is not None:
            return

        pts_layer = self.current_source_layer()
        if pts_layer is None:
            return

        anchor = self.current_anchor(pts_layer)
        cfg = get_trails_config(anchor) if anchor else None
        track_kwargs = tracks_kwargs_from_display_config(cfg) if cfg is not None else {}

        selected, active = self._snapshot_selection()
        try:
            payload = build_trails_payload(
                pts_layer,
                self._color_mode_getter(),
                cycle_override=self._resolved_cycle_getter(pts_layer),
            )
        except ValueError:
            return
        finally:
            # don't restore yet; add_tracks should happen inside preserved selection scope
            pass

        try:
            self._trails = self._viewer.add_tracks(
                payload.tracks_data,
                properties=payload.properties,
                color_by=payload.color_by,
                colormaps_dict=payload.colormaps_dict,
                name="trails",
                metadata={
                    "_source_points_layer_id": id(pts_layer),
                    "_dlc_trails_anchor": anchor,
                },
                **track_kwargs,
            )
            self._trails.visible = True
            self._trails_geom_sig = payload.geometry_signature
            self._trails_style_sig = payload.signature

            self.persist_current_config(visible=True)
        finally:
            self._restore_selection(selected, active)

    def update_style(self) -> None:
        """
        Recolor existing trails layer in place when geometry is unchanged.
        Falls back to a full rebuild if in-place update is not supported.
        """
        if self._trails is None:
            return

        pts_layer = self.current_source_layer()
        if pts_layer is None:
            return

        try:
            payload = build_trails_payload(
                pts_layer,
                self._color_mode_getter(),
                cycle_override=self._resolved_cycle_getter(pts_layer),
            )
        except ValueError:
            return

        try:
            new_props = dict(getattr(self._trails, "properties", {}) or {})
            new_props.update(payload.properties)
            self._trails.properties = new_props

            new_cmaps = dict(getattr(self._trails, "colormaps_dict", {}) or {})
            new_cmaps.update(payload.colormaps_dict)
            self._trails.colormaps_dict = new_cmaps

            self._trails.color_by = payload.color_by
            self._trails.visible = True
            self._trails_style_sig = payload.signature

            self._trails.metadata = dict(self._trails.metadata or {})
            self._trails.metadata["_source_points_layer_id"] = id(pts_layer)
            self._trails.metadata["_dlc_trails_anchor"] = self.current_anchor(pts_layer)
        except Exception:
            self.refresh()

    def on_points_visual_inputs_changed(self, *, checkbox_checked: bool) -> None:
        """
        Call this when the source layer coloring inputs changed (color mode, colormap,
        face-color cycles, etc.).
        """
        if checkbox_checked and self._trails is not None:
            self.update_style()

    def on_points_layer_added_or_rewired(self, *, checkbox_checked: bool) -> None:
        """
        Call when a managed points layer becomes available or changes ownership.
        """
        if checkbox_checked and self._trails is not None:
            self.refresh()

    def on_points_layer_removed(self, layer: Points) -> None:
        if self._trails is None:
            return

        src_id = (self._trails.metadata or {}).get("_source_points_layer_id")
        if src_id == id(layer):
            self.remove()

    def on_tracks_layer_removed(self, layer: Tracks) -> bool:
        """
        Returns True if the removed tracks layer was our trails layer.
        """
        if self._trails is None:
            return False

        if getattr(self, "_refreshing_trails", False):
            if layer is self._trails:
                self._trails = None
                self._trails_geom_sig = None
                self._trails_style_sig = None
                return True
            return False

        if layer is self._trails:
            self._trails = None
            self._trails_geom_sig = None
            self._trails_style_sig = None
            return True

        return False

    def _snapshot_selection(self):
        selected = [layer for layer in self._viewer.layers.selection if layer in self._viewer.layers]
        active = self._viewer.layers.selection.active
        return selected, active

    def _restore_selection(self, selected, active):
        try:
            self._viewer.layers.selection.clear()
            for layer in selected:
                if layer in self._viewer.layers:
                    self._viewer.layers.selection.add(layer)
            if active in self._viewer.layers:
                self._viewer.layers.selection.active = active
        except Exception:
            pass
