# src/napari_deeplabcut/ui/color_scheme_display.py
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from napari.layers import Points
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import napari_deeplabcut.core.io as io
from napari_deeplabcut import keypoints, misc
from napari_deeplabcut.config.models import DLCHeaderModel
from napari_deeplabcut.config.settings import DEFAULT_MULTI_ANIMAL_INDIVIDUAL_CMAP, DEFAULT_SINGLE_ANIMAL_CMAP
from napari_deeplabcut.keypoints import build_color_cycles
from napari_deeplabcut.ui.labels_and_dropdown import LabelPair

logger = logging.getLogger(__name__)


class ColorSchemeDisplay(QScrollArea):
    """Scrollable list of labels/ids and their associated colors."""

    added = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.scheme_dict: dict[str, str] = {}
        self._layout = QVBoxLayout()
        self._layout.setSpacing(0)

        # Container required by QScrollArea.setWidget
        self._container = QWidget(parent=self)
        self._build()

    @property
    def labels(self) -> list[QWidget]:
        out = []
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                out.append(w)
        return out

    def _build(self) -> None:
        self._container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        self._container.setLayout(self._layout)
        self._container.adjustSize()

        self.setWidget(self._container)
        self.setWidgetResizable(True)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.setBaseSize(100, 200)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def add_entry(self, name: str, color: str) -> None:
        self.scheme_dict[name] = color
        widget = LabelPair(color, name, self)
        self._layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignLeft)
        self.added.emit(widget)

    def update_color_scheme(self, new_color_scheme: dict[str, str]) -> None:
        logger.debug("Updating color scheme display with %d entries", len(new_color_scheme))
        self.scheme_dict = dict(new_color_scheme)
        names = list(new_color_scheme.keys())
        existing_widgets = self._layout.count()
        required_widgets = len(names)

        # Update existing widgets
        for idx in range(min(existing_widgets, required_widgets)):
            item = self._layout.itemAt(idx)
            if item is None:
                continue
            w = item.widget()
            if w is None:
                continue
            w.setVisible(True)
            w.part_name = names[idx]
            w.color = self.scheme_dict[names[idx]]

        # Hide extra widgets
        for idx in range(required_widgets, existing_widgets):
            item = self._layout.itemAt(idx)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.setVisible(False)

        # Add missing widgets
        for idx in range(existing_widgets, required_widgets):
            name = names[idx]
            self.add_entry(name, self.scheme_dict[name])

        self._container.adjustSize()
        self._container.updateGeometry()

    def reset(self) -> None:
        self.scheme_dict = {}
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.setVisible(False)
        self._container.adjustSize()
        self._container.updateGeometry()


def _to_hex(rgba: Any) -> str:
    """Convert RGB(A) float colors in [0, 1] to #rrggbb."""
    arr = np.asarray(rgba, dtype=float).ravel()
    if arr.size < 3:
        return "#000000"

    if arr.size == 3:
        arr = np.r_[arr, 1.0]

    arr = np.clip(arr[:4], 0, 1)
    r, g, b, _a = (arr * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


class ColorSchemeResolver:
    """
    Resolve which categories/colors should be shown in the color scheme display.

    Behavior:
    - ACTIVE mode:
        show only categories currently visible in the current frame/slice
    - CONFIG mode:
        show all configured categories from config.yaml when available;
        otherwise fall back to embedded header, then cycle keys
    """

    def __init__(
        self,
        viewer,
        get_color_mode: Callable[[], str],
        get_header_model: Callable[[dict], object | None],
    ):
        self.viewer = viewer
        self._get_color_mode = get_color_mode
        self._get_header_model = get_header_model
        self._config_header_cache: dict[str, DLCHeaderModel | None] = {}

    def is_multianimal(self, layer: Points) -> bool:
        md = layer.metadata or {}
        header = self._get_header_model(md)
        if header is None:
            return False
        try:
            inds = getattr(header, "individuals", None)
            return bool(inds and len(inds) > 0 and str(inds[0]) != "")
        except Exception:
            return False

    def get_config_colormap(self, layer: Points) -> str:
        md = layer.metadata or {}
        cmap = md.get("config_colormap")
        if isinstance(cmap, str) and cmap:
            return cmap
        return DEFAULT_SINGLE_ANIMAL_CMAP

    def get_active_color_property(self, layer: Points) -> str:
        if self.is_multianimal(layer) and self._get_color_mode() == str(keypoints.ColorMode.INDIVIDUAL):
            return "id"
        return "label"

    def get_face_color_cycles(self, layer: Points) -> dict[str, dict]:
        md = layer.metadata or {}
        header = self._get_header_model(md)
        if header is None:
            return {}

        config_cmap = self.get_config_colormap(layer)

        bodypart_cycles = build_color_cycles(header, config_cmap) or {}
        if self.is_multianimal(layer):
            individual_cycles = build_color_cycles(header, DEFAULT_MULTI_ANIMAL_INDIVIDUAL_CMAP) or {}
        else:
            individual_cycles = bodypart_cycles

        return {
            "label": bodypart_cycles.get("label", {}),
            "id": individual_cycles.get("id", {}),
        }

    # ------------------------------------------------------------------
    # Target layer / color property
    # ------------------------------------------------------------------
    def get_target_layer(self) -> Points | None:
        """Prefer active visible Points layer, else fallback to topmost visible Points layer."""
        active = self.viewer.layers.selection.active
        if isinstance(active, Points) and getattr(active, "visible", True):
            return active

        for layer in reversed(list(self.viewer.layers)):
            if isinstance(layer, Points) and getattr(layer, "visible", True):
                return layer

        return None

    def get_color_property(self, layer: Points) -> str | None:
        prop = self.get_active_color_property(layer)
        cycles = self.get_face_color_cycles(layer)
        if prop in cycles and cycles.get(prop):
            return prop
        if "label" in cycles and cycles.get("label"):
            return "label"
        if "id" in cycles and cycles.get("id"):
            return "id"
        return None

    # ------------------------------------------------------------------
    # Active/on-screen category extraction
    # ------------------------------------------------------------------
    def get_visible_categories(self, layer: Points, prop: str) -> list[str]:
        """
        Return categories currently visible in the current frame/slice.

        Conservative/safe definition of "visible":
        - matches current frame index on axis 0 when applicable
        - intersects with layer.shown if available
        """
        props = getattr(layer, "properties", {}) or {}
        values = props.get(prop, None)
        if values is None or len(values) == 0:
            return []

        values = np.asarray(values, dtype=object).ravel()
        mask = np.ones(len(values), dtype=bool)

        # Filter by current frame/slice when data has a leading frame axis
        try:
            data = np.asarray(layer.data)
            if len(data) == len(values) and data.ndim == 2 and data.shape[1] > 0:
                current_step = self.viewer.dims.current_step
                if len(current_step) > 0:
                    frame = current_step[0]
                    mask &= np.asarray(data[:, 0] == frame)
        except Exception:
            logger.debug("Failed to filter visible categories by current frame", exc_info=True)

        # Optional shown mask
        try:
            shown = getattr(layer, "shown", None)
            if shown is not None and len(shown) == len(values):
                mask &= np.asarray(shown, dtype=bool)
        except Exception:
            logger.debug("Failed to apply layer.shown mask to visible categories", exc_info=True)

        out: list[str] = []
        seen: set[str] = set()
        for v in values[mask]:
            if v in ("", None) or misc._is_nan_value(v):
                continue
            s = str(v)
            if s not in seen:
                seen.add(s)
                out.append(s)

        return out

    # ------------------------------------------------------------------
    # Config/header resolution
    # ------------------------------------------------------------------
    def _resolve_project_config_path(self, layer: Points) -> Path | None:
        """
        Best-effort resolution of the DeepLabCut config.yaml for a layer.

        Resolution order:
        1) metadata['project'] / config.yaml
        2) misc.find_project_config_path(project/root/source_h5)
        """
        md = layer.metadata or {}

        # 1) direct project metadata
        project = md.get("project")
        if isinstance(project, str) and project:
            cfg = Path(project) / "config.yaml"
            if cfg.is_file():
                return cfg

        # 2) best-effort search from available anchors
        anchors: list[str] = []
        for key in ("project", "root", "source_h5"):
            val = md.get(key)
            if isinstance(val, str) and val:
                anchors.append(val)

        for anchor in anchors:
            try:
                cfg_path = misc.find_project_config_path(anchor)
                if cfg_path:
                    p = Path(cfg_path)
                    if p.is_file():
                        return p
            except Exception:
                logger.debug("Could not resolve config from anchor=%r", anchor, exc_info=True)

        return None

    def _load_config_header(self, config_path: Path) -> DLCHeaderModel | None:
        """Load DLCHeaderModel from config.yaml with caching."""
        cache_key = str(config_path)
        if cache_key in self._config_header_cache:
            return self._config_header_cache[cache_key]

        hdr: DLCHeaderModel | None = None
        try:
            cfg = io.load_config(str(config_path))
            hdr = DLCHeaderModel.from_config(cfg)
        except Exception:
            logger.debug("Failed to load DLC header from config %s", config_path, exc_info=True)
            hdr = None

        self._config_header_cache[cache_key] = hdr
        return hdr

    def _get_config_header_for_layer(self, layer: Points) -> DLCHeaderModel | None:
        cfg_path = self._resolve_project_config_path(layer)
        if cfg_path is None:
            return None
        return self._load_config_header(cfg_path)

    def _categories_from_header(self, header: object | None, prop: str) -> list[str]:
        """Extract ordered categories from a DLCHeaderModel-like object."""
        if header is None:
            return []

        try:
            if prop == "label":
                return [str(x) for x in getattr(header, "bodyparts", [])]
            if prop == "id":
                individuals = [str(x) for x in getattr(header, "individuals", [])]
                # Single-animal headers often encode individuals as ['']
                individuals = [x for x in individuals if x != ""]
                return individuals
        except Exception:
            logger.debug("Failed to extract categories from header", exc_info=True)

        return []

    def get_config_categories(self, layer: Points, prop: str) -> list[str]:
        """
        Return all configured categories.

        Resolution order:
        1) config.yaml for the target layer's project
        2) embedded layer header
        3) cycle keys

        Special fallback:
        - if prop == 'id' and config/header individuals are absent/empty,
          fall back to bodyparts, then cycle keys
        """
        md = layer.metadata or {}
        cycles = md.get("face_color_cycles") or {}

        # 1) config.yaml
        cfg_header = self._get_config_header_for_layer(layer)
        names = self._categories_from_header(cfg_header, prop)
        if names:
            return names

        # If individual mode requested but config has no individuals, fall back to bodyparts
        if prop == "id":
            names = self._categories_from_header(cfg_header, "label")
            if names:
                return names

        # 2) embedded header
        embedded_header = self._get_header_model(md)
        names = self._categories_from_header(embedded_header, prop)
        if names:
            return names

        if prop == "id":
            names = self._categories_from_header(embedded_header, "label")
            if names:
                return names

        # 3) cycle keys
        mapping = cycles.get(prop) or {}
        if mapping:
            return list(mapping.keys())

        if prop == "id":
            mapping = cycles.get("label") or {}
            if mapping:
                return list(mapping.keys())

        return []

    # ------------------------------------------------------------------
    # Final resolution
    # ------------------------------------------------------------------
    def resolve(self, *, show_config_keypoints: bool) -> dict[str, str]:
        """
        Build the legend mapping {category_name: '#rrggbb'} for display.
        """
        layer = self.get_target_layer()
        if layer is None:
            return {}

        cycles = self.get_face_color_cycles(layer)
        if not cycles:
            return {}

        prop = self.get_active_color_property(layer)
        if prop is None:
            return {}

        mapping = cycles.get(prop) or {}
        if not mapping:
            # If in individual mode and id mapping is unavailable, fall back to label mapping
            if prop == "id":
                prop = "label"
                mapping = cycles.get(prop) or {}
            if not mapping:
                return {}

        if show_config_keypoints:
            names = self.get_config_categories(layer, prop)
        else:
            names = self.get_visible_categories(layer, prop)

        # Keep only entries that actually exist in the current mapping
        scheme: dict[str, str] = {}
        for name in names:
            color = mapping.get(name)
            if color is None:
                continue
            scheme[str(name)] = _to_hex(color)

        return scheme


class ColorSchemePanel(QWidget):
    """
    Dockable panel containing:
    - a checkbox to switch between active vs config preview
    - the scrollable ColorSchemeDisplay
    """

    def __init__(
        self,
        viewer,
        get_color_mode: Callable[[], str],
        get_header_model: Callable[[dict], object | None],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.viewer = viewer
        self._disposed = False
        self._wired_layers: set[int] = set()
        self._connections: list[tuple[object, object]] = []

        self._resolver = ColorSchemeResolver(
            viewer=viewer,
            get_color_mode=get_color_mode,
            get_header_model=get_header_model,
        )

        self._toggle = QCheckBox("Show all keypoints from config.yaml", self)
        self._toggle.setToolTip(
            "If checked, show all configured categories from the target layer's "
            "DeepLabCut config.yaml when available.\n"
            "If unchecked, show only the categories currently visible in the active layer/frame.\n\n"
            "In bodypart mode: show configured bodyparts.\n"
            "In individual mode: show configured individuals, falling back safely when unavailable."
        )

        self.display = ColorSchemeDisplay(parent=self)
        self.display.added.connect(self._wire_display_entry)

        layout = QVBoxLayout(self)
        layout.addWidget(self._toggle)
        layout.addWidget(self.display)

        # Parent-owned debounce timer: dies with the panel
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._on_update_timeout)

        self._toggle.toggled.connect(self.schedule_update)
        self.destroyed.connect(self._on_destroyed)

        self._connect_viewer_events()
        self.schedule_update()

    @property
    def show_config_keypoints(self) -> bool:
        if self._disposed:
            return False
        try:
            return bool(self._toggle.isChecked())
        except RuntimeError:
            # Underlying C++ widget already deleted
            return False

    def _connect(self, emitter, callback) -> None:
        """Connect and remember emitter/callback so we can disconnect later."""
        try:
            emitter.connect(callback)
            self._connections.append((emitter, callback))
        except Exception:
            logger.debug("Failed to connect emitter=%r callback=%r", emitter, callback, exc_info=True)

    def _disconnect_all(self) -> None:
        """Best-effort disconnect of all remembered external emitters."""
        while self._connections:
            emitter, callback = self._connections.pop()
            try:
                emitter.disconnect(callback)
            except Exception:
                # Already disconnected / emitter gone / teardown race
                pass

    def _on_destroyed(self, *args) -> None:
        """Mark disposed and disconnect from external emitters."""
        self._disposed = True
        try:
            self._update_timer.stop()
        except Exception:
            pass
        self._disconnect_all()
        self._wired_layers.clear()

    def closeEvent(self, event):  # type: ignore[override]
        self._disposed = True
        try:
            self._update_timer.stop()
        except Exception:
            pass
        self._disconnect_all()
        super().closeEvent(event)

    def _connect_viewer_events(self) -> None:
        self._connect(self.viewer.layers.selection.events.active, self.schedule_update)
        self._connect(self.viewer.layers.events.inserted, self._on_layers_changed)
        self._connect(self.viewer.layers.events.removed, self._on_layers_removed)
        self._connect(self.viewer.dims.events.current_step, self.schedule_update)

        for layer in list(self.viewer.layers):
            self._maybe_wire_layer(layer)

    def _on_layers_changed(self, event=None) -> None:
        if self._disposed:
            return

        layer = None

        try:
            layer = event.value if event is not None else None
        except Exception:
            layer = None

        if layer is None:
            try:
                layer = event.source[-1]
            except Exception:
                layer = None

        self._maybe_wire_layer(layer)
        self.schedule_update()

    def _on_layers_removed(self, event=None) -> None:
        if self._disposed:
            return
        # We don't try to disconnect per-layer emitters defensively here; just refresh.
        self.schedule_update()

    def _maybe_wire_layer(self, layer) -> None:
        if self._disposed or not isinstance(layer, Points):
            return

        lid = id(layer)
        if lid in self._wired_layers:
            return

        self._wired_layers.add(lid)

        for event_name in ("visible", "data", "properties", "shown", "current_properties"):
            try:
                emitter = getattr(layer.events, event_name)
            except AttributeError:
                # Optional event depending on napari version; not a problem.
                logger.debug(
                    "Skipping unavailable Points event '%s' for layer=%r",
                    event_name,
                    getattr(layer, "name", layer),
                )
                continue
            except Exception:
                logger.debug(
                    "Could not access Points event '%s' for layer=%r",
                    event_name,
                    getattr(layer, "name", layer),
                    exc_info=True,
                )
                continue

            self._connect(emitter, self.schedule_update)

    def schedule_update(self, event=None) -> None:
        """Debounced update to avoid refresh storms."""
        if self._disposed:
            return

        try:
            if self._update_timer.isActive():
                return
            self._update_timer.start(0)
        except RuntimeError:
            # Timer/widget already gone
            return

    def _on_update_timeout(self) -> None:
        if self._disposed:
            return
        self.update_scheme()

    def update_scheme(self) -> None:
        if self._disposed:
            return

        try:
            scheme = self._resolver.resolve(show_config_keypoints=self.show_config_keypoints)
        except RuntimeError:
            # Underlying Qt objects may already be gone during teardown races
            return
        except Exception:
            logger.debug("Failed to resolve color scheme", exc_info=True)
            return

        try:
            self.display.reset()
            if scheme:
                self.display.update_color_scheme(scheme)
        except RuntimeError:
            # display already deleted
            return

    def _wire_display_entry(self, widget) -> None:
        """Connect a legend entry click to selecting matching points in the layer."""
        try:
            widget.part_label.clicked.connect(lambda _checked=False, w=widget: self._on_display_entry_clicked(w))
        except Exception:
            logger.debug("Failed to wire ColorSchemeDisplay entry click", exc_info=True)

    def _on_display_entry_clicked(self, widget) -> None:
        """Select points in the target layer that match the clicked legend category."""
        if self._disposed:
            return

        category = getattr(widget, "part_name", None)
        if not category:
            return
        category = str(category)

        layer = self._resolver.get_target_layer()
        if layer is None:
            return

        prop = self._resolver.get_color_property(layer)
        if prop is None:
            return

        # Prefer matches in the current frame/slice first.
        indices = self._matching_point_indices(
            layer=layer,
            prop=prop,
            category=category,
            current_frame_only=True,
        )

        # If config view is showing categories not present in the current frame,
        # fall back to the first frame containing that category and jump there.
        # We could disable this if it's not intuitive, but this preserves interactivity.
        if not indices:
            indices = self._matching_point_indices(
                layer=layer,
                prop=prop,
                category=category,
                current_frame_only=False,
            )

            # Optional safe fallback: if "id" legend was derived from label-like data,
            # allow matching on label as a last resort.
            if not indices and prop == "id":
                indices = self._matching_point_indices(
                    layer=layer,
                    prop="label",
                    category=category,
                    current_frame_only=False,
                )

            if not indices:
                return

            # Jump to the frame of the first found point when possible.
            try:
                data = np.asarray(layer.data)
                if data.ndim == 2 and data.shape[1] > 0:
                    frame = int(data[indices[0], 0])
                    try:
                        self.viewer.dims.set_current_step(0, frame)
                    except AttributeError:
                        # Older napari fallback
                        self.viewer.dims.set_point(0, frame)
            except Exception:
                logger.debug("Failed to move to frame for clicked legend entry", exc_info=True)

            # Re-resolve in the new frame so selection reflects what is now visible.
            refreshed = self._matching_point_indices(
                layer=layer,
                prop=prop,
                category=category,
                current_frame_only=True,
            )
            if refreshed:
                indices = refreshed

        try:
            self.viewer.layers.selection.active = layer
        except Exception:
            logger.debug("Failed to activate target layer from legend click", exc_info=True)

        try:
            layer.mode = "select"
        except Exception:
            logger.debug("Failed to switch Points layer to select mode", exc_info=True)

        try:
            layer.selected_data = set(map(int, indices))
        except Exception:
            logger.debug("Failed to update selected_data from legend click", exc_info=True)

    def _matching_point_indices(
        self,
        *,
        layer: Points,
        prop: str,
        category: str,
        current_frame_only: bool,
    ) -> list[int]:
        """
        Return indices of points whose categorical property matches ``category``.

        If ``current_frame_only`` is True, restrict matches to the current frame/slice
        on axis 0 when possible, and prefer points that are currently shown.
        """
        props = getattr(layer, "properties", {}) or {}
        values = props.get(prop, None)
        if values is None or len(values) == 0:
            return []

        values = np.asarray(values, dtype=object).ravel()
        if len(values) == 0:
            return []

        matches = np.zeros(len(values), dtype=bool)
        for i, v in enumerate(values):
            if v in ("", None) or misc._is_nan_value(v):
                continue
            matches[i] = str(v) == category

        if not np.any(matches):
            return []

        # Restrict to current frame/slice when requested and when data supports it.
        if current_frame_only:
            try:
                data = np.asarray(layer.data)
                if len(data) == len(values) and data.ndim == 2 and data.shape[1] > 0:
                    current_step = self.viewer.dims.current_step
                    if len(current_step) > 0:
                        frame = current_step[0]
                        matches &= np.asarray(data[:, 0] == frame)
            except Exception:
                logger.debug("Failed to filter legend click selection by current frame", exc_info=True)

            if not np.any(matches):
                return []

        # Prefer currently shown points, but only if that leaves at least one result.
        try:
            shown = getattr(layer, "shown", None)
            if shown is not None and len(shown) == len(values):
                shown = np.asarray(shown, dtype=bool)
                shown_matches = matches & shown
                if np.any(shown_matches):
                    matches = shown_matches
        except Exception:
            logger.debug("Failed to apply layer.shown mask for legend click selection", exc_info=True)

        return np.flatnonzero(matches).astype(int).tolist()
