#  src/napari_deeplabcut/core/layer_lifecycle/display_settings.py
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

from napari.layers import Points

from ..layers import get_uniform_point_size

logger = logging.getLogger(__name__)


class PointsDisplaySource(str, Enum):
    TRACKING_RESULT = "tracking_result"
    MACHINE_LABELS = "machine_labels"


@dataclass(frozen=True, slots=True)
class PointsDisplaySettings:
    """Declarative display settings for Points layers.

    All fields are optional so callers can apply only the presentation concerns
    relevant for a given layer role.
    """

    symbol: str | None = None
    opacity: float | None = None
    border_width: float | None = None
    border_color: Any | None = None

    # If True, copy face_color and face_color_mode from the source layer.
    copy_face_colors_from_source: bool = False

    # If True, copy/derive size from the source layer.
    copy_size_from_source: bool = False


TRACKING_RESULT_POINTS_DISPLAY = PointsDisplaySettings(
    symbol="cross",
    opacity=0.85,
    border_width=0.15,
    border_color="green",
    copy_face_colors_from_source=True,
    copy_size_from_source=True,
)

MACHINE_LABELS_POINTS_DISPLAY = PointsDisplaySettings(
    symbol="x",
    opacity=0.85,
)


POINTS_DISPLAY_SETTINGS_BY_ROLE: dict[PointsDisplaySource, PointsDisplaySettings] = {
    PointsDisplaySource.TRACKING_RESULT: TRACKING_RESULT_POINTS_DISPLAY,
    PointsDisplaySource.MACHINE_LABELS: MACHINE_LABELS_POINTS_DISPLAY,
}


def apply_points_display_role(
    layer: Points,
    role: PointsDisplaySource,
    *,
    source: Points | None = None,
) -> Points:
    settings = POINTS_DISPLAY_SETTINGS_BY_ROLE.get(role)
    if settings is None:
        return layer

    return apply_points_display_settings(layer, settings, source=source)


def apply_points_display_settings(
    layer: Points,
    settings: PointsDisplaySettings,
    *,
    source: Points | None = None,
) -> Points:
    """Apply display settings to a Points layer.

    This helper is deliberately forgiving: display tweaks should not make layer
    creation fail. If napari changes a presentation attribute or rejects a value,
    the failure is logged at debug level and the remaining settings are applied.
    """

    if settings.symbol is not None:
        _safe_set(layer, "symbol", settings.symbol)

    if settings.opacity is not None:
        _safe_set(layer, "opacity", settings.opacity)

    if settings.copy_size_from_source and source is not None:
        _copy_point_size(layer, source)

    if settings.copy_face_colors_from_source and source is not None:
        _copy_face_colors(layer, source)

    if settings.border_width is not None:
        _safe_set(layer, "border_width", settings.border_width)

    if settings.border_color is not None:
        _safe_set(layer, "border_color", settings.border_color)

    return layer


def apply_tracking_result_display_settings(
    layer: Points,
    *,
    source: Points | None = None,
) -> Points:
    """Apply the standard visual style for tracking-result Points layers."""

    return apply_points_display_settings(
        layer,
        TRACKING_RESULT_POINTS_DISPLAY,
        source=source,
    )


def _safe_set(obj: Any, attr: str, value: Any) -> None:
    try:
        setattr(obj, attr, value)
    except Exception:
        logger.debug(
            "Failed to set display attribute %s=%r on layer %r",
            attr,
            value,
            getattr(obj, "name", obj),
            exc_info=True,
        )


def _copy_point_size(layer: Points, source: Points) -> None:
    try:
        layer.size = get_uniform_point_size(source)
        return
    except Exception:
        logger.debug(
            "Failed to derive uniform point size from source layer %r",
            getattr(source, "name", source),
            exc_info=True,
        )

    try:
        layer.size = deepcopy(source.size)
    except Exception:
        logger.debug(
            "Failed to copy point size from source layer %r",
            getattr(source, "name", source),
            exc_info=True,
        )


def _copy_face_colors(layer: Points, source: Points) -> None:
    try:
        layer.face_color = deepcopy(source.face_color)
        layer.face_color_mode = source.face_color_mode
    except Exception:
        logger.debug(
            "Failed to copy face colors from source layer %r",
            getattr(source, "name", source),
            exc_info=True,
        )
