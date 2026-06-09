# src/napari_deeplabcut/_tests/core/layer_manager/test_display_settings.py
from __future__ import annotations

import numpy as np
from napari.layers import Points

from napari_deeplabcut.core.layer_lifecycle.display_settings import (
    MACHINE_LABELS_POINTS_DISPLAY,
    POINTS_DISPLAY_SETTINGS_BY_ROLE,
    TRACKING_RESULT_POINTS_DISPLAY,
    PointsDisplaySettings,
    PointsDisplaySource,
    apply_points_display_role,
)


def make_points_layer(name: str = "pts") -> Points:
    """Non-empty layer so per-point symbol arrays can be inspected."""
    layer = Points(np.array([[0, 1, 2]], dtype=float))
    layer.name = name
    return layer


def _symbol_token(value) -> str:
    """Best-effort normalize napari Symbol/string values across versions."""
    try:
        value = value.value
    except Exception:
        pass

    text = str(value).strip().lower()

    # Defensive for enum repr-like strings, e.g. "Symbol.CROSS"
    if "." in text:
        text = text.rsplit(".", 1)[-1]

    return text


def _layer_symbol_tokens(layer: Points) -> tuple[str, ...]:
    """Return normalized symbol tokens currently stored on a Points layer."""
    symbols = np.asarray(layer.symbol, dtype=object).ravel()
    return tuple(_symbol_token(symbol) for symbol in symbols)


def _assert_configured_display_settings_applied(
    layer: Points,
    settings: PointsDisplaySettings,
) -> None:
    """Assert only the display fields that are explicitly configured."""
    if settings.symbol is not None:
        assert _symbol_token(settings.symbol) in _layer_symbol_tokens(layer)

    if settings.opacity is not None:
        assert layer.opacity == settings.opacity

    if settings.border_width is not None:
        assert getattr(layer, "border_width", None) == settings.border_width

    if settings.border_color is not None:
        # napari may normalize color strings to arrays internally, so avoid
        # over-specifying exact representation unless your pinned napari version
        # is stable enough. Keeping this light prevents cross-version brittleness.
        assert getattr(layer, "border_color", None) is not None


def test_display_settings_registry_matches_named_presets():
    assert POINTS_DISPLAY_SETTINGS_BY_ROLE[PointsDisplaySource.MACHINE_LABELS] is MACHINE_LABELS_POINTS_DISPLAY
    assert POINTS_DISPLAY_SETTINGS_BY_ROLE[PointsDisplaySource.TRACKING_RESULT] is TRACKING_RESULT_POINTS_DISPLAY


def test_apply_machine_label_display_role_uses_configured_presentation(qtbot):
    layer = make_points_layer("machine")

    apply_points_display_role(layer, PointsDisplaySource.MACHINE_LABELS)

    _assert_configured_display_settings_applied(
        layer,
        MACHINE_LABELS_POINTS_DISPLAY,
    )


def test_apply_tracking_display_role_uses_configured_presentation_and_copies_source_size(qtbot):
    source = make_points_layer("source")
    target = make_points_layer("tracking")

    source.size = 17

    apply_points_display_role(
        target,
        PointsDisplaySource.TRACKING_RESULT,
        source=source,
    )

    _assert_configured_display_settings_applied(
        target,
        TRACKING_RESULT_POINTS_DISPLAY,
    )

    if TRACKING_RESULT_POINTS_DISPLAY.copy_size_from_source:
        assert np.all(np.asarray(target.size) == 17)
