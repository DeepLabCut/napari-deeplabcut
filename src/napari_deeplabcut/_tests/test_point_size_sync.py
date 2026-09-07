"""Point-size synchronisation between the controls widget and a Points layer.

Covers the widget-side logic that decides *whether* to apply a size:
- reading the value lives in _tests/core/test_config_sync.py
- applying it to a layer lives in _tests/core/test_layers.py
"""

# src/napari_deeplabcut/_tests/test_point_size_sync.py

from __future__ import annotations

import logging

import numpy as np
import pytest
from napari.layers import Points

from napari_deeplabcut import _widgets


def _stub_config(monkeypatch, controls, tmp_path, size):
    """Resolve a config path for any layer and make it report `size`."""
    monkeypatch.setattr(
        controls,
        "_resolve_config_path_for_layer",
        lambda _layer: tmp_path / "config.yaml",
    )
    monkeypatch.setattr(
        _widgets,
        "load_point_size_from_config",
        lambda _path: size,
    )


# ---------------------------------------------------------------------------
# _maybe_initialize_layer_point_size_from_config
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qtbot")
def test_config_point_size_seeds_fresh_layer(monkeypatch, keypoint_controls, tmp_path):
    """A layer still at a default-ish size is seeded from config."""
    layer = Points(np.zeros((3, 2), dtype=float), size=6)
    _stub_config(monkeypatch, keypoint_controls, tmp_path, 14)

    keypoint_controls._maybe_initialize_layer_point_size_from_config(layer)

    np.testing.assert_array_equal(layer.size, np.full(3, 14.0))
    assert float(layer.current_size) == 14.0


@pytest.mark.usefixtures("qtbot")
def test_config_point_size_does_not_clobber_deliberate_size(monkeypatch, keypoint_controls, tmp_path):
    """An existing size above the conservative threshold is left alone."""
    layer = Points(np.zeros((3, 2), dtype=float), size=20)
    _stub_config(monkeypatch, keypoint_controls, tmp_path, 14)

    keypoint_controls._maybe_initialize_layer_point_size_from_config(layer)

    np.testing.assert_array_equal(layer.size, np.full(3, 20.0))


@pytest.mark.usefixtures("qtbot")
def test_config_point_size_honours_current_size_on_empty_layer(monkeypatch, keypoint_controls, tmp_path):
    """On an empty layer, current_size is the only real signal.

    get_uniform_point_size() falls back to its own default when there are no
    points, so checking it alone would silently overwrite a deliberately chosen
    size on a layer that has not been annotated yet.
    """
    layer = Points(np.empty((0, 2), dtype=float))
    layer.current_size = 20
    _stub_config(monkeypatch, keypoint_controls, tmp_path, 14)

    keypoint_controls._maybe_initialize_layer_point_size_from_config(layer)

    assert float(layer.current_size) == 20.0


@pytest.mark.usefixtures("qtbot")
def test_config_point_size_rejects_invalid_value(monkeypatch, keypoint_controls, tmp_path, caplog):
    """A bad config value is reported and leaves the layer untouched."""
    layer = Points(np.zeros((3, 2), dtype=float), size=6)
    _stub_config(monkeypatch, keypoint_controls, tmp_path, -5)

    with caplog.at_level(logging.WARNING):
        keypoint_controls._maybe_initialize_layer_point_size_from_config(layer)

    np.testing.assert_array_equal(layer.size, np.full(3, 6.0))
    assert float(layer.current_size) == 6.0
    assert "Invalid point size" in caplog.text


@pytest.mark.usefixtures("qtbot")
def test_config_point_size_skipped_when_config_has_no_value(monkeypatch, keypoint_controls, tmp_path):
    """No dotsize in the config means the layer keeps whatever it had."""
    layer = Points(np.zeros((3, 2), dtype=float), size=6)
    _stub_config(monkeypatch, keypoint_controls, tmp_path, None)

    keypoint_controls._maybe_initialize_layer_point_size_from_config(layer)

    np.testing.assert_array_equal(layer.size, np.full(3, 6.0))


# ---------------------------------------------------------------------------
# _on_active_points_size_changed
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qtbot")
def test_panel_size_change_applies_to_layer(monkeypatch, keypoint_controls):
    """The panel value reaches both existing points and the new-point default."""
    layer = Points(np.zeros((3, 2), dtype=float), size=6)
    monkeypatch.setattr(keypoint_controls, "_current_dlc_points_layer", lambda: layer)

    keypoint_controls._on_active_points_size_changed(12)

    np.testing.assert_array_equal(layer.size, np.full(3, 12.0))
    assert float(layer.current_size) == 12.0


@pytest.mark.usefixtures("qtbot")
def test_panel_size_change_rejects_invalid_value(monkeypatch, keypoint_controls, caplog):
    """An invalid size is ignored rather than propagating out of the Qt slot."""
    layer = Points(np.zeros((3, 2), dtype=float), size=6)
    monkeypatch.setattr(keypoint_controls, "_current_dlc_points_layer", lambda: layer)

    with caplog.at_level(logging.WARNING):
        keypoint_controls._on_active_points_size_changed(-5)

    np.testing.assert_array_equal(layer.size, np.full(3, 6.0))
    assert "Ignoring invalid point size" in caplog.text
