# src/napari_deeplabcut/_tests/core/test_layers.py

from __future__ import annotations

import numpy as np
from napari.layers import Points

from napari_deeplabcut.core.layers import set_uniform_point_size


def test_set_uniform_point_size_resizes_all_existing_points() -> None:
    """Existing points are resized uniformly regardless of selection."""
    layer = Points(
        np.zeros((3, 2), dtype=float),
        size=[4, 6, 8],
    )
    layer.selected_data = {0}
    original_current_size = float(layer.current_size)

    set_uniform_point_size(layer, 10, update_new=False)

    assert len(layer.size) == 3
    np.testing.assert_array_equal(
        layer.size,
        np.full(3, 10.0),
    )
    assert float(layer.current_size) == original_current_size


def test_set_uniform_point_size_updates_default_for_new_points() -> None:
    """With update_new enabled, subsequently added points inherit the size."""
    layer = Points(
        np.zeros((3, 2), dtype=float),
        size=[4, 6, 8],
    )
    layer.selected_data = {0}

    set_uniform_point_size(layer, 10, update_new=True)

    assert len(layer.size) == 3
    np.testing.assert_array_equal(
        layer.size,
        np.full(3, 10.0),
    )
    assert float(layer.current_size) == 10.0

    layer.add([1.0, 1.0])

    assert len(layer.size) == 4
    assert float(layer.size[-1]) == 10.0
