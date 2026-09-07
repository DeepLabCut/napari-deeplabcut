from __future__ import annotations

import numpy as np
from napari.layers import Points

from napari_deeplabcut.core.layers import set_uniform_point_size


def test_set_uniform_point_size_resizes_all_points_regardless_of_selection() -> None:
    """Uniform resizing must not be limited to the selected points.

    DeepLabCut treats point size as a layer-wide display setting. This protects
    that behavior from regressing to napari's selection-scoped current-size
    semantics.
    """
    n_points = 3
    layer = Points(
        np.zeros((n_points, 2)),
        size=[4, 6, 8],
    )
    layer.selected_data = {0}

    set_uniform_point_size(layer, 10)

    assert len(layer.size) == n_points
    np.testing.assert_array_equal(
        layer.size,
        np.full(n_points, 10.0),
    )
