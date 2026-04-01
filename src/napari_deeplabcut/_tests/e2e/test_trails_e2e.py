from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from napari.layers import Tracks

from napari_deeplabcut._widgets import KeypointControls
from napari_deeplabcut.core import keypoints


def _open_multianimal_points(viewer, tmp_path: Path, *, n_animals: int = 3, n_kpts: int = 2, n_frames: int = 4):
    rng = np.random.default_rng(123)
    data = rng.random((n_frames, n_animals * n_kpts * 2))

    cols = pd.MultiIndex.from_product(
        [
            ["me"],
            [f"animal_{i}" for i in range(n_animals)],
            [f"kpt_{i}" for i in range(n_kpts)],
            ["x", "y"],
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df = pd.DataFrame(data, columns=cols, index=range(n_frames))

    path = tmp_path / "three_animals.h5"
    df.to_hdf(path, key="data")

    layer = viewer.open(path, plugin="napari-deeplabcut")[0]
    return layer


def _trails_layer(controls) -> Tracks | None:
    """Current live trails layer managed by the extracted trails controller."""
    return controls._trails_controller.layer


def _expected_cycle_colors_from_controls(controls, points_layer):
    """
    Expected trails colors must come from the same resolved cycle path as the widget,
    not directly from raw metadata.
    """
    prop = "id" if controls.color_mode == str(keypoints.ColorMode.INDIVIDUAL) else "label"
    vals = list(dict.fromkeys(map(str, points_layer.properties[prop])))

    cycle = controls._resolved_cycle_for_layer(points_layer)
    out = []
    for v in vals:
        c = np.asarray(cycle[v], dtype=float)
        if c.shape[0] == 3:
            c = np.r_[c, 1.0]
        out.append(c)
    return np.asarray(out, dtype=float), vals


def _current_trails_cmap_colors(tracks_layer: Tracks):
    cmap = tracks_layer.colormaps_dict[tracks_layer.color_by]
    return np.asarray(cmap.colors)


def test_trails_mode_switch_does_not_fallback_to_track_id(viewer, tmp_path):
    points = _open_multianimal_points(viewer, tmp_path, n_animals=3, n_kpts=2, n_frames=4)
    controls = KeypointControls.get_layer_controls(points)
    assert controls is not None

    # Make the points layer active so trails source selection is deterministic
    viewer.layers.selection.active = points

    controls._trail_cb.setChecked(True)
    trails = _trails_layer(controls)
    assert trails is not None
    assert trails.color_by == "id_codes"

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")

        controls.color_mode = keypoints.ColorMode.BODYPART
        trails = _trails_layer(controls)
        assert trails is not None
        assert trails.color_by == "label_codes"

        controls.color_mode = keypoints.ColorMode.INDIVIDUAL
        trails = _trails_layer(controls)
        assert trails is not None
        assert trails.color_by == "id_codes"

    msgs = [str(w.message) for w in rec]
    assert not any("Falling back to track_id" in m for m in msgs), msgs
    assert not any("Previous color_by key" in m for m in msgs), msgs


def test_trails_repeated_mode_switch_keeps_expected_colormap(viewer, tmp_path):
    points = _open_multianimal_points(viewer, tmp_path, n_animals=3, n_kpts=2, n_frames=4)
    controls = KeypointControls.get_layer_controls(points)
    assert controls is not None

    viewer.layers.selection.active = points
    controls._trail_cb.setChecked(True)

    trails = _trails_layer(controls)
    assert trails is not None

    # Initial individual mode colors
    expected_id_colors, _ = _expected_cycle_colors_from_controls(controls, points)
    actual_id_colors = _current_trails_cmap_colors(trails)
    np.testing.assert_allclose(actual_id_colors, expected_id_colors)

    # Switch to bodypart mode
    controls.color_mode = keypoints.ColorMode.BODYPART
    trails = _trails_layer(controls)
    assert trails is not None
    assert trails.color_by == "label_codes"

    expected_label_colors, _ = _expected_cycle_colors_from_controls(controls, points)
    actual_label_colors = _current_trails_cmap_colors(trails)
    np.testing.assert_allclose(actual_label_colors, expected_label_colors)

    # Switch back to individual mode
    controls.color_mode = keypoints.ColorMode.INDIVIDUAL
    trails = _trails_layer(controls)
    assert trails is not None
    assert trails.color_by == "id_codes"

    actual_id_colors_2 = _current_trails_cmap_colors(trails)
    np.testing.assert_allclose(actual_id_colors_2, expected_id_colors)

    # And one more round trip for stability
    controls.color_mode = keypoints.ColorMode.BODYPART
    trails = _trails_layer(controls)
    assert trails is not None

    actual_label_colors_2 = _current_trails_cmap_colors(trails)
    np.testing.assert_allclose(actual_label_colors_2, expected_label_colors)


def test_trails_individual_mode_three_animals_have_three_distinct_mapped_colors(viewer, tmp_path):
    points = _open_multianimal_points(viewer, tmp_path, n_animals=3, n_kpts=1, n_frames=4)
    controls = KeypointControls.get_layer_controls(points)
    assert controls is not None

    viewer.layers.selection.active = points
    controls.color_mode = keypoints.ColorMode.INDIVIDUAL
    controls._trail_cb.setChecked(True)

    trails = _trails_layer(controls)
    assert trails is not None
    assert trails.color_by == "id_codes"

    cmap = trails.colormaps_dict["id_codes"]
    codes = np.asarray(trails.properties["id_codes"])

    # Only inspect the first instance of each animal
    uniq_codes = list(dict.fromkeys(codes.tolist()))
    mapped = np.asarray(cmap.map(np.asarray(uniq_codes, dtype=float)))
    unique_rows = np.unique(np.round(mapped, decimals=8), axis=0)

    assert unique_rows.shape[0] == 3, (
        f"Expected 3 unique colors for 3 animals, got {unique_rows.shape[0]} unique colors: {unique_rows}"
    )
