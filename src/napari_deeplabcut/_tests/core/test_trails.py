from __future__ import annotations

import numpy as np
import pytest
from napari.layers import Points, Tracks

from napari_deeplabcut import keypoints
from napari_deeplabcut.config.models import TrailsDisplayConfig
from napari_deeplabcut.core.trails import (
    _rgba_array,
    active_trails_color_property,
    build_trails_payload,
    categorical_colormap_from_points_layer,
    display_config_from_tracks_layer,
    is_multianimal_points_layer,
    tracks_kwargs_from_display_config,
    trails_geometry_signature,
    trails_signature,
    trails_track_ids,
)


def _make_points(
    data: np.ndarray,
    *,
    labels=None,
    ids=None,
    face_color_cycles=None,
    colormap_name="viridis",
) -> Points:
    properties = {}
    if labels is not None:
        properties["label"] = np.asarray(labels, dtype=object)
    if ids is not None:
        properties["id"] = np.asarray(ids, dtype=object)

    metadata = {
        "colormap_name": colormap_name,
        "face_color_cycles": face_color_cycles or {},
    }

    return Points(
        data=np.asarray(data, dtype=float),
        properties=properties,
        metadata=metadata,
        name="points",
    )


@pytest.fixture
def single_points_layer():
    data = np.array(
        [
            [0, 10, 20],
            [1, 11, 21],
            [2, 12, 22],
            [3, 13, 23],
        ],
        dtype=float,
    )
    labels = ["nose", "tail", "nose", "tail"]
    ids = ["", "", "", ""]
    face_color_cycles = {
        "label": {
            "nose": [1.0, 0.0, 0.0, 1.0],
            "tail": [0.0, 1.0, 0.0, 1.0],
        },
        "id": {
            "": [0.3, 0.3, 0.3, 1.0],
        },
    }
    return _make_points(
        data,
        labels=labels,
        ids=ids,
        face_color_cycles=face_color_cycles,
        colormap_name="magma",
    )


@pytest.fixture
def multi_points_layer():
    data = np.array(
        [
            [0, 10, 20],
            [0, 12, 22],
            [0, 14, 24],
            [0, 16, 26],
            [1, 11, 21],
            [1, 13, 23],
        ],
        dtype=float,
    )
    labels = ["nose", "tail", "nose", "tail", "nose", "tail"]
    ids = ["mouseA", "mouseA", "mouseB", "mouseB", "mouseA", "mouseA"]
    face_color_cycles = {
        "label": {
            "nose": [1.0, 0.0, 0.0, 1.0],
            "tail": [0.0, 1.0, 0.0, 1.0],
        },
        "id": {
            "mouseA": [0.2, 0.4, 0.6, 1.0],
            "mouseB": [0.8, 0.6, 0.2, 1.0],
        },
    }
    return _make_points(
        data,
        labels=labels,
        ids=ids,
        face_color_cycles=face_color_cycles,
        colormap_name="plasma",
    )


@pytest.fixture
def no_label_points_layer():
    data = np.array([[0, 1, 2], [1, 2, 3]], dtype=float)
    return _make_points(
        data,
        labels=None,
        ids=["animal1", "animal1"],
        face_color_cycles={},
    )


@pytest.fixture
def tracks_layer():
    data = np.array(
        [
            [0, 0, 10, 20],
            [0, 1, 11, 21],
            [1, 0, 30, 40],
        ],
        dtype=float,
    )
    layer = Tracks(
        data,
        tail_length=12,
        head_length=7,
        tail_width=3.5,
        opacity=0.4,
        blending="opaque",
        name="trails",
    )
    layer.visible = False
    return layer


def test_trails_signature_contains_expected_fields(single_points_layer):
    sig = trails_signature(single_points_layer, keypoints.ColorMode.BODYPART)

    assert sig[0] == id(single_points_layer)
    assert sig[1] == str(keypoints.ColorMode.BODYPART)
    assert sig[2] == "magma"
    assert sig[3] == 4
    assert sig[4] == ("nose", "tail", "nose", "tail")
    assert sig[5] == ("", "", "", "")


def test_trails_geometry_signature_contains_shape_and_properties(single_points_layer):
    sig = trails_geometry_signature(single_points_layer)

    assert sig[0] == id(single_points_layer)
    assert sig[1] == (4, 3)
    assert sig[2] == ("nose", "tail", "nose", "tail")
    assert sig[3] == ("", "", "", "")


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        (["animal1", "animal1"], True),
        (["", ""], False),
        ([1, 2], False),
        (None, False),
        ([], False),
    ],
)
def test_is_multianimal_points_layer(ids, expected):
    layer = _make_points(
        np.array([[0, 1, 2], [1, 2, 3]], dtype=float),
        labels=["nose", "tail"],
        ids=ids,
        face_color_cycles={},
    )
    assert is_multianimal_points_layer(layer) is expected


def test_active_trails_color_property_individual_mode_multi(multi_points_layer):
    color_prop, categories, is_multi = active_trails_color_property(
        multi_points_layer,
        keypoints.ColorMode.INDIVIDUAL,
    )

    assert color_prop == "id"
    assert is_multi is True
    np.testing.assert_array_equal(
        categories,
        np.array(["mouseA", "mouseA", "mouseB", "mouseB", "mouseA", "mouseA"], dtype=object),
    )


def test_active_trails_color_property_bodypart_mode_multi(multi_points_layer):
    color_prop, categories, is_multi = active_trails_color_property(
        multi_points_layer,
        keypoints.ColorMode.BODYPART,
    )

    assert color_prop == "label"
    assert is_multi is True
    np.testing.assert_array_equal(
        categories,
        np.array(["nose", "tail", "nose", "tail", "nose", "tail"], dtype=object),
    )


def test_active_trails_color_property_individual_mode_single_falls_back_to_label(single_points_layer):
    color_prop, categories, is_multi = active_trails_color_property(
        single_points_layer,
        keypoints.ColorMode.INDIVIDUAL,
    )

    assert color_prop == "label"
    assert is_multi is False
    np.testing.assert_array_equal(
        categories,
        np.array(["nose", "tail", "nose", "tail"], dtype=object),
    )


def test_active_trails_color_property_raises_without_labels(no_label_points_layer):
    with pytest.raises(ValueError, match="no 'label' property"):
        active_trails_color_property(no_label_points_layer, keypoints.ColorMode.BODYPART)


def test_rgba_array_converts_rgb_rgba_and_scalar():
    arr = _rgba_array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            7.0,
        ]
    )

    expected = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.5, 0.5, 0.5, 1.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(arr, expected)


def test_categorical_colormap_from_points_layer_uses_face_color_cycles(multi_points_layer):
    categories = np.array(["mouseB", "mouseA", "mouseB"], dtype=object)

    cmap, uniq_color, codes_norm = categorical_colormap_from_points_layer(
        multi_points_layer,
        "id",
        categories,
    )

    assert uniq_color == ["mouseB", "mouseA"]
    np.testing.assert_allclose(codes_norm, np.array([0.0, 1.0, 0.0]))
    assert cmap.name == "id_categorical"
    assert cmap.interpolation == "zero"

    expected_colors = np.array(
        [
            [0.8, 0.6, 0.2, 1.0],
            [0.2, 0.4, 0.6, 1.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(np.asarray(cmap.colors), expected_colors)
    np.testing.assert_allclose(np.asarray(cmap.controls), np.array([0.0, 0.5, 1.0]))


def test_categorical_colormap_from_points_layer_prefers_cycle_override(multi_points_layer):
    categories = np.array(["mouseB", "mouseA", "mouseB"], dtype=object)
    override = {
        "mouseA": [0.11, 0.22, 0.33, 1.0],
        "mouseB": [0.44, 0.55, 0.66, 1.0],
    }

    cmap, uniq_color, codes_norm = categorical_colormap_from_points_layer(
        multi_points_layer,
        "id",
        categories,
        cycle_override=override,
    )

    assert uniq_color == ["mouseB", "mouseA"]
    np.testing.assert_allclose(codes_norm, np.array([0.0, 1.0, 0.0]))
    expected_colors = np.array(
        [
            [0.44, 0.55, 0.66, 1.0],
            [0.11, 0.22, 0.33, 1.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(np.asarray(cmap.colors), expected_colors)


def test_categorical_colormap_from_points_layer_single_category_duplicates_color(single_points_layer):
    categories = np.array(["nose", "nose", "nose"], dtype=object)

    cmap, uniq_color, codes_norm = categorical_colormap_from_points_layer(
        single_points_layer,
        "label",
        categories,
    )

    assert uniq_color == ["nose"]
    np.testing.assert_allclose(codes_norm, np.array([0.0, 0.0, 0.0]))
    expected_colors = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(np.asarray(cmap.colors), expected_colors)
    np.testing.assert_allclose(np.asarray(cmap.controls), np.array([0.0, 1.0]))


def test_categorical_colormap_from_points_layer_falls_back_to_tab20(monkeypatch, multi_points_layer):
    class DummyCmap:
        colors = [
            (0.11, 0.22, 0.33),
            (0.44, 0.55, 0.66),
            (0.77, 0.88, 0.99),
        ]

    def fake_get_cmap(name):
        assert name == "tab20"
        return DummyCmap()

    monkeypatch.setattr("napari_deeplabcut.core.trails.plt.get_cmap", fake_get_cmap)

    categories = np.array(["missingA", "missingB", "missingA"], dtype=object)
    cmap, uniq_color, codes_norm = categorical_colormap_from_points_layer(
        multi_points_layer,
        "id",
        categories,
    )

    assert uniq_color == ["missingA", "missingB"]
    np.testing.assert_allclose(codes_norm, np.array([0.0, 1.0, 0.0]))
    expected_colors = np.array(
        [
            [0.11, 0.22, 0.33, 1.0],
            [0.44, 0.55, 0.66, 1.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(np.asarray(cmap.colors), expected_colors)


def test_categorical_colormap_from_points_layer_raises_on_empty_categories(single_points_layer):
    with pytest.raises(ValueError, match="No categories found"):
        categorical_colormap_from_points_layer(
            single_points_layer,
            "label",
            np.array([], dtype=object),
        )


def test_trails_track_ids_single_animal_groups_by_label(single_points_layer):
    track_ids = trails_track_ids(single_points_layer, is_multi=False)
    np.testing.assert_array_equal(track_ids, np.array([0, 1, 0, 1]))


def test_trails_track_ids_multi_animal_groups_by_id_and_label(multi_points_layer):
    track_ids = trails_track_ids(multi_points_layer, is_multi=True)
    np.testing.assert_array_equal(track_ids, np.array([0, 1, 2, 3, 0, 1]))


def test_trails_track_ids_raises_without_labels(no_label_points_layer):
    with pytest.raises(ValueError, match="no 'label' property"):
        trails_track_ids(no_label_points_layer, is_multi=True)


def test_build_trails_payload_multi_individual_mode(multi_points_layer):
    payload = build_trails_payload(multi_points_layer, keypoints.ColorMode.INDIVIDUAL)

    assert payload.color_by == "id_codes"
    assert set(payload.properties) == {"id_codes"}
    assert set(payload.colormaps_dict) == {"id_codes"}
    assert payload.signature == trails_signature(multi_points_layer, keypoints.ColorMode.INDIVIDUAL)
    assert payload.geometry_signature == trails_geometry_signature(multi_points_layer)

    np.testing.assert_array_equal(payload.tracks_data[:, 0], np.array([0, 1, 2, 3, 0, 1]))
    np.testing.assert_allclose(payload.tracks_data[:, 1:], multi_points_layer.data)
    np.testing.assert_allclose(
        payload.properties["id_codes"],
        np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
    )


def test_build_trails_payload_single_individual_mode_falls_back_to_label(single_points_layer):
    payload = build_trails_payload(single_points_layer, keypoints.ColorMode.INDIVIDUAL)

    assert payload.color_by == "label_codes"
    assert set(payload.properties) == {"label_codes"}
    assert set(payload.colormaps_dict) == {"label_codes"}
    assert payload.signature == trails_signature(single_points_layer, keypoints.ColorMode.INDIVIDUAL)
    assert payload.geometry_signature == trails_geometry_signature(single_points_layer)

    np.testing.assert_array_equal(payload.tracks_data[:, 0], np.array([0, 1, 0, 1]))
    np.testing.assert_allclose(payload.tracks_data[:, 1:], single_points_layer.data)
    np.testing.assert_allclose(payload.properties["label_codes"], np.array([0.0, 1.0, 0.0, 1.0]))


def test_tracks_kwargs_from_display_config_excludes_visible():
    cfg = TrailsDisplayConfig(
        tail_length=70,
        head_length=12,
        tail_width=4.5,
        opacity=0.75,
        blending="opaque",
        visible=False,
    )

    kwargs = tracks_kwargs_from_display_config(cfg)

    assert kwargs == {
        "tail_length": 70,
        "head_length": 12,
        "tail_width": 4.5,
        "opacity": 0.75,
        "blending": "opaque",
    }
    assert "visible" not in kwargs


def test_display_config_from_tracks_layer_reads_all_display_fields(tracks_layer):
    cfg = display_config_from_tracks_layer(tracks_layer)

    assert cfg == TrailsDisplayConfig(
        tail_length=12,
        head_length=7,
        tail_width=3.5,
        opacity=0.4,
        blending="opaque",
        visible=False,
    )


def test_display_config_from_tracks_layer_visible_override(tracks_layer):
    cfg = display_config_from_tracks_layer(tracks_layer, visible=True)
    assert cfg.visible is True
