from __future__ import annotations

import numpy as np
import pytest
from napari.layers import Points

from napari_deeplabcut import keypoints
from napari_deeplabcut.core.trails import (
    _rgba_array,
    active_trails_color_property,
    build_trails_payload,
    categorical_colormap_from_points_layer,
    is_multianimal_points_layer,
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
    ids = ["", "", "", ""]  # explicit single-animal convention
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
            [0, 10, 20],  # mouseA nose
            [0, 12, 22],  # mouseA tail
            [0, 14, 24],  # mouseB nose
            [0, 16, 26],  # mouseB tail
            [1, 11, 21],  # mouseA nose
            [1, 13, 23],  # mouseA tail
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


def test_trails_signature_contains_expected_fields(single_points_layer):
    sig = trails_signature(single_points_layer, keypoints.ColorMode.BODYPART)

    assert sig[0] == id(single_points_layer)
    assert sig[1] == str(keypoints.ColorMode.BODYPART)
    assert sig[2] == "magma"
    assert sig[3] == 4  # n_vertices
    assert sig[4] == 4  # n_labels
    assert sig[5] == 4  # n_ids


def test_trails_signature_changes_when_metadata_changes(single_points_layer):
    sig1 = trails_signature(single_points_layer, keypoints.ColorMode.BODYPART)

    single_points_layer.metadata["colormap_name"] = "viridis"
    sig2 = trails_signature(single_points_layer, keypoints.ColorMode.BODYPART)

    assert sig1 != sig2
    assert sig2[2] == "viridis"


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
            [1.0, 0.0, 0.0],  # RGB -> RGBA
            [0.0, 1.0, 0.0, 0.5],  # RGBA unchanged
            7.0,  # scalar -> fallback gray
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

    # first-seen order is preserved
    assert uniq_color == ["mouseB", "mouseA"]

    # codes: mouseB -> 0, mouseA -> 1, mouseB -> 0
    np.testing.assert_allclose(codes_norm, np.array([0.0, 1.0, 0.0]))

    assert cmap.name == "id_categorical"
    assert cmap.interpolation == "zero"

    # colors should come from metadata cycles in uniq_color order
    expected_colors = np.array(
        [
            [0.8, 0.6, 0.2, 1.0],  # mouseB
            [0.2, 0.4, 0.6, 1.0],  # mouseA
        ],
        dtype=float,
    )
    np.testing.assert_allclose(np.asarray(cmap.colors), expected_colors)

    # n_color=2 -> controls length = 3
    np.testing.assert_allclose(np.asarray(cmap.controls), np.array([0.0, 0.5, 1.0]))


def test_categorical_colormap_from_points_layer_single_category_duplicates_color(single_points_layer):
    categories = np.array(["nose", "nose", "nose"], dtype=object)

    cmap, uniq_color, codes_norm = categorical_colormap_from_points_layer(
        single_points_layer,
        "label",
        categories,
    )

    assert uniq_color == ["nose"]
    np.testing.assert_allclose(codes_norm, np.array([0.0, 0.0, 0.0]))

    # single category -> duplicated rows for zero interpolation colormap
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

    monkeypatch.setattr("napari_deeplabcut.core.layers.trails.plt.get_cmap", fake_get_cmap)

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

    # labels: nose, tail, nose, tail
    # first-seen order => nose -> 0, tail -> 1
    np.testing.assert_array_equal(track_ids, np.array([0, 1, 0, 1]))


def test_trails_track_ids_multi_animal_groups_by_id_and_label(multi_points_layer):
    track_ids = trails_track_ids(multi_points_layer, is_multi=True)

    # group keys:
    # mouseA|nose -> 0
    # mouseA|tail -> 1
    # mouseB|nose -> 2
    # mouseB|tail -> 3
    # mouseA|nose -> 0
    # mouseA|tail -> 1
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

    # first col = track ids, remainder = original data
    np.testing.assert_array_equal(payload.tracks_data[:, 0], np.array([0, 1, 2, 3, 0, 1]))
    np.testing.assert_allclose(payload.tracks_data[:, 1:], multi_points_layer.data)

    # in INDIVIDUAL mode, colors are based on ids: mouseA/mouseB
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

    np.testing.assert_array_equal(payload.tracks_data[:, 0], np.array([0, 1, 0, 1]))
    np.testing.assert_allclose(payload.tracks_data[:, 1:], single_points_layer.data)
    np.testing.assert_allclose(payload.properties["label_codes"], np.array([0.0, 1.0, 0.0, 1.0]))
