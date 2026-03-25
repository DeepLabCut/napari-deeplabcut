from __future__ import annotations

import numpy as np
import pytest
from napari.layers import Points

from napari_deeplabcut.core import keypoints
from napari_deeplabcut.core.layer_versioning import (
    detach_layer_change_hooks,
    layer_change_generations,
    mark_layer_content_changed,
    mark_layer_presentation_changed,
)
from napari_deeplabcut.core.trails import trails_geometry_signature, trails_signature


@pytest.fixture
def points_layer() -> Points:
    return Points(
        data=np.array(
            [
                [0, 1, 2],
                [1, 3, 4],
                [2, 5, 6],
                [3, 7, 8],
            ],
            dtype=float,
        ),
        properties={
            "label": np.array(["nose", "tail", "nose", "tail"], dtype=object),
            "id": np.array(["", "", "", ""], dtype=object),
        },
        metadata={
            "colormap_name": "magma",
        },
        name="points",
    )


def test_layer_change_generations_start_at_zero(points_layer: Points):
    generations = layer_change_generations(points_layer)

    assert generations.content == 0
    assert generations.presentation == 0


def test_layer_change_generations_bump_on_data_assignment(points_layer: Points):
    before = layer_change_generations(points_layer).content

    data = np.asarray(points_layer.data).copy()
    data[0, 0] += 1
    points_layer.data = data

    after = layer_change_generations(points_layer).content
    assert after == before + 1


def test_layer_change_generations_bump_on_properties_assignment(points_layer: Points):
    before = layer_change_generations(points_layer).content

    props = dict(points_layer.properties)
    props["label"] = np.array(["nose", "tail", "ear", "tail"], dtype=object)
    points_layer.properties = props

    after = layer_change_generations(points_layer).content
    assert after == before + 1


def test_layer_change_generations_bump_on_metadata_assignment(points_layer: Points):
    before = layer_change_generations(points_layer).presentation

    md = dict(points_layer.metadata or {})
    md["colormap_name"] = "viridis"
    points_layer.metadata = md

    after = layer_change_generations(points_layer).presentation
    assert after == before + 1


def test_manual_content_mark_bumps_generation(points_layer: Points):
    before = layer_change_generations(points_layer).content

    mark_layer_content_changed(points_layer)

    after = layer_change_generations(points_layer).content
    assert after == before + 1


def test_manual_presentation_mark_bumps_generation(points_layer: Points):
    before = layer_change_generations(points_layer).presentation

    mark_layer_presentation_changed(points_layer)

    after = layer_change_generations(points_layer).presentation
    assert after == before + 1


def test_trails_geometry_signature_tracks_content_generation(points_layer: Points):
    sig_before = trails_geometry_signature(points_layer)

    data = np.asarray(points_layer.data).copy()
    data[0, 0] += 1
    points_layer.data = data

    sig_after = trails_geometry_signature(points_layer)

    assert sig_before[0] == id(points_layer)
    assert sig_after[0] == id(points_layer)
    assert sig_after != sig_before


def test_trails_signature_tracks_presentation_generation(points_layer: Points):
    sig_before = trails_signature(points_layer, keypoints.ColorMode.BODYPART)

    md = dict(points_layer.metadata or {})
    md["colormap_name"] = "viridis"
    points_layer.metadata = md

    sig_after = trails_signature(points_layer, keypoints.ColorMode.BODYPART)

    assert sig_before[0] == id(points_layer)
    assert sig_after[0] == id(points_layer)
    assert sig_before[1] == str(keypoints.ColorMode.BODYPART)
    assert sig_after[1] == str(keypoints.ColorMode.BODYPART)
    assert sig_after != sig_before


def test_trails_signature_tracks_content_generation(points_layer: Points):
    sig_before = trails_signature(points_layer, keypoints.ColorMode.BODYPART)

    props = dict(points_layer.properties)
    props["label"] = np.array(["nose", "tail", "ear", "tail"], dtype=object)
    points_layer.properties = props

    sig_after = trails_signature(points_layer, keypoints.ColorMode.BODYPART)

    assert sig_after != sig_before


def test_detach_layer_change_hooks_reinstalls_cleanly(points_layer: Points):
    gens_before = layer_change_generations(points_layer)
    assert gens_before.content == 0
    assert gens_before.presentation == 0

    detach_layer_change_hooks(points_layer)

    gens_after = layer_change_generations(points_layer)
    assert gens_after.content == 0
    assert gens_after.presentation == 0

    data = np.asarray(points_layer.data).copy()
    data[0, 0] += 1
    points_layer.data = data

    assert layer_change_generations(points_layer).content == 1
