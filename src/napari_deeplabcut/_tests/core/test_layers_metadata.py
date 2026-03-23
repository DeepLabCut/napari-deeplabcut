from __future__ import annotations

import numpy as np
import pytest

from napari_deeplabcut.config.models import AnnotationKind
from napari_deeplabcut.core.layers import is_machine_layer, populate_keypoint_layer_metadata


class HeaderStub:
    """Minimal stand-in for misc.DLCHeader.

    In DLC single-animal projects, there is no individuals level; your code
    treats that as individuals == [''] so ids[0] is falsy. In multi-animal,
    individuals contains names like ['animal1', ...] so ids[0] is truthy. [3](https://forum.image.sc/t/how-to-generate-h5-files-from-pre-collected-data/61936)
    """

    def __init__(self, bodyparts=("bp1", "bp2"), individuals=("",)):
        self.bodyparts = list(bodyparts)
        self.individuals = list(individuals)


@pytest.fixture
def patch_color_cycles(monkeypatch):
    """Make build_color_cycles deterministic and independent of colormap internals."""
    import napari_deeplabcut.core.layers as layers_mod

    def fake_build_color_cycles(header, colormap):
        # Must return cycles for both candidate face_color properties.
        return {
            "label": {"bp1": "red", "bp2": "blue"},
            "id": {"": "gray", "animal1": "green"},
        }

    monkeypatch.setattr(layers_mod.misc, "build_color_cycles", fake_build_color_cycles)


# -----------------------------------------------------------------------------
# Face color/text selection: single animal vs multi animal
# -----------------------------------------------------------------------------


def test_single_animal_uses_label_for_face_color_and_text(patch_color_cycles):
    """Single-animal: no individuals dimension => ids[0] is falsy, use label. [3](https://forum.image.sc/t/how-to-generate-h5-files-from-pre-collected-data/61936)"""
    header = HeaderStub(bodyparts=("bp1", "bp2"), individuals=("",))  # single-animal sentinel
    md = populate_keypoint_layer_metadata(header)

    assert md["face_color"] == "label"
    assert md["text"] == "label"
    assert md["face_color_cycle"] == md["metadata"]["face_color_cycles"]["label"]


def test_multi_animal_uses_id_for_face_color_and_text(patch_color_cycles):
    """Multi-animal: individuals dimension present => ids[0] truthy, use id. [3](https://forum.image.sc/t/how-to-generate-h5-files-from-pre-collected-data/61936)"""
    header = HeaderStub(bodyparts=("bp1", "bp2"), individuals=("animal1", "animal2"))
    md = populate_keypoint_layer_metadata(header)

    assert md["face_color"] == "id"
    assert md["text"] == "{id}–{label}"
    assert md["face_color_cycle"] == md["metadata"]["face_color_cycles"]["id"]


# -----------------------------------------------------------------------------
# Robustness: must accept empty labels/ids/likelihood
# -----------------------------------------------------------------------------


def test_empty_ids_must_not_crash_defaults_to_label(patch_color_cycles):
    """Regression guard for the E2E failure: ids[0] must not be assumed."""
    header = HeaderStub(bodyparts=("bp1", "bp2"), individuals=("",))
    md = populate_keypoint_layer_metadata(header, labels=["bp1"], ids=[])

    # When ids is empty, treat like single-animal (label-based) rather than crashing.
    assert md["face_color"] == "label"
    assert md["text"] == "label"
    assert md["properties"]["id"] == []


def test_empty_labels_and_ids_produce_empty_properties(patch_color_cycles):
    header = HeaderStub(bodyparts=(), individuals=())
    md = populate_keypoint_layer_metadata(header, labels=[], ids=[], likelihood=None)

    assert md["properties"]["label"] == []
    assert md["properties"]["id"] == []
    assert md["properties"]["likelihood"].shape == (0,)
    assert md["properties"]["valid"].shape == (0,)
    assert md["face_color"] == "label"
    assert md["text"] == "label"


# -----------------------------------------------------------------------------
# Likelihood + pcutoff behavior
# -----------------------------------------------------------------------------


def test_valid_is_thresholded_by_pcutoff(patch_color_cycles):
    """Likelihood is per-frame confidence in DLC outputs; valid derived via cutoff. [1](https://deeplabcut.github.io/DeepLabCut/docs/HelperFunctions.html)[4](https://forum.image.sc/t/what-to-do-with-likelihoods-0-95/45897)"""
    header = HeaderStub(bodyparts=("bp1", "bp2"), individuals=("",))
    likelihood = np.array([0.2, 0.9], dtype=float)

    md = populate_keypoint_layer_metadata(
        header,
        labels=["bp1", "bp2"],
        ids=[""],  # single-animal sentinel
        likelihood=likelihood,
        pcutoff=0.6,
    )
    assert md["properties"]["valid"].tolist() == [False, True]


def test_default_likelihood_is_ones_of_len_labels(patch_color_cycles):
    """Default fallback behavior should be stable even when likelihood not provided."""
    header = HeaderStub(bodyparts=("bp1", "bp2"), individuals=("",))
    md = populate_keypoint_layer_metadata(header, labels=["bp1", "bp2"], ids=[""])

    assert np.all(md["properties"]["likelihood"] == np.ones(2))
    assert np.all(md["properties"]["valid"] == (np.ones(2) > 0.6))


# -----------------------------------------------------------------------------
# is_machine_layer
# -----------------------------------------------------------------------------


class LayerStub:
    def __init__(self, metadata):
        self.metadata = metadata


def test_is_machine_layer_true_for_enum_kind(caplog):
    layer = LayerStub(metadata={"io": {"kind": AnnotationKind.MACHINE}})
    assert is_machine_layer(layer) is True
    assert "literal 'machine' str" not in caplog.text


@pytest.mark.parametrize("k", ["machine", "MACHINE", "Machine"])
def test_is_machine_layer_true_for_string_kind_logs_info(caplog, k):
    layer = LayerStub(metadata={"io": {"kind": k}})
    assert is_machine_layer(layer) is True
    assert "literal 'machine' str was used for io.kind" in caplog.text


@pytest.mark.parametrize("metadata", [{}, {"io": {}}, {"io": {"kind": None}}, {"io": {"kind": AnnotationKind.GT}}])
def test_is_machine_layer_false_for_missing_or_non_machine(metadata):
    layer = LayerStub(metadata=metadata)
    assert is_machine_layer(layer) is False


def test_ids_as_pandas_series_single_animal_does_not_crash(patch_color_cycles):
    import pandas as pd

    header = HeaderStub(bodyparts=("bp1",), individuals=("",))
    md = populate_keypoint_layer_metadata(header, labels=["bp1"], ids=pd.Series([""], name="individuals"))
    assert md["face_color"] == "label"
    assert md["text"] == "label"


def test_ids_as_empty_pandas_series_does_not_crash_defaults_to_label(patch_color_cycles):
    import pandas as pd

    header = HeaderStub(bodyparts=("bp1",), individuals=("",))
    md = populate_keypoint_layer_metadata(header, labels=["bp1"], ids=pd.Series([], dtype=str, name="individuals"))
    assert md["face_color"] == "label"
    assert md["text"] == "label"
    assert md["properties"]["id"] == []


def test_ids_as_pandas_series_multi_animal_uses_id(patch_color_cycles):
    import pandas as pd

    header = HeaderStub(bodyparts=("bp1",), individuals=("animal1",))
    md = populate_keypoint_layer_metadata(header, labels=["bp1"], ids=pd.Series(["animal1"], name="individuals"))
    assert md["face_color"] == "id"
    assert md["text"] == "{id}–{label}"
