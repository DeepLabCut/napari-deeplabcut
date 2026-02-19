from __future__ import annotations

import numpy as np
import pytest

from napari_deeplabcut.core.io import form_df


def test_form_df_autofills_likelihood_when_missing():
    points = np.array([[0.0, 44.0, 55.0]])  # [frame,y,x]
    layer_metadata = {
        "header": {"columns": [("S", "", "bp1", "x"), ("S", "", "bp1", "y")]},
        "paths": ["labeled-data/test/img000.png"],
    }
    layer_properties = {"label": ["bp1"], "id": [""]}  # no likelihood

    df = form_df(points, layer_metadata, layer_properties)
    # Should contain finite coords
    assert np.isfinite(df.to_numpy()).any()


def test_form_df_rejects_properties_length_mismatch():
    points = np.array([[0.0, 44.0, 55.0]])
    layer_metadata = {"header": {"columns": [("S", "", "bp1", "x"), ("S", "", "bp1", "y")]}}
    layer_properties = {"label": ["bp1", "bp2"], "id": [""]}  # label length mismatch

    with pytest.raises(ValueError):
        form_df(points, layer_metadata, layer_properties)
