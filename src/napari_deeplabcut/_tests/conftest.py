import numpy as np
import pandas as pd
import pytest
from napari_deeplabcut import keypoints


@pytest.fixture  # TODO Hack to make this fixture session-scoped
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def points(tmp_path, viewer):
    n_rows = 10
    n_kpts = 3
    data = np.random.rand(n_rows, n_kpts * 2)
    cols = pd.MultiIndex.from_product([
        ["me"], [f"kpt_{i}" for i in range(n_kpts)], ["x", "y"]
    ], names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame(data, columns=cols, index=range(n_rows))
    dir_ = tmp_path / "folder"
    dir_.mkdir()
    output_path = dir_ / "fake_data.h5"
    df.to_hdf(output_path, key="data")
    layer = viewer.open(output_path, plugin="napari-deeplabcut")[0]
    return layer


@pytest.fixture
def store(viewer, points):
    return keypoints.KeypointStore(viewer, points)
