# src/napari_deeplabcut/_tests/conftest.py
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from qtpy.QtWidgets import QDockWidget
from skimage.io import imsave

from napari_deeplabcut import _writer, keypoints

# os.environ["NAPARI_DLC_HIDE_TUTORIAL"] = "True" # no longer on by default

os.environ["NAPARI_ASYNC"] = "0"  # avoid async teardown surprises in tests
# os.environ["PYTHONFAULTHANDLER"] = "1"  # better segfault traces in CI
# os.environ["QT_QPA_PLATFORM"] = "offscreen"  # headless QT for CI
# os.environ["QT_OPENGL"] = "software"  # avoid some CI issues with OpenGL
# os.environ["PYTEST_QT_API"] = "pyqt6" # only for local testing with pyqt6, we use pyside6 otherwise


@pytest.fixture
def viewer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()

    # Explicitly add the dock widgets
    # NOTE The old approach of opening every plugin menu
    # with "napari-deeplabcut" in the name is not reliable
    # and is not recommended.

    #  keypoints_dock_widget, keypoints_plugin_widget
    _, _ = viewer.window.add_plugin_dock_widget(
        "napari-deeplabcut",
        "Keypoint controls",
    )

    try:
        yield viewer
    finally:
        # proactively close dock widgets to drop any lingering Qt refs
        try:
            # close all added dock widgets (if any) before viewer is closed
            for dw in list(viewer.window._qt_window.findChildren(QDockWidget)):
                # defensive: some Qt objects can be None during shutdown
                if hasattr(dw, "close"):
                    dw.close()
        except Exception:
            # bail if viewer or its window is already gone
            pass


@pytest.fixture
def fake_keypoints():
    n_rows = 10
    n_animals = 2
    n_kpts = 3
    data = np.random.rand(n_rows, n_animals * n_kpts * 2)
    cols = pd.MultiIndex.from_product(
        [
            ["me"],
            [f"animal_{i}" for i in range(n_animals)],
            [f"kpt_{i}" for i in range(n_kpts)],
            ["x", "y"],
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df = pd.DataFrame(data, columns=cols, index=range(n_rows))
    return df


@pytest.fixture
def points(tmp_path_factory, viewer, fake_keypoints):
    output_path = str(tmp_path_factory.mktemp("folder") / "fake_data.h5")
    fake_keypoints.to_hdf(output_path, key="data")
    layer = viewer.open(output_path, plugin="napari-deeplabcut")[0]

    return layer


@pytest.fixture
def fake_image():
    return (np.random.rand(10, 10) * 255).astype(np.uint8)


@pytest.fixture
def images(tmp_path_factory, viewer, fake_image):
    output_path = str(tmp_path_factory.mktemp("folder") / "img.png")
    imsave(output_path, fake_image)
    return viewer.open(output_path, plugin="napari-deeplabcut")[0]


@pytest.fixture
def store(viewer, points):
    return keypoints.KeypointStore(viewer, points)


@pytest.fixture
def single_animal_store(tmp_path_factory, viewer, fake_keypoints):
    # Keep only columns for one animal
    df = fake_keypoints.xs("animal_0", level="individuals", axis=1)
    # Now df has levels: scorer, bodyparts, coords
    # Rebuild MultiIndex with an empty "individuals" level inserted
    df.columns = pd.MultiIndex.from_product(
        [
            [df.columns.levels[0][0]],  # scorer
            [""],  # single-animal: empty ID
            df.columns.levels[1],  # bodyparts
            df.columns.levels[2],  # coords
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )

    path = tmp_path_factory.mktemp("folder") / "single_animal_data.h5"
    df.to_hdf(path, key="data")
    layer = viewer.open(path, plugin="napari-deeplabcut")[0]

    return keypoints.KeypointStore(viewer, layer)


@pytest.fixture(scope="session")
def config_path(tmp_path_factory):
    cfg = {
        "scorer": "me",
        "bodyparts": list("abc"),
        "dotsize": 0,
        "pcutoff": 0,
        "colormap": "viridis",
        "video_sets": {
            "fake_video": [],
        },
    }
    path = str(tmp_path_factory.mktemp("configs") / "config.yaml")
    _writer._write_config(
        path,
        params=cfg,
    )
    return path


@pytest.fixture(scope="session")
def video_path(tmp_path_factory):
    output_path = str(tmp_path_factory.mktemp("data") / "fake_video.avi")
    h = w = 50
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        2,
        (w, h),
    )
    for _ in range(5):
        frame = np.random.randint(0, 255, (h, w, 3)).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return output_path


@pytest.fixture
def superkeypoints_assets(tmp_path, monkeypatch):
    """
    Create a fake module dir with the expected assets layout:

        module_dir/_reader_fake.py       -> patched as __file__
        module_dir/assets/fake.json
        module_dir/assets/fake.jpg

    This mirrors the code under test:
      Path(__file__).parent / "assets" / f"{super_animal}.json|.jpg"
    """
    module_dir = tmp_path / "module"
    assets_dir = module_dir / "assets"
    assets_dir.mkdir(parents=True)

    super_animal = "fake"
    data = {
        "SK1": [10.0, 20.0],
        "SK2": [40.0, 60.0],
    }

    # JSON with superkeypoints coordinates
    (assets_dir / f"{super_animal}.json").write_text(json.dumps(data))

    # Small 10x10 RGB diagram
    Image.new("RGB", (10, 10), "white").save(assets_dir / f"{super_animal}.jpg")

    # Patch the module's __file__ so that Path(__file__).parent == module_dir
    fake_module_file = module_dir / "_reader_fake.py"
    fake_module_file.write_text("# fake")
    monkeypatch.setattr("napari_deeplabcut._reader.__file__", str(fake_module_file))

    return {
        "module_dir": module_dir,
        "assets_dir": assets_dir,
        "super_animal": super_animal,
        "data": data,
    }


@pytest.fixture
def mapped_points(points, superkeypoints_assets, config_path):
    """
    Return a DLC Points layer that is ready for _map_keypoints():
      - metadata['project'] is set (so the widget can write config.yaml)
      - metadata['tables'] contains a mapping for two real bodyparts -> SK1/SK2
      - at least two rows have coordinates exactly on the SK1/SK2 positions
        and their labels are set to those bodyparts, guaranteeing a neighbor match.
    """
    layer = points  # DLC layer created via viewer.open(..., plugin="napari-deeplabcut")
    super_animal = superkeypoints_assets["super_animal"]
    superkpts = superkeypoints_assets["data"]

    # Required by _map_keypoints to locate and write config.yaml
    # NOTE: This relies on config_path pointing to a file directly under the
    # project directory, so that Path(config_path).parent is the project root.
    layer.metadata["project"] = str(Path(config_path).parent)
    header = layer.metadata["header"]
    bp1, bp2 = header.bodyparts[:2]

    # Inject a conversion table into metadata
    layer.metadata["tables"] = {super_animal: {bp1: "SK1", bp2: "SK2"}}

    # Ensure _map_keypoints finds matches:
    # Put the first two rows exactly on SK1/SK2 and set their labels accordingly.
    layer.data[0, 1:] = np.array(superkpts["SK1"], dtype=float)
    layer.properties["label"][0] = bp1

    layer.data[1, 1:] = np.array(superkpts["SK2"], dtype=float)
    layer.properties["label"][1] = bp2

    return layer, super_animal, bp1, bp2
