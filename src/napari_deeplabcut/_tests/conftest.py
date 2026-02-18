# src/napari_deeplabcut/_tests/conftest.py
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from qtpy.QtWidgets import QDockWidget
from skimage.io import imsave

from napari_deeplabcut import keypoints
from napari_deeplabcut.core import io as napari_dlc_io

# os.environ["NAPARI_DLC_HIDE_TUTORIAL"] = "True" # no longer on by default

os.environ["NAPARI_ASYNC"] = "0"  # avoid async teardown surprises in tests
# os.environ["PYTHONFAULTHANDLER"] = "1"  # better segfault traces in CI
# os.environ["QT_QPA_PLATFORM"] = "offscreen"  # headless QT for CI
# os.environ["QT_OPENGL"] = "software"  # avoid some CI issues with OpenGL
# os.environ["PYTEST_QT_API"] = "pyqt6" # only for local testing with pyqt6, we use pyside6 otherwise
logging.getLogger("napari_deeplabcut").propagate = True
logging.getLogger("napari-deeplabcut").propagate = True


@pytest.fixture(autouse=True)
def only_deeplabcut_debug_logs():
    """
    Show DEBUG logs only for napari-deeplabcut.
    Suppress DEBUG from all other libraries.
    """
    logging.getLogger()

    # Store original levels
    original_levels = {}

    try:
        for name, logger in logging.root.manager.loggerDict.items():
            if not isinstance(logger, logging.Logger):
                continue

            original_levels[name] = logger.level

            if not (name.startswith("napari_deeplabcut") or name.startswith("napari-deeplabcut")):
                logger.setLevel(logging.INFO)

        # Ensure our plugin is verbose
        logging.getLogger("napari_deeplabcut").setLevel(logging.DEBUG)

        yield
    finally:
        # Restore original logger levels
        for name, level in original_levels.items():
            logger = logging.getLogger(name)
            logger.setLevel(level)


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
    napari_dlc_io.write_config(
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
def superkeypoints_assets():
    super_animal = "superanimal_quadruped"
    json_path = Path(__file__).resolve().parents[1] / "assets" / f"{super_animal}.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {"data": data, "super_animal": super_animal}
