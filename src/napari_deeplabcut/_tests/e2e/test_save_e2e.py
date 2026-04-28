from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from napari.layers import Points

from napari_deeplabcut.config.models import DLCHeaderModel
from napari_deeplabcut.core.io import _read_hdf_any_key

from .utils import (
    _make_project_config_and_frames_no_gt,
    _set_or_add_bodypart_xy,
)


def _assert_single_animal_on_disk(path: Path, *, expected_bodyparts: tuple[str, ...] | None = None) -> pd.DataFrame:
    """
    Assert canonical SA on-disk format:
      columns = 3-level MultiIndex [scorer, bodyparts, coords]
    """
    df = _read_hdf_any_key(path)
    assert isinstance(df.columns, pd.MultiIndex), f"Expected MultiIndex columns in {path}"
    assert df.columns.nlevels == 3, (
        f"Expected SA 3-level columns in {path}, got {df.columns.nlevels}: {df.columns.names}"
    )
    assert list(df.columns.names) == ["scorer", "bodyparts", "coords"], (
        f"Expected SA names ['scorer','bodyparts','coords'], got {df.columns.names}"
    )

    if expected_bodyparts is not None:
        observed = tuple(dict.fromkeys(df.columns.get_level_values("bodyparts")))
        assert observed == expected_bodyparts, f"Expected bodyparts {expected_bodyparts}, got {observed}"

    return df


def _seed_single_animal_gt(
    labeled_folder: Path,
    *,
    scorer: str = "John",
    bodyparts: tuple[str, ...] = ("bodypart1",),
) -> Path:
    """
    Create a canonical SA GT file in labeled_folder:
      - 3-level columns
      - 1 labeled image row
      - first bodypart gets finite coords, the rest NaN
    """
    image_files = sorted(labeled_folder.glob("*.png"))
    assert image_files, f"No extracted frames found in {labeled_folder}"

    img_name = image_files[0].name
    dataset_name = labeled_folder.name

    cols = pd.MultiIndex.from_product(
        [[scorer], list(bodyparts), ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    idx = pd.MultiIndex.from_tuples(
        [("labeled-data", dataset_name, img_name)],
    )

    arr = np.full((1, len(cols)), np.nan, dtype=float)

    # Give the first bodypart one finite label so the file is non-empty
    arr[0, 0] = 10.0  # bodypart1 x
    arr[0, 1] = 20.0  # bodypart1 y

    df = pd.DataFrame(arr, index=idx, columns=cols)

    gt_path = labeled_folder / f"CollectedData_{scorer}.h5"
    df.to_hdf(gt_path, key="df_with_missing", mode="w")
    df.to_csv(gt_path.with_suffix(".csv"))

    return gt_path


def _append_bodypart_to_config(config_path: Path, bodypart: str) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bodyparts = list(cfg.get("bodyparts", []))
    if bodypart not in bodyparts:
        bodyparts.append(bodypart)
        cfg["bodyparts"] = bodyparts
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _header_model_for_layer(layer: Points) -> DLCHeaderModel:
    hdr = (layer.metadata or {}).get("header")
    if isinstance(hdr, DLCHeaderModel):
        return hdr
    return DLCHeaderModel.model_validate(hdr)


@pytest.mark.usefixtures("qtbot")
def test_single_animal_direct_h5_roundtrip_preserves_sa_format(
    viewer, keypoint_controls, qtbot, tmp_path, overwrite_confirm
):
    """
    Open a canonical SA GT .h5 directly, edit, save.

    This isolates the plain reader/writer path:
    - NO config layer
    - NO config merge
    - NO placeholder workflow

    If this test fails, then config merge is NOT required to reproduce the bug.
    """
    overwrite_confirm.capture()

    project, config_path, labeled_folder = _make_project_config_and_frames_no_gt(tmp_path)
    gt_path = _seed_single_animal_gt(labeled_folder, bodyparts=("bodypart1",))

    # Sanity: seed file really is canonical SA on disk
    _assert_single_animal_on_disk(gt_path, expected_bodyparts=("bodypart1",))

    viewer.open(str(gt_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len([ly for ly in viewer.layers if isinstance(ly, Points)]) == 1, timeout=10_000)
    qtbot.wait(100)

    layer = next(ly for ly in viewer.layers if isinstance(ly, Points))
    store = keypoint_controls.get_layer_store(layer)
    assert store is not None

    # Internal diagnostic: direct H5 read already normalizes SA -> canonical_4 with individuals=[""]
    hdr = _header_model_for_layer(layer)
    assert hdr.as_multiindex().nlevels == 4
    assert hdr.individuals == [""]

    # Edit existing bodypart and save
    _set_or_add_bodypart_xy(layer, store, "bodypart1", x=101.0, y=202.0)

    viewer.layers.selection.active = layer
    keypoint_controls.viewer.layers.selection.select_only(layer)
    keypoint_controls._save_layers_dialog(selected=True)
    qtbot.wait(100)

    # This is the real regression assertion.
    _assert_single_animal_on_disk(gt_path, expected_bodyparts=("bodypart1",))


@pytest.mark.usefixtures("qtbot")
def test_single_animal_gt_then_config_merge_preserves_sa_format(
    viewer, keypoint_controls, qtbot, tmp_path, overwrite_confirm
):
    """
      1) existing SA GT on disk
      2) config.yaml edited to add bodypart2
      3) open GT first
      4) open config.yaml
      5) save

    This isolates the 'config merge into existing GT layer' path.
    """
    overwrite_confirm.capture()

    project, config_path, labeled_folder = _make_project_config_and_frames_no_gt(tmp_path)
    gt_path = _seed_single_animal_gt(labeled_folder, bodyparts=("bodypart1",))
    _append_bodypart_to_config(config_path, "bodypart2")

    # Open GT first
    viewer.open(str(gt_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(
        lambda: len([ly for ly in viewer.layers if isinstance(ly, Points)]) == 1,
        timeout=10_000,
    )
    qtbot.wait(100)

    gt_layer = next(ly for ly in viewer.layers if isinstance(ly, Points))
    gt_store = keypoint_controls.get_layer_store(gt_layer)
    assert gt_store is not None

    # Then open config -> should merge and settle back to one Points layer
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(
        lambda: len([ly for ly in viewer.layers if isinstance(ly, Points)]) == 1,
        timeout=10_000,
    )
    qtbot.wait(100)

    pts_layers = [ly for ly in viewer.layers if isinstance(ly, Points)]
    assert len(pts_layers) == 1, f"Expected merged single Points layer, got {[p.name for p in pts_layers]}"

    layer = pts_layers[0]
    store = keypoint_controls.get_layer_store(layer)
    assert store is not None

    hdr = _header_model_for_layer(layer)
    assert "bodypart2" in hdr.bodyparts, f"Expected merged header to contain bodypart2, got {hdr.bodyparts}"
    assert "bodypart2" in [kp.label for kp in store._keypoints], (
        f"Store keypoints are stale after config merge: {store._keypoints}"
    )

    _set_or_add_bodypart_xy(layer, store, "bodypart2", x=77.0, y=88.0)

    viewer.layers.selection.active = layer
    keypoint_controls.viewer.layers.selection.select_only(layer)
    keypoint_controls._save_layers_dialog(selected=True)
    qtbot.wait(100)

    _assert_single_animal_on_disk(gt_path, expected_bodyparts=("bodypart1", "bodypart2"))


@pytest.mark.usefixtures("qtbot")
def test_single_animal_config_first_then_folder_new_bodypart_preserves_sa_format(
    viewer, keypoint_controls, qtbot, tmp_path, overwrite_confirm
):
    """
      1) existing SA GT on disk
      2) config.yaml edited to add bodypart2
      3) open config.yaml first
      4) open labeled-data folder
      5) add labels
      6) save

    This exercises the config-first / placeholder path.
    """
    overwrite_confirm.capture()

    _project, config_path, labeled_folder = _make_project_config_and_frames_no_gt(tmp_path)
    gt_path = _seed_single_animal_gt(labeled_folder, bodyparts=("bodypart1",))

    # Simulate user editing config.yaml outside the plugin
    _append_bodypart_to_config(config_path, "bodypart2")

    # Open config first -> placeholder points layer
    viewer.open(str(config_path), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: any(isinstance(ly, Points) for ly in viewer.layers), timeout=5_000)
    qtbot.wait(100)

    # Then open the labeled-data folder
    viewer.open(str(labeled_folder), plugin="napari-deeplabcut")
    qtbot.waitUntil(lambda: len(viewer.layers) >= 2, timeout=10_000)
    # qtbot.wait(500)

    pts_layers = [ly for ly in viewer.layers if isinstance(ly, Points)]
    assert pts_layers, "Expected at least one Points layer after config-first + folder open"

    # In this workflow the surviving layer is typically the placeholder / merged layer
    layer = pts_layers[0]
    store = keypoint_controls.get_layer_store(layer)
    assert store is not None

    hdr = _header_model_for_layer(layer)
    assert "bodypart2" in hdr.bodyparts, f"Expected config-first layer header to contain bodypart2, got {hdr.bodyparts}"

    _set_or_add_bodypart_xy(layer, store, "bodypart2", x=55.0, y=66.0)

    viewer.layers.selection.active = layer
    keypoint_controls.viewer.layers.selection.select_only(layer)
    keypoint_controls._save_layers_dialog(selected=True)
    qtbot.wait(100)

    # Expected good behavior: still canonical SA on disk
    _assert_single_animal_on_disk(gt_path, expected_bodyparts=("bodypart1", "bodypart2"))
