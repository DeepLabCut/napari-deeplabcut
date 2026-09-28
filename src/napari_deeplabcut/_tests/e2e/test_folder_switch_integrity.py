# src/napari_deeplabcut/_tests/e2e/test_folder_switch_integrity.py
"""Switching the image folder underneath a surviving Points layer.

Opening a labeled folder adopts its image context onto every non-Image layer. When the
new folder's frames cannot be mapped onto a layer's existing ones, that layer must not be
half-adopted: rebinding ``root`` while ``paths`` still names the previous folder yields an
annotation file written into one dataset but indexed against another.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from napari.layers import Image, Points

from .utils import (
    _make_project_with_two_labeled_folders,
    _make_two_projects_sharing_a_video_name,
    _read_h5_keypoints,
    _write_dlc_config,
    _write_frames,
)


def _points_layers(viewer):
    return [ly for ly in viewer.layers if isinstance(ly, Points)]


def _open_folder(viewer, qtbot, folder: Path, *, expect_points: bool) -> None:
    viewer.open(str(folder), plugin="napari-deeplabcut")
    if expect_points:
        qtbot.waitUntil(lambda: bool(_points_layers(viewer)), timeout=10_000)
    else:
        qtbot.waitUntil(
            lambda: any(isinstance(ly, Image) for ly in viewer.layers),
            timeout=10_000,
        )
    qtbot.wait(100)


def _remove_image_layers(viewer, qtbot) -> None:
    for layer in [ly for ly in viewer.layers if isinstance(ly, Image)]:
        viewer.layers.remove(layer)
    qtbot.wait(100)


def _dataset_names_in_index(df: pd.DataFrame) -> set[str]:
    """Dataset-folder component of each row key."""
    if isinstance(df.index, pd.MultiIndex):
        return {str(parts[-2]) for parts in df.index}
    return {Path(str(v)).parent.name for v in df.index}


@pytest.mark.usefixtures("qtbot")
def test_unmappable_folder_switch_leaves_points_layer_bound_to_its_own_dataset(
    viewer,
    keypoint_controls,
    qtbot,
    tmp_path,
) -> None:
    """videoB's frames cannot be mapped onto videoA's, so the layer must not be rebound."""
    _project, _config_path, folder_a, folder_b, _gt_path = _make_project_with_two_labeled_folders(tmp_path)

    _open_folder(viewer, qtbot, folder_a, expect_points=True)

    layer = _points_layers(viewer)[0]
    paths_before = list(layer.metadata.get("paths") or [])
    root_before = layer.metadata.get("root")
    assert paths_before, "Expected the folder reader to bind frame paths to the Points layer"

    _remove_image_layers(viewer, qtbot)
    _open_folder(viewer, qtbot, folder_b, expect_points=False)

    assert list(layer.metadata.get("paths") or []) == paths_before, (
        "Points layer frame paths changed after opening an unrelated folder"
    )
    assert layer.metadata.get("root") == root_before, (
        "Points layer root was rebound to a folder whose frames it does not contain"
    )


@pytest.mark.usefixtures("qtbot")
def test_unmappable_folder_switch_does_not_write_annotations_into_new_folder(
    viewer,
    keypoint_controls,
    qtbot,
    tmp_path,
    overwrite_confirm,
) -> None:
    """Saving after an unmappable switch must not drop a foreign-indexed file into videoB."""
    overwrite_confirm.capture()

    _project, _config_path, folder_a, folder_b, gt_path = _make_project_with_two_labeled_folders(tmp_path)

    _open_folder(viewer, qtbot, folder_a, expect_points=True)
    layer = _points_layers(viewer)[0]

    _remove_image_layers(viewer, qtbot)
    _open_folder(viewer, qtbot, folder_b, expect_points=False)

    viewer.layers.selection.select_only(layer)
    keypoint_controls._save_layers_dialog(selected=True)
    qtbot.wait(200)

    stray = sorted(p.name for p in folder_b.glob("CollectedData*"))
    assert not stray, f"Annotations were written into {folder_b.name}: {stray}"

    assert _dataset_names_in_index(_read_h5_keypoints(gt_path)) == {"videoA"}


@pytest.mark.usefixtures("qtbot")
def test_identical_frame_names_in_another_folder_do_not_migrate_the_layer(
    viewer,
    keypoint_controls,
    qtbot,
    tmp_path,
    overwrite_confirm,
) -> None:
    """videoB reuses videoA's frame names, which is the DLC norm, not evidence of identity.

    Matching on basenames alone would give a perfect 1:1 map here and silently carry the
    annotation onto a different dataset's frames.
    """
    overwrite_confirm.capture()

    _project, _config_path, folder_a, folder_b, gt_path = _make_project_with_two_labeled_folders(
        tmp_path,
        b_frames=("imgA000.png",),
    )

    _open_folder(viewer, qtbot, folder_a, expect_points=True)
    layer = _points_layers(viewer)[0]
    paths_before = list(layer.metadata.get("paths") or [])

    _remove_image_layers(viewer, qtbot)
    _open_folder(viewer, qtbot, folder_b, expect_points=False)

    assert Path(str(layer.metadata.get("root"))).name == "videoA"
    assert list(layer.metadata.get("paths") or []) == paths_before

    viewer.layers.selection.select_only(layer)
    keypoint_controls._save_layers_dialog(selected=True)
    qtbot.wait(200)

    stray = sorted(p.name for p in folder_b.glob("CollectedData*"))
    assert not stray, f"Annotations migrated into {folder_b.name}: {stray}"
    assert _dataset_names_in_index(_read_h5_keypoints(gt_path)) == {"videoA"}


@pytest.mark.usefixtures("qtbot")
def test_layer_does_not_follow_a_different_project_using_the_same_video_name(
    viewer,
    keypoint_controls,
    qtbot,
    tmp_path,
    overwrite_confirm,
) -> None:
    """Re-labelling the same footage in a fresh project is an ordinary DLC workflow.

    Both projects then hold `labeled-data/mouse1/img000.png`, which is identical at every
    canonicalization depth: the project root is not part of the key. A layer from the
    first project must not silently rebind to, or be saved into, the second.
    """
    overwrite_confirm.capture()

    proj = _make_two_projects_sharing_a_video_name(tmp_path)

    _open_folder(viewer, qtbot, proj.folder_a, expect_points=True)
    layer = _points_layers(viewer)[0]
    root_before = Path(str(layer.metadata.get("root")))
    project_before = layer.metadata.get("project")

    _remove_image_layers(viewer, qtbot)
    _open_folder(viewer, qtbot, proj.folder_b, expect_points=False)

    assert Path(str(layer.metadata.get("root"))) == root_before, (
        f"Layer rebound from {project_before} to another project's dataset of the same name"
    )

    viewer.layers.selection.select_only(layer)
    keypoint_controls._save_layers_dialog(selected=True)
    qtbot.wait(200)

    stray = sorted(p.name for p in proj.folder_b.glob("CollectedData*"))
    assert not stray, f"Annotations from {project_before} were written into project-B: {stray}"


def _write_multi_row_gt(path: Path, *, scorer: str, folder_name: str, rows: dict[str, list[float]]) -> Path:
    """GT file whose row keys may name frames that are not on disk."""
    cols = pd.MultiIndex.from_product(
        [[scorer], ["bodypart1", "bodypart2"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    index = pd.MultiIndex.from_tuples([("labeled-data", folder_name, name) for name in rows])
    df = pd.DataFrame(list(rows.values()), index=index, columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(path, key="df_with_missing", mode="w")
    df.to_csv(str(path).replace(".h5", ".csv"))
    return path


@pytest.mark.usefixtures("qtbot")
def test_keypoints_on_a_deleted_frame_do_not_spread_to_other_frames(
    viewer,
    keypoint_controls,
    qtbot,
    tmp_path,
    overwrite_confirm,
) -> None:
    """A row key with no file on disk must not be re-keyed onto a frame that does exist.

    Frame association is positional. Re-keying moves those keypoints onto whichever path
    took their index, and each save walks them one frame further down the folder.
    """
    overwrite_confirm.capture()

    project = tmp_path / "project"
    folder = _write_frames(project / "labeled-data" / "videoA", ("img001.png", "img002.png"))
    _write_dlc_config(project, bodyparts=("bodypart1", "bodypart2"))

    gt_path = _write_multi_row_gt(
        folder / "CollectedData_John.h5",
        scorer="John",
        folder_name="videoA",
        rows={
            "img000.png": [10.0, 20.0, 30.0, 40.0],  # deleted from disk, carries keypoints
            "img001.png": [np.nan, np.nan, np.nan, np.nan],
            "img002.png": [np.nan, np.nan, np.nan, np.nan],
        },
    )

    _open_folder(viewer, qtbot, folder, expect_points=True)
    layer = _points_layers(viewer)[0]

    assert len(layer.metadata.get("paths") or []) == 3, "The layer must keep its own frame list, not the folder's"

    viewer.layers.selection.select_only(layer)
    keypoint_controls._save_layers_dialog(selected=True)
    qtbot.wait(300)

    df = _read_h5_keypoints(gt_path)
    annotated = {str(idx[-1]) for idx, row in df.iterrows() if np.isfinite(row.to_numpy(dtype=float)).any()}
    assert annotated == {"img000.png"}, f"Keypoints spread to frames they were never placed on: {annotated}"


@pytest.mark.usefixtures("qtbot")
def test_frames_added_to_the_folder_still_remap(viewer, keypoint_controls, qtbot, tmp_path) -> None:
    """The DLC refine loop only adds frames, so it must keep working.

    `extract_outlier_frames` writes new frames into labeled-data/<video> and never removes
    one, so no annotated row can lose its path.
    """
    project = tmp_path / "project"
    folder = _write_frames(project / "labeled-data" / "videoA", ("img000.png", "img001.png"))
    _write_dlc_config(project, bodyparts=("bodypart1", "bodypart2"))

    _write_multi_row_gt(
        folder / "CollectedData_John.h5",
        scorer="John",
        folder_name="videoA",
        rows={"img000.png": [10.0, 20.0, 30.0, 40.0]},
    )

    _open_folder(viewer, qtbot, folder, expect_points=True)
    layer = _points_layers(viewer)[0]

    assert len(layer.metadata.get("paths") or []) == 2, "An added frame must not block the remap"
