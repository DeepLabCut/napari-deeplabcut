# src/napari_deeplabcut/_tests/e2e/test_folder_switch_integrity.py
"""Switching the image folder underneath a surviving Points layer.

Opening a labeled folder adopts its image context onto every non-Image layer. When the
new folder's frames cannot be mapped onto a layer's existing ones, that layer must not be
half-adopted: rebinding ``root`` while ``paths`` still names the previous folder yields an
annotation file written into one dataset but indexed against another.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from napari.layers import Image, Points

from .utils import _make_project_with_two_labeled_folders, _read_h5_keypoints


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
