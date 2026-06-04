from types import SimpleNamespace

import numpy as np
from napari.layers import Points

from napari_deeplabcut.core.layer_lifecycle import LayerLifecycleManager
from napari_deeplabcut.core.schemas.layer_identity import (
    DLC_LAYER_ROLE_KEY,
    DLC_SAVE_BEHAVIOR_KEY,
    LayerRole,
    LayerSaveBehavior,
    tag_config_placeholder_metadata,
    tag_dlc_annotation_metadata,
    tag_tracking_result_metadata,
)

from .test_manager import DummyViewer, connect_signal_recorders, make_image, make_points, mark_as_dlc_session_image


def make_dlc_points_with_header(
    header,
    *,
    name="pts",
    data=None,
    project="C:/project",
):
    layer = Points(np.zeros((0, 3)) if data is None else data)
    layer.name = name
    layer.metadata = {
        "header": header,
        "project": project,
    }
    return layer


def make_annotation_points_with_header(
    header,
    *,
    name="annotation",
    data=None,
):
    layer = make_dlc_points_with_header(
        header,
        name=name,
        data=np.array([[0, 1, 2]], dtype=float) if data is None else data,
    )
    layer.metadata = tag_dlc_annotation_metadata(layer.metadata)
    return layer


def make_config_placeholder_points_with_header(
    header,
    *,
    name="config",
):
    layer = make_dlc_points_with_header(
        header,
        name=name,
        data=np.zeros((0, 3)),
    )
    layer.metadata = tag_config_placeholder_metadata(
        layer.metadata,
        config_path="C:/project/config.yaml",
    )
    return layer


def make_tracking_points_with_header(
    header,
    *,
    name="tracking",
    data=None,
):
    layer = make_dlc_points_with_header(
        header,
        name=name,
        data=np.array([[0, 1, 2]], dtype=float) if data is None else data,
    )
    layer.metadata = tag_tracking_result_metadata(layer.metadata)
    return layer


def test_manager_save_behavior_defaults_to_napari_managed_for_generic_points(qtbot):
    viewer = DummyViewer()
    manager = LayerLifecycleManager(viewer=viewer)

    pts = make_points("generic")

    assert manager.save_behavior_for_points_layer(pts) is LayerSaveBehavior.NAPARI_MANAGED


def test_manager_save_behavior_for_dlc_annotation_is_plugin_managed(
    qtbot,
    make_real_header_factory,
):
    viewer = DummyViewer()
    manager = LayerLifecycleManager(viewer=viewer)

    header = make_real_header_factory(bodyparts=("nose", "tail"))
    pts = make_annotation_points_with_header(header)

    assert manager.save_behavior_for_points_layer(pts) is LayerSaveBehavior.PLUGIN_MANAGED


def test_manager_save_behavior_for_config_placeholder_is_napari_managed_until_promoted(
    qtbot,
    make_real_header_factory,
):
    viewer = DummyViewer()
    manager = LayerLifecycleManager(viewer=viewer)

    header = make_real_header_factory(bodyparts=("nose", "tail"))
    pts = make_config_placeholder_points_with_header(header)

    assert manager.save_behavior_for_points_layer(pts) is LayerSaveBehavior.NAPARI_MANAGED


def test_config_placeholder_without_save_context_is_not_promoted(
    qtbot,
    make_real_header_factory,
):
    viewer = DummyViewer()
    manager = LayerLifecycleManager(viewer=viewer)

    header = make_real_header_factory(bodyparts=("nose", "tail"))
    pts = make_config_placeholder_points_with_header(header)

    promoted = manager._maybe_promote_config_placeholder_points_layer(pts)

    assert promoted is False
    assert pts.metadata[DLC_LAYER_ROLE_KEY] == LayerRole.CONFIG_PLACEHOLDER.value
    assert DLC_SAVE_BEHAVIOR_KEY not in pts.metadata


def test_frames_first_then_config_placeholder_promotes_after_wiring(
    qtbot,
    monkeypatch,
    make_real_header_factory,
):
    img = mark_as_dlc_session_image(make_image("frames"))

    header = make_real_header_factory(bodyparts=("nose", "tail"))
    pts = make_config_placeholder_points_with_header(header)

    viewer = DummyViewer([img, pts])
    manager = LayerLifecycleManager(viewer=viewer)
    rec = connect_signal_recorders(manager)

    monkeypatch.setattr(manager, "validate_header", lambda layer: True)

    manager.on_insert(SimpleNamespace(value=img, index=0, source=viewer.layers))
    manager.on_insert(SimpleNamespace(value=pts, index=1, source=viewer.layers))

    assert manager.is_managed(pts) is True
    assert pts.metadata["root"] == "C:/project/labeled-data/test"
    assert pts.metadata["paths"] == ["img001.png", "img002.png"]

    assert pts.metadata[DLC_LAYER_ROLE_KEY] == LayerRole.DLC_ANNOTATION.value
    assert pts.metadata[DLC_SAVE_BEHAVIOR_KEY] == LayerSaveBehavior.PLUGIN_MANAGED.value

    assert rec.setup_points.count == 1
    req = rec.setup_points.calls[0][0]
    assert req.layer is pts
    assert req.layer.metadata[DLC_LAYER_ROLE_KEY] == LayerRole.DLC_ANNOTATION.value


def test_config_placeholder_first_then_frames_promotes_after_image_sync(
    qtbot,
    monkeypatch,
    make_real_header_factory,
):
    header = make_real_header_factory(bodyparts=("nose", "tail"))
    pts = make_config_placeholder_points_with_header(header)

    img = mark_as_dlc_session_image(make_image("frames"))

    viewer = DummyViewer([pts, img])
    manager = LayerLifecycleManager(viewer=viewer)

    monkeypatch.setattr(manager, "validate_header", lambda layer: True)

    manager.on_insert(SimpleNamespace(value=pts, index=0, source=viewer.layers))

    assert manager.is_managed(pts) is True
    assert pts.metadata[DLC_LAYER_ROLE_KEY] == LayerRole.CONFIG_PLACEHOLDER.value
    assert DLC_SAVE_BEHAVIOR_KEY not in pts.metadata

    manager.on_insert(SimpleNamespace(value=img, index=1, source=viewer.layers))

    assert pts.metadata["root"] == "C:/project/labeled-data/test"
    assert pts.metadata["paths"] == ["img001.png", "img002.png"]

    assert pts.metadata[DLC_LAYER_ROLE_KEY] == LayerRole.DLC_ANNOTATION.value
    assert pts.metadata[DLC_SAVE_BEHAVIOR_KEY] == LayerSaveBehavior.PLUGIN_MANAGED.value
