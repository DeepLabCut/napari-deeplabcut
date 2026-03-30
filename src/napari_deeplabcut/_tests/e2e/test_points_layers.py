import numpy as np
import pytest
from napari.layers import Points

from napari_deeplabcut.core.layers import populate_keypoint_layer_properties


@pytest.mark.usefixtures("qtbot")
def test_on_insert_empty_points_layer_does_not_crash(make_napari_viewer, make_real_header_factory):
    """
    Contract: inserting an empty Points layer must not crash.
    This guards against KeyError: nan coming from napari cycle colormap logic.
    """
    viewer = make_napari_viewer()

    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    header = make_real_header_factory(individuals=("animal1",))  # either is fine
    md = populate_keypoint_layer_properties(
        header,
        labels=[],  # empty properties
        ids=[],  # empty properties
        likelihood=np.array([], dtype=float),
        paths=[],
        colormap="viridis",
    )

    # napari Points layer coordinates: (N, D). For your plugin D=3 (frame,y,x)
    empty_data = np.empty((0, 3), dtype=float)

    # The test is simply: adding must not raise
    viewer.add_points(empty_data, **md)


@pytest.mark.usefixtures("qtbot")
def test_on_insert_empty_points_layer_does_not_enable_cycle_mode(make_napari_viewer, make_real_header_factory):
    """
    Contract: for empty layers, widget should not set face_color_mode='cycle'
    (or should otherwise avoid the cycle colormap path that crashes on nan).
    """
    viewer = make_napari_viewer()
    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    header = make_real_header_factory(individuals=("",))  # single animal
    md = populate_keypoint_layer_properties(
        header,
        labels=[],
        ids=[],
        likelihood=np.array([], dtype=float),
        paths=[],
        colormap="viridis",
    )
    layer = viewer.add_points(np.empty((0, 3), dtype=float), **md)

    assert isinstance(layer, Points)
    assert layer.data.shape[0] == 0

    # Allow either "direct" or something else, but cycle is unsafe for empties.
    assert layer.face_color_mode != "cycle"


@pytest.mark.usefixtures("qtbot")
def test_adopt_existing_empty_points_layer_does_not_crash(make_napari_viewer, make_real_header_factory):
    """
    Contract: adoption path must not crash for empty points layers.
    This exercises _adopt_existing_layers -> _handle_existing_points_layer.
    """
    viewer = make_napari_viewer()

    # Add layer BEFORE creating the widget (forces adoption path)
    header = make_real_header_factory(individuals=("animal1",))
    md = populate_keypoint_layer_properties(
        header,
        labels=[],
        ids=[],
        likelihood=np.array([], dtype=float),
        paths=[],
        colormap="viridis",
    )
    viewer.add_points(np.empty((0, 3), dtype=float), **md)

    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # If we got here without exception, adoption didn’t crash
    pts_layers = [ly for ly in viewer.layers if isinstance(ly, Points)]
    assert pts_layers, "Expected the empty Points layer to exist"


@pytest.mark.usefixtures("qtbot")
def test_layer_insert_does_not_crash_when_current_property_is_nan(make_napari_viewer, make_real_header_factory):
    """
    Contract: even if a property value is NaN (bad input), widget must not crash.
    It may fall back to direct mode or sanitize the property.
    """
    viewer = make_napari_viewer()

    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    header = make_real_header_factory(individuals=("",))
    md = populate_keypoint_layer_properties(
        header,
        labels=["bodypart1"],
        ids=[""],
        likelihood=np.array([1.0], dtype=float),
        paths=[],
        colormap="viridis",
    )

    # One point, but corrupt the property used for cycling
    data = np.array([[0.0, 10.0, 20.0]], dtype=float)
    md["properties"]["label"] = [np.nan]  # intentionally wrong

    # Must not raise on insertion
    layer = viewer.add_points(data, **md)
    # Plot cannot be formed because of the NaN,
    # but the layer must still be added and cycle mode must not be enabled.
    assert controls._matplotlib_canvas.df is None
    assert isinstance(layer, Points)
    assert layer.face_color_mode != "cycle"


@pytest.mark.usefixtures("qtbot")
def test_copy_paste_points_to_new_frame_does_not_crash_and_offsets_frame(
    make_napari_viewer,
    make_real_header_factory,
    qtbot,
):
    """
    Regression test for DLC's patched Points._paste_data.

    Scenario:
    - create a 3D (t, y, x) points layer
    - copy selected points on frame 0
    - move to frame 1
    - paste

    Contract:
    - must not crash
    - pasted points must appear on the current frame
    - point properties (e.g. labels) must be preserved
    """
    viewer = make_napari_viewer()

    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    # Add an image stack to make the time/frame axis explicit in a realistic way.
    viewer.add_image(
        np.zeros((2, 64, 64), dtype=np.uint8),
        name="frames",
        metadata={"paths": ["frame0.png", "frame1.png"]},
    )

    header = make_real_header_factory(individuals=("",))  # single-animal layout
    md = populate_keypoint_layer_properties(
        header,
        labels=["head", "tail"],
        ids=["", ""],
        likelihood=np.array([1.0, 1.0], dtype=float),
        paths=["frame0.png", "frame1.png"],
        colormap="viridis",
    )

    # two points on frame 0
    data = np.array(
        [
            [0.0, 10.0, 20.0],  # (t, y, x)
            [0.0, 30.0, 40.0],
        ],
        dtype=float,
    )

    layer = viewer.add_points(data, **md)

    assert isinstance(layer, Points)
    assert getattr(layer, "_dlc_controls", None) is controls, "Layer was not wired by KeypointControls"

    # frame 0: select and copy
    viewer.dims.set_point(0, 0)
    qtbot.wait(0)

    layer.selected_data = {0, 1}
    layer._copy_data()

    assert "data" in layer._clipboard
    assert "features" in layer._clipboard
    assert len(layer._clipboard["data"]) == 2

    # move to frame 1 and paste
    viewer.dims.set_point(0, 1)
    qtbot.wait(0)

    layer._paste_data()
    qtbot.wait(0)

    # original 2 + pasted 2
    assert len(layer.data) == 4

    pasted = np.asarray(layer.data)[-2:]
    np.testing.assert_array_equal(pasted[:, 0], np.array([1.0, 1.0]))
    np.testing.assert_allclose(pasted[:, 1:], data[:, 1:])

    # labels/features should be preserved for pasted points
    labels = list(layer.properties["label"])
    assert labels[-2:] == ["head", "tail"]


@pytest.mark.usefixtures("qtbot")
def test_copy_paste_same_frame_does_not_duplicate_existing_keypoints(
    make_napari_viewer,
    make_real_header_factory,
    qtbot,
):
    """
    Contract:
    If the copied keypoints are already annotated on the current frame,
    DLC's patched paste should not duplicate them.
    """
    viewer = make_napari_viewer()

    from napari_deeplabcut._widgets import KeypointControls

    controls = KeypointControls(viewer)
    viewer.window.add_dock_widget(controls, name="Keypoint controls", area="right")

    viewer.add_image(
        np.zeros((1, 64, 64), dtype=np.uint8),
        name="frame",
        metadata={"paths": ["frame0.png"]},
    )

    header = make_real_header_factory(individuals=("",))
    md = populate_keypoint_layer_properties(
        header,
        labels=["head", "tail"],
        ids=["", ""],
        likelihood=np.array([1.0, 1.0], dtype=float),
        paths=["frame0.png"],
        colormap="viridis",
    )

    data = np.array(
        [
            [0.0, 10.0, 20.0],
            [0.0, 30.0, 40.0],
        ],
        dtype=float,
    )

    layer = viewer.add_points(data, **md)
    assert isinstance(layer, Points)

    viewer.dims.set_point(0, 0)
    qtbot.wait(0)

    layer.selected_data = {0, 1}
    layer._copy_data()

    before = len(layer.data)
    layer._paste_data()
    qtbot.wait(0)

    # no duplicates expected on same frame
    assert len(layer.data) == before
