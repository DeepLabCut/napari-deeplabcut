import numpy as np
import pytest
from napari.layers import Points

from napari_deeplabcut.core.layers import populate_keypoint_layer_metadata


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
    md = populate_keypoint_layer_metadata(
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
    md = populate_keypoint_layer_metadata(
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
    md = populate_keypoint_layer_metadata(
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
    md = populate_keypoint_layer_metadata(
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
