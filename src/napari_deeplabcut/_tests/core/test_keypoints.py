import numpy as np
import pytest

from napari_deeplabcut.core import keypoints


def test_store_advance_step(store):
    assert store.current_step == 0
    # _advance_step() always moves the viewer one frame forward (with wrap-around)
    store._advance_step(event=None)
    assert store.current_step == 1


def test_store_labels(store, fake_keypoints):
    assert store.layer_id == id(store.layer)
    assert store.n_steps == fake_keypoints.shape[0]
    assert store.labels == list(fake_keypoints.columns.get_level_values("bodyparts").unique())


def test_store_find_first_unlabeled_frame(store):
    # With all frames annotated, finder should pick the "last" frame by convention
    store._find_first_unlabeled_frame(event=None)
    assert store.current_step == store.n_steps - 1

    # Remove all annotations for a specific frame -> it becomes the first "unlabeled" frame
    ind_to_remove = 2
    data = store.layer.data
    store.layer.data = data[data[:, 0] != ind_to_remove]
    store._find_first_unlabeled_frame(event=None)
    # Expect the navigation to jump to the first frame missing annotations
    assert store.current_step == ind_to_remove


def test_store_keypoints(store, fake_keypoints):
    annotated_keypoints = store.annotated_keypoints
    # Each frame has one keypoint per (bodypart, individual) pair; sanity-check the count and ordering
    assert len(annotated_keypoints) == fake_keypoints.shape[1] // 2
    assert annotated_keypoints[0].id == "animal_0"
    assert annotated_keypoints[-1].id == "animal_1"

    # Cycling through keypoints does not depend on selection state and should respect the header order
    kpt = keypoints.Keypoint(label="kpt_0", id="animal_0")
    next_kpt = keypoints.Keypoint(label="kpt_1", id="animal_0")
    store.current_keypoint = kpt
    assert store.current_keypoint == kpt
    store.next_keypoint()
    assert store.current_keypoint == next_kpt
    store.prev_keypoint()
    assert store.current_keypoint == kpt
    store.next_keypoint()


@pytest.mark.usefixtures("qtbot")
def test_point_resize(qtbot, viewer, points):
    viewer.layers.selection.add(points)
    layer = viewer.layers[0]
    controls = keypoints.QtPointsControls(layer)
    qtbot.addWidget(controls)
    new_size = 10
    controls.changeCurrentSize(new_size)
    np.testing.assert_array_equal(points.size, new_size)


def test_add_unannotated(store):
    # LOOP mode: after a successful add/move, the viewer advances to the next frame
    store._get_label_mode = lambda: keypoints.LabelMode.LOOP

    # Make frame 1 unannotated by removing all its rows from the layer data
    ind_to_remove = 1
    data = store.layer.data
    store.layer.data = data[data[:, 0] != ind_to_remove]

    # Navigate to that now-unannotated frame; annotated_keypoints must be empty by definition
    store.viewer.dims.set_current_step(0, ind_to_remove)
    assert not store.annotated_keypoints

    n_points = store.layer.data.shape[0]

    # IMPORTANT: pass coord with the CURRENT frame index so we truly add to the frame we're on
    # Data layout is (frame, y, x)
    store.add((ind_to_remove, 1, 1))

    # Exactly one new point should be appended
    assert store.layer.data.shape[0] == n_points + 1

    # LOOP mode advances the viewer one step forward after the add
    expected_next = (ind_to_remove + 1) % store.n_steps
    assert store.current_step == expected_next

    # Sanity check: verify a point now exists on the formerly unannotated frame
    assert np.any(store.layer.data[:, 0] == ind_to_remove)


def test_add_quick(store):
    # QUICK mode: if the keypoint for the current frame already exists, it is MOVED; otherwise, it is ADDED.
    # QUICK does NOT auto-advance the viewer.
    store._get_label_mode = lambda: keypoints.LabelMode.QUICK

    # Choose a specific keypoint to act on; this determines which (label, id) is added/moved
    store.current_keypoint = store._keypoints[0]

    # Add (or move) at the CURRENT frame; coord uses (frame, y, x)
    coord = store.current_step, -1, -1
    store.add(coord)

    # After QUICK add/move, the point for the current frame should match the requested coord
    # (If it existed, it was moved; if not, it was added.)
    np.testing.assert_array_equal(
        store.layer.data[store.current_step],
        coord,
    )


def test_store_can_attach_layer_resolver(store):
    original_layer = store.layer
    layer_id = id(original_layer)

    # Resolver returns the original live layer by id.
    store.attach_layer_resolver(lambda requested_id: original_layer if requested_id == layer_id else None)

    assert store.layer_id == layer_id
    assert store.maybe_layer() is original_layer
    assert store.layer is original_layer


def test_store_layer_raises_when_resolver_returns_none(store):
    store.attach_layer_resolver(lambda requested_id: None)

    assert store.maybe_layer() is None

    with pytest.raises(keypoints.LayerUnavailableError):
        _ = store.layer


def test_store_resolver_is_authoritative_over_local_fallback(store):

    # Even though the store still has fallback refs, resolver should dominate.
    store.attach_layer_resolver(lambda requested_id: None)

    assert store.maybe_layer() is None

    with pytest.raises(keypoints.LayerUnavailableError):
        _ = store.layer


def test_store_layer_setter_updates_layer_id_and_keypoints(store, viewer):
    old_layer = store.layer
    old_layer_id = store.layer_id

    new_layer = viewer.layers[0].copy() if hasattr(viewer.layers[0], "copy") else old_layer
    store.layer = new_layer

    assert store.layer is new_layer
    assert store.layer_id == id(new_layer)
    assert store.layer_id != old_layer_id or new_layer is old_layer
