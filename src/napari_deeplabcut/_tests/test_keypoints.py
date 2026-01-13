import weakref

import numpy as np

from napari_deeplabcut import keypoints


def test_store_advance_step(store):
    assert store.current_step == 0
    # _advance_step() always moves the viewer one frame forward (with wrap-around)
    store._advance_step(event=None)
    assert store.current_step == 1


def test_store_labels(store, fake_keypoints):
    assert store.n_steps == fake_keypoints.shape[0]
    # Labels are derived from the header (bodyparts level); this asserts the store mirrors header order
    assert store.labels == list(fake_keypoints.columns.get_level_values("bodyparts").unique())


def test_store_find_first_unlabeled_frame(store):
    # With all frames annotated, finder should pick the "last" frame by convention
    store._find_first_unlabeled_frame(event=None)
    assert store.current_step == store.n_steps - 1

    # Remove all annotations for a specific frame -> it becomes the first "unlabeled" frame
    ind_to_remove = 2
    data = weakref.proxy(store.layer).data
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
    store.layer.metadata["controls"].label_mode = "loop"

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
    keypoints._add(store, coord=(ind_to_remove, 1, 1))

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
    store.layer.metadata["controls"].label_mode = "quick"

    # Choose a specific keypoint to act on; this determines which (label, id) is added/moved
    store.current_keypoint = store._keypoints[0]

    # Add (or move) at the CURRENT frame; coord uses (frame, y, x)
    coord = store.current_step, -1, -1
    keypoints._add(store, coord=coord)

    # After QUICK add/move, the point for the current frame should match the requested coord
    # (If it existed, it was moved; if not, it was added.)
    np.testing.assert_array_equal(
        store.layer.data[store.current_step],
        coord,
    )
