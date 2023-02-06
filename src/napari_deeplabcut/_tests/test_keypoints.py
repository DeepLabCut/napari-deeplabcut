import numpy as np
from napari_deeplabcut import keypoints


def test_store(store, fake_keypoints):
    assert store.current_step == 0
    store._advance_step(event=None)
    assert store.current_step == 1

    assert store.n_steps == fake_keypoints.shape[0]
    assert store.labels == list(
        fake_keypoints.columns.get_level_values("bodyparts").unique()
    )

    annotated_keypoints = store.annotated_keypoints
    assert len(annotated_keypoints) == fake_keypoints.shape[1] // 2
    assert annotated_keypoints[0].id == "animal_0"
    assert annotated_keypoints[-1].id == "animal_1"
    kpt = keypoints.Keypoint(label="kpt_0", id="animal_0")
    next_kpt = keypoints.Keypoint(label="kpt_1", id="animal_0")
    store.current_keypoint = kpt
    assert store.current_keypoint == kpt
    store.next_keypoint()
    assert store.current_keypoint == next_kpt
    store.prev_keypoint()
    assert store.current_keypoint == kpt
    store.next_keypoint()

    store._find_first_unlabeled_frame(event=None)
    assert store.current_step == store.n_steps - 1
    # Remove a frame to test whether it is correctly found
    ind_to_remove = 2
    data = store.layer.data
    store.layer.data = data[data[:, 0] != ind_to_remove]
    store._find_first_unlabeled_frame(event=None)
    assert store.current_step == ind_to_remove


def test_point_resize(viewer, points):
    viewer.layers.selection.add(points)
    layer = viewer.layers[0]
    controls = keypoints.QtPointsControls(layer)
    new_size = 10
    controls.changeSize(new_size)
    np.testing.assert_array_equal(points.size, new_size)


def test_add_unnanotated(store):
    store.layer.metadata["controls"].label_mode = 'loop'
    ind_to_remove = 0
    data = store.layer.data
    store.layer.data = data[data[:, 0] != ind_to_remove]
    store.viewer.dims.set_current_step(0, ind_to_remove)
    assert not store.annotated_keypoints
    n_points = store.layer.data.shape[0]
    keypoints._add(store, coord=(0, 1, 1))
    assert store.layer.data.shape[0] == n_points + 1
    assert store.current_step == ind_to_remove + 1


def test_add_quick(store):
    store.layer.metadata["controls"].label_mode = 'quick'
    store.current_keypoint = store._keypoints[0]
    coord = store.current_step, -1, -1
    keypoints._add(store, coord=coord)
    np.testing.assert_array_equal(
        store.layer.data[store.current_step], coord,
    )
