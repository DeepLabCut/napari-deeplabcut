import dask.array as da
import numpy as np
import pandas as pd
import pytest
from skimage.io import imsave

from napari_deeplabcut import _reader


@pytest.mark.parametrize("ext", _reader.SUPPORTED_IMAGES)
def test_get_image_reader(ext):
    path = f"fake_path/img{ext}"
    assert _reader.get_image_reader(path) is not None
    assert _reader.get_image_reader([path]) is not None


def test_get_config_reader(config_path):
    assert _reader.get_config_reader(str(config_path)) is not None


def test_get_config_reader_invalid_path():
    assert _reader.get_config_reader("fake_data.h5") is None


def test_get_folder_parser(tmp_path_factory, fake_keypoints):
    folder = tmp_path_factory.mktemp("folder")
    frame = (np.random.rand(10, 10) * 255).astype(np.uint8)
    imsave(folder / "img1.png", frame)
    imsave(folder / "img2.png", frame)
    layers = _reader.get_folder_parser(folder)(None)
    # There should be only an Image layer
    assert len(layers) == 1
    assert layers[0][-1] == "image"

    # Add an annotation data file
    fake_keypoints.to_hdf(folder / "data.h5", key="data")
    layers = _reader.get_folder_parser(folder)(None)
    # There should now be an additional Points layer
    assert len(layers) == 2
    assert layers[-1][-1] == "points"


def test_get_folder_parser_wrong_input():
    assert _reader.get_folder_parser("") is None


def test_get_folder_parser_no_images(tmp_path_factory):
    folder = str(tmp_path_factory.mktemp("images"))
    with pytest.raises(OSError):
        _reader.get_folder_parser(folder)


def test_read_images(tmp_path_factory, fake_image):
    folder = tmp_path_factory.mktemp("folder")
    path = str(folder / "img.png")
    imsave(path, fake_image)
    _ = _reader.read_images(path)[0]


def test_read_config(config_path):
    dict_ = _reader.read_config(config_path)[0][1]
    assert dict_["name"].startswith("CollectedData_")
    assert config_path.startswith(dict_["metadata"]["project"])


def test_read_hdf_old_index(tmp_path_factory, fake_keypoints):
    path = str(tmp_path_factory.mktemp("folder") / "data.h5")
    old_index = [f"labeled-data/video/img{i}.png" for i in range(fake_keypoints.shape[0])]
    fake_keypoints.index = old_index
    fake_keypoints.to_hdf(path, key="data")
    layers = _reader.read_hdf(path)
    assert len(layers) == 1
    image_paths = layers[0][1]["metadata"]["paths"]
    assert len(image_paths) == len(fake_keypoints)
    assert isinstance(image_paths[0], str)
    assert "labeled-data" in image_paths[0]


def test_read_hdf_new_index(tmp_path_factory, fake_keypoints):
    path = str(tmp_path_factory.mktemp("folder") / "data.h5")
    new_index = pd.MultiIndex.from_product(
        [
            ["labeled-data"],
            ["video"],
            [f"img{i}.png" for i in range(fake_keypoints.shape[0])],
        ]
    )
    fake_keypoints.index = new_index
    fake_keypoints.to_hdf(path, key="data")
    layers = _reader.read_hdf(path)
    assert len(layers) == 1
    image_paths = layers[0][1]["metadata"]["paths"]
    assert len(image_paths) == len(fake_keypoints)
    assert isinstance(image_paths[0], str)
    assert "labeled-data" in image_paths[0]


def test_lazy_imread_single_image(tmp_path):
    img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    path = tmp_path / "img.png"
    imsave(path, img)

    result = _reader.lazy_imread(path, use_dask=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape


def test_lazy_imread_multiple_images_equal_shape(tmp_path):
    img1 = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    img2 = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    path1 = tmp_path / "img1.png"
    path2 = tmp_path / "img2.png"
    imsave(path1, img1)
    imsave(path2, img2)

    result = _reader.lazy_imread([path1, path2], use_dask=True)
    assert isinstance(result, da.Array)
    assert result.shape == (2, 10, 10, 3)


def test_lazy_imread_mixed_shapes(tmp_path):
    img1 = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    img2 = (np.random.rand(12, 10, 3) * 255).astype(np.uint8)
    path1 = tmp_path / "img1.png"
    path2 = tmp_path / "img2.png"
    imsave(path1, img1)
    imsave(path2, img2)

    # Should fail when stacking mixed shapes without padding
    with pytest.raises(ValueError):
        _ = _reader.lazy_imread([path1, path2], use_dask=False, stack=True)


def test_read_images_list_input(tmp_path):
    img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    path1 = tmp_path / "img1.png"
    path2 = tmp_path / "img2.png"
    imsave(path1, img)
    imsave(path2, img)

    layers = _reader.read_images([path1, path2])
    assert len(layers) == 1
    data, params, kind = layers[0]
    assert kind == "image"
    assert "paths" in params["metadata"]
    assert isinstance(data, da.Array)
    assert data.shape[0] == 2


def test_read_images_empty_list():
    with pytest.raises(OSError):
        _reader.read_images([])


def test_video_init_and_properties(video_path):
    """Ensure Video object initializes and exposes correct properties."""
    vid = _reader.Video(video_path)

    assert len(vid) == 5  # number of frames created by fixture
    assert vid.width == 50
    assert vid.height == 50

    vid.close()
    assert not vid.stream.isOpened()


def test_video_read_single_frame(video_path):
    """Check that we can read at least one frame correctly."""
    vid = _reader.Video(video_path)
    vid.set_to_frame(0)
    frame = vid.read_frame()

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (50, 50, 3)
    assert frame.dtype == np.uint8

    vid.close()


def test_video_reader_invalid_path():
    """Invalid path should raise ValueError."""
    with pytest.raises(ValueError):
        _ = _reader.Video("")


def test_read_video_output(video_path):
    """Test the full read_video() API returns expected tuple structure."""
    layers = _reader.read_video(video_path)
    assert len(layers) == 1

    data, params, kind = layers[0]

    # basic structure
    assert kind == "image"
    assert isinstance(data, da.Array)
    assert "root" in params["metadata"]

    # shape & dtype
    assert data.shape[0] == 5  # number of frames
    frame = data[0].compute()
    assert frame.shape == (50, 50, 3)
    assert frame.dtype == np.uint8


def test_get_video_reader_dispatch(video_path):
    assert _reader.get_video_reader(video_path) is not None
    assert _reader.is_video(str(video_path))
    assert _reader.get_video_reader("file.txt") is None
    assert not _reader.is_video("file.png")


def test_lazy_imread_list_no_stack(tmp_path):
    img = (np.random.rand(8, 9, 3) * 255).astype(np.uint8)
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    imsave(p1, img)
    imsave(p2, img)
    res = _reader.lazy_imread([p1, p2], use_dask=True, stack=False)
    assert isinstance(res, list) and len(res) == 2
    assert all(isinstance(x, da.Array) for x in res)


def test_read_images_list_metadata_paths(tmp_path):
    img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    p1, p2 = tmp_path / "img1.png", tmp_path / "img2.png"
    imsave(p1, img)
    imsave(p2, img)
    [(data, params, kind)] = _reader.read_images([p2, p1])  # unordered input
    assert params["metadata"]["paths"]  # exists
    assert len(params["metadata"]["paths"]) == 2
    # natsorted is applied; assert the order is deterministic by name
    assert params["metadata"]["paths"][0].endswith("img1.png")
    assert params["metadata"]["paths"][1].endswith("img2.png")


def test_lazy_imread_grayscale_and_rgba(tmp_path):
    import cv2

    gray = (np.random.rand(10, 10) * 255).astype(np.uint8)
    rgba = (np.random.rand(10, 10, 4) * 255).astype(np.uint8)
    p1, p2 = tmp_path / "g.png", tmp_path / "r.png"
    cv2.imwrite(str(p1), gray)  # cv2 writes BGR or grayscale by default
    cv2.imwrite(str(p2), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    res = _reader.lazy_imread([p1, p2], use_dask=False, stack=False)
    assert all(img.shape[-1] == 3 for img in res)
