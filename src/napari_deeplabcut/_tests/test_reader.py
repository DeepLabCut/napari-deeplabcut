import cv2
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from skimage.io import imsave

from napari_deeplabcut import _reader

FAKE_EXTENSION = ".notanimage"


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


def test_read_images_mixed_extensions_list_input(tmp_path):
    """List input with mixed extensions (.jpg and .png) should stack and preserve order (natsorted)."""
    img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    p_jpg = tmp_path / "img1.jpg"
    p_png = tmp_path / "img2.png"
    imsave(p_jpg, img)
    imsave(p_png, img)

    layers = _reader.read_images([p_jpg, p_png])
    assert len(layers) == 1
    data, params, kind = layers[0]
    assert kind == "image"
    assert isinstance(data, da.Array)
    assert data.shape == (2, 10, 10, 3)

    # Ensure metadata paths include both files and are natsorted
    paths = params["metadata"]["paths"]
    assert len(paths) == 2
    assert paths[0].endswith("img1.jpg")
    assert paths[1].endswith("img2.png")


def test_read_images_mixed_extensions_directory_ignores_unsupported(tmp_path):
    """Directory with .jpg, .png and an unsupported .tif should only include supported images."""
    img = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    p_jpg = tmp_path / "a.jpg"
    p_png = tmp_path / "b.png"
    p_fake = tmp_path / f"c{FAKE_EXTENSION}"  # unsupported by SUPPORTED_IMAGES in this reader
    imsave(p_jpg, img)
    imsave(p_png, img)
    imsave(p_fake, img)

    layers = _reader.read_images(tmp_path)  # pass directory
    assert len(layers) == 1
    data, params, kind = layers[0]
    assert kind == "image"
    assert isinstance(data, da.Array)
    assert data.shape[0] == 2  # only jpg & png included

    # Check metadata "paths" reflect only supported files
    paths = params["metadata"]["paths"]
    assert len(paths) == 2
    assert any(p.endswith("a.jpg") for p in paths)
    assert any(p.endswith("b.png") for p in paths)
    assert all(not p.endswith(FAKE_EXTENSION) for p in paths)


def test_lazy_imread_mixed_extensions_list(tmp_path):
    """_lazy_imread / lazy_imread should support mixed extensions with stacking and RGB normalization."""
    img1 = (np.random.rand(12, 9, 3) * 255).astype(np.uint8)
    img2 = (np.random.rand(12, 9, 3) * 255).astype(np.uint8)
    p_jpg = tmp_path / "m1.jpg"
    p_png = tmp_path / "m2.png"
    imsave(p_jpg, img1)
    imsave(p_png, img2)

    result = _reader._lazy_imread([p_jpg, p_png], use_dask=True, stack=True)
    assert isinstance(result, da.Array)
    assert result.shape == (2, 12, 9, 3)
    first = result[0].compute()
    assert first.dtype == np.uint8
    assert first.shape == (12, 9, 3)


def test_read_images_mixed_extensions_globs(tmp_path):
    """Mixed-extension globs should be supported by passing a list of patterns."""
    img = (np.random.rand(7, 7, 3) * 255).astype(np.uint8)
    p1 = tmp_path / "g1.jpg"
    p2 = tmp_path / "g2.png"
    p3 = tmp_path / f"g3{FAKE_EXTENSION}"  # unsupported
    imsave(p1, img)
    imsave(p2, img)
    imsave(p3, img)

    layers = _reader.read_images([str(tmp_path / "*.jpg"), str(tmp_path / "*.png")])
    assert len(layers) == 1
    data, params, kind = layers[0]
    assert kind == "image"
    assert isinstance(data, da.Array)
    assert data.shape[0] == 2

    paths = params["metadata"]["paths"]
    assert len(paths) == 2
    assert any(p.endswith("g1.jpg") for p in paths)
    assert any(p.endswith("g2.png") for p in paths)
    assert all(not p.endswith(FAKE_EXTENSION) for p in paths)


def test_read_images_mixed_extensions_tuple_input(tmp_path):
    """Tuples should behave like lists for mixed extensions."""
    img = (np.random.rand(6, 6, 3) * 255).astype(np.uint8)
    p1 = tmp_path / "t1.jpg"
    p2 = tmp_path / "t2.png"
    imsave(p1, img)
    imsave(p2, img)

    layers = _reader.read_images((p1, p2))  # tuple
    assert len(layers) == 1
    data, params, kind = layers[0]
    assert kind == "image"
    assert isinstance(data, da.Array)
    assert data.shape == (2, 6, 6, 3)

    paths = params["metadata"]["paths"]
    assert len(paths) == 2
    assert paths[0].endswith("t1.jpg")
    assert paths[1].endswith("t2.png")


def test_lazy_imread_single_image(tmp_path):
    img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    path = tmp_path / "img.png"
    imsave(path, img)

    result = _reader._lazy_imread(path, use_dask=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape


def test_lazy_imread_multiple_images_equal_shape(tmp_path):
    img1 = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    img2 = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    path1 = tmp_path / "img1.png"
    path2 = tmp_path / "img2.png"
    imsave(path1, img1)
    imsave(path2, img2)

    result = _reader._lazy_imread([path1, path2], use_dask=True)
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
        _ = _reader._lazy_imread([path1, path2], use_dask=False, stack=True)


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


def test_read_images_single_glob_pattern(tmp_path):
    """A single glob pattern string (e.g., 'dir/*.png') should expand to multiple images."""
    img = (np.random.rand(5, 5, 3) * 255).astype(np.uint8)
    p1 = tmp_path / "g_a.png"
    p2 = tmp_path / "g_b.png"
    p3 = tmp_path / "g_c.jpg"  # different extension; should be excluded by '*.png' glob

    imsave(p1, img)
    imsave(p2, img)
    imsave(p3, img)

    layers = _reader.read_images(str(tmp_path / "*.png"))
    assert len(layers) == 1
    data, params, kind = layers[0]
    assert kind == "image"
    # Both .png images must be included
    assert data.shape[0] == 2
    paths = params["metadata"]["paths"]
    assert len(paths) == 2
    assert any(p.endswith("g_a.png") for p in paths)
    assert any(p.endswith("g_b.png") for p in paths)
    # Ensure the .jpg file is not included by the '*.png' glob
    assert all(not p.endswith("g_c.jpg") for p in paths)


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
    res = _reader._lazy_imread([p1, p2], use_dask=True, stack=False)
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
    gray = (np.random.rand(10, 10) * 255).astype(np.uint8)
    rgba = (np.random.rand(10, 10, 4) * 255).astype(np.uint8)
    p1, p2 = tmp_path / "g.png", tmp_path / "r.png"
    cv2.imwrite(str(p1), gray)  # cv2 writes BGR or grayscale by default
    cv2.imwrite(str(p2), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    res = _reader._lazy_imread([p1, p2], use_dask=False, stack=False)
    assert all(img.shape[-1] == 3 for img in res)


@pytest.mark.parametrize(
    "exists",
    [
        True,
        False,
    ],
)
def test_load_superkeypoints(monkeypatch, tmp_path, exists):
    """Test loading of superkeypoints JSON with and without the file present."""
    module_dir = tmp_path / "module"
    assets_dir = module_dir / "assets"
    assets_dir.mkdir(parents=True)

    super_animal = "fake"
    json_path = assets_dir / f"{super_animal}.json"

    if exists:
        json_path.write_text('{"SK1": [1, 2]}')

    # Patch module __file__
    fake_file = module_dir / "_reader_fake.py"
    fake_file.write_text("# fake module")
    monkeypatch.setattr("napari_deeplabcut._reader.__file__", str(fake_file))

    if exists:
        assert _reader._load_superkeypoints(super_animal) == {"SK1": [1, 2]}
    else:
        with pytest.raises(FileNotFoundError):
            _reader._load_superkeypoints(super_animal)


@pytest.mark.parametrize(
    "exists",
    [
        True,
        False,
    ],
)
def test_load_superkeypoints_diagram(monkeypatch, tmp_path, exists):
    """Test loading of superkeypoints diagram with and without the file present."""
    module_dir = tmp_path / "module"
    assets_dir = module_dir / "assets"
    assets_dir.mkdir(parents=True)

    super_animal = "fake"
    jpg_path = assets_dir / f"{super_animal}.jpg"

    if exists:
        Image.new("RGB", (10, 10), "white").save(jpg_path)

    # Patch module __file__
    fake_file = module_dir / "_reader_fake.py"
    fake_file.write_text("# fake")
    monkeypatch.setattr("napari_deeplabcut._reader.__file__", str(fake_file))

    if exists:
        array, meta, layer_type = _reader._load_superkeypoints_diagram(super_animal)
        assert layer_type == "images"
        assert meta == {"root": ""}
        assert tuple(array.shape[-3:-1]) == (10, 10)
    else:
        with pytest.raises(FileNotFoundError):
            _reader._load_superkeypoints_diagram(super_animal)
