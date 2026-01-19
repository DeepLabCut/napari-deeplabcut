import numpy as np
import pandas as pd
import pytest
from PIL import Image
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


def test_video(video_path):
    video = _reader.Video(video_path)
    video.close()
    assert not video.stream.isOpened()


def test_video_wrong_path():
    with pytest.raises(ValueError):
        _ = _reader.Video("")


def test_read_video(video_path):
    array, dict_ = _reader.read_video(video_path)[0]
    assert dict_["metadata"].get("root")
    assert array.shape[0] == 5
    assert array[0].compute().shape == (50, 50, 3)


def test_load_superkeypoints(superkeypoints_json, monkeypatch):
    module_dir = superkeypoints_json["module_dir"]
    super_animal = superkeypoints_json["super_animal"]
    expected = superkeypoints_json["data"]

    # Fake the module's __file__
    fake_module_file = module_dir / "_reader_fake.py"
    fake_module_file.write_text("# fake module")
    monkeypatch.setattr("napari_deeplabcut._reader.__file__", str(fake_module_file))

    out = _reader._load_superkeypoints(super_animal)
    assert out == expected


def test_load_superkeypoints_diagram(superkeypoints_json, monkeypatch):
    module_dir = superkeypoints_json["module_dir"]
    assets_dir = superkeypoints_json["assets_dir"]
    super_animal = superkeypoints_json["super_animal"]

    # Create diagram file in module_dir/assets/

    img = Image.new("RGB", (10, 10), "white")
    img.save(assets_dir / f"{super_animal}.jpg")

    # Patch module __file__ to point to module_dir
    fake_module_file = module_dir / "_reader_fake.py"
    fake_module_file.write_text("# fake")
    monkeypatch.setattr("napari_deeplabcut._reader.__file__", str(fake_module_file))

    # Run it
    array, meta, layer_type = _reader._load_superkeypoints_diagram(super_animal)

    assert layer_type == "images"
    assert meta == {"root": ""}
    assert tuple(array.shape[-3:-1]) == (10, 10)
