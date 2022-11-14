import os
import pandas as pd
import pytest
from napari_deeplabcut import misc, _reader


def test_unsorted_unique_numeric():
    seq = [4, 3, 2, 1, 0, 1, 2, 3, 4, 5]
    out = misc.unsorted_unique(seq)
    assert list(out) == [4, 3, 2, 1, 0, 5]


def test_unsorted_unique_string():
    seq = ["c", "b", "d", "b", "a", "b"]
    out = misc.unsorted_unique(seq)
    assert list(out) == ["c", "b", "d", "a"]


def test_encode_categories():
    categories = list("abcdabcd")
    inds, map_ = misc.encode_categories(categories, return_map=True)
    assert list(inds) == [0, 1, 2, 3, 0, 1, 2, 3]
    assert map_ == dict(zip(list("abcd"), range(4)))
    inds = misc.encode_categories(categories, return_map=False)


@pytest.mark.parametrize(
    "path",
    ["/home/to/fake/path", "C:\\Users\\with\\fake\\name"],
)
def test_to_os_dir_sep(path):
    sep_wrong = "\\" if os.path.sep == "/" else "/"
    assert sep_wrong not in misc.to_os_dir_sep(path)


def test_to_os_dir_sep_invalid():
    with pytest.raises(ValueError):
        misc.to_os_dir_sep("/home\\home")


def test_guarantee_multiindex_rows():
    fake_index = [
        f"labeled-data/subfolder_{i}/image_{j}"
        for i in range(3) for j in range(10)
    ]
    df = pd.DataFrame(index=fake_index)
    misc.guarantee_multiindex_rows(df)
    assert isinstance(df.index, pd.MultiIndex)

    # Substitute index with frame numbers
    frame_numbers = list(range(df.shape[0]))
    df.index = frame_numbers
    misc.guarantee_multiindex_rows(df)
    assert df.index.to_list() == frame_numbers


@pytest.mark.parametrize("n_colors", range(1, 11))
def test_build_color_cycle(n_colors):
    color_cycle = misc.build_color_cycle(n_colors)
    assert color_cycle.shape[0] == n_colors
    # Test whether all colors are different
    assert len(set(map(tuple, color_cycle))) == n_colors


def test_dlc_header():
    n_animals = 2
    n_keypoints = 3
    scorer = "me"
    animals = [f"animal_{n}" for n in range(n_animals)]
    keypoints = [f"kpt_{n}" for n in range(n_keypoints)]
    fake_columns = pd.MultiIndex.from_product([
        [scorer], animals, keypoints, ["x", "y", "likelihood"],
    ], names=["scorer", "individuals", "bodyparts", "coords"])
    header = misc.DLCHeader(fake_columns)
    assert header.scorer == scorer
    header.scorer = "you"
    assert header.scorer == "you"
    assert header.individuals == animals
    assert header.bodyparts == keypoints
    assert header.coords == ["x", "y", "likelihood"]


def test_dlc_header_from_config_multi(config_path):
    config = _reader._load_config(config_path)
    config["multianimalproject"] = True
    config["individuals"] = ["animal"]
    config["multianimalbodyparts"] = list("abc")
    config["uniquebodyparts"] = list("de")
    header = misc.DLCHeader.from_config(config)
    assert header.individuals != [""]


def test_cycle_enum():
    enum = misc.CycleEnum("Test", list("AB"))
    assert next(enum).value == "a"
    assert next(enum).value == "b"
    assert next(enum).value == "a"
    assert next(enum).value == "b"
    assert enum["a"] == enum.A
