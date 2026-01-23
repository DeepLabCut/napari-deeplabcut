from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_deeplabcut import _reader, misc


def test_unsorted_unique_numeric():
    seq = [4, 3, 2, 1, 0, 1, 2, 3, 4, 5]
    out = misc.unsorted_unique(seq)
    assert list(out) == [4, 3, 2, 1, 0, 5]


def test_unsorted_unique_string():
    seq = ["c", "b", "d", "b", "a", "b"]
    out = misc.unsorted_unique(seq)
    assert list(out) == ["c", "b", "d", "a"]


# Tests for new OS-agnostic path canonicalization in misc.canonicalize_path and misc.encode_categories
def test_canonicalize_path():
    p = "root/sub1/sub2/file.png"
    assert misc.canonicalize_path(p) == "sub1/sub2/file.png"
    p = "root/sub/file.png"
    assert misc.canonicalize_path(p, n=2) == "sub/file.png"
    p = "root/sub/file.png"
    assert misc.canonicalize_path(p, n=1) == "file.png"
    p = "a/b/c"
    assert misc.canonicalize_path(p, n=10) == "a/b/c"
    p = Path("a/b/c/d.txt")
    assert misc.canonicalize_path(p, n=3) == "b/c/d.txt"
    assert misc.canonicalize_path("") == ""
    assert misc.canonicalize_path(".") == ""
    assert misc.canonicalize_path("/") == ""
    p = "a/b/c/"
    # Path("a/b/c/") collapses trailing slash; last 3 parts are a/b/c
    assert misc.canonicalize_path(p, n=3) == "a/b/c"
    # parts[-0:] is equivalent to parts[:], so full path returned
    p = "a/b/c"
    assert misc.canonicalize_path(p, n=0) == "a/b/c"
    p = "a/b/c/d"
    assert misc.canonicalize_path(p, n=-1) == "b/c/d"
    # Path(123) raises TypeError -> fallback to str(p)
    out = misc.canonicalize_path(123, n=3)  # type: ignore[arg-type]
    assert out == "123"


def test_canonicalize_path_converts_backslashes():
    # Important: behavior depends on Path.parts semantics of the running OS.
    p = r"a\b\c\file.png"
    out = misc.canonicalize_path(p, n=3)

    # Always normalize separators to POSIX style
    assert "\\" not in out

    # Portable expected result: mirror "last n parts" logic + normalization
    parts = Path(p).parts
    expected = str(Path(*parts[-3:])).replace("\\", "/")
    assert out == expected


def test_canonicalize_path_mixed_separators_normalized_to_posix():
    # '/' splits into parts on POSIX; '\ ' stays within a component but gets replaced.
    p = r"frames\\test\video0/img001.png"
    out = misc.canonicalize_path(p, n=3)
    assert "\\" not in out
    assert out.endswith("video0/img001.png")


def test_canonicalize_path_exception_fallback_still_replaces_backslashes():
    class Weird:
        def __str__(self):
            return r"x\y\z"

    out = misc.canonicalize_path(Weird(), n=3)  # type: ignore[arg-type]
    assert out == "x/y/z"
    assert "\\" not in out


# encode_categories tests utils
def _expected_unique(categories, *, is_path, do_sort):
    """Compute expected unique list according to encode_categories semantics."""
    if is_path:
        cats = [misc.canonicalize_path(c) for c in categories]
    else:
        cats = list(categories)

    # stable unique in first-seen order
    unique = list(dict.fromkeys(cats))

    if do_sort:
        # mimic natural sort used in misc.encode_categories
        from natsort import natsorted

        unique = natsorted(unique)

    return cats, unique


def _expected_inds(cats, unique):
    m = {k: i for i, k in enumerate(unique)}
    return np.array([m[c] for c in cats], dtype=int)


@pytest.mark.parametrize("return_map", [False, True])
@pytest.mark.parametrize("is_path", [False, True])
@pytest.mark.parametrize("do_sort", [False, True])
def test_encode_categories_all_branches_basic(return_map, is_path, do_sort):
    """
    Full branch coverage across return_map/is_path/do_sort
    """
    categories = list("abcdabcd")

    cats, unique_expected = _expected_unique(categories, is_path=is_path, do_sort=do_sort)
    inds_expected = _expected_inds(cats, unique_expected)

    if return_map:
        inds, unique = misc.encode_categories(categories, return_unique=True, is_path=is_path, do_sort=do_sort)
        assert isinstance(unique, list)
        assert unique == unique_expected
        assert np.array_equal(inds, inds_expected)
    else:
        inds = misc.encode_categories(categories, return_unique=False, is_path=is_path, do_sort=do_sort)
        assert np.array_equal(inds, inds_expected)

    # dtype guarantee
    assert isinstance(inds_expected, np.ndarray)
    if return_map:
        assert isinstance(inds, np.ndarray)
        assert inds.dtype == int


@pytest.mark.parametrize("do_sort", [False, True])
def test_encode_categories_return_map_consistency(do_sort):
    """
    Ensures inds returned with return_map=False matches inds from return_map=True
    (for the same is_path/do_sort settings).
    """
    categories = ["b", "a", "b", "a", "c"]

    inds_a, unique = misc.encode_categories(categories, return_unique=True, is_path=False, do_sort=do_sort)
    inds_b = misc.encode_categories(categories, return_unique=False, is_path=False, do_sort=do_sort)
    assert np.array_equal(inds_a, inds_b)

    # sanity: unique produces a valid mapping
    m = {k: i for i, k in enumerate(unique)}
    assert list(inds_a) == [m[c] for c in categories]


def test_encode_categories_do_sort_changes_indexing():
    """
    Verifies the behavioral difference between do_sort=True and do_sort=False.
    """
    categories = ["b", "a", "b", "a"]

    inds_sorted, unique_sorted = misc.encode_categories(categories, return_unique=True, is_path=False, do_sort=True)
    assert unique_sorted == ["a", "b"]
    assert list(inds_sorted) == [1, 0, 1, 0]

    inds_unsorted, unique_unsorted = misc.encode_categories(
        categories, return_unique=True, is_path=False, do_sort=False
    )
    assert unique_unsorted == ["b", "a"]  # first-seen stable order
    assert list(inds_unsorted) == [0, 1, 0, 1]


def test_encode_categories_natural_sort_img2_before_img10():
    """
    Tests natural sorting (natsort): img2 comes before img10 when do_sort=True.
    """
    categories = ["img10.png", "img2.png", "img1.png"]

    inds, unique = misc.encode_categories(categories, return_unique=True, is_path=False, do_sort=True)
    assert unique == ["img1.png", "img2.png", "img10.png"]

    m = {k: i for i, k in enumerate(unique)}
    assert list(inds) == [m["img10.png"], m["img2.png"], m["img1.png"]]


@pytest.mark.parametrize(
    "categories",
    [
        # mixed OS separators
        [r"C:\data\frames\test\img001.png", r"/data/frames/test/img002.png", r"C:\data\frames\test\img001.png"],
        # Path objects
        [Path("/data/frames/test/img010.png"), Path("/data/frames/test/img002.png")],
    ],
)
def test_encode_categories_path_canonicalization(categories):
    """
    Ensures canonicalization normalizes separators and retains the last components
    (default canonicalize_path n=3).
    """
    inds, unique = misc.encode_categories(categories, return_unique=True, is_path=True, do_sort=True)

    # Canonical keys should use POSIX separators and keep last 3 components.
    # e.g. frames/test/img001.png
    assert all("\\" not in u for u in unique)
    assert all(len(Path(u).parts) <= 3 for u in unique)

    # inds should reference the unique list correctly
    m = {k: i for i, k in enumerate(unique)}
    canon = [misc.canonicalize_path(c) for c in categories]
    assert list(inds) == [m[c] for c in canon]


def test_encode_categories_is_path_false_does_not_canonicalize():
    """
    If is_path=False, we do not canonicalize.
    Mixed separators remain distinct categories.
    """
    categories = [r"frames\test\img001.png", "frames/test/img001.png"]
    inds, unique = misc.encode_categories(categories, return_unique=True, is_path=False, do_sort=False)

    assert unique == categories  # distinct, first-seen order preserved
    assert list(inds) == [0, 1]


def test_encode_categories_empty_input():
    """
    Empty categories should return an empty indices array (and empty unique list if requested).
    """
    categories = []
    inds = misc.encode_categories(categories, return_unique=False)
    assert isinstance(inds, np.ndarray)
    assert inds.dtype == int
    assert inds.size == 0

    inds2, unique = misc.encode_categories(categories, return_unique=True)
    assert inds2.size == 0
    assert unique == []


@pytest.mark.parametrize("is_path", [False, True])
@pytest.mark.parametrize("do_sort", [False, True])
def test_encode_categories_numeric_categories(is_path, do_sort):
    """
    Non-string categories should still work.
    When is_path=True, canonicalize_path() falls back to str(p) on exceptions.
    """
    categories = [10, 2, 10, 3]

    inds, unique = misc.encode_categories(categories, return_unique=True, is_path=is_path, do_sort=do_sort)

    # unique elements should be stringified if is_path=True (due to canonicalize_path fallback)
    if is_path:
        assert all(isinstance(u, str) for u in unique)
    else:
        # could be ints directly
        assert all(isinstance(u, int) for u in unique)

    # verify mapping consistency
    m = {k: i for i, k in enumerate(unique)}
    if is_path:
        cats = [misc.canonicalize_path(c) for c in categories]
    else:
        cats = categories
    assert list(inds) == [m[c] for c in cats]


def test_merge_multiple_scorers_no_likelihood(fake_keypoints):
    temp = fake_keypoints.copy(deep=True)
    temp.columns = temp.columns.set_levels(["you"], level="scorer")
    df = fake_keypoints.merge(temp, left_index=True, right_index=True)
    df = misc.merge_multiple_scorers(df)
    pd.testing.assert_frame_equal(df, fake_keypoints)


def test_merge_multiple_scorers(fake_keypoints):
    new_columns = pd.MultiIndex.from_product(
        fake_keypoints.columns.levels[:-1] + [["x", "y", "likelihood"]],
        names=fake_keypoints.columns.names,
    )
    fake_keypoints = fake_keypoints.reindex(new_columns, axis=1)
    fake_keypoints.loc(axis=1)[:, :, :, "likelihood"] = 1
    temp = fake_keypoints.copy(deep=True)
    temp.columns = temp.columns.set_levels(["you"], level="scorer")
    fake_keypoints.iloc[:5] = np.nan
    temp.iloc[5:] = np.nan
    df = fake_keypoints.merge(temp, left_index=True, right_index=True)
    df = misc.merge_multiple_scorers(df)
    pd.testing.assert_index_equal(df.columns, fake_keypoints.columns)
    assert not df.isna().any(axis=None)


def test_guarantee_multiindex_rows():
    fake_index = [f"labeled-data/subfolder_{i}/image_{j}" for i in range(3) for j in range(10)]
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
    fake_columns = pd.MultiIndex.from_product(
        [
            [scorer],
            animals,
            keypoints,
            ["x", "y", "likelihood"],
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
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
