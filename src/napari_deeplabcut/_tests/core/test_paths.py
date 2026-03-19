from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import napari_deeplabcut.core.paths as paths_mod


# -----------------------------------------------------------------------------
# canonicalize_path
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "n", "expected"),
    [
        # basic POSIX cases
        ("root/sub1/sub2/file.png", 3, "sub1/sub2/file.png"),
        ("root/sub/file.png", 2, "sub/file.png"),
        ("root/sub/file.png", 1, "file.png"),
        ("a/b/c", 10, "a/b/c"),
        (Path("a/b/c/d.txt"), 3, "b/c/d.txt"),
        # empty / degenerate inputs
        ("", 3, ""),
        (".", 3, ""),
        ("..", 3, ""),
        ("/", 3, ""),
        ("a/b/c/", 3, "a/b/c"),
        # non-string coercion
        (123, 3, "123"),
        # Windows / mixed separators
        (r"a\b\c\file.png", 3, "b/c/file.png"),
        (r"frames\\test\video0/img001.png", 3, "test/video0/img001.png"),
        # string-based normalization, not filesystem resolution
        ("./a/b/../c/d", 3, "b/c/d"),
        # invalid n
        ("a/b/c", 0, ValueError),
        ("a/b/c/d", -1, ValueError),
    ],
)
def test_canonicalize_path_contract(value, n, expected):
    is_exc_class = inspect.isclass(expected) and issubclass(expected, Exception)
    if is_exc_class:
        with pytest.raises(expected):
            paths_mod.canonicalize_path(value, n=n)
        return

    assert paths_mod.canonicalize_path(value, n=n) == expected


def test_canonicalize_path_removes_backslashes():
    out = paths_mod.canonicalize_path(r"a\b\c\file.png", n=3)
    assert "\\" not in out


def test_canonicalize_path_stringifies_objects_and_normalizes_separators():
    class Weird:
        def __str__(self):
            return r"x\y\z"

    out = paths_mod.canonicalize_path(Weird(), n=3)  # type: ignore[arg-type]
    assert out == "x/y/z"
    assert "\\" not in out


def test_canonicalize_path_returns_empty_string_when_stringify_fails():
    class BadPath:
        def __str__(self):
            raise RuntimeError("boom")

    assert paths_mod.canonicalize_path(BadPath(), n=3) == ""


# -----------------------------------------------------------------------------
# PathMatchPolicy / find_matching_depth
# -----------------------------------------------------------------------------


def test_path_match_policy_ordered_depths():
    assert paths_mod.PathMatchPolicy.ORDERED_DEPTHS.depths == (3, 2, 1)


def test_find_matching_depth_prefers_deepest_first_match():
    old_paths = [
        "/project/labeled-data/mouse1/img001.png",
        "/project/labeled-data/mouse1/img002.png",
    ]
    new_paths = [
        "/other/root/labeled-data/mouse1/img002.png",
        "/other/root/labeled-data/mouse1/img003.png",
    ]

    # exact overlap at depth=3 -> labeled-data/mouse1/img002.png
    assert paths_mod.find_matching_depth(old_paths, new_paths) == 3


def test_find_matching_depth_falls_back_to_shallower_depth():
    old_paths = ["/a/b/c/img001.png"]
    new_paths = ["/x/y/z/img001.png"]

    # depth=3 -> b/c/img001 vs y/z/img001  (no overlap)
    # depth=2 -> c/img001 vs z/img001      (no overlap)
    # depth=1 -> img001                    (overlap)
    assert paths_mod.find_matching_depth(old_paths, new_paths) == 1


def test_find_matching_depth_returns_none_when_no_overlap():
    old_paths = ["/a/b/c/img001.png"]
    new_paths = ["/x/y/z/img999.png"]

    assert paths_mod.find_matching_depth(old_paths, new_paths) is None


@pytest.mark.parametrize(
    ("old_paths", "new_paths"),
    [
        ([], ["/x/y/z/img001.png"]),
        (["/a/b/c/img001.png"], []),
        ([], []),
    ],
)
def test_find_matching_depth_returns_none_for_empty_inputs(old_paths, new_paths):
    assert paths_mod.find_matching_depth(old_paths, new_paths) is None


# -----------------------------------------------------------------------------
# config.yaml / DLC artifact heuristics
# -----------------------------------------------------------------------------


def test_is_config_yaml_true_only_for_existing_config_yaml(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.touch()

    other = tmp_path / "not_config.yaml"
    other.touch()

    assert paths_mod.is_config_yaml(cfg) is True
    assert paths_mod.is_config_yaml(other) is False
    assert paths_mod.is_config_yaml(tmp_path / "missing_config.yaml") is False


def test_is_config_yaml_returns_false_for_bad_input():
    assert paths_mod.is_config_yaml(None) is False


def test_has_dlc_datafiles_detects_collecteddata_and_machinelabels(tmp_path: Path):
    folder = tmp_path / "dataset"
    folder.mkdir()

    assert paths_mod.has_dlc_datafiles(folder) is False

    (folder / "CollectedData_Jane.h5").touch()
    assert paths_mod.has_dlc_datafiles(folder) is True

    # also cover another supported pattern
    other = tmp_path / "dataset2"
    other.mkdir()
    (other / "machinelabels_alex.csv").touch()
    assert paths_mod.has_dlc_datafiles(other) is True


def test_has_dlc_datafiles_returns_false_for_missing_or_non_directory(tmp_path: Path):
    missing = tmp_path / "missing"
    file_path = tmp_path / "a_file.txt"
    file_path.touch()

    assert paths_mod.has_dlc_datafiles(missing) is False
    assert paths_mod.has_dlc_datafiles(file_path) is False


def test_looks_like_dlc_labeled_folder_true_when_artifacts_present(tmp_path: Path):
    folder = tmp_path / "some_folder"
    folder.mkdir()
    (folder / "CollectedData_Jane.csv").touch()

    assert paths_mod.looks_like_dlc_labeled_folder(folder) is True


def test_looks_like_dlc_labeled_folder_true_inside_labeled_data(tmp_path: Path):
    folder = tmp_path / "project" / "labeled-data" / "mouse1"
    folder.mkdir(parents=True)

    assert paths_mod.looks_like_dlc_labeled_folder(folder) is True


def test_looks_like_dlc_labeled_folder_false_for_regular_folder(tmp_path: Path):
    folder = tmp_path / "images"
    folder.mkdir()

    assert paths_mod.looks_like_dlc_labeled_folder(folder) is False


def test_should_force_dlc_reader_true_for_config_yaml(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.touch()

    assert paths_mod.should_force_dlc_reader(cfg) is True


def test_should_force_dlc_reader_true_for_labeled_folder(tmp_path: Path):
    folder = tmp_path / "project" / "labeled-data" / "mouse1"
    folder.mkdir(parents=True)

    assert paths_mod.should_force_dlc_reader(folder) is True


def test_should_force_dlc_reader_false_for_empty_or_regular_inputs(tmp_path: Path):
    regular = tmp_path / "images"
    regular.mkdir()

    assert paths_mod.should_force_dlc_reader([]) is False
    assert paths_mod.should_force_dlc_reader(regular) is False
    assert paths_mod.should_force_dlc_reader([regular]) is False


# -----------------------------------------------------------------------------
# root-anchor inference
# -----------------------------------------------------------------------------


def test_infer_root_anchor_prefers_explicit_root(tmp_path: Path):
    opened = tmp_path / "project" / "labeled-data" / "mouse1"
    opened.mkdir(parents=True)

    explicit = tmp_path / "explicit-anchor"
    explicit.mkdir()

    assert paths_mod.infer_root_anchor(opened, explicit_root=explicit) == str(explicit)


def test_infer_root_anchor_returns_directory_when_opening_folder(tmp_path: Path):
    folder = tmp_path / "dataset"
    folder.mkdir()

    assert paths_mod.infer_root_anchor(folder) == str(folder)


def test_infer_root_anchor_returns_parent_when_opening_file(tmp_path: Path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    file_path = folder / "CollectedData_Jane.h5"
    file_path.touch()

    assert paths_mod.infer_root_anchor(file_path) == str(folder)


def test_infer_root_anchor_returns_none_for_missing_path(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    assert paths_mod.infer_root_anchor(missing) is None


# -----------------------------------------------------------------------------
# project-root discovery
# -----------------------------------------------------------------------------


def test_find_nearest_project_root_finds_config_in_current_directory(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yaml").touch()

    assert paths_mod.find_nearest_project_root(project) == str(project)


def test_find_nearest_project_root_finds_parent_project_from_nested_file(tmp_path: Path):
    project = tmp_path / "project"
    dataset = project / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)
    (project / "config.yaml").touch()

    img = dataset / "img001.png"
    img.touch()

    assert paths_mod.find_nearest_project_root(img) == str(project)


def test_find_nearest_project_root_respects_max_levels(tmp_path: Path):
    project = tmp_path / "project"
    deep = project / "a" / "b" / "c" / "d" / "e" / "f"
    deep.mkdir(parents=True)
    (project / "config.yaml").touch()

    # too shallow: cannot reach project root
    assert paths_mod.find_nearest_project_root(deep, max_levels=2) is None

    # enough levels: can reach project root
    assert paths_mod.find_nearest_project_root(deep, max_levels=6) == str(project)


def test_find_nearest_project_root_returns_none_when_no_config(tmp_path: Path):
    folder = tmp_path / "no_project_here"
    folder.mkdir()

    assert paths_mod.find_nearest_project_root(folder) is None


# -----------------------------------------------------------------------------
# anchor candidate selection
# -----------------------------------------------------------------------------


def test_choose_anchor_candidate_prefers_explicit_root(tmp_path: Path):
    opened = tmp_path / "project" / "labeled-data" / "mouse1"
    opened.mkdir(parents=True)

    explicit = tmp_path / "explicit-root"
    explicit.mkdir()

    result = paths_mod.choose_anchor_candidate(
        opened=opened,
        explicit_root=explicit,
        prefer_project_root=True,
    )

    assert result == str(explicit)


def test_choose_anchor_candidate_uses_inferred_anchor_by_default(tmp_path: Path):
    folder = tmp_path / "dataset"
    folder.mkdir()

    result = paths_mod.choose_anchor_candidate(opened=folder)
    assert result == str(folder)


def test_choose_anchor_candidate_can_elevate_to_project_root(tmp_path: Path):
    project = tmp_path / "project"
    dataset = project / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)
    (project / "config.yaml").touch()

    result = paths_mod.choose_anchor_candidate(
        opened=dataset,
        prefer_project_root=True,
    )

    assert result == str(project)


def test_choose_anchor_candidate_keeps_local_anchor_when_no_project_root_found(tmp_path: Path):
    dataset = tmp_path / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)

    result = paths_mod.choose_anchor_candidate(
        opened=dataset,
        prefer_project_root=True,
    )

    assert result == str(dataset)


def test_choose_anchor_candidate_returns_none_when_anchor_cannot_be_inferred(tmp_path: Path):
    missing = tmp_path / "missing"

    result = paths_mod.choose_anchor_candidate(
        opened=missing,
        prefer_project_root=True,
    )

    assert result is None


# -----------------------------------------------------------------------------
# artifact check wrapper
# -----------------------------------------------------------------------------


def test_anchor_contains_dlc_artifacts_true_when_datafiles_present(tmp_path: Path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    (folder / "CollectedData_Jane.h5").touch()

    assert paths_mod.anchor_contains_dlc_artifacts(folder) is True


def test_anchor_contains_dlc_artifacts_false_when_no_datafiles(tmp_path: Path):
    folder = tmp_path / "dataset"
    folder.mkdir()

    assert paths_mod.anchor_contains_dlc_artifacts(folder) is False


def test_anchor_contains_dlc_artifacts_swallows_exceptions(monkeypatch):
    monkeypatch.setattr(paths_mod, "has_dlc_datafiles", lambda anchor: (_ for _ in ()).throw(RuntimeError("boom")))

    assert paths_mod.anchor_contains_dlc_artifacts("/tmp/whatever") is False
