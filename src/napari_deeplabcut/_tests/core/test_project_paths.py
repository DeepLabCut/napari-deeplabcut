from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import napari_deeplabcut.core.project_paths as paths_mod
from napari_deeplabcut.core.project_paths import (
    coerce_paths_to_dlc_row_keys,
    dataset_folder_has_files,
    infer_dlc_project_from_config,
    infer_dlc_project_from_labeled_folder,
    resolve_project_root_from_config,
    target_dataset_folder_for_config,
)


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
# normalize_anchor_candidate
# -----------------------------------------------------------------------------
def test_normalize_anchor_candidate_returns_directory_for_directory(tmp_path: Path):
    folder = tmp_path / "dataset"
    folder.mkdir()

    assert paths_mod.normalize_anchor_candidate(folder) == folder.resolve()


def test_normalize_anchor_candidate_returns_parent_for_file(tmp_path: Path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    file_path = folder / "CollectedData_Jane.h5"
    file_path.touch()

    assert paths_mod.normalize_anchor_candidate(file_path) == folder.resolve()


def test_normalize_anchor_candidate_returns_path_for_missing_path(tmp_path: Path):
    missing = tmp_path / "does_not_exist"

    result = paths_mod.normalize_anchor_candidate(missing)
    assert result == missing.resolve()


def test_normalize_anchor_candidate_returns_none_for_none():
    assert paths_mod.normalize_anchor_candidate(None) is None


# -----------------------------------------------------------------------------
# find_nearest_config
# -----------------------------------------------------------------------------
def test_find_nearest_config_finds_config_in_current_directory(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    cfg = project / "config.yaml"
    cfg.touch()

    assert paths_mod.find_nearest_config(project) == cfg.resolve()


def test_find_nearest_config_finds_parent_project_from_nested_file(tmp_path: Path):
    project = tmp_path / "project"
    dataset = project / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)
    cfg = project / "config.yaml"
    cfg.touch()

    img = dataset / "img001.png"
    img.touch()

    assert paths_mod.find_nearest_config(img) == cfg.resolve()


def test_find_nearest_config_respects_max_levels(tmp_path: Path):
    project = tmp_path / "project"
    deep = project / "a" / "b" / "c" / "d" / "e" / "f"
    deep.mkdir(parents=True)
    cfg = project / "config.yaml"
    cfg.touch()

    assert paths_mod.find_nearest_config(deep, max_levels=2) is None
    assert paths_mod.find_nearest_config(deep, max_levels=6) == cfg.resolve()


def test_find_nearest_config_returns_none_when_no_config(tmp_path: Path):
    folder = tmp_path / "no_project_here"
    folder.mkdir()

    assert paths_mod.find_nearest_config(folder) is None


# -----------------------------------------------------------------------------
# infer_labeled_data_folder_from_paths
# -----------------------------------------------------------------------------
def test_infer_labeled_data_folder_from_paths_uses_fallback_root_when_already_dataset(tmp_path: Path):
    dataset = tmp_path / "project" / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)

    result = paths_mod.infer_labeled_data_folder_from_paths(
        [],
        fallback_root=dataset,
    )

    assert result == dataset.resolve()


def test_infer_labeled_data_folder_from_paths_builds_folder_from_project_and_relpaths(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()

    result = paths_mod.infer_labeled_data_folder_from_paths(
        ["labeled-data/mouse1/img001.png"],
        project_root=project,
    )

    assert result == (project / "labeled-data" / "mouse1").resolve()


def test_infer_labeled_data_folder_from_paths_returns_none_without_dataset_name(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()

    result = paths_mod.infer_labeled_data_folder_from_paths(
        ["images/img001.png"],
        project_root=project,
    )

    assert result is None


def test_infer_labeled_data_folder_from_paths_returns_none_without_project_root():
    result = paths_mod.infer_labeled_data_folder_from_paths(
        ["labeled-data/mouse1/img001.png"],
        project_root=None,
    )

    assert result is None


# -----------------------------------------------------------------------------
# infer_dlc_project
# -----------------------------------------------------------------------------
def test_infer_dlc_project_prefers_explicit_root_and_finds_config(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    cfg = project / "config.yaml"
    cfg.touch()

    other = tmp_path / "other"
    other.mkdir()

    ctx = paths_mod.infer_dlc_project(
        explicit_root=project,
        anchor_candidates=[other],
        prefer_project_root=True,
    )

    assert ctx.project_root == project.resolve()
    assert ctx.config_path == cfg.resolve()
    assert ctx.root_anchor == project.resolve()


def test_infer_dlc_project_keeps_local_anchor_when_prefer_project_root_false(tmp_path: Path):
    project = tmp_path / "project"
    dataset = project / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)
    cfg = project / "config.yaml"
    cfg.touch()

    ctx = paths_mod.infer_dlc_project(
        anchor_candidates=[dataset],
        prefer_project_root=False,
    )

    assert ctx.project_root == project.resolve()
    assert ctx.config_path == cfg.resolve()
    assert ctx.root_anchor == dataset.resolve()


def test_infer_dlc_project_returns_best_effort_without_config(tmp_path: Path):
    dataset = tmp_path / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)

    ctx = paths_mod.infer_dlc_project(
        anchor_candidates=[dataset],
        prefer_project_root=True,
    )

    assert ctx.project_root is None
    assert ctx.config_path is None
    assert ctx.root_anchor == dataset.resolve()


def test_infer_dlc_project_uses_dataset_candidate_when_no_anchor_candidates(tmp_path: Path):
    dataset = tmp_path / "project" / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)

    ctx = paths_mod.infer_dlc_project(
        dataset_candidates=[dataset],
    )

    assert ctx.dataset_folder == dataset.resolve()
    assert ctx.root_anchor == dataset.resolve()


# -----------------------------------------------------------------------------
# infer_dlc_project_from_opened
# -----------------------------------------------------------------------------
def test_infer_dlc_project_from_opened_uses_opened_path(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    ctx = paths_mod.infer_dlc_project_from_opened(dataset)

    assert ctx.root_anchor == dataset.resolve()
    assert ctx.project_root is None
    assert ctx.config_path is None


def test_infer_dlc_project_from_opened_can_find_project_root(tmp_path: Path):
    project = tmp_path / "project"
    dataset = project / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)
    cfg = project / "config.yaml"
    cfg.touch()

    ctx = paths_mod.infer_dlc_project_from_opened(dataset)

    assert ctx.project_root == project.resolve()
    assert ctx.config_path == cfg.resolve()
    assert ctx.root_anchor == project.resolve()


# -----------------------------------------------------------------------------
# infer_dlc_project_from_points_meta
# -----------------------------------------------------------------------------
def test_infer_dlc_project_from_points_meta_infers_dataset_and_project(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    cfg = project / "config.yaml"
    cfg.touch()

    pts_meta = SimpleNamespace(
        project=str(project),
        root=None,
        paths=["labeled-data/mouse1/img001.png"],
    )

    ctx = paths_mod.infer_dlc_project_from_points_meta(pts_meta)

    assert ctx.project_root == project.resolve()
    assert ctx.config_path == cfg.resolve()
    assert ctx.dataset_folder == (project / "labeled-data" / "mouse1").resolve()
    assert ctx.root_anchor == project.resolve()


def test_infer_dlc_project_from_points_meta_uses_root_as_dataset_fallback(tmp_path: Path):
    dataset = tmp_path / "project" / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)

    pts_meta = SimpleNamespace(
        project=None,
        root=str(dataset),
        paths=[],
    )

    ctx = paths_mod.infer_dlc_project_from_points_meta(pts_meta)

    assert ctx.dataset_folder == dataset.resolve()
    assert ctx.root_anchor == dataset.resolve()


# -----------------------------------------------------------------------------
# infer_dlc_project_from_image_layer
# -----------------------------------------------------------------------------
def test_infer_dlc_project_from_image_layer_uses_metadata_project(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    cfg = project / "config.yaml"
    cfg.touch()

    layer = SimpleNamespace(
        metadata={"project": str(project)},
        source=SimpleNamespace(path=None),
    )

    ctx = paths_mod.infer_dlc_project_from_image_layer(layer)

    assert ctx.project_root == project.resolve()
    assert ctx.config_path == cfg.resolve()
    assert ctx.root_anchor == project.resolve()


def test_infer_dlc_project_from_image_layer_falls_back_to_source_path(tmp_path: Path):
    project = tmp_path / "project"
    videos = project / "videos"
    videos.mkdir(parents=True)
    cfg = project / "config.yaml"
    cfg.touch()

    video = videos / "demo.mp4"
    video.touch()

    layer = SimpleNamespace(
        metadata={},
        source=SimpleNamespace(path=str(video)),
    )

    ctx = paths_mod.infer_dlc_project_from_image_layer(layer)

    assert ctx.project_root == project.resolve()
    assert ctx.config_path == cfg.resolve()
    assert ctx.root_anchor == project.resolve()


def test_infer_dlc_project_from_image_layer_returns_best_effort_without_config(tmp_path: Path):
    videos = tmp_path / "videos"
    videos.mkdir()

    video = videos / "demo.mp4"
    video.touch()

    layer = SimpleNamespace(
        metadata={},
        source=SimpleNamespace(path=str(video)),
    )

    ctx = paths_mod.infer_dlc_project_from_image_layer(layer)

    assert ctx.project_root is None
    assert ctx.config_path is None
    assert ctx.root_anchor == videos.resolve()


def test_resolve_project_root_from_config(tmp_path):
    project = tmp_path / "my-project"
    project.mkdir()
    cfg = project / "config.yaml"
    cfg.write_text("scorer: test\n", encoding="utf-8")

    assert resolve_project_root_from_config(cfg) == project
    assert resolve_project_root_from_config(project / "not_config.yaml") is None
    assert resolve_project_root_from_config(project / "config.yml") is None
    assert resolve_project_root_from_config(project / "missing" / "config.yaml") is None


def test_coerce_paths_to_dlc_row_keys_for_projectless_folder(tmp_path):
    source_root = tmp_path / "session_42"
    source_root.mkdir()

    inside_abs = source_root / "img001.png"
    nested_abs = source_root / "nested" / "img_nested.png"
    nested_abs.parent.mkdir()
    outside_abs = tmp_path / "elsewhere" / "img999.png"
    outside_abs.parent.mkdir()

    rewritten, unresolved = coerce_paths_to_dlc_row_keys(
        [
            inside_abs,
            "img002.png",
            "labeled-data\\session_42\\img003.png",
            nested_abs,
            outside_abs,
        ],
        source_root=source_root,
    )

    assert rewritten == [
        "labeled-data/session_42/img001.png",
        "labeled-data/session_42/img002.png",
        "labeled-data/session_42/img003.png",
        nested_abs.as_posix(),
        outside_abs.as_posix(),
    ]
    assert unresolved == (3, 4)


def test_target_dataset_folder_and_existing_files_guard(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    cfg = project / "config.yaml"
    cfg.write_text("scorer: John\n", encoding="utf-8")

    target = target_dataset_folder_for_config(cfg, dataset_name="session_42")
    assert target == project / "labeled-data" / "session_42"
    assert not dataset_folder_has_files(target)

    target.mkdir(parents=True)
    assert not dataset_folder_has_files(target)

    (target / "img001.png").write_bytes(b"x")
    assert dataset_folder_has_files(target)


def test_infer_dlc_project_from_config_returns_explicit_project_context(tmp_path):
    project = tmp_path / "my-project"
    project.mkdir()

    config_path = project / "config.yaml"
    config_path.write_text("scorer: John\n", encoding="utf-8")

    ctx = infer_dlc_project_from_config(config_path)

    assert ctx.root_anchor == project
    assert ctx.project_root == project
    assert ctx.config_path == config_path
    assert ctx.dataset_folder is None


def test_infer_dlc_project_from_config_rejects_invalid_path(tmp_path):
    project = tmp_path / "my-project"
    project.mkdir()

    bad_config = project / "not_config.yaml"
    bad_config.write_text("scorer: John\n", encoding="utf-8")

    with pytest.raises(ValueError):
        infer_dlc_project_from_config(bad_config)


# -----------------------------------------------------------------------------
# infer_dlc_project_from_labeled_folder
# -----------------------------------------------------------------------------
def test_infer_dlc_project_from_labeled_folder_without_config_uses_dataset_as_anchor(
    tmp_path: Path,
):
    dataset = tmp_path / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)

    ctx = infer_dlc_project_from_labeled_folder(dataset)

    assert ctx.dataset_folder == dataset.resolve()
    assert ctx.root_anchor == dataset.resolve()
    assert ctx.project_root is None
    assert ctx.config_path is None


def test_infer_dlc_project_from_labeled_folder_with_config_prefers_project_root(
    tmp_path: Path,
):
    project = tmp_path / "project"
    dataset = project / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)

    cfg = project / "config.yaml"
    cfg.touch()

    ctx = infer_dlc_project_from_labeled_folder(
        dataset,
        prefer_project_root=True,
    )

    assert ctx.dataset_folder == dataset.resolve()
    assert ctx.project_root == project.resolve()
    assert ctx.config_path == cfg.resolve()
    assert ctx.root_anchor == project.resolve()


def test_infer_dlc_project_from_labeled_folder_with_config_can_keep_dataset_anchor(
    tmp_path: Path,
):
    project = tmp_path / "project"
    dataset = project / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)

    cfg = project / "config.yaml"
    cfg.touch()

    ctx = infer_dlc_project_from_labeled_folder(
        dataset,
        prefer_project_root=False,
    )

    assert ctx.dataset_folder == dataset.resolve()
    assert ctx.project_root == project.resolve()
    assert ctx.config_path == cfg.resolve()
    assert ctx.root_anchor == dataset.resolve()


def test_infer_dlc_project_from_labeled_folder_respects_max_levels(tmp_path: Path):
    project = tmp_path / "project"
    dataset = project / "a" / "b" / "c" / "labeled-data" / "mouse1"
    dataset.mkdir(parents=True)

    cfg = project / "config.yaml"
    cfg.touch()

    # Too shallow: should not reach project/config.yaml
    ctx = infer_dlc_project_from_labeled_folder(dataset, max_levels=2)
    assert ctx.dataset_folder == dataset.resolve()
    assert ctx.root_anchor == dataset.resolve()
    assert ctx.project_root is None
    assert ctx.config_path is None

    # Deep enough: should find config and elevate to project root by default
    ctx = infer_dlc_project_from_labeled_folder(dataset, max_levels=5)
    assert ctx.dataset_folder == dataset.resolve()
    assert ctx.project_root == project.resolve()
    assert ctx.config_path == cfg.resolve()
    assert ctx.root_anchor == project.resolve()


def test_infer_dlc_project_from_labeled_folder_rejects_unusable_input():
    with pytest.raises(ValueError, match="Could not normalize labeled folder"):
        infer_dlc_project_from_labeled_folder(object())
