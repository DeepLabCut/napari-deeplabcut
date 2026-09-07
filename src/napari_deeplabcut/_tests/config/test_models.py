# src/napari_deeplabcut/_tests/config/test_models.py

# src/napari_deeplabcut/_tests/config/test_models.py

from __future__ import annotations

import pytest

from napari_deeplabcut.config.models import DLCHeaderModel


def assert_categorical_header_invariant(header: DLCHeaderModel) -> None:
    """Assert that keypoint identity values remain categorical strings.

    ``individuals`` and ``bodyparts`` become the ``id`` and ``label``
    properties of napari Points layers. They must remain strings,
    even when users provide numeric values to prevent numeric identity
    columns from being interpreted as continuous color values downstream,
    which would render the colormap as a gradient instead of discrete colors.
    """
    assert all(isinstance(value, str) for column in header.columns for value in column)

    if header.names is not None:
        assert all(isinstance(name, str) for name in header.names)

    assert header.scorer is None or isinstance(header.scorer, str)
    assert all(isinstance(value, str) for value in header.scorers)
    assert all(isinstance(value, str) for value in header.individuals)
    assert all(isinstance(value, str) for value in header.bodyparts)
    assert all(isinstance(value, str) for value in header.coords)

    assert all(
        isinstance(individual, str) and isinstance(bodypart, str)
        for individual, bodypart in header.form_individual_bodypart_pairs()
    )


def test_direct_header_input_normalizes_identity_values_to_strings() -> None:
    """Protect the invariant for headers created outside ``from_config``."""
    header = DLCHeaderModel(
        columns=[
            (123, 1, 10, "x"),
            (123, 1, 10, "y"),
            (123, 2, 20, "x"),
            (123, 2, 20, "y"),
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )

    assert header.columns == [
        ("123", "1", "10", "x"),
        ("123", "1", "10", "y"),
        ("123", "2", "20", "x"),
        ("123", "2", "20", "y"),
    ]
    assert header.individuals == ["1", "2"]
    assert header.bodyparts == ["10", "20"]

    assert_categorical_header_invariant(header)


def test_header_level_names_are_normalized_to_strings() -> None:
    """Keep the complete portable header representation string-based."""
    header = DLCHeaderModel(
        columns=[("scorer", "nose", "x")],
        names=[1, 2, 3],
    )

    assert header.names == ["1", "2", "3"]
    assert_categorical_header_invariant(header)


def test_single_animal_config_preserves_categorical_identity_values() -> None:
    """Protect label coloring for numeric bodyparts in single-animal configs."""
    header = DLCHeaderModel.from_config(
        {
            "multianimalproject": False,
            "scorer": 123,
            "bodyparts": [1, "tail", 3.5],
        }
    )

    assert header.scorer == "123"
    assert header.individuals == [""]
    assert header.bodyparts == ["1", "tail", "3.5"]
    assert header.coords == ["x", "y"]

    assert_categorical_header_invariant(header)


def test_multi_animal_config_preserves_categorical_identity_values() -> None:
    """Protect id and label coloring for numeric multi-animal config values."""
    header = DLCHeaderModel.from_config(
        {
            "multianimalproject": True,
            "scorer": 123,
            "individuals": [1, 2],
            "multianimalbodyparts": [10, "tail"],
            "uniquebodyparts": [99],
        }
    )

    assert header.scorer == "123"
    assert header.individuals == ["1", "2", "single"]
    assert header.bodyparts == ["10", "tail", "99"]
    assert header.coords == ["x", "y"]

    assert_categorical_header_invariant(header)


@pytest.mark.parametrize(
    ("columns", "names"),
    [
        (
            [(123, 10, "x")],
            ["scorer", "bodyparts", "coords"],
        ),
        (
            [[123, 10, "x"]],
            ["scorer", "bodyparts", "coords"],
        ),
        (
            [(123, 1, 10, "x")],
            ["scorer", "individuals", "bodyparts", "coords"],
        ),
        (
            [[123, 1, 10, "x"]],
            ["scorer", "individuals", "bodyparts", "coords"],
        ),
    ],
)
def test_supported_direct_header_representations_preserve_categorical_values(
    columns,
    names,
) -> None:
    """Protect string identity values across all valid direct input forms."""
    header = DLCHeaderModel(columns=columns, names=names)

    assert_categorical_header_invariant(header)
