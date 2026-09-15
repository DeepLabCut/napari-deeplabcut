"""Guard the test suite that ships inside the built distribution.

The expected list is hard-coded rather than discovered because discovery cannot help
here: a subpackage missing from the wheel is missing from disk too.
"""

# src/napari_deeplabcut/_tests/test_packaging.py

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXPECTED_SUBPACKAGES = (
    "compat",
    "config",
    "core",
    "core.io",
    "core.layer_manager",
    "e2e",
    "tracking",
    "tracking.ui",
    "ui",
    "utils",
)


@pytest.mark.parametrize("subpackage", EXPECTED_SUBPACKAGES)
def test_test_subpackage_is_importable(subpackage: str) -> None:
    """Fail if a test subpackage was excluded from the installed distribution."""
    name = f"napari_deeplabcut._tests.{subpackage}"
    assert importlib.util.find_spec(name) is not None, (
        f"{name} is not importable. If this fails in the wheel job, the subpackage did "
        f"not reach the built distribution; check the sdist and wheel contents."
    )


@pytest.mark.parametrize("subpackage", EXPECTED_SUBPACKAGES)
def test_test_subpackage_is_a_regular_package(subpackage: str) -> None:
    """Fail if a test subpackage lost its ``__init__.py`` and became a namespace."""
    name = f"napari_deeplabcut._tests.{subpackage}"
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin is not None, (
        f"{name} resolves to an implicit namespace package, so it has no __init__.py. "
        f"Add one: without it pytest imports its modules under bare top-level names, "
        f"which collide with same-named modules elsewhere in the suite."
    )


def test_expected_subpackages_are_not_stale() -> None:
    """Fail if a test subpackage exists on disk but is not covered by the list above."""
    here = Path(__file__).parent
    on_disk = {
        ".".join(path.relative_to(here).parts)
        for path in here.rglob("*")
        if path.is_dir() and not path.name.startswith((".", "__"))
    }

    unlisted = on_disk - set(EXPECTED_SUBPACKAGES)
    assert not unlisted, (
        f"Test subpackages not listed in EXPECTED_SUBPACKAGES: {sorted(unlisted)}. "
        f"Add them, and give each an __init__.py to keep it a regular package."
    )
