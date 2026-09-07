"""
``[tool.setuptools.packages.find]`` discovers implicit namespaces by default, but explicit
``__init__.py`` files ensure these test directories remain regular packages. The installed-wheel
check below fails if any expected test subpackage is absent.

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
        f"{name} is not importable. If this fails in the wheel job, the directory is "
        f"probably missing __init__.py and was excluded from the distribution."
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
        f"Add them, and give each an __init__.py so it ships in the wheel."
    )
