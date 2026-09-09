import configparser
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
TOX_INI = REPO_ROOT / "tox.ini"

# Both files only exist in a source checkout; an installed-wheel test run has
# nothing to compare.
pytestmark = pytest.mark.skipif(
    not (PYPROJECT.is_file() and TOX_INI.is_file()),
    reason="requires a source checkout",
)


def test_napari_floor_matches_declared_lower_bound():
    """Keep the tox floor env aligned with the napari lower bound in pyproject."""
    declared_floor = re.search(
        r"""["']napari\s*(?:\[[^\]]*\])?\s*>=\s*([0-9][^,"'\s]*)""",
        PYPROJECT.read_text(encoding="utf-8"),
    )
    assert declared_floor is not None, f"Expected a napari lower bound in {PYPROJECT.name}"

    # interpolation=None: tox commands are free to contain '%'.
    config = configparser.ConfigParser(interpolation=None)
    config.read(TOX_INI, encoding="utf-8")
    commands_pre = config["testenv:napari-floor"]["commands_pre"]

    pinned_floor = re.search(
        r"""pip install ["']napari==([^"']+)["']""",
        commands_pre,
    )
    assert pinned_floor is not None, "Expected the napari-floor environment to install an exact napari version"

    assert pinned_floor.group(1) == declared_floor.group(1), (
        "The napari floor in tox.ini does not match the lower bound declared in "
        f"pyproject.toml: {pinned_floor.group(1)} != {declared_floor.group(1)}"
    )
