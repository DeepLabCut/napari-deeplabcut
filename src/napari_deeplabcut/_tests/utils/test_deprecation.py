from __future__ import annotations

import pytest

from napari_deeplabcut.utils.deprecations import (
    DeprecationMode,
    NapariDLCDeprecationWarning,
    deprecated,
    deprecation_mode,
)


def test_deprecated_function_warns():
    @deprecated(
        since="0.9.0",
        remove_in="1.1.0",
        replacement="new_func",
    )
    def old_func():
        return 123

    with pytest.warns(NapariDLCDeprecationWarning, match="new_func"):
        assert old_func() == 123


def test_deprecated_function_can_error_via_context():
    @deprecated(since="0.9.0", remove_in="1.1.0")
    def old_func():
        return 123

    with pytest.raises(RuntimeError, match="deprecated"):
        with deprecation_mode(DeprecationMode.ERROR):
            old_func()


def test_deprecated_class_warns():
    @deprecated(since="0.9.0", replacement="NewThing")
    class OldThing:
        def __init__(self, x):
            self.x = x

    with pytest.warns(NapariDLCDeprecationWarning, match="NewThing"):
        obj = OldThing(5)

    assert obj.x == 5
