from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

from napari.settings import get_settings
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QToolButton

_PACKAGE = "napari_deeplabcut"


@lru_cache(maxsize=1)
def _pkg_root():
    return files(_PACKAGE)


def _assets_traversable():
    return _pkg_root() / "assets"


def _normalize_theme(theme: str | None = None) -> str:
    """
    Return a normalized launch-time theme choice.

    Rules
    -----
    - explicit `theme` parameter wins
    - otherwise use napari settings
    - unknown values fall back to 'dark'
    """
    if theme is None:
        try:
            theme = str(get_settings().appearance.theme)
        except Exception:
            theme = "dark"

    theme = str(theme).strip().lower()
    if theme in {"dark", "light", "system"}:
        return theme
    return "dark"


def _help_info_icon_name(theme: str | None = None) -> str:
    """
    Pick the SVG file for the current theme.

    Convention
    ----------
    - light  -> black/dark icon
    - dark   -> light/gray icon
    - system -> default to light/gray icon unless caller passes an override
    """
    resolved = _normalize_theme(theme)
    if resolved == "light":
        return "info_icon_dark.svg"
    return "info_icon_light.svg"


def apply_help_info_icon(
    button: QToolButton,
    *,
    scale: float = 1.2,
    theme: str | None = None,
    tooltip: str = "Hover for more details",
) -> None:
    """
    Apply the local SVG help/info icon to a button.

    This intentionally preserves the same sizing pattern you had before:
    - set icon
    - then scale the existing iconSize directly
    """
    icon_path = _assets_traversable() / _help_info_icon_name(theme)
    button.setIcon(QIcon(str(icon_path)))
    button.setIconSize(button.iconSize() * scale)
    button.setCursor(Qt.WhatsThisCursor)
    button.setAutoRaise(True)
    button.setToolTip(tooltip)
