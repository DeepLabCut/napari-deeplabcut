from qtpy.QtCore import QSettings

DEFAULT_SINGLE_ANIMAL_CMAP = "rainbow"
DEFAULT_MULTI_ANIMAL_INDIVIDUAL_CMAP = "Set3"

_OVERWRITE_CONFIRM_ENABLED_KEY = "napari_deeplabcut/overwrite/confirm_enabled"
AUTO_OPEN_KEYPOINT_CONTROLS_KEY = "napari_deeplabcut/ui/auto_open_keypoint_controls"


def get_overwrite_confirmation_enabled() -> bool:
    """Return whether overwrite confirmation dialogs are enabled."""
    settings = QSettings()
    return settings.value(_OVERWRITE_CONFIRM_ENABLED_KEY, True, type=bool)


def set_overwrite_confirmation_enabled(enabled: bool) -> None:
    """Persist whether overwrite confirmation dialogs are enabled."""
    settings = QSettings()
    settings.setValue(_OVERWRITE_CONFIRM_ENABLED_KEY, bool(enabled))


def get_auto_open_keypoint_controls() -> bool:
    """Return whether keypoint controls should be auto-opened."""
    settings = QSettings()
    return settings.value(AUTO_OPEN_KEYPOINT_CONTROLS_KEY, True, type=bool)


def set_auto_open_keypoint_controls(enabled: bool) -> None:
    """Persist whether keypoint controls should be auto-opened."""
    settings = QSettings()
    settings.setValue(AUTO_OPEN_KEYPOINT_CONTROLS_KEY, bool(enabled))
