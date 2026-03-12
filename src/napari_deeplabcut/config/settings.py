from qtpy.QtCore import QSettings

_OVERWRITE_CONFIRM_ENABLED_KEY = "napari_deeplabcut/overwrite/confirm_enabled"


def get_overwrite_confirmation_enabled() -> bool:
    """Return whether overwrite confirmation dialogs are enabled."""
    settings = QSettings()
    return settings.value(_OVERWRITE_CONFIRM_ENABLED_KEY, True, type=bool)


def set_overwrite_confirmation_enabled(enabled: bool) -> None:
    """Persist whether overwrite confirmation dialogs are enabled."""
    settings = QSettings()
    settings.setValue(_OVERWRITE_CONFIRM_ENABLED_KEY, bool(enabled))
