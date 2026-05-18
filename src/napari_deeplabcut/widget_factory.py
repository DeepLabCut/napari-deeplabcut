# src/napari_deeplabcut/widget_factory.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._widgets import KeypointControls
    from .tracking._widgets import TrackingControls


def get_existing_keypoint_controls(viewer) -> KeypointControls | None:
    from ._widgets import KeypointControls

    wdg = KeypointControls.get_existing(viewer)
    if wdg is None:
        return None
    if not KeypointControls.is_docked(viewer, wdg):
        return None
    return wdg


def get_or_create_keypoint_controls(viewer) -> KeypointControls:
    from ._widgets import KeypointControls

    return KeypointControls(viewer)


def get_existing_tracking_controls(viewer) -> TrackingControls | None:
    from .tracking._widgets import TrackingControls

    return TrackingControls.get_existing(viewer)


def get_or_create_tracking_controls(viewer) -> TrackingControls:
    from .tracking._widgets import TrackingControls

    return TrackingControls(viewer)
