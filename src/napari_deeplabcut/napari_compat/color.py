# src/napari_deeplabcut/napari_compat/color.py
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_color_manager_guess_continuous() -> None:
    """
    Napari-compat patch: override guess_continuous so that only floating
    dtypes are treated as continuous.

    Why:
    Napari historically guessed "continuous" for floats OR integer arrays
    with many unique values, which is problematic for categorical features
    like body-part labels.

    Compatibility:
    - Supports old napari calls: guess_continuous(color_map)
    - Supports newer napari calls: guess_continuous(color_map, feature_name)
    - Gracefully no-ops if napari internals move again
    """
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        logger.debug("Skipping color_manager patch (numpy import failed): %r", e)
        return

    targets = []

    try:
        from napari.layers.utils import color_manager

        targets.append(("napari.layers.utils.color_manager", color_manager))
    except Exception as e:  # pragma: no cover
        logger.debug("Could not import napari.layers.utils.color_manager: %r", e)

    try:
        from napari.layers.utils import color_manager_utils

        targets.append(("napari.layers.utils.color_manager_utils", color_manager_utils))
    except Exception as e:  # pragma: no cover
        logger.debug("Could not import napari.layers.utils.color_manager_utils: %r", e)

    if not targets:
        logger.debug("Skipping color_manager patch (no compatible napari module found)")
        return

    def guess_continuous(color_map, feature_name=None, *args, **kwargs):
        """
        Treat only floating dtypes as continuous.

        Parameters
        ----------
        color_map : array-like
            Feature/property values.
        feature_name : str | None
            Accepted for compatibility with newer napari versions.
        *args, **kwargs
            Ignored for forward compatibility.
        """
        try:
            dtype = getattr(color_map, "dtype", None)
            if dtype is None:
                color_map = np.asarray(color_map)
                dtype = color_map.dtype
            return np.issubdtype(dtype, np.floating)
        except Exception:  # pragma: no cover
            return False

    for module_name, module in targets:
        try:
            module.guess_continuous = guess_continuous
            logger.debug("Patched %s.guess_continuous", module_name)
        except Exception as e:  # pragma: no cover
            logger.debug("Skipping patch for %s (assignment failed): %r", module_name, e)
