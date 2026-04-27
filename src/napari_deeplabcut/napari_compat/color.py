from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_color_manager_guess_continuous() -> None:
    """
    Napari-compat patch: override color_manager.guess_continuous.
    By default :
    The property is guessed as continuous if it is a float or contains over 16 elements.
    We override this to guess continuous only if it is a float,
    to avoid issues with categorical properties with many categories (e.g. body parts).

    - Uses try/except around private imports.
    - If napari internals change, we log and no-op.
    """
    try:
        import numpy as np
        from napari.layers.utils import color_manager
    except Exception as e:  # pragma: no cover
        logger.debug("Skipping color_manager patch (napari import failed): %r", e)
        return

    def guess_continuous(property_):
        try:
            return issubclass(property_.dtype.type, np.floating)
        except Exception:  # pragma: no cover
            return False

    try:
        color_manager.guess_continuous = guess_continuous
    except Exception as e:  # pragma: no cover
        logger.debug("Skipping color_manager patch (assignment failed): %r", e)
        return
