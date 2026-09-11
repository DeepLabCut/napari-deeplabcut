# src/napari_deeplabcut/napari_compat/proxy.py
"""Unwrapping of napari's ``PublicOnlyProxy``.

napari wraps the viewer it hands to plugin dock widgets in
``napari.utils._proxies.PublicOnlyProxy``, and re-wraps recursively: any
napari-typed object reached through it comes back wrapped. The proxy emits a
``FutureWarning`` on every single-underscore attribute get *or* set, and napari's
own source already carries the commented-out ``raise AttributeError`` meant to
replace that warning.

``unwrap`` is deliberately the single place that opts out, so there is one thing
to audit when napari changes its internals.
"""

from __future__ import annotations

from typing import Any

__all__ = ["unwrap"]


def unwrap(obj: Any) -> Any:
    """Return the object behind any proxy wrapper, following nested wrappers.

    Handles both the ``__wrapped__`` convention used by ``wrapt``/``PublicOnlyProxy``
    and the ``_obj`` convention, tolerates objects that are not proxies at all, and
    guards against reference cycles.

    ``__wrapped__`` is checked first and is a dunder, which ``PublicOnlyProxy`` does
    not treat as private, so reading it raises no warning. ``_obj`` is only consulted
    once ``__wrapped__`` is absent, i.e. on an object that is not a ``PublicOnlyProxy``,
    so that lookup cannot warn either.
    """
    current = obj
    seen: set[int] = set()

    while True:
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is None:
            wrapped = getattr(current, "_obj", None)

        if wrapped is None or wrapped is current or id(wrapped) in seen:
            return current

        seen.add(id(current))
        current = wrapped
