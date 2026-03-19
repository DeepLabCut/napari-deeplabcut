from __future__ import annotations

import functools
import inspect
import os
import warnings
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


class NapariDLCDeprecationWarning(DeprecationWarning):
    """Package-specific deprecation warning."""

    pass


class DeprecationMode(str, Enum):
    WARN = "warn"
    ERROR = "error"
    IGNORE = "ignore"
    PASS = "pass"  # synonym for "do nothing, still run"


@dataclass(frozen=True)
class DeprecationPolicy:
    mode: DeprecationMode = DeprecationMode.WARN
    category: type[Warning] = NapariDLCDeprecationWarning
    stacklevel: int = 2
    envvar: str = "NAPARI_DLC_DEPRECATIONS"


_DEFAULT_POLICY = DeprecationPolicy()
_policy_override: ContextVar[DeprecationMode | None] = ContextVar("_policy_override", default=None)


def _parse_mode(value: str | None) -> DeprecationMode | None:
    if value is None:
        return None
    try:
        return DeprecationMode(value.strip().lower())
    except Exception:
        return None


def _effective_mode(
    *,
    explicit_mode: DeprecationMode | None,
    policy: DeprecationPolicy,
) -> DeprecationMode:
    if explicit_mode is not None:
        return explicit_mode

    ctx = _policy_override.get()
    if ctx is not None:
        return ctx

    env = _parse_mode(os.getenv(policy.envvar))
    if env is not None:
        return env

    return policy.mode


def build_deprecation_message(
    *,
    name: str,
    since: str | None = None,
    remove_in: str | None = None,
    replacement: str | None = None,
    details: str | None = None,
    kind: str = "API",
) -> str:
    parts: list[str] = [f"{kind} {name!r} is deprecated"]

    if since:
        parts[-1] += f" since {since}"
    if remove_in:
        parts[-1] += f" and will be removed in {remove_in}"
    parts[-1] += "."

    if replacement:
        parts.append(f"Use {replacement!r} instead.")
    if details:
        parts.append(details)

    return " ".join(parts)


def _emit_deprecation(
    message: str,
    *,
    mode: DeprecationMode,
    category: type[Warning],
    stacklevel: int,
) -> None:
    if mode in (DeprecationMode.PASS, DeprecationMode.IGNORE):
        return
    if mode is DeprecationMode.ERROR:
        raise RuntimeError(message)
    warnings.warn(message, category=category, stacklevel=stacklevel)


def deprecated(
    *,
    since: str | None = None,
    remove_in: str | None = None,
    replacement: str | None = None,
    details: str | None = None,
    mode: DeprecationMode | None = None,
    category: type[Warning] | None = None,
    stacklevel: int | None = None,
    policy: DeprecationPolicy = _DEFAULT_POLICY,
    add_to_docstring: bool = True,
) -> Callable[[F], F]:
    """
    Mark a function or class as deprecated.

    Behavior can be controlled globally via policy/env/context, or per-usage via `mode=...`.

    Args:
        since: Version when the deprecation was introduced.
        remove_in: Version when the deprecated API will be removed.
        replacement: Optional API to use instead of the deprecated one.
        details: Additional details to include in the deprecation message.
        mode: Optional override for this deprecation's behavior.
        category: Optional warning category to use.
        stacklevel: Optional stack level for the warning.
        policy: Deprecation policy to use.
        add_to_docstring: Whether to add the deprecation message to the docstring.
    """
    warning_category = category or policy.category
    warning_stacklevel = stacklevel or policy.stacklevel

    def decorator(obj: F) -> F:
        obj_name = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
        kind = "class" if inspect.isclass(obj) else "function"
        message = build_deprecation_message(
            name=obj_name,
            since=since,
            remove_in=remove_in,
            replacement=replacement,
            details=details,
            kind=kind,
        )

        if add_to_docstring:
            note = f"\n\n.. deprecated:: {since or 'unknown'}\n   {message}\n"
            obj.__doc__ = (obj.__doc__ or "") + note

        if inspect.isclass(obj):
            cls = cast(type[Any], obj)
            original_init = cls.__init__

            @functools.wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                effective = _effective_mode(explicit_mode=mode, policy=policy)
                _emit_deprecation(
                    message,
                    mode=effective,
                    category=warning_category,
                    stacklevel=warning_stacklevel,
                )
                original_init(self, *args, **kwargs)

            cls.__init__ = wrapped_init  # type: ignore[method-assign]
            return cast(F, cls)

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            effective = _effective_mode(explicit_mode=mode, policy=policy)
            _emit_deprecation(
                message,
                mode=effective,
                category=warning_category,
                stacklevel=warning_stacklevel,
            )
            return obj(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


class deprecation_mode:
    """
    Context manager to temporarily override deprecation behavior.

    Example:
        with deprecation_mode(DeprecationMode.ERROR):
            old_api()
    """

    def __init__(self, mode: DeprecationMode):
        self.mode = mode
        self._token = None

    def __enter__(self):
        self._token = _policy_override.set(self.mode)
        return self

    def __exit__(self, exc_type, exc, tb):
        assert self._token is not None
        _policy_override.reset(self._token)
        return False
