import os


def _get_int_env(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    if minimum is not None:
        value = max(value, minimum)

    return value
