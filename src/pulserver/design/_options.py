"""Options of a plugin: the module constants a parameter file sets when the plugin is loaded."""

from __future__ import annotations

import contextlib
import contextvars
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

#: The scalar types an option holds; a TOML file states each of them.
OptionValue = bool | int | float | str


class OptionError(ValueError):
    """An option a parameter file sets that the plugin does not declare or take."""


@dataclass
class _Loading:
    given: Mapping[str, Any]
    resolved: dict[str, OptionValue] = field(default_factory=dict)


_loading: contextvars.ContextVar[_Loading | None] = contextvars.ContextVar(
    "pulserver_plugin_options", default=None
)


def option(
    name: str, default: OptionValue, choices: tuple[OptionValue, ...] = ()
) -> OptionValue:
    """Return the value of a plugin option: the parameter file's where it sets ``name``, else ``default``.

    A plugin assigns it to a module constant, which its class body reads, so
    the option shapes the sequence function and the protocol the plugin
    declares. Outside :func:`~pulserver.design.load_plugin` it returns
    ``default``.

    Parameters
    ----------
    name : str
        The option's name in a parameter file.
    default : bool, int, float or str
        Its value where the file does not set it. Its type is the option's:
        an int is taken for a float, and nothing else is converted.
    choices : tuple, default=()
        The values it takes, ``default`` among them; any value of its type
        when empty.

    Raises
    ------
    ValueError
        If ``default`` is not among ``choices``.
    OptionError
        If ``name`` is declared twice in one plugin, or the file's value is
        not of the option's type or not among ``choices``.
    """
    if choices and default not in choices:
        raise ValueError(f"option {name}: default {default!r} is not one of {choices}")
    loading = _loading.get()
    if loading is None:
        return default
    if name in loading.resolved:
        raise OptionError(f"option {name} is declared twice")
    value = loading.given.get(name, default)
    if not _same_type(value, default):
        raise OptionError(f"option {name} is a {type(default).__name__}, not {value!r}")
    if choices and value not in choices:
        raise OptionError(f"option {name} is one of {choices}, not {value!r}")
    if isinstance(default, float):
        value = float(value)
    loading.resolved[name] = value
    return value


def _same_type(value: Any, default: OptionValue) -> bool:
    if isinstance(default, bool) or isinstance(value, bool):
        return isinstance(value, bool) and isinstance(default, bool)
    if isinstance(default, float):
        return isinstance(value, int | float)
    return isinstance(value, type(default))


@contextlib.contextmanager
def loading_options(given: Mapping[str, Any]) -> Iterator[dict[str, OptionValue]]:
    """Make :func:`option` read ``given`` while a plugin module executes; yield the options it resolves.

    Raises
    ------
    OptionError
        On leaving, if ``given`` names an option the module does not declare.
    """
    loading = _Loading(given)
    token = _loading.set(loading)
    try:
        yield loading.resolved
    finally:
        _loading.reset(token)
    unknown = sorted(set(given) - set(loading.resolved))
    if unknown:
        declared = sorted(loading.resolved) or "none"
        raise OptionError(
            f"the plugin does not declare the options {unknown}; it declares {declared}"
        )


def read_options(path: Path | str) -> dict[str, OptionValue]:
    """Return the options a parameter file sets, a TOML table of option names and values.

    Raises
    ------
    OptionError
        If the file is not TOML, or a value is not a bool, int, float or string.
    """
    try:
        table = tomllib.loads(Path(path).read_text())
    except tomllib.TOMLDecodeError as error:
        raise OptionError(f"not TOML: {error}") from None
    for name, value in table.items():
        if not isinstance(value, OptionValue):
            raise OptionError(
                f"option {name} is {value!r}; an option is a bool, an "
                "int, a float or a string"
            )
    return table
