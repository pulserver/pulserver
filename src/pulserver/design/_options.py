"""Options of a plugin: module constants a user entry of the protocol sets when the plugin is loaded."""

from __future__ import annotations

import contextlib
import contextvars
import inspect
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..protocol._keys import NUM_USER_ENTRIES
from ..protocol._schema import MAX_DROPDOWN_OPTIONS

#: The scalar types an option holds.
OptionValue = bool | int | float | str

#: The module global holding the options a plugin module declares, by name.
_DECLARED = "__pulserver_options__"


class OptionError(ValueError):
    """An option a plugin is loaded with that it does not declare or take, or a declaration it cannot list."""


@dataclass(frozen=True)
class Declared:
    """An option as a plugin module declares it, and the value the module was loaded with."""

    name: str
    choices: tuple[OptionValue, ...]
    default: OptionValue
    user: int
    label: str
    value: OptionValue


@dataclass
class _Loading:
    given: Mapping[str, Any]
    resolved: dict[str, OptionValue] = field(default_factory=dict)


_loading: contextvars.ContextVar[_Loading | None] = contextvars.ContextVar(
    "pulserver_plugin_options", default=None
)


def option(
    name: str,
    default: OptionValue,
    choices: tuple[OptionValue, ...] = (),
    *,
    user: int,
    label: str,
) -> OptionValue:
    """Declare a plugin option at user entry ``user`` and return the value the plugin is loaded with.

    A plugin assigns it to a module constant at module level, before its
    :class:`~pulserver.design.SequencePlugin` subclass, whose class body reads
    it, so the option shapes the sequence function and the protocol the plugin
    declares. The subclass lists the option as ``user<user>_value``, the index
    of its value among ``choices`` (0 or 1 for a flag), named ``label`` and
    the code of each choice in ``user<user>_name``. A design call loads the
    plugin at the options its request holds in those entries; outside
    :func:`~pulserver.design.load_plugin` the value is ``default``.

    Parameters
    ----------
    name : str
        The option's name, which a design's manifest records it by.
    default : bool, int, float or str
        Its value where the request does not state it.
    choices : tuple, default=()
        Its values, ``default`` among them, at most five; a flag, whose default
        is a bool, takes none and chooses between ``False`` and ``True``.
    user : int
        The user entry it is listed at.
    label : str
        The option's name on the console.

    Raises
    ------
    OptionError
        If a flag is given choices, any other option is not, ``default`` is
        not among them, there are more than five or ``user`` is not a user
        entry; or, while a plugin loads, if ``name`` or ``user`` is declared
        twice or the value it is loaded with is not among the choices.
    """
    if isinstance(default, bool):
        if choices:
            raise OptionError(f"option {name}: a flag takes no choices")
        choices = (False, True)
    elif not choices:
        raise OptionError(
            f"option {name}: an option that is not a flag names its choices"
        )
    if default not in choices:
        raise OptionError(f"option {name}: default {default!r} is not one of {choices}")
    if len(choices) > MAX_DROPDOWN_OPTIONS:
        raise OptionError(
            f"option {name}: {len(choices)} choices; an option offers at most "
            f"{MAX_DROPDOWN_OPTIONS}"
        )
    if not 0 <= user < NUM_USER_ENTRIES:
        raise OptionError(
            f"option {name}: user{user} is not a user entry, user0 to "
            f"user{NUM_USER_ENTRIES - 1}"
        )
    declared: dict[str, Declared] = inspect.currentframe().f_back.f_globals.setdefault(
        _DECLARED, {}
    )
    value = default
    loading = _loading.get()
    if loading is not None:
        if name in loading.resolved:
            raise OptionError(f"option {name} is declared twice")
        taken = next(
            (d.name for d in declared.values() if d.user == user and d.name != name),
            None,
        )
        if taken is not None:
            raise OptionError(f"options {taken} and {name} are both at user{user}")
        value = loading.given.get(name, default)
        if value not in choices or type(value) is not type(
            choices[choices.index(value)]
        ):
            raise OptionError(f"option {name} is one of {choices}, not {value!r}")
        value = choices[choices.index(value)]
        loading.resolved[name] = value
    declared[name] = Declared(name, tuple(choices), default, user, label, value)
    return value


def declared_options(module: Mapping[str, Any]) -> dict[str, Declared]:
    """Return the options the module whose globals are ``module`` declares, by name."""
    return dict(module.get(_DECLARED, {}))


@contextlib.contextmanager
def loading_options(given: Mapping[str, Any]) -> Iterator[dict[str, OptionValue]]:
    """Make :func:`option` take ``given`` while a plugin module executes; yield the options it resolves.

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
